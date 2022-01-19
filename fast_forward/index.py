import abc
import time
import pickle
import logging
from enum import Enum
from pathlib import Path
from queue import PriorityQueue
from collections import OrderedDict, defaultdict
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Set, Tuple, Union

import numpy as np
from tqdm import tqdm
from scipy.spatial.distance import cosine

from fast_forward.ranking import Ranking
from fast_forward.encoder import QueryEncoder
from fast_forward.util import interpolate


LOGGER = logging.getLogger(__name__)


class Mode(Enum):
    """Enum used to set the retrieval mode of an index."""

    PASSAGE = 1
    MAXP = 2
    FIRSTP = 3
    AVEP = 4


class Index(abc.ABC):
    """Abstract base class for Fast-Forward indexes."""

    def __init__(
        self,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """
        super().__init__()
        self.encoder = encoder
        self.mode = mode
        self._encoder_batch_size = encoder_batch_size

    def encode(self, queries: Sequence[str]) -> List[np.ndarray]:
        """Encode queries.

        Args:
            queries (Sequence[str]): The queries to encode.

        Raises:
            RuntimeError: When no query encoder exists.

        Returns:
            List[np.ndarray]: The query representations.
        """
        if self._encoder is None:
            raise RuntimeError("This index does not have a query encoder.")

        result = []
        for i in range(0, len(queries), self._encoder_batch_size):
            batch = queries[i : i + self._encoder_batch_size]
            result.extend(self._encoder.encode(batch))
        return result

    @property
    def encoder(self) -> QueryEncoder:
        """Return the query encoder.

        Returns:
            QueryEncoder: The encoder.
        """
        return self._encoder

    @encoder.setter
    def encoder(self, encoder: QueryEncoder) -> None:
        """Set the query encoder.

        Args:
            encoder (QueryEncoder): The encoder.
        """
        assert encoder is None or isinstance(encoder, QueryEncoder)
        self._encoder = encoder

    @property
    def mode(self) -> Mode:
        """Return the indexing mode.

        Returns:
            Mode: The mode.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: Mode) -> None:
        """Set the indexing mode.

        Args:
            mode (Mode): The indexing mode.
        """
        assert isinstance(mode, Mode)
        self._mode = mode

    @property
    def doc_ids(self) -> Set[str]:
        """Return all unique document IDs.

        Returns:
            Set[str]: The document IDs.
        """
        return self._get_doc_ids()

    @abc.abstractmethod
    def _get_doc_ids(self) -> Set[str]:
        """Return all unique document IDs.

        Returns:
            Set[str]: The document IDs.
        """
        pass

    @property
    def psg_ids(self) -> Set[str]:
        """Return all unique passage IDs.

        Returns:
            Set[str]: The passage IDs.
        """
        return self._get_psg_ids()

    @abc.abstractmethod
    def _get_psg_ids(self) -> Set[str]:
        """Return all unique passage IDs.

        Returns:
            Set[str]: The passage IDs.
        """
        pass

    @abc.abstractmethod
    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        """Add vector representations and corresponding IDs to the index. Each vector is guaranteed to
        have either a document or passage ID associated.

        Args:
            vectors (np.ndarray): The representations, shape (num_vectors, dim).
            doc_ids (Sequence[Union[str, None]]): The corresponding document IDs (may be duplicate).
            psg_ids (Sequence[Union[str, None]]): The corresponding passage IDs (must be unique).
        """
        pass

    def add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[str] = None,
        psg_ids: Sequence[str] = None,
    ) -> None:
        """Add vector representations and corresponding IDs to the index. Only one of "doc_ids" and "psg_ids"
        may be None. For performance reasons, this function should not be called frequently with few items.

        Args:
            vectors (np.ndarray): The representations, shape (num_vectors, dim).
            doc_id (Sequence[str], optional): The corresponding document IDs (may be duplicate). Defaults to None.
            psg_id (Sequence[str], optional): The corresponding passage IDs (must be unique). Defaults to None.

        Raises:
            ValueError: When there are no document IDs and no passage IDs.
        """
        if doc_ids is None and psg_ids is None:
            raise ValueError(
                'At least one of "doc_ids" and "psg_ids" must be provided.'
            )

        num_vectors = vectors.shape[0]
        if num_vectors < 100:
            LOGGER.warning(
                'calling "Index.add()" repeatedly with few vectors may be slow'
            )
        if doc_ids is None:
            doc_ids = [None] * num_vectors
        if psg_ids is None:
            psg_ids = [None] * num_vectors

        assert num_vectors == len(doc_ids) == len(psg_ids)
        self._add(vectors, doc_ids, psg_ids)

    @abc.abstractmethod
    def _get_vectors(
        self, ids: Iterable[str], mode: Mode
    ) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        """Return:
            * A single array containing all vectors necessary to compute the scores for each document/passage.
            * For each document/passage (in the same order as the IDs), either
                * a list of integers (MAXP, AVEP),
                * a single integer (FIRSTP, PASSAGE),
                * None (the document/passage is not indexed and has no vector)

        The integers will be used to get the corresponding representations from the array.
        The output of this function depends on the current mode.

        Args:
            ids (Iterable[str]): The document/passage IDs to get the representations for.
            mode (Mode): The index mode.

        Returns:
            Tuple[np.ndarray, List[Union[List[int], int, None]]]: The vectors and corresponding indices.
        """
        pass

    def _compute_scores(self, q_rep: np.ndarray, ids: Iterable[str]) -> Iterator[float]:
        """Compute scores based on the current mode.

        Args:
            q_rep (np.ndarray): Query representation.
            ids (Iterable[str]): Document/passage IDs.

        Yields:
            float: The scores, preserving the order of the IDs.
        """
        vectors, id_indices = self._get_vectors(ids, self.mode)
        all_scores = np.dot(q_rep, vectors.T)

        for ind in id_indices:
            if ind is None:
                yield None
            else:
                if self.mode == Mode.MAXP:
                    yield np.max(all_scores[ind])
                elif self.mode == Mode.AVEP:
                    yield np.average(all_scores[ind])
                elif self.mode in (Mode.FIRSTP, Mode.PASSAGE):
                    yield all_scores[ind]

    def _early_stopping(
        self,
        ids: Iterable[str],
        dense_scores: Iterable[float],
        sparse_scores: Iterable[float],
        alpha: float,
        cutoff: int,
    ) -> Dict[str, float]:
        """Interpolate scores with early stopping.

        Args:
            ids (Iterable[str]): Document/passage IDs.
            dense_scores (Iterable[float]): Corresponding dense scores.
            sparse_scores (Iterable[float]): Corresponding sparse scores.
            alpha (float): Interpolation parameter.
            cutoff (int): Cut-off depth.

        Returns:
            Dict[str, float]: Document/passage IDs mapped to scores.
        """
        result = {}
        relevant_scores = PriorityQueue(cutoff)
        min_relevant_score = float("-inf")
        max_dense_score = float("-inf")
        for id, dense_score, sparse_score in zip(ids, dense_scores, sparse_scores):
            if relevant_scores.qsize() >= cutoff:

                # check if approximated max possible score is too low to make a difference
                min_relevant_score = relevant_scores.get_nowait()
                max_possible_score = (
                    alpha * sparse_score + (1 - alpha) * max_dense_score
                )

                # early stopping
                if max_possible_score <= min_relevant_score:
                    break

            if dense_score is None:
                LOGGER.warning(f"{id} not indexed, skipping")
                continue

            max_dense_score = max(max_dense_score, dense_score)
            score = alpha * sparse_score + (1 - alpha) * dense_score
            result[id] = score

            # the new score might be ranked higher than the one we removed
            relevant_scores.put_nowait(max(score, min_relevant_score))
        return result

    def get_scores(
        self,
        ranking: Ranking,
        queries: Dict[str, str],
        alpha: Union[float, Iterable[float]] = 0.0,
        cutoff: int = None,
        early_stopping: bool = False,
    ) -> Dict[float, Ranking]:
        """Compute corresponding dense scores for a ranking and interpolate.

        Args:
            ranking (Ranking): The ranking to compute scores for and interpolate with.
            queries (Dict[str, str]): Query IDs mapped to queries.
            alpha (Union[float, Iterable[float]], optional): Interpolation weight(s). Defaults to 0.0.
            cutoff (int, optional): Cut-off depth (documents/passages per query). Defaults to None.
            early_stopping (bool, optional): Whether to use early stopping. Defaults to False.

        Raises:
            ValueError: When the cut-off depth is missing for early stopping.

        Returns:
            Dict[float, Ranking]: Alpha mapped to interpolated scores.
        """
        if isinstance(alpha, float):
            alpha = [alpha]

        if early_stopping and cutoff is None:
            raise ValueError("A cut-off depth is required for early stopping.")

        t0 = time.time()

        # batch encode queries
        q_id_list = list(ranking)
        q_reps = self.encode([queries[q_id] for q_id in q_id_list])

        result = {}
        if not early_stopping:
            # here we can simply compute the dense scores once and interpolate for each alpha
            dense_run = defaultdict(OrderedDict)
            for q_id, q_rep in zip(tqdm(q_id_list), q_reps):
                ids = list(ranking[q_id].keys())
                for id, score in zip(ids, self._compute_scores(q_rep, ids)):
                    if score is None:
                        LOGGER.warning(f"{id} not indexed, skipping")
                    else:
                        dense_run[q_id][id] = score
            for a in alpha:
                result[a] = interpolate(
                    ranking, Ranking(dense_run, sort=False), a, sort=True
                )
                if cutoff is not None:
                    result[a].cut(cutoff)
        else:
            # early stopping requries the ranking to be sorted
            # this should normally be the case anyway
            if not ranking.is_sorted:
                LOGGER.warning("input ranking not sorted. sorting...")
                ranking.sort()

            # since early stopping depends on alpha, we have to run the algorithm more than once
            for a in alpha:
                run = defaultdict(OrderedDict)
                for q_id, q_rep in zip(tqdm(q_id_list), q_reps):
                    ids, sparse_scores = zip(*ranking[q_id].items())
                    dense_scores = self._compute_scores(q_rep, ids)
                    scores = self._early_stopping(
                        ids, dense_scores, sparse_scores, a, cutoff
                    )
                    for id, score in scores.items():
                        run[q_id][id] = score
                result[a] = Ranking(run, sort=True, copy=False)
                result[a].cut(cutoff)

        LOGGER.info(f"computed scores in {time.time() - t0}s")
        return result


class InMemoryIndex(Index):
    """Fast-Forward index that is held in memory."""

    def __init__(
        self,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """
        self._vectors = None
        self._doc_ids = []
        self._psg_ids = []
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        super().__init__(encoder, mode, encoder_batch_size)

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        if self._vectors is None:
            idx = 0
            self._vectors = vectors.copy()
        else:
            idx = self._vectors.shape[0]
            self._vectors = np.append(self._vectors, vectors, axis=0)

        for doc_id, psg_id in zip(doc_ids, psg_ids):
            if doc_id is not None:
                self._doc_id_to_idx[doc_id].append(idx)
            if psg_id is not None:
                assert psg_id not in self._psg_id_to_idx
                self._psg_id_to_idx[psg_id] = idx
            idx += 1

        self._doc_ids.extend(doc_ids)
        self._psg_ids.extend(psg_ids)

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(
        self, ids: Iterable[str], mode: Mode
    ) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        # a list of all vectors to take from the main vector array
        vector_indices = []

        # for each ID, keep a list of indices to get the corresponding vectors from "vector_indices"
        id_indices = []
        i = 0

        if mode in (Mode.MAXP, Mode.AVEP):
            for id in ids:
                if id in self._doc_id_to_idx:
                    doc_indices = self._doc_id_to_idx[id]
                    vector_indices.extend(doc_indices)
                    id_indices.append(list(range(i, i + len(doc_indices))))
                    i += len(doc_indices)
                else:
                    id_indices.append(None)
        elif mode == Mode.FIRSTP:
            for id in ids:
                if id in self._doc_id_to_idx:
                    vector_indices.append(self._doc_id_to_idx[id][0])
                    id_indices.append(i)
                    i += 1
                else:
                    id_indices.append(None)
        elif mode == Mode.PASSAGE:
            for id in ids:
                if id in self._psg_id_to_idx:
                    vector_indices.append(self._psg_id_to_idx[id])
                    id_indices.append(i)
                    i += 1
                else:
                    id_indices.append(None)
        else:
            LOGGER.error(f"invalid mode: {mode}")
        return self._vectors[vector_indices], id_indices

    def save(self, target: Path) -> None:
        """Save the index in a file on disk.

        Args:
            target (Path): Target file to create.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"writing {target}")
        with open(target, "wb") as fp:
            pickle.dump((self._vectors, self._doc_ids, self._psg_ids), fp)

    @classmethod
    def from_disk(
        cls,
        index_file: Path,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> "InMemoryIndex":
        """Read an index from disk.

        Args:
            index_file (Path): The index file.
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.

        Returns:
            InMemoryIndex: The index.
        """
        LOGGER.info(f"reading {index_file}")
        with open(index_file, "rb") as fp:
            vectors, doc_ids, psg_ids = pickle.load(fp)

        index = cls(encoder, mode, encoder_batch_size)
        if vectors is not None:
            index.add(vectors, doc_ids, psg_ids)
        index.mode = mode
        return index


def create_coalesced_index(
    source_index: Index,
    target_index: Index,
    delta: float,
    distance: Callable[[np.ndarray, np.ndarray], float] = cosine,
    buffer_size: int = None,
) -> None:
    """Create a compressed index using sequential coalescing.

    Args:
        source_index (Index): The source index. Should contain multiple vectors for each document.
        target_index (Index): The target index. Must be empty.
        delta (float): The coalescing threshold.
        distance (Callable[[np.ndarray, np.ndarray], float]): The distance function. Defaults to cosine.
        buffer_size (int, optional): Use a buffer instead of adding all vectors at the end. Defaults to None.
    """
    assert len(target_index.doc_ids) == 0
    buffer_size = buffer_size or len(source_index.doc_ids)

    def _coalesce(P):
        P_new = []
        A = []
        A_avg = None
        first_iteration = True
        for v in P:
            if first_iteration:
                first_iteration = False
            elif distance(v, A_avg) >= delta:
                P_new.append(A_avg)
                A = []
            A.append(v)
            A_avg = np.mean(A, axis=0)
        P_new.append(A_avg)
        return P_new

    vectors, doc_ids = [], []
    for doc_id in tqdm(source_index.doc_ids):

        # check if buffer is full
        if len(vectors) == buffer_size:
            target_index.add(np.array(vectors), doc_ids=doc_ids)
            vectors, doc_ids = [], []

        v_old, _ = source_index._get_vectors([doc_id], Mode.MAXP)
        v_new = _coalesce(v_old)
        vectors.extend(v_new)
        doc_ids.extend([doc_id] * len(v_new))

    if len(vectors) > 0:
        target_index.add(np.array(vectors), doc_ids=doc_ids)

    assert source_index.doc_ids == target_index.doc_ids
