import abc
import logging
import time
from collections import OrderedDict, defaultdict
from enum import Enum
from queue import PriorityQueue
from typing import Callable, Dict, Iterable, Iterator, List, Sequence, Set, Tuple, Union

import numpy as np
from scipy.spatial.distance import cosine
from tqdm import tqdm

from fast_forward.encoder import QueryEncoder
from fast_forward.ranking import Ranking, interpolate

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
    @abc.abstractmethod
    def dim(self) -> int:
        """Return the dimensionality of the vectors in the index.

        Returns:
            int: The dimensionality.
        """
        pass

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
    def __len__(self) -> int:
        """The number of vectors in the index.

        Returns:
            int: The number of vectors.
        """
        pass

    @abc.abstractmethod
    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Union[Sequence[str], None],
        psg_ids: Union[Sequence[str], None],
    ) -> None:
        """Add vector representations and corresponding IDs to the index. Each vector is guaranteed to
        have either a document or passage ID associated.

        Args:
            vectors (np.ndarray): The representations, shape (num_vectors, dim).
            doc_ids (Union[Sequence[str], None]): The corresponding document IDs (may be duplicate).
            psg_ids (Union[Sequence[str], None]): The corresponding passage IDs (must be unique).
        """
        pass

    def add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[str] = None,
        psg_ids: Sequence[str] = None,
    ) -> None:
        """Add vector representations and corresponding IDs to the index. Only one of `doc_ids` and `psg_ids`
        may be None.

        Args:
            vectors (np.ndarray): The representations, shape (num_vectors, dim).
            doc_id (Sequence[str], optional): The corresponding document IDs (may be duplicate). Defaults to None.
            psg_id (Sequence[str], optional): The corresponding passage IDs (must be unique). Defaults to None.

        Raises:
            ValueError: When there are no document IDs and no passage IDs.
            ValueError: When vector and index dimensionalities don't match.
            RuntimeError: When items can't be added to the index for any reason.
        """
        if doc_ids is None and psg_ids is None:
            raise ValueError(
                "At least one of `doc_ids` and `psg_ids` must be provided."
            )

        num_vectors, dim = vectors.shape
        if doc_ids is not None:
            assert num_vectors == len(doc_ids)
        if psg_ids is not None:
            assert num_vectors == len(psg_ids)

        if dim != self.dim:
            raise ValueError(
                f"Vector dimensionality ({dim}) does not match index dimensionality ({self.dim})"
            )

        self._add(vectors, doc_ids, psg_ids)

    @abc.abstractmethod
    def _get_vectors(self, ids: Iterable[str]) -> Tuple[np.ndarray, List[List[int]]]:
        """Return:
            * A single array containing all vectors necessary to compute the scores for each document/passage.
            * For each document/passage (in the same order as the IDs), a list of integers (depending on the mode).

        The integers will be used to get the corresponding representations from the array.
        The output of this function depends on the current mode.

        Args:
            ids (Iterable[str]): The document/passage IDs to get the representations for.

        Returns:
            Tuple[np.ndarray, List[List[int]]]: The vectors and corresponding indices.
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
        vectors, id_indices = self._get_vectors(ids)
        all_scores = np.dot(q_rep, vectors.T)

        for ind in id_indices:
            if len(ind) == 0:
                yield None
            else:
                if self.mode == Mode.MAXP:
                    yield np.max(all_scores[ind])
                elif self.mode == Mode.AVEP:
                    yield np.average(all_scores[ind])
                elif self.mode in (Mode.FIRSTP, Mode.PASSAGE):
                    yield all_scores[ind][0]

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
                    scores = _interpolate_early_stopping(
                        ids, dense_scores, sparse_scores, a, cutoff
                    )
                    for id, score in scores.items():
                        run[q_id][id] = score
                result[a] = Ranking(run, sort=True, copy=False)
                result[a].cut(cutoff)

        LOGGER.info(f"computed scores in {time.time() - t0}s")
        return result


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
    source_index.mode = Mode.MAXP

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

        v_old, _ = source_index._get_vectors([doc_id])
        v_new = _coalesce(v_old)
        vectors.extend(v_new)
        doc_ids.extend([doc_id] * len(v_new))

    if len(vectors) > 0:
        target_index.add(np.array(vectors), doc_ids=doc_ids)

    assert source_index.doc_ids == target_index.doc_ids


def _interpolate_early_stopping(
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
            max_possible_score = alpha * sparse_score + (1 - alpha) * max_dense_score

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
