import abc
import logging
from enum import Enum
from queue import PriorityQueue
from time import perf_counter
from typing import Callable, Dict, Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine
from tqdm import tqdm

from fast_forward.encoder import QueryEncoder
from fast_forward.ranking import Ranking

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
            encoder_batch_size (int, optional): Encoder batch size. Defaults to 32.
        """
        super().__init__()
        self.query_encoder = encoder
        self.mode = mode
        self._encoder_batch_size = encoder_batch_size

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Encode queries.

        Args:
            queries (Sequence[str]): The queries to encode.

        Raises:
            RuntimeError: When no query encoder exists.

        Returns:
            np.ndarray: The query representations.
        """
        if self._query_encoder is None:
            raise RuntimeError("This index does not have a query encoder.")

        result = []
        for i in range(0, len(queries), self._encoder_batch_size):
            batch = queries[i : i + self._encoder_batch_size]
            result.append(self._query_encoder.encode(batch))
        return np.concatenate(result)

    @property
    def query_encoder(self) -> QueryEncoder:
        """Return the query encoder.

        Returns:
            QueryEncoder: The encoder.
        """
        return self._query_encoder

    @query_encoder.setter
    def query_encoder(self, encoder: QueryEncoder) -> None:
        """Set the query encoder.

        Args:
            encoder (QueryEncoder): The encoder.
        """
        assert encoder is None or isinstance(encoder, QueryEncoder)
        self._query_encoder = encoder

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

    def __call__(self, ranking: Ranking) -> Ranking:
        """Compute scores for a ranking.

        Args:
            ranking (Ranking): The ranking to compute scores for. Must have queries attached.

        Returns:
            Ranking: The updated ranking.
        """
        t0 = perf_counter()

        # TODO: error when no queries

        new_df = ranking._df.copy(deep=False)

        # map doc/passage IDs to unique numbers (0 to n)
        id_df = new_df[["id"]].drop_duplicates().reset_index(drop=True)
        id_df["id_no"] = id_df.index
        new_df = new_df.merge(id_df, on="id", suffixes=[None, "_"])

        # get all unique queries and query IDs and map to unique numbers (0 to m)
        query_df = new_df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
        query_df["q_no"] = query_df.index
        new_df = new_df.merge(query_df, on="q_id", suffixes=[None, "_"])

        # batch encode queries
        query_vectors = self.encode_queries(list(query_df["query"]))

        # get all required vectors from the FF index
        vectors, id_to_vec_idxs = self._get_vectors(id_df["id"].to_list())

        # compute indices for query vectors and doc/passage vectors in current arrays
        select_query_vectors = []
        select_vectors = []
        select_scores = []
        c = 0
        for id_no, q_no in zip(new_df["id_no"], new_df["q_no"]):
            vec_idxs = id_to_vec_idxs[id_no]
            select_vectors.extend(vec_idxs)
            select_scores.append(list(range(c, c + len(vec_idxs))))
            c += len(vec_idxs)
            select_query_vectors.extend([q_no] * len(vec_idxs))

        # compute all dot products (scores)
        q_reps = query_vectors[select_query_vectors]
        d_reps = vectors[select_vectors]
        scores = np.sum(q_reps * d_reps, axis=1)

        # select aggregation operation based on current mode
        if self.mode == Mode.MAXP:
            op = np.max
        elif self.mode == Mode.AVEP:
            op = np.average
        else:
            op = lambda x: x[0]

        # insert FF scores in the correct rows
        new_df["ff_score"] = new_df.index.map(lambda i: op(scores[select_scores[i]]))

        LOGGER.info(f"computed scores in {perf_counter() - t0}s")
        return Ranking(
            new_df,
            name=ranking.name,
            dtype=ranking._df.dtypes["score"],
            copy=False,
            is_sorted=True,
        )


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
