import abc
import logging
from enum import Enum
from time import perf_counter
from typing import Iterable, List, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd

from fast_forward.encoder import Encoder
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
        query_encoder: Encoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> None:
        """Constructor.

        Args:
            query_encoder (Encoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Retrieval mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Encoder batch size. Defaults to 32.
        """
        super().__init__()
        self.query_encoder = query_encoder
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
            raise RuntimeError("Index does not have a query encoder.")

        result = []
        for i in range(0, len(queries), self._encoder_batch_size):
            batch = queries[i : i + self._encoder_batch_size]
            result.append(self._query_encoder(batch))
        return np.concatenate(result)

    @property
    def query_encoder(self) -> Encoder:
        """Return the query encoder.

        Returns:
            Encoder: The encoder.
        """
        return self._query_encoder

    @query_encoder.setter
    def query_encoder(self, encoder: Encoder) -> None:
        """Set the query encoder.

        Args:
            encoder (Encoder): The encoder.
        """
        assert encoder is None or isinstance(encoder, Encoder)
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
            raise ValueError("At least one of doc_ids and psg_ids must be provided.")

        num_vectors, dim = vectors.shape
        if doc_ids is not None:
            assert num_vectors == len(doc_ids)
        if psg_ids is not None:
            assert num_vectors == len(psg_ids)

        if dim != self.dim:
            raise ValueError(
                f"Vector dimensionality ({dim}) does not match index dimensionality ({self.dim})."
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

    def _compute_scores(
        self, df: pd.DataFrame, query_vectors: np.ndarray
    ) -> pd.DataFrame:
        """Computes scores for a data frame.
        The input data frame needs a "q_no" column with unique query numbers.

        Args:
            df (pd.DataFrame): Input data frame.
            query_vectors (np.ndarray): All query vectors indexed by "q_no".

        Returns:
            pd.DataFrame: Data frame with computed scores.
        """
        # map doc/passage IDs to unique numbers (0 to n)
        id_df = df[["id"]].drop_duplicates().reset_index(drop=True)
        id_df["id_no"] = id_df.index

        # attach doc/passage numbers to data frame
        df = df.merge(id_df, on="id", suffixes=[None, "_"]).reset_index(drop=True)

        # get all required vectors from the FF index
        vectors, id_to_vec_idxs = self._get_vectors(id_df["id"].to_list())

        # compute indices for query vectors and doc/passage vectors in current arrays
        select_query_vectors = []
        select_vectors = []
        select_scores = []
        c = 0
        for id_no, q_no in zip(df["id_no"], df["q_no"]):
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

        def _mapfunc(i):
            scores_i = select_scores[i]
            if len(scores_i) == 0:
                return np.nan
            return op(scores[select_scores[i]])

        # insert FF scores in the correct rows
        df["ff_score"] = df.index.map(_mapfunc)
        return df

    def _early_stopping(
        self,
        df: pd.DataFrame,
        query_vectors: np.ndarray,
        cutoff: int,
        alpha: float,
        intervals: Iterable[int],
    ) -> pd.DataFrame:
        """Compute scores with early stopping for a data frame.
        The input data frame needs a "q_no" column with unique query numbers.

        Args:
            df (pd.DataFrame): Input data frame.
            query_vectors (np.ndarray): All query vectors indexed by "q_no".
            cutoff (int): Cut-off depth for early stopping.
            alpha (float): Interpolation parameter.
            intervals (Iterable[int]): Intervals do compute scores at.

        Returns:
            pd.DataFrame: Data frame with computed scores.
        """

        # remaining query IDs that do not meet the early stopping criterion yet
        q_ids_left = pd.unique(df["q_id"])

        # data frame for computed scores
        scores_df = None

        # a and b are the interval for which the scores are computed in each step
        a = 0
        for b in intervals:

            # take the next chunk with b-a docs/passages for each query
            chunk = df.loc[df["q_id"].isin(q_ids_left)].groupby("q_id").nth(range(a, b))

            # stop if no pairs are left
            if len(chunk) == 0:
                break

            # compute scores for the chunk and concat with scores_df
            out = self._compute_scores(chunk, query_vectors)[["orig_index", "ff_score"]]
            scores_df = out if scores_df is None else pd.concat([scores_df, out])

            # join computed scores with other scores and compute interpolated scores
            tmp_result = scores_df.join(
                chunk, on="orig_index", lsuffix=None, rsuffix="_"
            )
            tmp_result["es_score"] = (
                alpha * tmp_result["score"] + (1 - alpha) * tmp_result["ff_score"]
            )

            # identify which queries still do not meet the early stopping criterion
            q_ids_left = (
                tmp_result.groupby("q_id")
                .filter(
                    lambda g: g["es_score"].nlargest(cutoff).iat[-1]
                    < alpha * g["score"].iat[-1] + (1 - alpha) * g["ff_score"].max()
                )["q_id"]
                .drop_duplicates()
                .to_list()
            )

            a = b
        return scores_df.join(df, on="orig_index", lsuffix=None, rsuffix="_")

    def __call__(
        self,
        ranking: Ranking,
        early_stopping: int = None,
        early_stopping_alpha: float = None,
        early_stopping_intervals: Iterable[int] = None,
    ) -> Ranking:
        """Compute scores for a ranking.

        Args:
            ranking (Ranking): The ranking to compute scores for. Must have queries attached.
            early_stopping (int, optional): Perform early stopping at this cut-off depth. Defaults to None.
            early_stopping_alpha (float, optional): Interpolation parameter for early stopping. Defaults to None.
            early_stopping_intervals (Iterable[int], optional): Intervals for early stopping. Defaults to None.

        Returns:
            Ranking: Ranking with the computed scores.

        Raises:
            ValueError: When the ranking has no queries attached.
            ValueError: When early stopping is enabled but arguments are missing.
        """
        if not ranking.has_queries:
            raise ValueError("Input ranking has no queries attached.")

        if early_stopping is not None and (
            early_stopping_alpha is None or early_stopping_intervals is None
        ):
            raise ValueError("Early stopping requires alpha ind intervals.")

        t0 = perf_counter()

        # get all unique queries and query IDs and map to unique numbers (0 to m)
        query_df = (
            ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
        )
        query_df["q_no"] = query_df.index

        # attach query numbers to data frame
        df = ranking._df.merge(query_df, on="q_id", suffixes=[None, "_"])

        # batch encode queries
        query_vectors = self.encode_queries(list(query_df["query"]))

        if early_stopping is not None:
            # early stopping splits the data frame, hence we need to keep track of the original index
            df["orig_index"] = df.index
            result = self._early_stopping(
                df,
                query_vectors,
                early_stopping,
                early_stopping_alpha,
                early_stopping_intervals,
            )
        else:
            result = self._compute_scores(df, query_vectors)
        result["score"] = result["ff_score"]

        LOGGER.info("computed scores in %s seconds", perf_counter() - t0)
        return Ranking(
            result,
            name="fast-forward",
            dtype=ranking._df.dtypes["score"],
            copy=False,
            is_sorted=False,
        )
