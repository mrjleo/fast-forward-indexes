import abc
import logging
from collections.abc import Iterable, Iterator, Sequence
from enum import Enum
from time import perf_counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from fast_forward.encoder.base import Encoder
from fast_forward.quantizer import Quantizer
from fast_forward.ranking import Ranking

LOGGER = logging.getLogger(__name__)


class Mode(Enum):
    """Enum used to set the ranking mode of an index."""

    PASSAGE = 1
    MAXP = 2
    FIRSTP = 3
    AVEP = 4


IDSequence = Sequence[str | None]


class Index(abc.ABC):
    """Abstract base class for Fast-Forward indexes."""

    _query_encoder: Encoder | None = None
    _quantizer: Quantizer | None = None

    def __init__(
        self,
        query_encoder: Encoder | None = None,
        quantizer: Quantizer | None = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
    ) -> None:
        """Create an index.

        :param query_encoder: The query encoder to use.
        :param quantizer: The quantizer to use.
        :param mode: The ranking mode.
        :param encoder_batch_size: The encoder batch size.
        """
        super().__init__()
        if query_encoder is not None:
            self.query_encoder = query_encoder
        self.mode = mode
        if quantizer is not None:
            self.quantizer = quantizer
        self._encoder_batch_size = encoder_batch_size

    def encode_queries(self, queries: Sequence[str]) -> np.ndarray:
        """Encode queries.

        :param queries: The queries to encode.
        :raises RuntimeError: When no query encoder exists.
        :return: The query representations.
        """
        if self.query_encoder is None:
            raise RuntimeError("Index does not have a query encoder.")

        result = []
        for i in range(0, len(queries), self._encoder_batch_size):
            batch = queries[i : i + self._encoder_batch_size]
            result.append(self.query_encoder(batch))
        return np.concatenate(result)

    @property
    def query_encoder(self) -> Encoder | None:
        """Return the query encoder if it exists.

        :return: The query encoder (if any).
        """
        return self._query_encoder

    @query_encoder.setter
    def query_encoder(self, encoder: Encoder) -> None:
        """Set the query encoder.

        :param encoder: The query encoder to set.
        """
        assert isinstance(encoder, Encoder)
        self._query_encoder = encoder

    @property
    def quantizer(self) -> Quantizer | None:
        """Return the quantizer if it exists.

        :return: The quantizer (if any).
        """
        return self._quantizer

    def _on_quantizer_set(self) -> None:
        """Handle a quantizer being attached to this index."""
        pass

    @quantizer.setter
    def quantizer(self, quantizer: Quantizer) -> None:
        """Set the quantizer.

        This is only possible before any vectors are added to the index.

        :param quantizer: The new quantizer.
        :raises RuntimeError: When the index is not empty.
        """
        assert isinstance(quantizer, Quantizer)

        if len(self) > 0:
            raise RuntimeError("Quantizers can only be attached to empty indexes.")
        self._quantizer = quantizer
        self._on_quantizer_set()
        quantizer.set_attached()

    @property
    def mode(self) -> Mode:
        """Return the ranking mode.

        :return: The ranking mode.
        """
        return self._mode

    @mode.setter
    def mode(self, mode: Mode) -> None:
        """Set the ranking mode.

        :param mode: The ranking mode to set.
        """
        assert isinstance(mode, Mode)
        self._mode = mode

    @abc.abstractmethod
    def _get_internal_dim(self) -> int | None:
        pass

    @property
    def dim(self) -> int | None:
        """Return the dimensionality of the vector index.

        If no vectors exist, return `None`.
        If a quantizer is used, return the dimension of the codes.

        :return: The dimensionality (if any).
        """
        if self._quantizer is not None:
            return self._quantizer.dims[0]
        return self._get_internal_dim()

    @abc.abstractmethod
    def _get_doc_ids(self) -> set[str]:
        pass

    @property
    def doc_ids(self) -> set[str]:
        """Return all unique document IDs.

        :return: The document IDs.
        """
        return self._get_doc_ids()

    @abc.abstractmethod
    def _get_psg_ids(self) -> set[str]:
        pass

    @property
    def psg_ids(self) -> set[str]:
        """Return all unique passage IDs.

        :return: The passage IDs.
        """
        return self._get_psg_ids()

    @abc.abstractmethod
    def _get_num_vectors(self) -> int:
        pass

    def __len__(self) -> int:
        """Return the index size.

        :return: The number of vectors in the index.
        """
        return self._get_num_vectors()

    @abc.abstractmethod
    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence,
        psg_ids: IDSequence,
    ) -> None:
        """Add vector representations and corresponding IDs to the index.

        Document IDs may have duplicates, passage IDs are assumed to be unique.
        Vectors may be quantized.

        Specific to index implementation.

        :param vectors:
            The representations, shape `(num_vectors, dim)` or
            `(num_vectors, quantized_dim)`.
        :param doc_ids: The corresponding document IDs.
        :param psg_ids: The corresponding passage IDs.
        """
        pass

    def add(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence | None = None,
        psg_ids: IDSequence | None = None,
    ) -> None:
        """Add vector representations and corresponding IDs to the index.

        Only one of `doc_ids` and `psg_ids` may be `None`.
        Individual IDs in the sequence may also be `None`, but each vector must have at
        least one associated ID.
        Document IDs may have duplicates, passage IDs must be unique.

        :param vectors: The representations, shape `(num_vectors, dim)`.
        :param doc_ids: The corresponding document IDs (may be duplicate).
        :param psg_ids: The corresponding passage IDs (must be unique).
        :raises ValueError: When the number of IDs does not match the number of vectors.
        :raises ValueError:
            When the input vector and index dimensionalities do not match.
        :raises ValueError: When a vector has neither a document nor a passage ID.
        :raises RuntimeError: When items can't be added to the index for any reason.
        """
        num_vectors, dim = vectors.shape

        if doc_ids is None:
            doc_ids = [None] * num_vectors
        if psg_ids is None:
            psg_ids = [None] * num_vectors
        if not len(doc_ids) == len(psg_ids) == num_vectors:
            raise ValueError("Number of IDs does not match number of vectors.")

        if self.dim is not None and dim != self.dim:
            raise ValueError(
                f"Input vector dimensionality ({dim}) does not match "
                f"index dimensionality ({self.dim})."
            )

        for doc_id, psg_id in zip(doc_ids, psg_ids):
            if doc_id is None and psg_id is None:
                raise ValueError("Vector has neither document nor passage ID.")

        self._add(
            vectors if self.quantizer is None else self.quantizer.encode(vectors),
            doc_ids,
            psg_ids,
        )

    @abc.abstractmethod
    def _get_vectors(self, ids: Iterable[str]) -> tuple[np.ndarray, list[str]]:
        """Get vectors and corresponding IDs from the index.

        Return a tuple containing:
            * A single array of all vectors necessary to compute the scores for each
                document/passage.
            * A list of document/passage IDs to identify each vector in the array.

        The output of this function depends on the current mode.

        If a quantizer is used, this function returns quantized vectors.

        Specific to index implementation.

        :param ids: The document/passage IDs to get the representations for.
        :raises IndexError: When a requested ID is not found in the index.
        :return: The vectors and corresponding indices.
        """
        pass

    def _compute_scores(
        self, data: pd.DataFrame | pd.Series, query_vectors: np.ndarray
    ) -> pd.DataFrame:
        """Compute scores for a data frame.

        The input data frame needs a "q_no" column with unique query numbers.

        :param data: Input data frame (or series).
        :param query_vectors: All query vectors indexed by "q_no".
        :return: Data frame with computed scores.
        """
        # get all required vectors and corresponding IDs from the FF index
        vectors, vec_ids = self._get_vectors(data["id"].unique())
        if self.quantizer is not None:
            vectors = self.quantizer.decode(vectors)

        # merge data frames so "id_idx" can be used to index arrays
        df_vec_ids = pd.DataFrame(vec_ids, columns=["id"])
        df_vec_ids["id_idx"] = df_vec_ids.index
        df_merged = df_vec_ids.merge(data[["id", "q_no"]], how="left", on="id")

        # compute all dot products (scores)
        q_reps = query_vectors[df_merged["q_no"].tolist()]
        d_reps = vectors[df_merged["id_idx"].tolist()]
        df_merged["ff_score"] = np.sum(q_reps * d_reps, axis=1)

        # select aggregation operation based on current mode
        if self.mode == Mode.MAXP:
            op = "max"
        elif self.mode == Mode.AVEP:
            op = "mean"
        else:
            op = "first"
        df_agg = df_merged.groupby(["id", "q_no"], as_index=False).aggregate(op)

        return data.merge(df_agg, on=["id", "q_no"])

    def _early_stopping(
        self,
        df: pd.DataFrame,
        query_vectors: np.ndarray,
        cutoff: int,
        alpha: float,
        depths: Iterable[int],
    ) -> pd.DataFrame:
        """Compute scores with early stopping for a data frame.

        The input data frame needs a "q_no" column with unique query numbers.

        :param df: Input data frame.
        :param query_vectors: All query vectors indexed by "q_no".
        :param cutoff: Cut-off depth for early stopping.
        :param alpha: Interpolation parameter.
        :param depths: Depths to compute scores at.
        :return: Data frame with computed scores.
        """
        # data frame for computed scores
        scores_so_far = pd.DataFrame()

        # [a, b] is the interval for which the scores are computed in each step
        a = 0
        for b in sorted(depths):
            if b < cutoff:
                continue

            # identify queries which do not meet the early stopping criterion
            if a == 0:
                # first iteration: take all queries
                q_ids_left = pd.unique(df["q_id"])
            else:
                # subsequent iterations: compute ES criterion
                q_ids_left = (
                    scores_so_far.groupby("q_id")
                    .filter(
                        lambda g: g["int_score"].nlargest(cutoff).iat[-1]
                        < alpha * g["score"].iat[-1] + (1 - alpha) * g["ff_score"].max()
                    )["q_id"]
                    .drop_duplicates()
                    .to_list()
                )
            LOGGER.info("depth %s: %s queries left", b, len(q_ids_left))

            # take the next chunk with b-a docs/passages for each query
            chunk = (
                df.loc[df["q_id"].isin(q_ids_left)]
                .groupby("q_id")
                .nth(list(range(a, b)))
            )

            # stop if no pairs are left
            if len(chunk) == 0:
                break

            # compute scores for the chunk and merge
            out = self._compute_scores(chunk, query_vectors)[["orig_index", "ff_score"]]
            chunk_scores = chunk.merge(out, on="orig_index", suffixes=(None, "_"))

            # compute interpolated scores
            chunk_scores["int_score"] = (
                alpha * chunk_scores["score"] + (1 - alpha) * chunk_scores["ff_score"]
            )

            scores_so_far = pd.concat(
                [scores_so_far, chunk_scores],
                axis=0,
            )

            a = b
        return scores_so_far.join(df, on="orig_index", lsuffix="", rsuffix="_")

    def __call__(
        self,
        ranking: Ranking,
        early_stopping: int | None = None,
        early_stopping_alpha: float | None = None,
        early_stopping_depths: Iterable[int] | None = None,
        batch_size: int | None = None,
    ) -> Ranking:
        """Compute scores for a ranking.

        :param ranking: The ranking to compute scores for. Must have queries attached.
        :param early_stopping: Perform early stopping at this cut-off depth.
        :param early_stopping_alpha: Interpolation parameter for early stopping.
        :param early_stopping_depths: Depths for early stopping.
        :param batch_size: How many queries to process at once.
        :raises ValueError: When the ranking has no queries attached.
        :raises ValueError: When early stopping is enabled but arguments are missing.
        :return: Ranking with the computed scores.
        """
        if not ranking.has_queries:
            raise ValueError("Input ranking has no queries attached.")
        if early_stopping is not None and (
            early_stopping_alpha is None or early_stopping_depths is None
        ):
            raise ValueError("Early stopping requires alpha and depths.")
        t0 = perf_counter()

        # get all unique queries and query IDs and map to unique numbers (0 to m)
        query_df = (
            ranking._df[["q_id", "query"]].drop_duplicates().reset_index(drop=True)
        )
        query_df["q_no"] = query_df.index
        df_with_q_no = ranking._df.merge(query_df, on="q_id", suffixes=(None, "_"))

        # early stopping splits the data frame, hence we need to keep track of the
        # original index
        df_with_q_no["orig_index"] = df_with_q_no.index

        # batch encode queries
        query_vectors = self.encode_queries(list(query_df["query"]))

        def _get_result(df: pd.DataFrame) -> pd.DataFrame:
            if early_stopping is None:
                return self._compute_scores(df, query_vectors)

            assert early_stopping_alpha is not None
            assert early_stopping_depths is not None
            return self._early_stopping(
                df,
                query_vectors,
                early_stopping,
                early_stopping_alpha,
                early_stopping_depths,
            )

        num_queries = len(query_df)
        if batch_size is None or batch_size >= num_queries:
            result = _get_result(df_with_q_no)
        else:
            # assign batch indices to query IDs
            df_with_q_no["batch_idx"] = (
                df_with_q_no.groupby("q_id").ngroup() / batch_size
            ).astype(int)

            chunks = []
            num_batches = int(num_queries / batch_size) + 1
            for batch_idx in tqdm(range(num_batches)):
                batch = df_with_q_no[df_with_q_no["batch_idx"] == batch_idx]
                chunks.append(_get_result(batch))
            result = pd.concat(chunks)

        result["score"] = result["ff_score"]
        LOGGER.info("computed scores in %s seconds", perf_counter() - t0)
        return Ranking(
            result,
            name="fast-forward",
            dtype=ranking._df.dtypes["score"],
            copy=False,
            is_sorted=False,
        )

    @abc.abstractmethod
    def _batch_iter(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, IDSequence, IDSequence]]:
        """Iterate over the index in batches.

        If a quantizer is used, the vectors are the quantized codes.
        When an ID does not exist, it must be set to `None`.

        Specific to index implementation.

        :param batch_size: The batch size.
        :return: Iterator yielding vectors, document IDs, passage IDs (in batches).
        """
        pass

    def batch_iter(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, IDSequence, IDSequence]]:
        """Iterate over all vectors, document IDs, and passage IDs in batches.

        IDs may be either strings or `None`.

        :param batch_size: Batch size.
        :return: Iterator yielding batches of vectors, document IDs (if any),
            passage IDs (if any).
        """
        if self._quantizer is None:
            yield from self._batch_iter(batch_size)

        else:
            for batch in self._batch_iter(batch_size):
                yield self._quantizer.decode(batch[0]), batch[1], batch[2]

    def __iter__(
        self,
    ) -> Iterator[tuple[np.ndarray, str | None, str | None]]:
        """Iterate over all vectors, document IDs, and passage IDs.

        :return: Iterator yielding vectors, document IDs (if any), passage IDs (if any).
        """
        for vectors, doc_ids, psg_ids in self.batch_iter(2**9):
            yield from zip(vectors, doc_ids, psg_ids)
