import logging
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from pathlib import Path

LOGGER = logging.getLogger(__name__)

Run = Mapping[str, Mapping[str, float]]


def _attach_queries(df: pd.DataFrame, queries: Mapping[str, str]) -> pd.DataFrame:
    """Attach queries to a data frame.

    :param df: The data frame (same format as used by rankings).
    :param queries: Query IDs mapped to queries.
    :raises ValueError: When the queries are incomplete.
    :return: The data frame with queries attached.
    """
    if not set(pd.unique(df["q_id"])).issubset(set(queries.keys())):
        raise ValueError("Queries are incomplete.")
    return df.merge(
        pd.DataFrame(queries.items(), columns=["q_id", "query"]), how="left", on="q_id"
    )


def _add_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """Add a new column ("rank") to a data frame.

    The new column contains ranks of documents based on the scores w.r.t. the queries.
    Ranks start from `1`.

    :param df: The data frame to add the column to.
    :return: The data frame with the new column added.
    """
    df_ranks = df.groupby("q_id").cumcount().to_frame() + 1
    df_ranks.columns = ("rank",)
    return df.join(df_ranks)


def _minmax_normalize_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of a data frame with normalized scores (min-max).

    If all scores are equal, they are set to `0`.

    :param df: The input data frame.
    :return: A copy of the data frame with normalized scores.
    """
    new_df = df.copy()
    min_val = new_df["score"].min()
    max_val = new_df["score"].max()
    if min_val == max_val:
        LOGGER.warning("all scores are equal, setting scores to 0")
        new_df["score"] = 0
    else:
        new_df["score"] = (new_df["score"] - min_val) / (max_val - min_val)
    return new_df


class Ranking:
    """Represents rankings of documents/passages w.r.t. queries."""

    def __init__(
        self,
        df: pd.DataFrame,
        name: str | None = None,
        queries: Mapping[str, str] | None = None,
        dtype: np.dtype = np.dtype(np.float32),
        copy: bool = True,
        is_sorted: bool = False,
    ) -> None:
        """Create a ranking from an existing data frame. Removes rows with NaN scores.

        :param df: Data frame containing IDs and scores.
        :param name: Method name.
        :param queries: Query IDs mapped to queries.
        :param dtype: How the scores should be represented in the data frame.
        :param copy: Whether to copy the data frame.
        :param is_sorted: Whether the data frame is already sorted (by score).
        :raises ValueError: When the queries are incomplete.
        """
        super().__init__()
        self.name = name

        cols = ["q_id", "id", "score"]
        if "query" in df.columns:
            cols += ["query"]
        self._df = df.loc[:, cols].dropna()
        if copy:
            self._df = self._df.copy()

        for col, dt in (
            ("score", dtype),
            ("q_id", str),
            ("id", str),
        ):
            if col in self._df.columns and self._df[col].dtype != dt:
                self._df[col] = self._df[col].astype(dt)

        if not is_sorted:
            self._df.sort_values(by=["q_id", "score"], inplace=True, ascending=False)
        self._df.reset_index(inplace=True, drop=True)

        self._q_ids = set(pd.unique(self._df["q_id"]))
        if queries is not None:
            self._df = _attach_queries(self._df, queries)

    @property
    def has_queries(self) -> bool:
        """Whether the ranking has queries attached.

        :return: Whether queries exist.
        """
        return "query" in self._df.columns

    @property
    def q_ids(self) -> set[str]:
        """The set of (unique) query IDs in this ranking.

        Only queries with at least one scored document are considered.

        :return: The query IDs.
        """
        return self._q_ids

    def __getitem__(self, q_id: str) -> dict[str, float]:
        """Return the ranking for a query.

        :param q_id: The query ID.
        :return: Document/passage IDs mapped to scores.
        """
        return dict(self._df[self._df["q_id"] == q_id][["id", "score"]].values)

    def __len__(self) -> int:
        """Return the number of queries.

        :return: The number of queries.
        """
        return len(self._q_ids)

    def __iter__(self) -> Iterator[str]:
        """Yield all query IDs.

        :yield: A query ID.
        """
        yield from self._q_ids

    def __contains__(self, key: object) -> bool:
        """Check whether a query ID has associated document/passage IDs.

        :param key: The query ID.
        :return: Whether the query ID is in the ranking.
        """
        return key in self._q_ids

    def __eq__(self, o: object) -> bool:
        """Check if this ranking is identical to another one.

        Only takes IDs and scores into account.

        :param o: The other ranking.
        :return: Whether the two rankings are identical.
        """
        if not isinstance(o, Ranking):
            return False

        df1 = self._df.sort_values(["q_id", "id"]).reset_index(drop=True)
        df2 = o._df.sort_values(["q_id", "id"]).reset_index(drop=True)

        cols = ["q_id", "id", "score"]
        return df1[cols].equals(df2[cols])

    def __add__(self, o: "Ranking | float") -> "Ranking":
        """Add a constant or the scores of another ranking to this ranking's scores.

        Missing scores in either ranking are treated as zero.

        :param o: A ranking or a constant.
        :return: The resulting ranking with added scores.
        """
        if isinstance(o, Ranking):
            new_df = self._df.merge(
                o._df, on=["q_id", "id"], suffixes=(None, "_other"), how="outer"
            ).fillna(0)
            new_df["score"] = new_df["score"] + new_df["score_other"]
            is_sorted = False
        elif isinstance(o, int | float):
            new_df = self._df.copy()
            new_df["score"] += o
            is_sorted = True
        else:
            return NotImplemented

        return Ranking(
            new_df,
            name=self.name,
            dtype=self._df.dtypes["score"],
            copy=False,
            is_sorted=is_sorted,
        )

    __radd__ = __add__

    def __mul__(self, o: float) -> "Ranking":
        """Multiply this ranking's scores by a constant.

        :param o: A constant.
        :return: The resulting ranking with multiplied scores.
        """
        if not isinstance(o, int | float):
            return NotImplemented

        new_df = self._df.copy()
        new_df["score"] *= o

        return Ranking(
            new_df,
            name=self.name,
            dtype=self._df.dtypes["score"],
            copy=False,
            is_sorted=True,
        )

    __rmul__ = __mul__

    def __repr__(self) -> str:
        """Return a string representation of this ranking.

        :return: The string representation.
        """
        return self._df.__repr__()

    def attach_queries(self, queries: Mapping[str, str]) -> "Ranking":
        """Attach queries to the ranking.

        :param queries: Query IDs mapped to queries.
        :raises ValueError: When the queries are incomplete.
        :return: The ranking with queries attached.
        """
        return Ranking(
            self._df,
            self.name,
            queries=queries,
            dtype=self._df.dtypes["score"],
            copy=True,
            is_sorted=True,
        )

    def normalize(self) -> "Ranking":
        """Normalize the scores (min-max) to be in `[0, 1]`.

        If all scores are equal, they are set to `0`.

        :return: A ranking with normalized scores.
        """
        return Ranking(
            _minmax_normalize_scores(self._df),
            self.name,
            dtype=self._df.dtypes["score"],
            copy=False,
            is_sorted=True,
        )

    def cut(self, cutoff: int) -> "Ranking":
        """For each query, remove all but the top-`k` scoring documents/passages.

        :param cutoff: Number of best scores per query to keep (`k`).
        :return: The resulting ranking.
        """
        return Ranking(
            self._df.groupby("q_id").head(cutoff).reset_index(drop=True),
            name=self.name,
            dtype=self._df.dtypes["score"],
            copy=True,
            is_sorted=True,
        )

    def interpolate(
        self,
        other: "Ranking",
        alpha: float,
        normalize: bool = False,
    ) -> "Ranking":
        """Interpolate as `score = self.score * alpha + other.score * (1 - alpha)`.

        Missing scores in either ranking are treated as zero.

        :param other: Ranking to interpolate with.
        :param alpha: Interpolation parameter.
        :param normalize: Perform min-max normalization.
        :return: The resulting ranking.
        """
        df1 = self._df if not normalize else _minmax_normalize_scores(self._df)
        df2 = other._df if not normalize else _minmax_normalize_scores(other._df)

        # during normalization the data frames are copied already
        new_df = df1.merge(
            df2,
            on=["q_id", "id"],
            suffixes=(None, "_other"),
            copy=not normalize,
            how="outer",
        ).fillna(0)
        new_df["score"] = alpha * new_df["score"] + (1 - alpha) * new_df["score_other"]
        return Ranking(
            new_df,
            name=self.name,
            dtype=self._df.dtypes["score"],
            copy=False,
            is_sorted=False,
        )

    def rr_scores(self, k: int = 60) -> "Ranking":
        """Re-score documents/passages using reciprocal rank.

        A score is computed as `1 / (rank + k)`.

        Used by [RRF](https://dl.acm.org/doi/10.1145/1571941.1572114).

        :param k: RR scoring parameter.
        :return: A ranking with RR scores.
        """
        new_df = _add_ranks(self._df)
        new_df["score"] = 1 / (new_df["rank"] + k)
        return Ranking(
            new_df,
            name=self.name,
            dtype=self._df.dtypes["score"],
            copy=False,
            is_sorted=True,
        )

    def save(
        self,
        target: "Path",
    ) -> None:
        """Save the ranking in a TREC runfile.

        :param target: Output file.
        """
        df_out = _add_ranks(self._df)
        df_out["name"] = str(self.name)
        df_out["q0"] = "Q0"
        target.parent.mkdir(parents=True, exist_ok=True)
        df_out.to_csv(
            target,
            sep="\t",
            columns=["q_id", "q0", "id", "rank", "score", "name"],
            index=False,
            header=False,
        )

    @classmethod
    def from_run(
        cls,
        run: Run,
        name: str | None = None,
        queries: Mapping[str, str] | None = None,
        dtype: np.dtype = np.dtype(np.float32),
    ) -> "Ranking":
        """Create a Ranking object from a TREC run.

        :param run: TREC run.
        :param name: Method name.
        :param queries: Query IDs mapped to queries.
        :param dtype: How the score should be represented in the data frame.
        :return: The resulting ranking.
        """
        df = pd.DataFrame.from_dict(dict(run)).stack().reset_index()
        df.columns = ("id", "q_id", "score")
        return cls(df, name=name, queries=queries, dtype=dtype, copy=False)

    @classmethod
    def from_file(
        cls,
        f: "Path",
        queries: Mapping[str, str] | None = None,
        dtype: np.dtype = np.dtype(np.float32),
    ) -> "Ranking":
        """Create a Ranking object from a runfile in TREC format.

        :param f: TREC runfile to read.
        :param queries: Query IDs mapped to queries.
        :param dtype: How the score should be represented in the data frame.
        :return: The resulting ranking.
        """
        df = pd.read_csv(
            f,
            sep=r"\s+",
            skipinitialspace=True,
            header=None,
            names=["q_id", "q0", "id", "rank", "score", "name"],
        )
        return cls(df, name=df["name"][0], queries=queries, dtype=dtype, copy=False)
