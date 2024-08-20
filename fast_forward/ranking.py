"""
.. include:: docs/ranking.md
"""

import logging
from pathlib import Path
from typing import Dict, Iterator, Set, Union

import numpy as np
import pandas as pd

Run = Dict[str, Dict[str, Union[float, int]]]

LOGGER = logging.getLogger(__name__)


def _attach_queries(df: pd.DataFrame, queries: Dict[str, str]) -> pd.DataFrame:
    """Attach queries to a data frame.

    Args:
        df (pd.DataFrame): The data frame (same format as used by rankings).
        queries (Dict[str, str]): Query IDs mapped to queries.

    Raises:
        ValueError: When the queries are incomplete.

    Returns:
        pd.DataFrame: The data frame with queries attached.
    """
    if not set(pd.unique(df["q_id"])).issubset(set(queries.keys())):
        raise ValueError("Queries are incomplete.")
    return df.merge(
        pd.DataFrame(queries.items(), columns=["q_id", "query"]), how="left", on="q_id"
    )


class Ranking(object):
    """Represents rankings of documents/passages w.r.t. queries."""

    def __init__(
        self,
        df: pd.DataFrame,
        name: str = None,
        queries: Dict[str, str] = None,
        dtype: np.dtype = np.float32,
        copy: bool = True,
        is_sorted: bool = False,
    ) -> None:
        """Create a ranking from an existing data frame. Removes rows with NaN scores.

        Args:
            df (pd.DataFrame): Data frame containing IDs and scores.
            name (str, optional): Method name. Defaults to None. Defaults to None.
            queries (Dict[str, str], optional): Query IDs mapped to queries. Defaults to None.
            dtype (np.dtype, optional): How the scores should be represented in the data frame. Defaults to np.float32.
            copy (bool, optional): Whether to copy the data frame. Defaults to True.
            is_sorted (bool, optional): Whether the data frame is already sorted (by score). Defaults to False.

        Raises:
            ValueError: When the queries are incomplete.
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

        Returns:
            bool: Whether queries exist.
        """
        return "query" in self._df.columns

    @property
    def q_ids(self) -> Set[str]:
        """The set of (unique) query IDs in this ranking. Only queries with at least one scored document are considered.

        Returns:
            Set[str]: The query IDs.
        """
        return self._q_ids

    def __getitem__(self, q_id: str) -> Dict[str, float]:
        """Return the ranking for a query.

        Args:
            q_id (str): The query ID.

        Returns:
            Dict[str, float]: Document/passage IDs mapped to scores.
        """
        return dict(self._df[self._df["q_id"] == q_id][["id", "score"]].values)

    def __len__(self) -> int:
        """Return the number of queries.

        Returns:
            int: The number of queries.
        """
        return len(self._q_ids)

    def __iter__(self) -> Iterator[str]:
        """Yield all query IDs.

        Yields:
            str: The query IDs.
        """
        yield from self._q_ids

    def __contains__(self, key: object) -> bool:
        """Check whether a query ID is in the ranking.

        Args:
            key (object): The query ID.

        Returns:
            bool: Whether the query ID has associated document/passage IDs.
        """
        return key in self._q_ids

    def __eq__(self, o: object) -> bool:
        """Check if this ranking is identical to another one. Only takes IDs and scores into account.

        Args:
            o (object): The other ranking.

        Returns:
            bool: Whether the two rankings are identical.
        """
        if not isinstance(o, Ranking):
            return False

        df1 = self._df.sort_values(["q_id", "id"]).reset_index(drop=True)
        df2 = o._df.sort_values(["q_id", "id"]).reset_index(drop=True)

        cols = ["q_id", "id", "score"]
        return df1[cols].equals(df2[cols])

    def __repr__(self) -> str:
        """Return a string representation of this ranking.

        Returns:
            str: The string representation.
        """
        return self._df.__repr__()

    def attach_queries(self, queries: Dict[str, str]) -> "Ranking":
        """Attach queries to the ranking.

        Args:
            queries (Dict[str, str]): Query IDs mapped to queries.

        Raises:
            ValueError: When the queries are incomplete.

        Returns:
            Ranking: The ranking with queries attached.
        """
        return Ranking(
            self._df,
            self.name,
            queries=queries,
            dtype=self._df.dtypes["score"],
            copy=True,
            is_sorted=True,
        )

    def cut(self, cutoff: int) -> "Ranking":
        """For each query, remove all but the top-`k` scoring documents/passages.

        Args:
            cutoff (int): Number of best scores per query to keep (`k`).

        Returns:
            Ranking: The resulting ranking.
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
    ) -> "Ranking":
        """Interpolate as `score = self.score * alpha + other.score * (1 - alpha)`.

        Args:
            other (Ranking): Ranking to interpolate with.
            alpha (float): Interpolation parameter.

        Returns:
            Ranking: The resulting ranking.
        """
        # preserves order by score
        new_df = self._df.merge(other._df, on=["q_id", "id"], suffixes=[None, "_other"])
        new_df["score"] = alpha * new_df["score"] + (1 - alpha) * new_df["score_other"]
        result = Ranking(
            new_df,
            name=self.name,
            dtype=self._df.dtypes["score"],
            copy=False,
            is_sorted=False,
        )
        return result

    def save(
        self,
        target: Path,
    ) -> None:
        """Save the ranking in a TREC runfile.

        Args:
            target (Path): Output file.
        """
        df_ranks = self._df.groupby("q_id").cumcount().to_frame()
        df_ranks.columns = ("rank",)
        df_out = self._df.join(df_ranks)
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
        name: str = None,
        queries: Dict[str, str] = None,
        dtype: np.dtype = np.float32,
    ) -> "Ranking":
        """Create a Ranking object from a TREC run.

        Args:
            run (Run): TREC run.
            name (str, optional): Method name. Defaults to None. Defaults to None.
            queries (Dict[str, str], optional): Query IDs mapped to queries. Defaults to None.
            dtype (np.dtype, optional): How the score should be represented in the data frame. Defaults to np.float32.

        Returns:
            Ranking: The resulting ranking.
        """
        df = pd.DataFrame.from_dict(run).stack().reset_index()
        df.columns = ("id", "q_id", "score")
        return cls(df, name=name, queries=queries, dtype=dtype, copy=False)

    @classmethod
    def from_file(
        cls,
        f: Path,
        queries: Dict[str, str] = None,
        dtype: np.dtype = np.float32,
    ) -> "Ranking":
        """Create a Ranking object from a runfile in TREC format.

        Args:
            f (Path): TREC runfile to read.
            queries (Dict[str, str], optional): Query IDs mapped to queries. Defaults to None.
            dtype (np.dtype, optional): How the score should be represented in the data frame. Defaults to np.float32.

        Returns:
            Ranking: The resulting ranking.
        """
        df = pd.read_csv(
            f,
            sep=r"\s+",
            skipinitialspace=True,
            header=None,
            names=["q_id", "q0", "id", "rank", "score", "name"],
        )
        return cls(df, name=df["name"][0], queries=queries, dtype=dtype, copy=False)
