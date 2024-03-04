import logging
from heapq import heappop, heappush
from pathlib import Path
from typing import Dict, Iterator, Set, Union

import numpy as np
import pandas as pd

Run = Dict[str, Dict[str, Union[float, int]]]

LOGGER = logging.getLogger(__name__)


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
        """Create a ranking from an existing data frame.

        Args:
            df (pd.DataFrame): Data frame containing IDs and scores.
            name (str, optional): Method name. Defaults to None. Defaults to None.
            queries (Dict[str, str], optional): Query IDs mapped to queries. Defaults to None.
            dtype (np.dtype, optional): How the scores should be represented in the data frame. Defaults to np.float32.
            copy (bool, optional): Whether to copy the data frame. Defaults to True.
            is_sorted (bool, optional): Whether the data frame is already sorted (by score). Defaults to False.
        """
        super().__init__()
        self.name = name

        cols = ["q_id", "id", "score"] + [
            col for col in ("query", "ff_score") if col in df.columns
        ]
        if copy:
            self._df = df.loc[:, cols].copy()
        else:
            self._df = df.loc[:, cols]

        self._df["score"] = self._df["score"].astype(dtype)
        if "ff_score" in df.columns:
            self._df["ff_score"] = self._df["ff_score"].astype(dtype)

        self._q_ids = set(pd.unique(self._df["q_id"]))

        if not is_sorted:
            self._sort()

        if queries is not None:
            self.attach_queries(queries)

    def _sort(self) -> None:
        """Sort the ranking by scores (in-place)."""
        self._df.sort_values(by=["q_id", "score"], inplace=True, ascending=False)
        self._df.reset_index(inplace=True, drop=True)

    @property
    def has_queries(self) -> bool:
        """Whether the ranking has queries attached.

        Returns:
            bool: Whether queries exist.
        """
        return "query" in self._df.columns

    @property
    def has_ff_scores(self) -> bool:
        """Whether the ranking has semantic scores.

        Returns:
            bool: Whether semantic scores exist.
        """
        return "ff_score" in self._df.columns

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
            bool: Wherther the query ID has associated document/passage IDs.
        """
        return key in self._q_ids

    def __eq__(self, o: object) -> bool:
        """Check if this ranking is identical to another one. Only takes IDs and scores into account.

        Args:
            o (object): The other ranking.

        Returns:
            bool: Whether the two rankings are identical.
        """
        if not isinstance(o, Ranking) or self.has_ff_scores != o.has_ff_scores:
            return False

        df1 = self._df.sort_values(["q_id", "id"]).reset_index(drop=True)
        df2 = o._df.sort_values(["q_id", "id"]).reset_index(drop=True)

        cols = ["q_id", "id", "score"]
        if self.has_ff_scores:
            cols += ["ff_score"]

        return df1[cols].equals(df2[cols])

    def __repr__(self) -> str:
        """Return the run a string representation of this ranking.

        Returns:
            str: The string representation.
        """
        return self._df.__repr__()

    def attach_queries(self, queries: Dict[str, str]) -> None:
        """Attach queries to this ranking (in-place).

        Args:
            queries (Dict[str, str]): Query IDs mapped to queries.
        """
        if set(queries.keys()) != self._q_ids:
            raise ValueError("Queries are incomplete")
        q_df = pd.DataFrame(queries.items(), columns=["q_id", "query"])
        self._df = self._df.merge(q_df, how="left", on="q_id")

    def cut(self, cutoff: int) -> "Ranking":
        """For each query, remove all but the top-k scoring documents/passages.

        Args:
            cutoff (int): Number of best scores per query to keep (k).

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
        self, alpha: float, cutoff: int = None, early_stopping: bool = False
    ) -> "Ranking":
        """Interpolate scores as `score * alpha + ff_score * (1 - alpha)`

        Args:
            alpha (float): Interpolation parameter.
            cutoff (int, optional): Cut-off depth. Defaults to None.
            early_stopping (bool, optional): Use early stopping (requires cut-off depth). Defaults to None.

        Returns:
            Ranking: The resulting ranking.
        """

        if early_stopping and cutoff is None:
            LOGGER.warning("No cut-off depth provided, disabling early stopping")
            early_stopping = False

        if not early_stopping:
            new_df = self._df.dropna().copy()
            new_df["score"] = alpha * new_df["score"] + (1 - alpha) * new_df["ff_score"]
            result = Ranking(
                new_df.drop(columns="ff_score"),
                name=self.name,
                dtype=self._df.dtypes["score"],
                copy=False,
                is_sorted=False,
            )
            if cutoff is not None:
                return result.cut(cutoff)
            return result

        def _es(q_df):
            heap = []
            min_relevant_score = float("-inf")
            max_ff_score = float("-inf")
            for id, score, ff_score in zip(q_df["id"], q_df["score"], q_df["ff_score"]):
                if len(heap) >= cutoff:
                    # check if approximated max possible score is too low to make a difference
                    min_relevant_score, min_relevant_id = heappop(heap)
                    max_possible_score = alpha * score + (1 - alpha) * max_ff_score

                    # early stopping
                    if max_possible_score <= min_relevant_score:
                        heappush(heap, (min_relevant_score, min_relevant_id))
                        break

                max_ff_score = max(max_ff_score, ff_score)
                next_score = alpha * score + (1 - alpha) * ff_score
                if next_score > min_relevant_score:
                    heappush(heap, (next_score, id))
                else:
                    heappush(heap, (min_relevant_score, min_relevant_id))
            return reversed([heappop(heap) for _ in range(len(heap))])

        q_dfs_out = []
        for q_id in self.q_ids:
            q_df = self._df[self._df["q_id"] == q_id].dropna()
            q_df_out = pd.DataFrame(_es(q_df), columns=["score", "id"])
            q_df_out["q_id"] = q_id
            if "query" in q_df.columns:
                q_df_out["query"] = q_df["query"].values[0]
            q_dfs_out.append(q_df_out)

        return Ranking(
            pd.concat(q_dfs_out).reset_index(),
            name=self.name,
            dtype=self._df.dtypes["score"],
            copy=False,
            is_sorted=True,
        )

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
            delim_whitespace=True,
            skipinitialspace=True,
            header=None,
            names=["q_id", "q0", "id", "rank", "score", "name"],
        )
        return cls(df, name=df["name"][0], queries=queries, dtype=dtype, copy=False)
