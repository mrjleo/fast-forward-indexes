import csv
from pathlib import Path
from copy import deepcopy
from typing import Dict, Iterator
from collections import OrderedDict, defaultdict


Run = Dict[str, Dict[str, float]]


class Ranking(object):
    """Represents rankings of documents/passages w.r.t. queries."""

    def __init__(
        self, run: Run, name: str = None, sort: bool = True, copy: bool = True
    ) -> None:
        """Constructor.

        Args:
            run (Run): Run to create ranking from.
            name (str, optional): Method name. Defaults to None.
            sort (bool, optional): Whether to sort the documents/passages by score. Defaults to True.
            copy (bool, optional): Whether to make a deep copy of the run to avoid side effects. Defaults to True.
        """
        super().__init__()
        self.name = name
        self.is_sorted = sort
        if copy:
            self.run = deepcopy(run)
        else:
            self.run = run
        if sort:
            self.sort()
        self.q_ids = set(self.run.keys())

    def sort(self) -> None:
        """Sort the ranking by scores (in-place)."""
        for q_id, d in self.run.items():
            self.run[q_id] = OrderedDict(
                sorted(d.items(), key=lambda x: x[1], reverse=True)
            )
        self.is_sorted = True

    def cut(self, cutoff: int) -> None:
        """For each query, remove all but the top-k scoring documents/passages.

        Args:
            cutoff (int): Number of best scores per query to keep (k).
        """
        if not self.is_sorted:
            self.sort()
        for q_id in self.q_ids:
            self.run[q_id] = OrderedDict(list(self.run[q_id].items())[:cutoff])

    def __getitem__(self, q_id: str) -> Dict[str, float]:
        """Return the ranking for a query.

        Args:
            q_id (str): The query ID.

        Returns:
            Dict[str, float]: Document/passage IDs mapped to scores.
        """
        return self.run[q_id]

    def __len__(self) -> int:
        """Return the number of queries.

        Returns:
            int: The number of queries.
        """
        return len(self.q_ids)

    def __iter__(self) -> Iterator[str]:
        """Yield all query IDs.

        Yields:
            str: The query IDs.
        """
        yield from self.q_ids

    def __contains__(self, key: object) -> bool:
        """Check whether a query ID is in the ranking.

        Args:
            key (object): The query ID.

        Returns:
            bool: Wherther the query ID has associated document/passage IDs.
        """
        return key in self.q_ids

    def __eq__(self, o: object) -> bool:
        """Check if this ranking is identical to another one.

        Args:
            o (object): The other ranking.

        Returns:
            bool: Whether the two rankings are identical.
        """
        if not isinstance(o, Ranking):
            return False

        if self.q_ids != o.q_ids:
            return False

        for q_id in self.q_ids:
            if self[q_id] != o[q_id]:
                return False
        return True

    def __repr__(self) -> str:
        """Return the run a string representation of this ranking.

        Returns:
            str: The string representation.
        """
        return self.run.__repr__()

    def save(self, target: Path,) -> None:
        """Save the ranking in a TREC runfile.

        Args:
            target (Path): Output file.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp, delimiter="\t")
            for q_id in self:
                ranking = sorted(self[q_id].keys(), key=self[q_id].get, reverse=True)
                for rank, id in enumerate(ranking, 1):
                    score = self[q_id][id]
                    writer.writerow([q_id, "Q0", id, rank, score, str(self.name)])

    @classmethod
    def from_file(cls, fname: Path) -> "Ranking":
        """Create a Ranking object from a runfile in TREC format.

        Args:
            fname (Path): TREC runfile to read.

        Returns:
            Ranking: The resulting ranking.
        """
        run = defaultdict(dict)
        with open(fname, encoding="utf-8") as fp:
            for line in fp:
                q_id, _, id, _, score, name = line.split()
                run[q_id][id] = float(score)
        return cls(run, name, sort=True, copy=False)
