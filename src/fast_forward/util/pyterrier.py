import pandas as pd
import pyterrier as pt

from fast_forward.index import Index
from fast_forward.ranking import Ranking


class FFScore(pt.Transformer):
    """PyTerrier transformer that computes scores using a Fast-Forward index."""

    def __init__(self, index: Index) -> None:
        """Create an FFScore transformer.

        Args:
            index (Index): The Fast-Forward index.
        """
        self._index = index
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute the scores for all query-document pairs in the data frame.
        The previous scores are moved to the "score_0" column.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the computed scores.
        """
        ff_scores = self._index(
            Ranking(
                df.rename(columns={"qid": "q_id", "docno": "id"}),
                copy=False,
                is_sorted=True,
            )
        )._df.rename(columns={"q_id": "qid", "id": "docno"})

        return df[["qid", "docno", "score"]].merge(
            ff_scores[["qid", "docno", "score", "query"]],
            on=["qid", "docno"],
            suffixes=["_0", None],
        )

    def __repr__(self) -> str:
        """Return a string representation.
        The representation is unique w.r.t. the index and its query encoder.

        Returns:
            str: The representation.
        """
        return f"{self.__class__.__name__}({id(self._index)}, {id(self._index._query_encoder)})"


class FFInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: float) -> None:
        """Create an FFInterpolate transformer.

        Args:
            alpha (float): The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `alpha * score_0 + (1 - alpha) * score`.

        Args:
            df (pd.DataFrame): The PyTerrier data frame.

        Returns:
            pd.DataFrame: A new data frame with the interpolated scores.
        """
        new_df = df[["qid", "docno", "query"]].copy()
        new_df["score"] = self.alpha * df["score_0"] + (1 - self.alpha) * df["score"]
        return new_df
