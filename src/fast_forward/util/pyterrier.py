import pandas as pd
import pyterrier as pt

from fast_forward.index import Index
from fast_forward.ranking import Ranking


class FFScore(pt.Transformer):
    """PyTerrier transformer that computes scores using a Fast-Forward index."""

    def __init__(self, index: Index) -> None:
        """Create an FFScore transformer.

        :param index: The Fast-Forward index.
        """
        self._index = index
        super().__init__()

    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        """Compute the scores for all query-document pairs in the data frame.
        The previous scores are moved to the "score_0" column.

        :param topics_or_res: The PyTerrier data frame.
        :return: A data frame with the computed scores.
        """
        ff_scores = self._index(
            Ranking(
                topics_or_res.rename(columns={"qid": "q_id", "docno": "id"}),
                copy=False,
                is_sorted=True,
            )
        )._df.rename(columns={"q_id": "qid", "id": "docno"})

        return topics_or_res[["qid", "docno", "score"]].merge(
            ff_scores[["qid", "docno", "score", "query"]],
            on=["qid", "docno"],
            suffixes=("_0", None),
        )

    def __repr__(self) -> str:
        """Return a string representation.
        The representation is unique w.r.t. the index and its query encoder.

        :return: The representation.
        """
        return f"{self.__class__.__name__}({id(self._index)}, {id(self._index._query_encoder)})"


class FFInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: float) -> None:
        """Create an FFInterpolate transformer.

        :param alpha: The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def transform(self, topics_or_res: pd.DataFrame) -> pd.DataFrame:
        """Interpolate the scores for all query-document pairs in the data frame as
        `alpha * score_0 + (1 - alpha) * score`.

        :param topics_or_res: The PyTerrier data frame.
        :return: A data frame with the interpolated scores.
        """
        new_df = topics_or_res[["qid", "docno", "query"]].copy()
        new_df["score"] = (
            self.alpha * topics_or_res["score_0"]
            + (1 - self.alpha) * topics_or_res["score"]
        )
        return new_df
