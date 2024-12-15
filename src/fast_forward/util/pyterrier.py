""".. include:: ../docs/pyterrier.md"""  # noqa: D400, D415

from typing import TYPE_CHECKING

import pyterrier as pt

from fast_forward.ranking import Ranking

if TYPE_CHECKING:
    import pandas as pd

    from fast_forward.index.base import Index


class FFScore(pt.Transformer):
    """PyTerrier transformer that computes scores using a Fast-Forward index."""

    def __init__(self, index: "Index") -> None:
        """Create an FFScore transformer.

        :param index: The Fast-Forward index.
        """
        self._index = index
        super().__init__()

    def transform(self, inp: "pd.DataFrame") -> "pd.DataFrame":
        """Compute the scores for all query-document pairs in the data frame.

        The previous scores are moved to the "score_0" column.

        :param inp: The PyTerrier data frame.
        :return: A data frame with the computed scores.
        """
        ff_scores = self._index(
            Ranking(
                inp.rename(columns={"qid": "q_id", "docno": "id"}),
                copy=False,
                # since this ranking is only used for scoring, sorting is not necessary
                is_sorted=True,
            )
        )._df.rename(columns={"q_id": "qid", "id": "docno"})

        result = ff_scores[["qid", "docno", "score", "query"]].merge(
            inp[["qid", "docno", "score"]],
            on=["qid", "docno"],
            suffixes=(None, "_0"),
        )
        return pt.model.add_ranks(result, single_query=False)

    def __repr__(self) -> str:
        """Return a string representation.

        The representation is unique w.r.t. the index and its query encoder.

        :return: The representation.
        """
        cls_name = self.__class__.__name__
        index_id = id(self._index)
        encoder_id = id(self._index._query_encoder)
        return f"{cls_name}({index_id}, {encoder_id})"


class FFInterpolate(pt.Transformer):
    """PyTerrier transformer that interpolates scores computed by `FFScore`."""

    def __init__(self, alpha: float) -> None:
        """Create an FFInterpolate transformer.

        :param alpha: The interpolation parameter.
        """
        # attribute name needs to be exactly this for pyterrier.GridScan to work
        self.alpha = alpha
        super().__init__()

    def transform(self, inp: "pd.DataFrame") -> "pd.DataFrame":
        """Interpolate the scores as `alpha * score_0 + (1 - alpha) * score`.

        :param inp: The PyTerrier data frame.
        :return: A data frame with the interpolated scores.
        """
        new_df = inp[["qid", "docno", "query"]].copy()
        new_df["score"] = self.alpha * inp["score_0"] + (1 - self.alpha) * inp["score"]
        return pt.model.add_ranks(new_df, single_query=False)
