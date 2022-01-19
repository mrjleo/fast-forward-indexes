import logging
from collections import defaultdict
from fast_forward.ranking import Ranking


LOGGER = logging.getLogger(__name__)


def interpolate(
    r1: Ranking, r2: Ranking, alpha: float, name: str = None, sort: bool = True
) -> Ranking:
    """Interpolate scores. For each query-doc pair:
        * If the pair has only one score, ignore it.
        * If the pair has two scores, interpolate: r1 * alpha + r2 * (1 - alpha).

    Args:
        r1 (Ranking): Scores from the first retriever.
        r2 (Ranking): Scores from the second retriever.
        alpha (float): Interpolation weight.
        name (str, optional): Ranking name. Defaults to None.
        sort (bool, optional): Whether to sort the documents by score. Defaults to True.

    Returns:
        Ranking: Interpolated ranking.
    """
    assert r1.q_ids == r2.q_ids
    results = defaultdict(dict)
    for q_id in r1:
        for doc_id in r1[q_id].keys() & r2[q_id].keys():
            results[q_id][doc_id] = (
                alpha * r1[q_id][doc_id] + (1 - alpha) * r2[q_id][doc_id]
            )
    return Ranking(results, name=name, sort=sort, copy=False)
