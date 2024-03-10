"""
.. include:: ../docs/util.md
"""

from typing import Callable

import numpy as np
import pandas as pd
from tqdm import tqdm

from fast_forward.index import Index, Mode
from fast_forward.ranking import Ranking


def to_ir_measures(ranking: Ranking) -> pd.DataFrame:
    """Return a ranking as a data frame suitable for the ir-measures library.

    Args:
        ranking (Ranking): The input ranking.

    Returns:
        pd.DataFrame: The data frame.
    """
    return ranking._df[["q_id", "id", "score"]].rename(
        columns={"q_id": "query_id", "id": "doc_id"}
    )


def cos_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine distance of two vectors.

    Args:
        a (np.ndarray): First vector.
        b (np.ndarray): Second vector.

    Returns:
        np.ndarray: Cosine distance.
    """
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def create_coalesced_index(
    source_index: Index,
    target_index: Index,
    delta: float,
    distance_function: Callable[[np.ndarray, np.ndarray], float] = cos_dist,
    buffer_size: int = None,
) -> None:
    """Create a compressed index using sequential coalescing.

    Args:
        source_index (Index): The source index. Should contain multiple vectors for each document.
        target_index (Index): The target index. Must be empty.
        delta (float): The coalescing threshold.
        distance_function (Callable[[np.ndarray, np.ndarray], float]): The distance function. Defaults to cos_dist.
        buffer_size (int, optional): Use a buffer instead of adding all vectors at the end. Defaults to None.
    """
    assert len(target_index.doc_ids) == 0
    buffer_size = buffer_size or len(source_index.doc_ids)
    source_index.mode = Mode.MAXP

    def _coalesce(P):
        P_new = []
        A = []
        A_avg = None
        first_iteration = True
        for v in P:
            if first_iteration:
                first_iteration = False
            elif distance_function(v, A_avg) >= delta:
                P_new.append(A_avg)
                A = []
            A.append(v)
            A_avg = np.mean(A, axis=0)
        P_new.append(A_avg)
        return P_new

    vectors, doc_ids = [], []
    for doc_id in tqdm(source_index.doc_ids):

        # check if buffer is full
        if len(vectors) == buffer_size:
            target_index.add(np.array(vectors), doc_ids=doc_ids)
            vectors, doc_ids = [], []

        v_old, _ = source_index._get_vectors([doc_id])
        v_new = _coalesce(v_old)
        vectors.extend(v_new)
        doc_ids.extend([doc_id] * len(v_new))

    if len(vectors) > 0:
        target_index.add(np.array(vectors), doc_ids=doc_ids)

    assert source_index.doc_ids == target_index.doc_ids
