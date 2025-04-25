from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from fast_forward.index.base import Mode

if TYPE_CHECKING:
    from collections.abc import Iterable


def get_indices(
    ids: "Iterable[str]",
    mode: Mode,
    doc_id_to_idx: dict[str, list[int]],
    psg_id_to_idx: dict[str, int],
) -> tuple[list[int], list[str]]:
    """Find the array indices for a list of document/passage IDs.

    :param ids: IDs to return vectors for.
    :param mode: The ranking mode.
    :param doc_id_to_idx: Document IDs mapped to non-chunked indices.
    :param psg_id_to_idx: Passage IDs mapped to non-chunked indices.
    :raises IndexError: When the ID cannot be found in the index.
    :return: The indices and corresponding IDs.
    """
    indices = []
    ids_ = []
    for id in ids:
        if mode in (Mode.MAXP, Mode.AVEP):
            cur_indices = doc_id_to_idx.get(id, [])
        elif mode == Mode.FIRSTP:
            cur_indices = doc_id_to_idx.get(id, [])[:1]
        else:
            psg_idx = psg_id_to_idx.get(id)
            cur_indices = [] if psg_idx is None else [psg_idx]

        if len(cur_indices) == 0:
            raise IndexError(f"ID {id} not found in the index.")
        indices.extend(cur_indices)
        ids_.extend([id] * len(cur_indices))
    return indices, ids_


class ChunkIndexer:
    """Utility class for retrieving vectors from chunked indexes."""

    def __init__(
        self,
        chunks: list[np.ndarray] | list[np.memmap],
        doc_id_to_idx: dict[str, list[int]],
        psg_id_to_idx: dict[str, int],
    ) -> None:
        """Create a chunk indexer.

        The first chunk may have a different size than the others.

        :param chunks: A list of chunks.
        :param doc_id_to_idx: Document IDs mapped to non-chunked indices.
        :param psg_id_to_idx: Passage IDs mapped to non-chunked indices.
        """
        self._chunks = chunks
        self._doc_id_to_idx = doc_id_to_idx
        self._psg_id_to_idx = psg_id_to_idx

    def _get_chunk_indices(self, idx: int) -> tuple[int, int]:
        """Given an index, compute the chunk index and index within that chunk.

        The input index may be in `[0, N]`, where `N` is the total number of vectors.

        :param idx: The input index.
        :return: Chunk index and index within that chunk.
        """
        # the first chunk might be larger
        if idx < self._chunks[0].shape[0]:
            return 0, idx

        idx_ = idx - self._chunks[0].shape[0]
        return (
            int(idx_ / self._chunks[1].shape[0]) + 1,
            idx_ % self._chunks[1].shape[0],
        )

    def __call__(
        self, ids: "Iterable[str]", mode: Mode
    ) -> tuple[np.ndarray, list[str]]:
        """Retrieve vectors for the given IDs from the chunks.

        :param ids: IDs to return vectors for.
        :raises IndexError: When the ID cannot be found in the index.
        :return: The vectors and corresponding IDs.
        """
        indices, ids_ = get_indices(ids, mode, self._doc_id_to_idx, self._psg_id_to_idx)

        if len(indices) == 0:
            return np.array([]), []
        if len(self._chunks) == 1:
            return self._chunks[0][indices], ids_

        # group indexes by chunk and index each chunk individually
        items_by_chunk = defaultdict(lambda: ([], []))
        for i, idx in enumerate(indices):
            chunk_idx, idx_in_chunk = self._get_chunk_indices(idx)
            items_by_chunk[chunk_idx][0].append(idx_in_chunk)
            items_by_chunk[chunk_idx][1].append(ids_[i])

        result = []
        out_ids = []
        for chunk_idx, (idx_in_chunk, ids_in_chunk) in items_by_chunk.items():
            result.append(self._chunks[chunk_idx][idx_in_chunk])
            out_ids.extend(ids_in_chunk)

        return np.concatenate(result), out_ids
