import logging
from collections import defaultdict
from typing import Iterable, List, Sequence, Set, Tuple, Union

import numpy as np

from fast_forward.encoder import QueryEncoder
from fast_forward.index import Index, Mode

LOGGER = logging.getLogger(__name__)


class InMemoryIndex(Index):
    """Fast-Forward index that is held in memory."""

    def __init__(
        self,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        init_size: int = 2**14,
        alloc_size: int = 2**14,
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
            init_size (int, optional): Initial index size. Defaults to 2**14.
            alloc_size (int, optional): Size of array allocated when index is full. Defaults to 2**14.
        """
        self._chunks = []
        self._init_size = init_size
        self._alloc_size = alloc_size
        self._cur_chunk_idx = 0
        self._dim = None
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        super().__init__(encoder, mode, encoder_batch_size)

    def __len__(self) -> int:
        # account for the fact that the first chunk might be larger
        if len(self._chunks) == 1:
            return self._cur_chunk_idx
        else:
            return (
                self._chunks[0].shape[0]
                + (len(self._chunks) - 2) * self._alloc_size
                + self._cur_chunk_idx
            )

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        # if this is the first call to _add, no chunks exist
        if len(self._chunks) == 0:
            self._dim = vectors.shape[1]
            self._chunks.append(np.zeros((self._init_size, self._dim)))

        # assign passage and document IDs
        j = len(self)
        for doc_id, psg_id in zip(doc_ids, psg_ids):
            if doc_id is not None:
                self._doc_id_to_idx[doc_id].append(j)
            if psg_id is not None:
                if psg_id not in self._psg_id_to_idx:
                    self._psg_id_to_idx[psg_id] = j
                else:
                    LOGGER.error(f"passage ID {psg_id} already exists")
            j += 1

        # add vectors to chunks
        added = 0
        num_vectors = vectors.shape[0]
        while added < num_vectors:
            cur_chunk_size = self._chunks[-1].shape[0]

            # if current chunk is full, add a new one
            if self._cur_chunk_idx == cur_chunk_size:
                LOGGER.debug("adding new chunk")
                self._chunks.append(np.zeros((self._alloc_size, self._dim)))
                self._cur_chunk_idx = 0

            to_add = min(
                num_vectors - added,
                cur_chunk_size,
                cur_chunk_size - self._cur_chunk_idx,
            )
            self._chunks[-1][self._cur_chunk_idx : self._cur_chunk_idx + to_add] = (
                vectors[added : added + to_add]
            )
            added += to_add
            self._cur_chunk_idx += to_add

    def consolidate(self) -> None:
        """Copy all chunks of the index to one contiguous section in the memory."""
        if len(self._chunks) < 2:
            return
        self._cur_chunk_idx = len(self)
        self._chunks = [np.stack(self._chunks)]

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(self, ids: Iterable[str]) -> Tuple[np.ndarray, List[List[int]]]:
        items_by_chunk = defaultdict(list)
        for id in ids:
            if self.mode in (Mode.MAXP, Mode.AVEP) and id in self._doc_id_to_idx:
                idxs = self._doc_id_to_idx[id]
            elif self.mode == Mode.FIRSTP and id in self._doc_id_to_idx:
                idxs = [self._doc_id_to_idx[id][0]]
            elif self.mode == Mode.PASSAGE and id in self._psg_id_to_idx:
                idxs = [self._psg_id_to_idx[id]]
            else:
                LOGGER.warning(f"no vectors for {id}")
                idxs = []

            for idx in idxs:

                # account for the fact that the first chunk might be larger
                if idx < self._chunks[0].shape[0]:
                    chunk_idx = 0
                    idx_in_chunk = idx
                else:
                    idx -= self._chunks[0].shape[0]
                    chunk_idx = int(idx / self._alloc_size) + 1
                    idx_in_chunk = idx % self._alloc_size

                items_by_chunk[chunk_idx].append((idx_in_chunk, id))

        result_vectors = []
        result_ids = defaultdict(list)
        items_so_far = 0

        for chunk_idx, items in items_by_chunk.items():
            idxs, ids_ = zip(*items)
            result_vectors.append(self._chunks[chunk_idx][list(idxs)])

            for i, id_in_chunk in enumerate(ids_):
                result_ids[id_in_chunk].append(i + items_so_far)

            items_so_far += len(items)
        return np.concatenate(result_vectors), [result_ids[id] for id in ids]
