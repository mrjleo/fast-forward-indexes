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
        chunk_size: int = 2**14,
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
            chunk_size (int, optional): Index chunk size (i.e., size of allocated arrays). Defaults to 2**14.
        """
        self._chunks = []
        self._chunk_size = chunk_size
        self._cur_chunk_idx = 0
        self._dim = None

        self._doc_ids = []
        self._psg_ids = []
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}

        super().__init__(encoder, mode, encoder_batch_size)

    def _add_chunk(self) -> None:
        """Add a chunk to the index."""
        self._chunks.append(np.zeros((self._chunk_size, self._dim)))

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        # if this is the first call to _add, no chunks exist
        if len(self._chunks) == 0:
            self._dim = vectors.shape[1]
            self._add_chunk()

        # assign passage and document IDs
        j = (len(self._chunks) - 1) * self._chunk_size + self._cur_chunk_idx
        for doc_id, psg_id in zip(doc_ids, psg_ids):
            if doc_id is not None:
                self._doc_id_to_idx[doc_id].append(j)
            if psg_id is not None:
                assert psg_id not in self._psg_id_to_idx
                self._psg_id_to_idx[psg_id] = j
            j += 1

        self._doc_ids.extend(doc_ids)
        self._psg_ids.extend(psg_ids)

        # add vectors to chunks
        i = 0
        num_vectors = vectors.shape[0]
        while i < num_vectors:
            remaining = num_vectors - i
            to_add = min(
                remaining, self._chunk_size, self._chunk_size - self._cur_chunk_idx
            )
            self._chunks[-1][self._cur_chunk_idx : self._cur_chunk_idx + to_add] = (
                vectors[i : i + to_add]
            )
            i += to_add
            self._cur_chunk_idx += to_add

            if self._cur_chunk_idx == self._chunk_size:
                self._add_chunk()
                self._cur_chunk_idx = 0

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
                chunk_idx = int(idx / self._chunk_size)
                idx_in_chunk = idx % self._chunk_size
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
