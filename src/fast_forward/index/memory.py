import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from fast_forward.index.base import IDSequence, Index, Mode
from fast_forward.index.util import ChunkIndexer

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from fast_forward.encoder.base import Encoder
    from fast_forward.quantizer import Quantizer

LOGGER = logging.getLogger(__name__)


class InMemoryIndex(Index):
    """Fast-Forward index that is held entirely in memory."""

    def __init__(
        self,
        query_encoder: "Encoder | None" = None,
        quantizer: "Quantizer | None" = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
        init_size: int = 2**16,
        alloc_size: int = 2**16,
    ) -> None:
        """Create an index in memory.

        :param query_encoder: The query encoder to use.
        :param quantizer: The quantizer to use.
        :param mode: The ranking mode.
        :param encoder_batch_size: Batch size for the query encoder.
        :param init_size: Size of initial chunk (number of vectors).
        :param alloc_size: Size of additionally allocated chunks (number of vectors).
        """
        self._chunks = []
        self._init_size = init_size
        self._alloc_size = alloc_size
        self._idx_in_cur_chunk = 0
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        self._chunk_indexer = ChunkIndexer(
            self._chunks, self._doc_id_to_idx, self._psg_id_to_idx
        )

        super().__init__(
            query_encoder=query_encoder,
            quantizer=quantizer,
            mode=mode,
            encoder_batch_size=encoder_batch_size,
        )

    def _get_num_vectors(self) -> int:
        # account for the fact that the first chunk might be larger
        if len(self._chunks) < 2:
            return self._idx_in_cur_chunk
        return (
            self._chunks[0].shape[0]
            + (len(self._chunks) - 2) * self._alloc_size
            + self._idx_in_cur_chunk
        )

    def _get_internal_dim(self) -> int | None:
        if len(self._chunks) > 0:
            return self._chunks[0].shape[-1]
        return None

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence,
        psg_ids: IDSequence,
    ) -> None:
        # if this is the first call to _add, no chunks exist
        if len(self._chunks) == 0:
            self._chunks.append(
                np.zeros((self._init_size, vectors.shape[-1]), dtype=vectors.dtype)
            )

        # assign passage and document IDs
        for i, doc_id in enumerate(doc_ids, len(self)):
            if doc_id is not None:
                self._doc_id_to_idx[doc_id].append(i)

        for i, psg_id in enumerate(psg_ids, len(self)):
            if psg_id is None:
                continue
            if psg_id in self._psg_id_to_idx:
                raise RuntimeError(f"Passage ID {psg_id} already exists.")
            self._psg_id_to_idx[psg_id] = i

        # add vectors to chunks
        added = 0
        num_vectors = vectors.shape[0]
        while added < num_vectors:
            cur_chunk_size, dim = self._chunks[-1].shape

            # if current chunk is full, add a new one
            if self._idx_in_cur_chunk == cur_chunk_size:
                LOGGER.debug("adding new chunk")
                self._chunks.append(np.zeros((self._alloc_size, dim)))
                self._idx_in_cur_chunk = 0
                cur_chunk_size = self._alloc_size

            to_add = min(
                num_vectors - added,
                cur_chunk_size,
                cur_chunk_size - self._idx_in_cur_chunk,
            )
            self._chunks[-1][
                self._idx_in_cur_chunk : self._idx_in_cur_chunk + to_add
            ] = vectors[added : added + to_add]
            added += to_add
            self._idx_in_cur_chunk += to_add

    def consolidate(self) -> None:
        """Combine all chunks of the index in one contiguous section in the memory."""
        # combine all chunks up to the last one entirely, and take only whats in use of
        # the last one
        self._chunks = [
            np.concatenate(
                self._chunks[:-1] + [self._chunks[-1][: self._idx_in_cur_chunk]]
            )
        ]
        self._idx_in_cur_chunk = self._chunks[0].shape[0]
        self._chunk_indexer._chunks = self._chunks

    def _get_doc_ids(self) -> set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(self, ids: "Iterable[str]") -> tuple[np.ndarray, list[str]]:
        return self._chunk_indexer(ids, self.mode)

    def _batch_iter(
        self, batch_size: int
    ) -> "Iterator[tuple[np.ndarray, IDSequence, IDSequence]]":
        LOGGER.info("creating ID mappings for this index")
        idx_to_doc_id = {
            idx: doc_id
            for doc_id, idxs in tqdm(self._doc_id_to_idx.items())
            for idx in idxs
        }
        idx_to_psg_id = {
            idx: psg_id for psg_id, idx in tqdm(self._psg_id_to_idx.items())
        }

        num_vectors = len(self)
        for i in range(0, num_vectors, batch_size):
            j = min(i + batch_size, num_vectors)

            # the current batch is between i (incl.) and j (excl.)
            i_chunk_idx, i_idx_in_chunk = self._chunk_indexer._get_chunk_indices(i)
            j_chunk_idx, j_idx_in_chunk = self._chunk_indexer._get_chunk_indices(j - 1)

            arrays = []

            # if the batch spans multiple chunks, collect them in a list
            while i_chunk_idx < j_chunk_idx:
                arrays.append(self._chunks[i_chunk_idx][i_idx_in_chunk:])
                i_chunk_idx += 1
                i_idx_in_chunk = 0

            # now i_chunk_idx == j_chunk_idx
            arrays.append(
                self._chunks[i_chunk_idx][i_idx_in_chunk : j_idx_in_chunk + 1]
            )

            yield (
                np.concatenate(arrays),
                list(map(idx_to_doc_id.get, range(i, j))),
                list(map(idx_to_psg_id.get, range(i, j))),
            )
