import logging
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from fast_forward.index.base import IDSequence, Index, Mode

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

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
        init_size: int = 2**14,
        alloc_size: int = 2**14,
    ) -> None:
        """Create an index in memory.

        :param query_encoder: The query encoder to use.
        :param quantizer: The quantizer to use.
        :param mode: The ranking mode.
        :param encoder_batch_size: Batch size for the query encoder.
        :param init_size: Initial index size (number of vectors).
        :param alloc_size: Shard size (number of vectors) allocated when index is full.
        """
        self._shards = []
        self._init_size = init_size
        self._alloc_size = alloc_size
        self._idx_in_cur_shard = 0
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}

        super().__init__(
            query_encoder=query_encoder,
            quantizer=quantizer,
            mode=mode,
            encoder_batch_size=encoder_batch_size,
        )

    def _get_num_vectors(self) -> int:
        # account for the fact that the first shard might be larger
        if len(self._shards) < 2:
            return self._idx_in_cur_shard
        return (
            self._shards[0].shape[0]
            + (len(self._shards) - 2) * self._alloc_size
            + self._idx_in_cur_shard
        )

    def _get_internal_dim(self) -> int | None:
        if len(self._shards) > 0:
            return self._shards[0].shape[-1]
        return None

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence,
        psg_ids: IDSequence,
    ) -> None:
        # if this is the first call to _add, no shards exist
        if len(self._shards) == 0:
            self._shards.append(
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

        # add vectors to shards
        added = 0
        num_vectors = vectors.shape[0]
        while added < num_vectors:
            cur_shard_size, dim = self._shards[-1].shape

            # if current shard is full, add a new one
            if self._idx_in_cur_shard == cur_shard_size:
                LOGGER.debug("adding new shard")
                self._shards.append(np.zeros((self._alloc_size, dim)))
                self._idx_in_cur_shard = 0
                cur_shard_size = self._alloc_size

            to_add = min(
                num_vectors - added,
                cur_shard_size,
                cur_shard_size - self._idx_in_cur_shard,
            )
            self._shards[-1][
                self._idx_in_cur_shard : self._idx_in_cur_shard + to_add
            ] = vectors[added : added + to_add]
            added += to_add
            self._idx_in_cur_shard += to_add

    def consolidate(self) -> None:
        """Combine all shards of the index in one contiguous section in the memory."""
        # combine all shards up to the last one entirely, and take only whats in use of
        # the last one
        self._shards = [
            np.concatenate(
                self._shards[:-1] + [self._shards[-1][: self._idx_in_cur_shard]]
            )
        ]
        self._idx_in_cur_shard = self._shards[0].shape[0]

    def _get_doc_ids(self) -> set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> set[str]:
        return set(self._psg_id_to_idx.keys())

    def _index(self, idxs: "Sequence[int]") -> np.ndarray:
        """Return vectors from the internal array(s).

        This method retrieves vectors from the correct shards (if there are any).

        :param idxs: Indices to return vectors for.
        :return: The vectors in the same order as the provided indices.
        """
        if len(idxs) == 0:
            return np.array([])

        # if there is no sharding, simply use numpy indexing
        if len(self._shards) == 1:
            return self._shards[0][idxs]

        # otherwise, group indexes by shard and index each shared individually
        items_by_shard = defaultdict(lambda: ([], []))
        for i, idx in enumerate(idxs):
            # the first shard might be larger
            if idx < self._shards[0].shape[0]:
                shard_idx = 0
                idx_in_shard = idx
            else:
                idx_ = idx - self._shards[0].shape[0]
                shard_idx = int(idx_ / self._alloc_size) + 1
                idx_in_shard = idx_ % self._alloc_size
            items_by_shard[shard_idx][0].append(idx_in_shard)
            items_by_shard[shard_idx][1].append(i)

        result = []
        ordering = []
        for shard_idx, (idxs_in_shard, i_) in items_by_shard.items():
            result.append(self._shards[shard_idx][idxs_in_shard])
            ordering.extend(i_)

        return np.concatenate(result)[np.argsort(ordering)]

    def _get_idxs(self, id: str) -> list[int]:
        """Find the internal array indices for a document/passage ID.

        Takes ranking mode into account.

        :param id: The ID to return the indices for.
        :raises IndexError: When the ID cannot be found in the index.
        :return: The internal array indices.
        """
        if self.mode in (Mode.MAXP, Mode.AVEP):
            return self._doc_id_to_idx.get(id, [])
        if self.mode == Mode.FIRSTP:
            return self._doc_id_to_idx.get(id, [])[:1]

        psg_idx = self._psg_id_to_idx.get(id)
        return [] if psg_idx is None else [psg_idx]

    def _get_vectors(self, ids: "Iterable[str]") -> tuple[np.ndarray, list[str]]:
        idxs = []
        ids_ = []
        for id in ids:
            cur_idxs = self._get_idxs(id)
            if len(cur_idxs) == 0:
                raise IndexError(f"ID {id} not found in the index.")
            idxs.extend(cur_idxs)
            ids_.extend([id] * len(cur_idxs))
        return self._index(idxs), ids_

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
            idxs = range(i, min(i + batch_size, num_vectors))
            yield (
                self._index(idxs),
                list(map(idx_to_doc_id.get, idxs)),
                list(map(idx_to_psg_id.get, idxs)),
            )
