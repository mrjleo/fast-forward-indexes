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
        dim: int,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        init_size: int = 2**14,
        alloc_size: int = 2**14,
        dtype: np.dtype = np.float32,
    ) -> None:
        """Constructor.

        Args:
            dim (int): Vector dimension.
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
            init_size (int, optional): Initial index size. Defaults to 2**14.
            alloc_size (int, optional): Size of shard allocated when index is full. Defaults to 2**14.
            dtype (np.dtype, optional): Vector dtype. Defaults to np.float32.
        """
        self._shards = []
        self._init_size = init_size
        self._alloc_size = alloc_size
        self._dtype = dtype
        self._idx_in_cur_shard = 0
        self._dim = dim
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        super().__init__(encoder, mode, encoder_batch_size)

    def __len__(self) -> int:
        # account for the fact that the first shard might be larger
        if len(self._shards) < 2:
            return self._idx_in_cur_shard
        else:
            return (
                self._shards[0].shape[0]
                + (len(self._shards) - 2) * self._alloc_size
                + self._idx_in_cur_shard
            )

    @property
    def dim(self) -> int:
        return self._dim

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        # if this is the first call to _add, no shards exist
        if len(self._shards) == 0:
            self._shards.append(
                np.zeros((self._init_size, self._dim), dtype=self._dtype)
            )

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

        # add vectors to shards
        added = 0
        num_vectors = vectors.shape[0]
        while added < num_vectors:
            cur_shard_size = self._shards[-1].shape[0]

            # if current shard is full, add a new one
            if self._idx_in_cur_shard == cur_shard_size:
                LOGGER.debug("adding new shard")
                self._shards.append(np.zeros((self._alloc_size, self._dim)))
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
        # combine all shards up to the last one entirely, and take only whats in use of the last one
        self._shards = [
            np.concatenate(
                self._shards[:-1] + [self._shards[-1][: self._idx_in_cur_shard]]
            )
        ]
        self._idx_in_cur_shard = self._shards[0].shape[0]

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(self, ids: Iterable[str]) -> Tuple[np.ndarray, List[List[int]]]:
        items_by_shard = defaultdict(list)
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
                # account for the fact that the first shard might be larger
                if idx < self._shards[0].shape[0]:
                    shard_idx = 0
                    idx_in_shard = idx
                else:
                    idx -= self._shards[0].shape[0]
                    shard_idx = int(idx / self._alloc_size) + 1
                    idx_in_shard = idx % self._alloc_size
                items_by_shard[shard_idx].append((idx_in_shard, id))

        result_vectors = []
        result_ids = defaultdict(list)
        items_so_far = 0
        for shard_idx, items in items_by_shard.items():
            idxs, ids_ = zip(*items)
            result_vectors.append(self._shards[shard_idx][list(idxs)])
            for i, id_in_shard in enumerate(ids_):
                result_ids[id_in_shard].append(i + items_so_far)
            items_so_far += len(items)

        if len(result_vectors) == 0:
            return np.array([], dtype=self._dtype), []
        return np.concatenate(result_vectors), [result_ids[id] for id in ids]
