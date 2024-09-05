import logging
from collections import defaultdict
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import numpy as np
from tqdm import tqdm

from fast_forward.encoder import Encoder
from fast_forward.index import IDSequence, Index, Mode
from fast_forward.quantizer import Quantizer

LOGGER = logging.getLogger(__name__)


class InMemoryIndex(Index):
    """Fast-Forward index that is held entirely in memory."""

    def __init__(
        self,
        query_encoder: Encoder = None,
        quantizer: Quantizer = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
        init_size: int = 2**14,
        alloc_size: int = 2**14,
    ) -> None:
        """Create an index.

        Args:
            query_encoder (Encoder, optional): The query encoder to use. Defaults to None.
            quantizer (Quantizer, optional): The quantizer to use. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.MAXP.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
            init_size (int, optional): Initial index size. Defaults to 2**14.
            alloc_size (int, optional): Size of shard allocated when index is full. Defaults to 2**14.
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

    def _get_internal_dim(self) -> Optional[int]:
        if len(self._shards) > 0:
            return self._shards[0].shape[-1]
        return None

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Optional[str]],
        psg_ids: Sequence[Optional[str]],
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
        # combine all shards up to the last one entirely, and take only whats in use of the last one
        self._shards = [
            np.concatenate(
                self._shards[:-1] + [self._shards[-1][: self._idx_in_cur_shard]]
            )
        ]
        self._idx_in_cur_shard = self._shards[0].shape[0]

    @property
    def doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    @property
    def psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _index_shards(self, idx: int) -> Tuple[int, int]:
        """Given an index in `[0, N]`, where `N` is the total number of vectors,
        compute the corresponding shard index and index within that shard.

        Args:
            idx (int): The input index.

        Returns:
            Tuple[int, int]: Shard index and index within that shard.
        """
        # the first shard might be larger
        if idx < self._shards[0].shape[0]:
            shard_idx = 0
            idx_in_shard = idx
        else:
            idx -= self._shards[0].shape[0]
            shard_idx = int(idx / self._alloc_size) + 1
            idx_in_shard = idx % self._alloc_size
        return shard_idx, idx_in_shard

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
                LOGGER.warning("no vectors for %s", id)
                idxs = []

            for idx in idxs:
                shard_idx, idx_in_shard = self._index_shards(idx)
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
            return np.array([]), []
        return np.concatenate(result_vectors), [result_ids[id] for id in ids]

    def _batch_iter(
        self, batch_size: int
    ) -> Iterator[Tuple[np.ndarray, IDSequence, IDSequence]]:
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
            i_shard_idx, i_idx_in_shard = self._index_shards(i)
            j_shard_idx, j_idx_in_shard = self._index_shards(j - 1)

            arrays = []

            # if the batch spans multiple shards, collect them in a list
            while i_shard_idx < j_shard_idx:
                arrays.append(self._shards[i_shard_idx][i_idx_in_shard:])
                i_shard_idx += 1
                i_idx_in_shard = 0

            # now i_shard_idx == j_shard_idx
            arrays.append(
                self._shards[i_shard_idx][i_idx_in_shard : j_idx_in_shard + 1]
            )

            yield (
                np.concatenate(arrays),
                list(map(idx_to_doc_id.get, range(i, j))),
                list(map(idx_to_psg_id.get, range(i, j))),
            )
