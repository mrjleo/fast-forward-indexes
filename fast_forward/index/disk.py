import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

import fast_forward
from fast_forward.encoder import Encoder
from fast_forward.index import Index, Mode
from fast_forward.index.memory import InMemoryIndex

LOGGER = logging.getLogger(__name__)


class OnDiskIndex(Index):
    """Fast-Forward index that is read from disk.

    Uses HDF5 via h5py under the hood. The buffer (ds_buffer_size) works around a h5py limitation.
    More information: https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing
    """

    def __init__(
        self,
        index_file: Path,
        dim: int,
        query_encoder: Encoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        init_size: int = 2**14,
        resize_min_val: int = 2**10,
        hdf5_chunk_size: int = None,
        dtype: np.dtype = np.float32,
        max_id_length: int = 8,
        overwrite: bool = False,
        ds_buffer_size: int = 2**10,
    ) -> None:
        """Constructor.

        Args:
            index_file (Path): Index file to create (or overwrite).
            dim (int): Vector dimension.
            query_encoder (Encoder, optional): Query encoder. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Batch size for query encoder. Defaults to 32.
            init_size (int, optional): Initial size to allocate (number of vectors). Defaults to 2**14.
            resize_min_val (int, optional): Minimum number of vectors to increase index size by. Defaults to 2**10.
            hdf5_chunk_size (int, optional): Override chunk size used by HDF5. Defaults to None.
            dtype (np.dtype, optional): Vector dtype. Defaults to np.float32.
            max_id_length (int, optional): Maximum length of document and passage IDs (number of characters). Defaults to 8.
            overwrite (bool, optional): Overwrite index file if it exists. Defaults to False.
            ds_buffer_size (int, optional): Maximum number of vectors to retrieve from the HDF5 dataset at once. Defaults to 2**10.

        Raises:
            ValueError: When the file exists and `overwrite=False`.
        """
        if index_file.exists() and not overwrite:
            raise ValueError(f"File {index_file} exists.")

        super().__init__(query_encoder, mode, encoder_batch_size)
        self._index_file = index_file.absolute()
        self._resize_min_val = resize_min_val
        self._ds_buffer_size = ds_buffer_size
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}

        with h5py.File(self._index_file, "w") as fp:
            fp.attrs["num_vectors"] = 0
            fp.attrs["ff_version"] = fast_forward.__version__
            fp.create_dataset(
                "vectors",
                (init_size, dim),
                dtype,
                maxshape=(None, dim),
                chunks=True if hdf5_chunk_size is None else (hdf5_chunk_size, dim),
            )
            fp.create_dataset(
                "doc_ids",
                (init_size,),
                f"S{max_id_length}",
                maxshape=(None,),
                chunks=True if hdf5_chunk_size is None else (hdf5_chunk_size,),
            )
            fp.create_dataset(
                "psg_ids",
                (init_size,),
                f"S{max_id_length}",
                maxshape=(None,),
                chunks=True if hdf5_chunk_size is None else (hdf5_chunk_size,),
            )

    def __len__(self) -> int:
        with h5py.File(self._index_file, "r") as fp:
            return fp.attrs["num_vectors"]

    @property
    def dim(self) -> int:
        with h5py.File(self._index_file, "r") as fp:
            return fp["vectors"].shape[1]

    def to_memory(self, buffer_size=None) -> InMemoryIndex:
        """Load the index entirely into memory.

        Args:
            buffer_size (int, optional): Use a buffer instead of adding all vectors at once. Defaults to None.

        Returns:
            InMemoryIndex: The loaded index.
        """
        with h5py.File(self._index_file, "r") as fp:
            index = InMemoryIndex(
                dim=self.dim,
                query_encoder=self._query_encoder,
                mode=self.mode,
                encoder_batch_size=self._encoder_batch_size,
                init_size=len(self),
                dtype=fp["vectors"].dtype,
            )

            buffer_size = buffer_size or fp.attrs["num_vectors"]
            for i_low in range(0, fp.attrs["num_vectors"], buffer_size):
                i_up = min(i_low + buffer_size, fp.attrs["num_vectors"])

                # we can only add vectors of the same type (doc IDs, passage IDs, or both) in one batch
                has_doc_id, has_psg_id, has_both_ids = [], [], []
                vecs = fp["vectors"][i_low:i_up]
                doc_ids = fp["doc_ids"].asstr()[i_low:i_up]
                psg_ids = fp["psg_ids"].asstr()[i_low:i_up]
                for j, (doc_id, psg_id) in enumerate(zip(doc_ids, psg_ids)):
                    if len(doc_id) == 0:
                        has_psg_id.append(j)
                    elif len(psg_id) == 0:
                        has_doc_id.append(j)
                    else:
                        has_both_ids.append(j)

                if len(has_doc_id) > 0:
                    index.add(
                        vecs[has_doc_id],
                        doc_ids=doc_ids[has_doc_id],
                    )
                if len(has_psg_id) > 0:
                    index.add(
                        vecs[has_psg_id],
                        psg_ids=psg_ids[has_psg_id],
                    )
                if len(has_both_ids) > 0:
                    index.add(
                        vecs[has_both_ids],
                        doc_ids=doc_ids[has_both_ids],
                        psg_ids=psg_ids[has_both_ids],
                    )
        return index

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Union[Sequence[str], None],
        psg_ids: Union[Sequence[str], None],
    ) -> None:
        with h5py.File(self._index_file, "a") as fp:
            num_new_vecs = vectors.shape[0]
            capacity = fp["vectors"].shape[0]

            # check if we have enough space, resize if necessary
            cur_num_vectors = fp.attrs["num_vectors"]
            space_left = capacity - cur_num_vectors
            if num_new_vecs > space_left:
                new_size = max(
                    capacity + num_new_vecs - space_left, self._resize_min_val
                )
                LOGGER.debug("resizing index from %s to %s", capacity, new_size)
                fp["vectors"].resize(new_size, axis=0)
                fp["doc_ids"].resize(new_size, axis=0)
                fp["psg_ids"].resize(new_size, axis=0)

            # check all IDs first before adding anything
            doc_id_size = fp["doc_ids"].dtype.itemsize
            psg_id_size = fp["psg_ids"].dtype.itemsize
            add_doc_ids, add_psg_ids = [], []
            if doc_ids is not None:
                for i, doc_id in enumerate(doc_ids):
                    if len(doc_id) > doc_id_size:
                        raise RuntimeError(
                            f"Document ID {doc_id} is longer than the maximum ({doc_id_size} characters)."
                        )
                    add_doc_ids.append((doc_id, cur_num_vectors + i))
            if psg_ids is not None:
                for i, psg_id in enumerate(psg_ids):
                    if len(psg_id) > psg_id_size:
                        raise RuntimeError(
                            f"Passage ID {psg_id} is longer than the maximum ({psg_id_size} characters)."
                        )
                    add_psg_ids.append((psg_id, cur_num_vectors + i))

            # add new IDs to index and in-memory mappings
            if doc_ids is not None:
                for doc_id, idx in add_doc_ids:
                    self._doc_id_to_idx[doc_id].append(idx)
                fp["doc_ids"][
                    cur_num_vectors : cur_num_vectors + num_new_vecs
                ] = doc_ids
            if psg_ids is not None:
                for psg_id, idx in add_psg_ids:
                    self._psg_id_to_idx[psg_id] = idx
                fp["psg_ids"][
                    cur_num_vectors : cur_num_vectors + num_new_vecs
                ] = psg_ids

            # add new vectors
            fp["vectors"][cur_num_vectors : cur_num_vectors + num_new_vecs] = vectors
            fp.attrs["num_vectors"] += num_new_vecs

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(self, ids: Iterable[str]) -> Tuple[np.ndarray, List[List[int]]]:
        idx_pairs = []
        with h5py.File(self._index_file, "r") as fp:
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
                    idx_pairs.append((id, idx))

            # h5py requires accessing the dataset with sorted indices
            idx_pairs.sort(key=lambda x: x[1])
            id_to_idxs = defaultdict(list)
            vec_idxs = []
            for id_idx, (id, vec_idx) in enumerate(idx_pairs):
                vec_idxs.append(vec_idx)
                id_to_idxs[id].append(id_idx)

            # reading all vectors at once slows h5py down significantly, so we read them in chunks and concatenate
            vectors = np.concatenate(
                [
                    fp["vectors"][vec_idxs[i : i + self._ds_buffer_size]]
                    for i in range(0, len(vec_idxs), self._ds_buffer_size)
                ]
            )
            return vectors, [id_to_idxs[id] for id in ids]

    @classmethod
    def load(
        cls,
        index_file: Path,
        encoder: Encoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        resize_min_val: int = 2**10,
        ds_buffer_size: int = 2**10,
    ) -> "OnDiskIndex":
        """Open an existing index on disk.

        Args:
            index_file (Path): Index file to open.
            encoder (Encoder, optional): Query encoder. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Batch size for query encoder. Defaults to 32.
            resize_min_val (int, optional): Minimum number of vectors to increase index size by. Defaults to 2**10.
            ds_buffer_size (int, optional): Maximum number of vectors to retrieve from the HDF5 dataset at once. Defaults to 2**10.

        Returns:
            OnDiskIndex: The index.
        """
        index = cls.__new__(cls)
        super(OnDiskIndex, index).__init__(encoder, mode, encoder_batch_size)
        index._index_file = index_file.absolute()
        index._resize_min_val = resize_min_val
        index._ds_buffer_size = ds_buffer_size

        # read ID mappings
        index._doc_id_to_idx = defaultdict(list)
        index._psg_id_to_idx = {}
        with h5py.File(index._index_file, "r") as fp:
            for i, (doc_id, psg_id) in tqdm(
                enumerate(
                    zip(
                        fp["doc_ids"].asstr()[: fp.attrs["num_vectors"]],
                        fp["psg_ids"].asstr()[: fp.attrs["num_vectors"]],
                    ),
                ),
                total=fp.attrs["num_vectors"],
            ):
                if len(doc_id) > 0:
                    index._doc_id_to_idx[doc_id].append(i)
                if len(psg_id) > 0:
                    index._psg_id_to_idx[psg_id] = i
        return index
