import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

import h5py
import numpy as np
from tqdm import tqdm

import fast_forward
from fast_forward.encoder import Encoder
from fast_forward.index import IDSequence, Index, Mode
from fast_forward.index.memory import InMemoryIndex
from fast_forward.quantizer import Quantizer

LOGGER = logging.getLogger(__name__)


class OnDiskIndex(Index):
    """Fast-Forward index that is read on-demand from disk.

    Uses HDF5 via h5py under the hood. The buffer (`ds_buffer_size`) works around a [h5py limitation](https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing).
    """

    def __init__(
        self,
        index_file: Path,
        query_encoder: Encoder = None,
        quantizer: Quantizer = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
        init_size: int = 2**14,
        resize_min_val: int = 2**10,
        hdf5_chunk_size: int = None,
        max_id_length: int = 8,
        overwrite: bool = False,
        ds_buffer_size: int = 2**10,
    ) -> None:
        """Create an index.

        Args:
            index_file (Path): Index file to create (or overwrite).
            query_encoder (Encoder, optional): Query encoder. Defaults to None.
            quantizer (Quantizer, optional): The quantizer to use. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.MAXP.
            encoder_batch_size (int, optional): Batch size for query encoder. Defaults to 32.
            init_size (int, optional): Initial size to allocate (number of vectors). Defaults to 2**14.
            resize_min_val (int, optional): Minimum number of vectors to increase index size by. Defaults to 2**10.
            hdf5_chunk_size (int, optional): Override chunk size used by HDF5. Defaults to None.
            max_id_length (int, optional): Maximum length of document and passage IDs (number of characters). Defaults to 8.
            overwrite (bool, optional): Overwrite index file if it exists. Defaults to False.
            ds_buffer_size (int, optional): Maximum number of vectors to retrieve from the HDF5 dataset at once. Defaults to 2**10.

        Raises:
            ValueError: When the file exists and `overwrite=False`.
        """
        if index_file.exists() and not overwrite:
            raise ValueError(f"File {index_file} exists.")

        self._index_file = index_file.absolute()
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}

        self._init_size = init_size
        self._resize_min_val = resize_min_val
        self._hdf5_chunk_size = hdf5_chunk_size
        self._max_id_length = max_id_length
        self._ds_buffer_size = ds_buffer_size

        LOGGER.debug("creating file %s", self._index_file)
        with h5py.File(self._index_file, "w") as fp:
            fp.attrs["num_vectors"] = 0
            fp.attrs["ff_version"] = fast_forward.__version__

        super().__init__(
            query_encoder=query_encoder,
            quantizer=quantizer,
            mode=mode,
            encoder_batch_size=encoder_batch_size,
        )

    @Index.quantizer.setter
    def quantizer(self, quantizer: Quantizer) -> None:
        # call the setter of the super class
        Index.quantizer.fset(self, quantizer)

        # serialize the quantizer and store it on disk
        with h5py.File(self._index_file, "a") as fp:
            if "quantizer" in fp:
                del fp["quantizer"]

            meta, attributes, data = self.quantizer.serialize()
            fp.create_group("quantizer/meta").attrs.update(meta)
            fp.create_group("quantizer/attributes").attrs.update(attributes)
            data_group = fp.create_group("quantizer/data")
            for k, v in data.items():
                data_group.create_dataset(k, data=v)

    def _create_ds(self, fp: h5py.File, dim: int, dtype: np.dtype) -> None:
        """Create the HDF5 datasets for vectors and IDs.

        Args:
            fp (h5py.File): Index file (write permissions).
            dim (int): Dimension of the vectors.
            dtype (np.dtype): Type of the vectors.
        """
        fp.create_dataset(
            "vectors",
            (self._init_size, dim),
            dtype,
            maxshape=(None, dim),
            chunks=(
                True if self._hdf5_chunk_size is None else (self._hdf5_chunk_size, dim)
            ),
        )
        fp.create_dataset(
            "doc_ids",
            (self._init_size,),
            f"S{self._max_id_length}",
            maxshape=(None,),
            chunks=(
                True if self._hdf5_chunk_size is None else (self._hdf5_chunk_size,)
            ),
        )
        fp.create_dataset(
            "psg_ids",
            (self._init_size,),
            f"S{self._max_id_length}",
            maxshape=(None,),
            chunks=(
                True if self._hdf5_chunk_size is None else (self._hdf5_chunk_size,)
            ),
        )

    def __len__(self) -> int:
        with h5py.File(self._index_file, "r") as fp:
            return fp.attrs["num_vectors"]

    def _get_internal_dim(self) -> Optional[int]:
        with h5py.File(self._index_file, "r") as fp:
            if "vectors" in fp:
                return fp["vectors"].shape[1]
        return None

    def to_memory(self, buffer_size=None) -> InMemoryIndex:
        """Load the index entirely into memory.

        Args:
            buffer_size (int, optional): Use batches instead of adding all vectors at once. Defaults to None.

        Returns:
            InMemoryIndex: The loaded index.
        """
        index = InMemoryIndex(
            query_encoder=self._query_encoder,
            quantizer=self._quantizer,
            mode=self.mode,
            encoder_batch_size=self._encoder_batch_size,
            init_size=len(self),
        )
        with h5py.File(self._index_file, "r") as fp:
            buffer_size = buffer_size or fp.attrs["num_vectors"]
            for i_low in range(0, fp.attrs["num_vectors"], buffer_size):
                i_up = min(i_low + buffer_size, fp.attrs["num_vectors"])

                # IDs that don't exist will be returned as empty strings here
                doc_ids = fp["doc_ids"].asstr()[i_low:i_up]
                doc_ids[doc_ids == ""] = None
                psg_ids = fp["psg_ids"].asstr()[i_low:i_up]
                psg_ids[psg_ids == ""] = None
                index._add(fp["vectors"][i_low:i_up], doc_ids=doc_ids, psg_ids=psg_ids)
        return index

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Optional[str]],
        psg_ids: Sequence[Optional[str]],
    ) -> None:
        with h5py.File(self._index_file, "a") as fp:
            # if this is the first call to _add, no datasets exist
            if "vectors" not in fp:
                self._create_ds(fp, vectors.shape[-1], vectors.dtype)

            # check all IDs first before adding anything
            doc_id_size = fp["doc_ids"].dtype.itemsize
            for doc_id in doc_ids:
                if doc_id is not None and len(doc_id) > doc_id_size:
                    raise RuntimeError(
                        f"Document ID {doc_id} is longer than the maximum ({doc_id_size} characters)."
                    )
            psg_id_size = fp["psg_ids"].dtype.itemsize
            for psg_id in psg_ids:
                if psg_id is not None and len(psg_id) > psg_id_size:
                    raise RuntimeError(
                        f"Passage ID {psg_id} is longer than the maximum ({psg_id_size} characters)."
                    )

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

            # add new document IDs to index and in-memory mappings
            doc_id_idxs, non_null_doc_ids = [], []
            for i, doc_id in enumerate(doc_ids):
                if doc_id is not None:
                    self._doc_id_to_idx[doc_id].append(cur_num_vectors + i)
                    doc_id_idxs.append(cur_num_vectors + i)
                    non_null_doc_ids.append(doc_id)
            fp["doc_ids"][doc_id_idxs] = non_null_doc_ids

            # add new passage IDs to index and in-memory mappings
            psg_id_idxs, non_null_psg_ids = [], []
            for i, psg_id in enumerate(psg_ids):
                if psg_id is not None:
                    self._psg_id_to_idx[psg_id] = cur_num_vectors + i
                    psg_id_idxs.append(cur_num_vectors + i)
                    non_null_psg_ids.append(psg_id)
            fp["psg_ids"][psg_id_idxs] = non_null_psg_ids

            # add new vectors
            fp["vectors"][cur_num_vectors : cur_num_vectors + num_new_vecs] = vectors
            fp.attrs["num_vectors"] += num_new_vecs

    @property
    def doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    @property
    def psg_ids(self) -> Set[str]:
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

    def _batch_iter(
        self, batch_size: int
    ) -> Iterator[Tuple[np.ndarray, IDSequence, IDSequence]]:
        with h5py.File(self._index_file, "r") as fp:
            num_vectors = fp.attrs["num_vectors"]
            for i in range(0, num_vectors, batch_size):
                j = min(i + batch_size, num_vectors)
                doc_ids = fp["doc_ids"].asstr()[i:j]
                doc_ids[doc_ids == ""] = None
                psg_ids = fp["psg_ids"].asstr()[i:j]
                psg_ids[psg_ids == ""] = None
                yield (
                    fp["vectors"][i:j],
                    doc_ids.tolist(),
                    psg_ids.tolist(),
                )

    @classmethod
    def load(
        cls,
        index_file: Path,
        query_encoder: Encoder = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
        resize_min_val: int = 2**10,
        ds_buffer_size: int = 2**10,
    ) -> "OnDiskIndex":
        """Open an existing index on disk.

        Args:
            index_file (Path): Index file to open.
            query_encoder (Encoder, optional): Query encoder. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.MAXP.
            encoder_batch_size (int, optional): Batch size for query encoder. Defaults to 32.
            resize_min_val (int, optional): Minimum value to increase index size by. Defaults to 2**10.
            ds_buffer_size (int, optional): Maximum number of vectors to retrieve from the HDF5 dataset at once. Defaults to 2**10.

        Returns:
            OnDiskIndex: The index.
        """
        LOGGER.debug("reading file %s", index_file)

        index = cls.__new__(cls)
        super(OnDiskIndex, index).__init__(
            query_encoder=query_encoder,
            quantizer=None,
            mode=mode,
            encoder_batch_size=encoder_batch_size,
        )
        index._index_file = index_file.absolute()
        index._resize_min_val = resize_min_val
        index._ds_buffer_size = ds_buffer_size

        # deserialize quantizer if any
        with h5py.File(index_file, "r") as fp:
            if "quantizer" in fp:
                index._quantizer = Quantizer.deserialize(
                    dict(fp["quantizer/meta"].attrs),
                    dict(fp["quantizer/attributes"].attrs),
                    {k: v[:] for k, v in fp["quantizer/data"].items()},
                )

        # read ID mappings
        with h5py.File(index_file, "r") as fp:
            index._doc_id_to_idx = defaultdict(list)
            index._psg_id_to_idx = {}

            num_vectors = fp.attrs["num_vectors"]
            if num_vectors == 0:
                return index

            doc_id_iter = fp["doc_ids"].asstr()[:num_vectors]
            psg_id_iter = fp["psg_ids"].asstr()[:num_vectors]
            for i, (doc_id, psg_id) in tqdm(
                enumerate(zip(doc_id_iter, psg_id_iter)),
                total=num_vectors,
            ):
                if len(doc_id) > 0:
                    index._doc_id_to_idx[doc_id].append(i)
                if len(psg_id) > 0:
                    index._psg_id_to_idx[psg_id] = i
        return index
