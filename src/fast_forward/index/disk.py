import logging
from collections import defaultdict
from typing import TYPE_CHECKING, cast

import h5py
import numpy as np
from tqdm import tqdm

import fast_forward
from fast_forward.index.base import IDSequence, Index, Mode
from fast_forward.index.memory import InMemoryIndex
from fast_forward.quantizer import Quantizer

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from pathlib import Path

    from fast_forward.encoder.base import Encoder

LOGGER = logging.getLogger(__name__)


# h5py does not play nice with pyright, so we add lots of ignores in this class
class OnDiskIndex(Index):
    """Fast-Forward index that is read on-demand from disk.

    Uses HDF5 via h5py under the hood. The `max_indexing_size` argument works around an
    [h5py limitation](https://docs.h5py.org/en/latest/high/dataset.html#fancy-indexing).
    """

    def __init__(
        self,
        index_file: "Path",
        query_encoder: "Encoder | None" = None,
        quantizer: Quantizer | None = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
        init_size: int = 2**14,
        resize_min_val: int = 2**10,
        hdf5_chunk_size: int | None = None,
        max_id_length: int = 8,
        overwrite: bool = False,
        max_indexing_size: int = 2**10,
    ) -> None:
        """Create an index on disk.

        :param index_file: The index file to create (or overwrite).
        :param query_encoder: The query encoder.
        :param quantizer: The quantizer to use.
        :param mode: The ranking mode.
        :param encoder_batch_size: Batch size for the query encoder.
        :param init_size: Initial size to allocate (number of vectors).
        :param resize_min_val: Minimum number of vectors to increase index size by.
        :param hdf5_chunk_size: Override chunk size used by HDF5.
        :param max_id_length:
            Maximum length of document and passage IDs (number of characters).
        :param overwrite: Overwrite index file if it exists.
        :param max_indexing_size:
            Maximum number of vectors to retrieve from the HDF5 dataset at once.
        :raises ValueError: When the file exists and `overwrite=False`.
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
        self._max_indexing_size = max_indexing_size

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

    def _on_quantizer_set(self) -> None:
        assert self.quantizer is not None

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

        :param fp: Index file (write permissions).
        :param dim: Dimension of the vectors.
        :param dtype: Type of the vectors.
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

    def _get_num_vectors(self) -> int:
        with h5py.File(self._index_file, "r") as fp:
            return fp.attrs["num_vectors"]  # pyright: ignore[reportReturnType]

    def _get_internal_dim(self) -> int | None:
        with h5py.File(self._index_file, "r") as fp:
            if "vectors" in fp:
                return fp["vectors"].shape[1]  # pyright: ignore[reportAttributeAccessIssue]
        return None

    def to_memory(self, batch_size: int | None = None) -> InMemoryIndex:
        """Load the index entirely into memory.

        :param batch_size: Use batches instead of adding all vectors at once.
        :return: The loaded index.
        """
        index = InMemoryIndex(
            query_encoder=self._query_encoder,
            quantizer=self._quantizer,
            mode=self.mode,
            encoder_batch_size=self._encoder_batch_size,
            init_size=len(self),
        )
        with h5py.File(self._index_file, "r") as fp:
            num_vectors = cast(int, fp.attrs["num_vectors"])

            batch_size = batch_size or num_vectors
            for i_low in range(0, num_vectors, batch_size):
                i_up = min(i_low + batch_size, num_vectors)

                doc_ids = fp["doc_ids"].asstr()[i_low:i_up]  # pyright: ignore[reportAttributeAccessIssue]
                psg_ids = fp["psg_ids"].asstr()[i_low:i_up]  # pyright: ignore[reportAttributeAccessIssue]
                vectors = fp["vectors"][i_low:i_up]  # pyright: ignore[reportIndexIssue]

                # IDs that don't exist will be returned as empty strings here
                doc_ids[doc_ids == ""] = None  # pyright: ignore[reportIndexIssue]
                psg_ids[psg_ids == ""] = None  # pyright: ignore[reportIndexIssue]
                index._add(vectors, doc_ids=doc_ids, psg_ids=psg_ids)  # pyright: ignore[reportArgumentType]
        return index

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence,
        psg_ids: IDSequence,
    ) -> None:
        with h5py.File(self._index_file, "a") as fp:
            # if this is the first call to _add, no datasets exist
            if "vectors" not in fp:
                self._create_ds(fp, vectors.shape[-1], vectors.dtype)

            # check all IDs first before adding anything
            doc_id_size = fp["doc_ids"].dtype.itemsize  # pyright: ignore[reportAttributeAccessIssue]
            for doc_id in doc_ids:
                if doc_id is not None and len(doc_id) > doc_id_size:
                    raise RuntimeError(
                        f"Document ID {doc_id} is longer than the maximum "
                        f"({doc_id_size} characters)."
                    )
            psg_id_size = fp["psg_ids"].dtype.itemsize  # pyright: ignore[reportAttributeAccessIssue]
            for psg_id in psg_ids:
                if psg_id is not None and len(psg_id) > psg_id_size:
                    raise RuntimeError(
                        f"Passage ID {psg_id} is longer than the maximum "
                        f"({psg_id_size} characters)."
                    )

            num_new_vecs = vectors.shape[0]
            capacity = fp["vectors"].shape[0]  # pyright: ignore[reportAttributeAccessIssue]

            # check if we have enough space, resize if necessary
            cur_num_vectors = cast(int, fp.attrs["num_vectors"])
            space_left = capacity - cur_num_vectors
            if num_new_vecs > space_left:
                new_size = max(
                    cur_num_vectors + num_new_vecs,
                    capacity + self._resize_min_val,
                )
                LOGGER.debug("resizing index from %s to %s", capacity, new_size)
                for ds in ("vectors", "doc_ids", "psg_ids"):
                    fp[ds].resize(new_size, axis=0)  # pyright: ignore[reportAttributeAccessIssue]

            # add new document IDs to index and in-memory mappings
            doc_id_idxs, non_null_doc_ids = [], []
            for i, doc_id in enumerate(doc_ids):
                if doc_id is not None:
                    self._doc_id_to_idx[doc_id].append(cur_num_vectors + i)
                    doc_id_idxs.append(cur_num_vectors + i)
                    non_null_doc_ids.append(doc_id)
            fp["doc_ids"][doc_id_idxs] = non_null_doc_ids  # pyright: ignore[reportIndexIssue]

            # add new passage IDs to index and in-memory mappings
            psg_id_idxs, non_null_psg_ids = [], []
            for i, psg_id in enumerate(psg_ids):
                if psg_id is not None:
                    self._psg_id_to_idx[psg_id] = cur_num_vectors + i
                    psg_id_idxs.append(cur_num_vectors + i)
                    non_null_psg_ids.append(psg_id)
            fp["psg_ids"][psg_id_idxs] = non_null_psg_ids  # pyright: ignore[reportIndexIssue]

            # add new vectors
            fp["vectors"][cur_num_vectors : cur_num_vectors + num_new_vecs] = vectors  # pyright: ignore[reportIndexIssue]
            fp.attrs["num_vectors"] += num_new_vecs  # pyright: ignore[reportOperatorIssue]

    def _get_doc_ids(self) -> set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(self, ids: "Iterable[str]") -> tuple[np.ndarray, list[list[int]]]:
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

            # reading all vectors at once slows h5py down significantly, so we read them
            # in chunks and concatenate
            vectors = np.concatenate(  # pyright: ignore[reportCallIssue]
                [
                    fp["vectors"][vec_idxs[i : i + self._max_indexing_size]]  # pyright: ignore[reportIndexIssue, reportArgumentType]
                    for i in range(0, len(vec_idxs), self._max_indexing_size)
                ]
            )
            return vectors, [id_to_idxs[id] for id in ids]

    def _batch_iter(
        self, batch_size: int
    ) -> "Iterator[tuple[np.ndarray, IDSequence, IDSequence]]":
        with h5py.File(self._index_file, "r") as fp:
            num_vectors = cast(int, fp.attrs["num_vectors"])
            for i in range(0, num_vectors, batch_size):
                j = min(i + batch_size, num_vectors)
                doc_ids = fp["doc_ids"].asstr()[i:j]  # pyright: ignore[reportAttributeAccessIssue]
                psg_ids = fp["psg_ids"].asstr()[i:j]  # pyright: ignore[reportAttributeAccessIssue]
                doc_ids[doc_ids == ""] = None  # pyright: ignore[reportIndexIssue]
                psg_ids[psg_ids == ""] = None  # pyright: ignore[reportIndexIssue]
                yield (
                    fp["vectors"][i:j],  # pyright: ignore[reportIndexIssue, reportReturnType]
                    doc_ids.tolist(),  # pyright: ignore[reportAttributeAccessIssue]
                    psg_ids.tolist(),  # pyright: ignore[reportAttributeAccessIssue]
                )

    @classmethod
    def load(
        cls,
        index_file: "Path",
        query_encoder: "Encoder | None" = None,
        mode: Mode = Mode.MAXP,
        encoder_batch_size: int = 32,
        resize_min_val: int = 2**10,
        max_indexing_size: int = 2**10,
    ) -> "OnDiskIndex":
        """Open an existing index on disk.

        :param index_file: The index file to open.
        :param query_encoder: The query encoder.
        :param mode: The ranking mode.
        :param encoder_batch_size: Batch size for the query encoder.
        :param resize_min_val: Minimum value to increase index size by.
        :param max_indexing_size:
            Maximum number of vectors to retrieve from the HDF5 dataset at once.
        :return: The index.
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
        index._max_indexing_size = max_indexing_size

        # deserialize quantizer if any
        with h5py.File(index_file, "r") as fp:
            if "quantizer" in fp:
                index._quantizer = Quantizer.deserialize(
                    dict(fp["quantizer/meta"].attrs),  # pyright: ignore[reportArgumentType]
                    dict(fp["quantizer/attributes"].attrs),  # pyright: ignore[reportArgumentType]
                    {k: v[:] for k, v in fp["quantizer/data"].items()},  # pyright: ignore[reportAttributeAccessIssue]
                )

            # read ID mappings
            index._doc_id_to_idx = defaultdict(list)
            index._psg_id_to_idx = {}

            num_vectors = cast(int, fp.attrs["num_vectors"])
            if num_vectors == 0:
                return index

            doc_id_iter = fp["doc_ids"].asstr()[:num_vectors]  # pyright: ignore[reportAttributeAccessIssue]
            psg_id_iter = fp["psg_ids"].asstr()[:num_vectors]  # pyright: ignore[reportAttributeAccessIssue]
            for i, (doc_id, psg_id) in tqdm(
                enumerate(zip(doc_id_iter, psg_id_iter)),
                total=num_vectors,
            ):
                if len(doc_id) > 0:
                    index._doc_id_to_idx[doc_id].append(i)
                if len(psg_id) > 0:
                    index._psg_id_to_idx[psg_id] = i
        return index
