import logging
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, Union

import h5py
import numpy as np
from tqdm import tqdm

import fast_forward
from fast_forward.encoder import QueryEncoder
from fast_forward.index import Index, Mode

LOGGER = logging.getLogger(__name__)


class OnDiskIndex(Index):
    """Fast-Forward index that is read from disk (HDF5)."""

    def __init__(
        self,
        index_file: Path,
        dim: int,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        init_size: int = 2**14,
        resize_min_val: int = 2**10,
        hdf5_chunk_size: int = None,
        dtype: np.dtype = np.float32,
        overwrite: bool = False,
    ) -> None:
        """Constructor.

        Args:
            index_file (Path): Index file to create (or overwrite).
            dim (int): Vector dimension.
            encoder (QueryEncoder, optional): Query encoder. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Batch size for query encoder. Defaults to 32.
            init_size (int, optional): Initial size to allocate (number of vectors). Defaults to 2**14.
            resize_min_val (int, optional): Minimum number of vectors to increase index size by. Defaults to 2**10.
            hdf5_chunk_size (int, optional): Override chunk size used by HDF5. Defaults to None.
            dtype (np.dtype, optional): Vector dtype. Defaults to np.float32.
            overwrite (bool, optional): Overwrite index file if it exists. Defaults to False.

        Raises:
            ValueError: When the file exists and `overwrite=False`.
        """
        if index_file.exists() and not overwrite:
            raise ValueError(f"File {index_file} exists")

        super().__init__(encoder, mode, encoder_batch_size)
        self._index_file = index_file.absolute()
        self._resize_min_val = resize_min_val
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
                "S8",  # TODO
                maxshape=(None,),
                chunks=True if hdf5_chunk_size is None else (hdf5_chunk_size,),
            )
            fp.create_dataset(
                "psg_ids",
                (init_size,),
                "S8",  # TODO
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
                LOGGER.debug(f"resizing index from {capacity} to {new_size}")
                fp["vectors"].resize(new_size, axis=0)
                fp["doc_ids"].resize(new_size, axis=0)
                fp["psg_ids"].resize(new_size, axis=0)

            # add new items
            fp["vectors"][cur_num_vectors : cur_num_vectors + num_new_vecs] = vectors

            if doc_ids is not None:
                fp["doc_ids"][
                    cur_num_vectors : cur_num_vectors + num_new_vecs
                ] = doc_ids

                for i, doc_id in enumerate(doc_ids):
                    self._doc_id_to_idx[doc_id].append(cur_num_vectors + i)

            if psg_ids is not None:
                fp["psg_ids"][
                    cur_num_vectors : cur_num_vectors + num_new_vecs
                ] = psg_ids

                for i, psg_id in enumerate(psg_ids):
                    self._psg_id_to_idx[psg_id] = cur_num_vectors + i

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
                    LOGGER.warning(f"no vectors for {id}")
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
            return fp["vectors"][vec_idxs], [id_to_idxs[id] for id in ids]

    @classmethod
    def load(
        cls,
        index_file: Path,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
        resize_min_val: int = 2**10,
    ) -> "OnDiskIndex":
        """Open an existing index on disk.

        Args:
            index_file (Path): Index file to open.
            encoder (QueryEncoder, optional): Query encoder. Defaults to None.
            mode (Mode, optional): Ranking mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Batch size for query encoder. Defaults to 32.
            resize_min_val (int, optional): Minimum number of vectors to increase index size by. Defaults to 2**10.

        Returns:
            OnDiskIndex: The index.
        """
        index = cls.__new__(cls)
        super(OnDiskIndex, index).__init__(encoder, mode, encoder_batch_size)
        index._index_file = index_file.absolute()
        index._resize_min_val = resize_min_val

        # read ID mappings
        index._doc_id_to_idx = defaultdict(list)
        index._psg_id_to_idx = {}
        with h5py.File(index._index_file, "r") as fp:
            for i, (doc_id, psg_id) in tqdm(
                enumerate(
                    zip(fp["doc_ids"].asstr()[:], fp["psg_ids"].asstr()[:]),
                ),
                total=fp.attrs["num_vectors"],
            ):
                if len(doc_id) > 0:
                    index._doc_id_to_idx[doc_id].append(i)
                if len(psg_id) > 0:
                    index._psg_id_to_idx[psg_id] = i
        return index
