import logging
from pathlib import Path
from typing import Iterable, List, Sequence, Set, Tuple, Union

import h5py
import numpy as np

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
        chunksize: int = None,
        dtype: np.dtype = np.float32,
        overwrite: bool = False,
    ) -> None:
        if index_file.exists() and not overwrite:
            raise ValueError(f"File {index_file} exists")

        super().__init__(encoder, mode, encoder_batch_size)

        self.index_file = index_file.absolute()
        with h5py.File(self.index_file, "w") as fp:
            fp.create_dataset(
                "vectors",
                (init_size, dim),
                dtype,
                maxshape=(None, dim),
                chunks=True if chunksize is None else (chunksize, dim),
            )
            fp["vectors"].attrs["num_vectors"] = 0
            fp.attrs["ff_version"] = fast_forward.__version__

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        n, d = vectors.shape
        with h5py.File(self.index_file, "a") as fp:
            assert d == fp["vectors"].shape[1]
            cur_num_vectors = fp["vectors"].attrs["num_vectors"]
            fp["vectors"][cur_num_vectors : cur_num_vectors + n] = vectors
            fp["vectors"].attrs["num_vectors"] += n

            for i, (doc_id, psg_id) in enumerate(zip(doc_ids, psg_ids)):
                if doc_id is not None:
                    ds = fp.require_dataset(
                        f"/ids/doc/{doc_id}", (1,), dtype=h5py.vlen_dtype(np.uint)
                    )
                    ds[0] = np.append(ds[0], [cur_num_vectors + i])

                if psg_id is not None:
                    ds = fp.require_dataset(f"/ids/psg/{psg_id}", (1,), dtype=np.uint)
                    ds[0] = cur_num_vectors + i

    def _get_doc_ids(self) -> Set[str]:
        with h5py.File(self.index_file, "r") as fp:
            return set(fp["/ids/doc"].keys())

    def _get_psg_ids(self) -> Set[str]:
        with h5py.File(self.index_file, "r") as fp:
            return set(fp["/ids/psg"].keys())

    def _get_vectors(self, ids: Iterable[str]) -> Tuple[np.ndarray, List[List[int]]]:
        result_vectors = []
        id_idxs = []
        c = 0
        with h5py.File(self.index_file, "r") as fp:
            for id in ids:
                if self.mode in (Mode.MAXP, Mode.AVEP) and id in fp["/ids/doc"]:
                    idxs = fp[f"/ids/doc/{id}"][0]
                elif self.mode == Mode.FIRSTP and id in fp["/ids/doc"]:
                    idxs = [fp[f"/ids/doc/{id}"][0][0]]
                elif self.mode == Mode.PASSAGE and id in fp["/ids/psg"]:
                    idxs = [fp[f"/ids/psg/{id}"][0]]
                else:
                    LOGGER.warning(f"no vectors for {id}")
                    idxs = []

                result_vectors.append(fp["vectors"][idxs])
                id_idxs.append(list(range(c, c + len(idxs))))
                c += len(idxs)
            return np.concatenate(result_vectors), id_idxs

    def from_file(cls) -> "OnDiskIndex":
        pass
