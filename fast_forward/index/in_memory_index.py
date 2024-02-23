import logging
import pickle
from collections import defaultdict
from pathlib import Path
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
    ) -> None:
        """Constructor.

        Args:
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.
        """
        self._vectors = None
        self._doc_ids = []
        self._psg_ids = []
        self._doc_id_to_idx = defaultdict(list)
        self._psg_id_to_idx = {}
        super().__init__(encoder, mode, encoder_batch_size)

    def _add(
        self,
        vectors: np.ndarray,
        doc_ids: Sequence[Union[str, None]],
        psg_ids: Sequence[Union[str, None]],
    ) -> None:
        if self._vectors is None:
            idx = 0
            self._vectors = vectors.copy()
        else:
            idx = self._vectors.shape[0]
            self._vectors = np.append(self._vectors, vectors, axis=0)

        for doc_id, psg_id in zip(doc_ids, psg_ids):
            if doc_id is not None:
                self._doc_id_to_idx[doc_id].append(idx)
            if psg_id is not None:
                assert psg_id not in self._psg_id_to_idx
                self._psg_id_to_idx[psg_id] = idx
            idx += 1

        self._doc_ids.extend(doc_ids)
        self._psg_ids.extend(psg_ids)

    def _get_doc_ids(self) -> Set[str]:
        return set(self._doc_id_to_idx.keys())

    def _get_psg_ids(self) -> Set[str]:
        return set(self._psg_id_to_idx.keys())

    def _get_vectors(
        self, ids: Iterable[str], mode: Mode
    ) -> Tuple[np.ndarray, List[Union[List[int], int, None]]]:
        # a list of all vectors to take from the main vector array
        vector_indices = []

        # for each ID, keep a list of indices to get the corresponding vectors from "vector_indices"
        id_indices = []
        i = 0

        if mode in (Mode.MAXP, Mode.AVEP):
            for id in ids:
                if id in self._doc_id_to_idx:
                    doc_indices = self._doc_id_to_idx[id]
                    vector_indices.extend(doc_indices)
                    id_indices.append(list(range(i, i + len(doc_indices))))
                    i += len(doc_indices)
                else:
                    id_indices.append(None)
        elif mode == Mode.FIRSTP:
            for id in ids:
                if id in self._doc_id_to_idx:
                    vector_indices.append(self._doc_id_to_idx[id][0])
                    id_indices.append(i)
                    i += 1
                else:
                    id_indices.append(None)
        elif mode == Mode.PASSAGE:
            for id in ids:
                if id in self._psg_id_to_idx:
                    vector_indices.append(self._psg_id_to_idx[id])
                    id_indices.append(i)
                    i += 1
                else:
                    id_indices.append(None)
        else:
            LOGGER.error(f"invalid mode: {mode}")
        return self._vectors[vector_indices], id_indices

    def save(self, target: Path) -> None:
        """Save the index in a file on disk.

        Args:
            target (Path): Target file to create.
        """
        target.parent.mkdir(parents=True, exist_ok=True)
        LOGGER.info(f"writing {target}")
        with open(target, "wb") as fp:
            pickle.dump((self._vectors, self._doc_ids, self._psg_ids), fp)

    @classmethod
    def from_disk(
        cls,
        index_file: Path,
        encoder: QueryEncoder = None,
        mode: Mode = Mode.PASSAGE,
        encoder_batch_size: int = 32,
    ) -> "InMemoryIndex":
        """Read an index from disk.

        Args:
            index_file (Path): The index file.
            encoder (QueryEncoder, optional): The query encoder to use. Defaults to None.
            mode (Mode, optional): Indexing mode. Defaults to Mode.PASSAGE.
            encoder_batch_size (int, optional): Query encoder batch size. Defaults to 32.

        Returns:
            InMemoryIndex: The index.
        """
        LOGGER.info(f"reading {index_file}")
        with open(index_file, "rb") as fp:
            vectors, doc_ids, psg_ids = pickle.load(fp)

        index = cls(encoder, mode, encoder_batch_size)
        if vectors is not None:
            index.add(vectors, doc_ids, psg_ids)
        index.mode = mode
        return index
