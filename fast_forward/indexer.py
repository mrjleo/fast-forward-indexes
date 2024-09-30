"""
.. include:: docs/indexer.md
"""

import logging
from typing import Dict, Iterable

import numpy as np
from tqdm import tqdm

from fast_forward.encoder import Encoder
from fast_forward.index import IDSequence, Index
from fast_forward.quantizer import Quantizer

LOGGER = logging.getLogger(__name__)


class Indexer(object):
    """Utility class for indexing collections."""

    def __init__(
        self,
        index: Index,
        encoder: Encoder = None,
        batch_size: int = 32,
        quantizer: Quantizer = None,
        quantizer_fit_batches: int = 2**8,
    ) -> None:
        """Instantiate an indexer.

        Args:
            index (Index): The target index.
            encoder (Encoder, optional): Document/passage encoder. Defaults to None.
            batch_size (int, optional): How many items to process at once. Defaults to 32.
            quantizer (Quantizer, optional): A quantizer to be trained and attached to the index. Defaults to None.
            quantizer_fit_batches (int, optional): How many of the first batches to use to fit the quantizer. Defaults to 2**8.
        """
        self._index = index
        self._encoder = encoder
        self._batch_size = batch_size
        self._quantizer = quantizer
        self._quantizer_fit_batches = quantizer_fit_batches

        if quantizer is not None:
            self._buf_vectors, self._buf_doc_ids, self._buf_psg_ids = [], [], []
            LOGGER.warning(
                "quantizer is set, inputs will be buffered and index will remain empty until the quantizer has been fit"
            )

    def _index_batch(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence = None,
        psg_ids: IDSequence = None,
    ) -> None:
        """Add a batch to the index.

        If this indexer has a quantizer to be trained, the inputs will be buffered until the desired amount of data for fitting
        has been obtained. Afterwards, the quantizer is fit and attached to the index, and all buffered inputs are added at once.

        Args:
            vectors (np.ndarray): The vectors.
            doc_ids (IDSequence, optional): Corresponding document IDs. Defaults to None.
            psg_ids (IDSequence, optional): Corresponding passage IDs. Defaults to None.
        """
        if self._quantizer is None:
            self._index.add(vectors, doc_ids, psg_ids)
            return

        self._buf_vectors.append(vectors)
        self._buf_doc_ids.append(doc_ids)
        self._buf_psg_ids.append(psg_ids)

        if len(self._buf_vectors) >= self._quantizer_fit_batches:
            LOGGER.info(
                "fitting quantizer (%s batches of size %s)",
                len(self._buf_vectors),
                self._batch_size,
            )

            self._quantizer.fit(np.concatenate(self._buf_vectors))
            self._index.quantizer = self._quantizer
            self._quantizer = None

            LOGGER.info("adding buffered vectors to index")
            for vectors, doc_ids, psg_ids in zip(
                self._buf_vectors, self._buf_doc_ids, self._buf_psg_ids
            ):
                self._index.add(vectors, doc_ids, psg_ids)

            del self._buf_vectors
            del self._buf_doc_ids
            del self._buf_psg_ids

    def from_dicts(self, data: Iterable[Dict[str, str]]) -> None:
        """Index data from dictionaries.

        The dictionaries should have the key "text" and at least one of "doc_id" and "psg_id".

        Args:
            data (Iterable[Dict[str, str]]): An iterable of the dictionaries.

        Raises:
            RuntimeError: When no encoder is set.
        """
        if self._encoder is None:
            raise RuntimeError("An encoder is required.")

        texts, doc_ids, psg_ids = [], [], []
        for d in tqdm(data):
            texts.append(d["text"])
            doc_ids.append(d.get("doc_id"))
            psg_ids.append(d.get("psg_id"))

            if len(texts) == self._batch_size:
                self._index_batch(
                    self._encoder(texts), doc_ids=doc_ids, psg_ids=psg_ids
                )
                texts, doc_ids, psg_ids = [], [], []

        if len(texts) > 0:
            self._index_batch(self._encoder(texts), doc_ids=doc_ids, psg_ids=psg_ids)

    def from_index(self, index: Index) -> None:
        """Transfer vectors and IDs from another index.

        Args:
            index (Index): The source index.
        """
        for vectors, doc_ids, psg_ids in tqdm(index.batch_iter(self._batch_size)):
            self._index_batch(vectors, doc_ids, psg_ids)
