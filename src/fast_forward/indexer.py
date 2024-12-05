"""
.. include:: docs/indexer.md
"""

import logging
from typing import Dict, Iterable, Sequence

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
        encoder_batch_size: int = 128,
        batch_size: int = 2**16,
        quantizer: Quantizer = None,
        quantizer_fit_batches: int = 1,
    ) -> None:
        """Instantiate an indexer.

        Optionally, a quantizer can be automatically be fit on the first batch(es) to be indexed. This requires the index to be empty.
        If a quantizer is provided, the first batch(es) will be buffered and used to fit the quantizer.

        Args:
            index (Index): The target index.
            encoder (Encoder, optional): Document/passage encoder. Defaults to None.
            encoder_batch_size (int, optional): Encoder batch size. Defaults to 128.
            batch_size (int, optional): How many vectors to add to the index at once. Defaults to 2**16.
            quantizer (Quantizer, optional): A quantizer to be fit and attached to the index. Defaults to None.
            quantizer_fit_batches (int, optional): How many of the first batches to use to fit the quantizer. Defaults to 1.

        Raises:
            ValueError: When a quantizer is provided that has already been fit.
            ValueError: When a quantizer is provided and the index already has vectors.
        """
        self._index = index
        self._encoder = encoder
        self._encoder_batch_size = encoder_batch_size
        self._batch_size = batch_size
        self._quantizer = quantizer
        self._quantizer_fit_batches = quantizer_fit_batches

        if quantizer is not None:
            if quantizer._trained:
                raise ValueError(
                    "The quantizer is already fit and should be attached to the index directly."
                )

            if len(index) > 0:
                raise ValueError(
                    "The index must be empty for a quantizer to be attached."
                )

            self._buf_vectors, self._buf_doc_ids, self._buf_psg_ids = [], [], []
            if quantizer_fit_batches > 1:
                LOGGER.warning(
                    "inputs will be buffered and index will remain empty until the quantizer has been fit"
                )

    def _index_batch(
        self,
        vectors: np.ndarray,
        doc_ids: IDSequence = None,
        psg_ids: IDSequence = None,
    ) -> None:
        """Add a batch to the index.

        If this indexer has a quantizer to be fit, the inputs will be buffered until the desired amount of data for fitting
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
                "fitting quantizer (%s batch(es), batch size %s)",
                len(self._buf_vectors),
                self._batch_size,
            )

            last_batch_size = self._buf_vectors[-1].shape[0]
            if last_batch_size < self._batch_size:
                LOGGER.warning(
                    "the size of the last batch (%s) is smaller than the batch size (%s)",
                    last_batch_size,
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

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        """Encode a list of strings (respecting the encoder batch size).

        Args:
            texts (Sequence[str]): The pieces of text to encode.

        Returns:
            np.ndarray: The vector representations.
        """
        result = []
        for i in range(0, len(texts), self._encoder_batch_size):
            batch = texts[i : i + self._encoder_batch_size]
            result.append(self._encoder(batch))
        return np.concatenate(result)

    def from_dicts(self, data: Iterable[Dict[str, str]]) -> None:
        """Index data from dictionaries.

        The dictionaries should have the key "text" and at least one of "doc_id" and "psg_id".

        Args:
            data (Iterable[Dict[str, str]]): An iterable of the dictionaries.

        Raises:
            RuntimeError: When no encoder exists.
        """
        if self._encoder is None:
            raise RuntimeError("An encoder is required.")

        texts, doc_ids, psg_ids = [], [], []
        for d in tqdm(data):
            texts.append(d["text"])
            doc_ids.append(d.get("doc_id"))
            psg_ids.append(d.get("psg_id"))

            if len(texts) == self._batch_size:
                self._index_batch(self._encode(texts), doc_ids=doc_ids, psg_ids=psg_ids)
                texts, doc_ids, psg_ids = [], [], []

        if len(texts) > 0:
            self._index_batch(self._encode(texts), doc_ids=doc_ids, psg_ids=psg_ids)

    def from_index(self, index: Index) -> None:
        """Transfer vectors and IDs from another index.

        If the source index uses quantized representations, the vectors are reconstructed first.

        Args:
            index (Index): The source index.
        """
        for vectors, doc_ids, psg_ids in tqdm(index.batch_iter(self._batch_size)):
            self._index_batch(vectors, doc_ids, psg_ids)
