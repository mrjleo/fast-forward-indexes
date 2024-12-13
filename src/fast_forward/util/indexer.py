import logging
from typing import TYPE_CHECKING, TypedDict

import numpy as np
from tqdm import tqdm

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from fast_forward.encoder.base import Encoder
    from fast_forward.index.base import IDSequence, Index
    from fast_forward.quantizer import Quantizer

LOGGER = logging.getLogger(__name__)


class IndexingDict(TypedDict):
    """Dictionary that represents a document/passage.

    Consumed by `Indexer.from_dicts`.
    """

    text: str
    doc_id: str | None
    psg_id: str | None


class Indexer:
    """Utility class for indexing collections."""

    def __init__(
        self,
        index: "Index",
        encoder: "Encoder | None" = None,
        encoder_batch_size: int = 128,
        batch_size: int = 2**16,
        quantizer: "Quantizer | None" = None,
        quantizer_fit_batches: int = 1,
    ) -> None:
        """Create an indexer.

        Optionally, a quantizer can automatically be fit on the first batch(es) to be
        indexed. This requires the index to be empty.
        If a quantizer is provided, the first batch(es) will be buffered and used to fit
        the quantizer.

        :param index: The target index.
        :param encoder: Document/passage encoder.
        :param encoder_batch_size: Encoder batch size.
        :param batch_size: How many vectors to add to the index at once.
        :param quantizer: A quantizer to be fit and attached to the index.
        :param quantizer_fit_batches: How many batches to use to fit the quantizer.
        :raises ValueError: When a quantizer is provided that has already been fit.
        :raises ValueError: When a quantizer is provided and the index is not empty.
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
                    "The quantizer is already fit. "
                    "It should be attached to the index directly."
                )

            if len(index) > 0:
                raise ValueError(
                    "The index must be empty for a quantizer to be attached."
                )

            self._buf_vectors, self._buf_doc_ids, self._buf_psg_ids = [], [], []
            if quantizer_fit_batches > 1:
                LOGGER.warning(
                    "inputs will be buffered and index will remain empty until the "
                    "quantizer has been fit"
                )

    def _index_batch(
        self,
        vectors: np.ndarray,
        doc_ids: "IDSequence | None" = None,
        psg_ids: "IDSequence | None" = None,
    ) -> None:
        """Add a batch to the index.

        If this indexer has a quantizer to be fit, the inputs will be buffered until the
        desired amount of data for fitting has been obtained. Afterwards, the quantizer
        is fit and attached to the index, and all buffered inputs are added at once.

        :param vectors: The vectors.
        :param doc_ids: Corresponding document IDs.
        :param psg_ids: Corresponding passage IDs.
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
                    "the size of the last batch (%s) is smaller than %s",
                    last_batch_size,
                    self._batch_size,
                )

            self._quantizer.fit(np.concatenate(self._buf_vectors))
            self._index.quantizer = self._quantizer
            self._quantizer = None

            LOGGER.info("adding buffered vectors to index")
            for b_vectors, b_doc_ids, b_psg_ids in zip(
                self._buf_vectors, self._buf_doc_ids, self._buf_psg_ids
            ):
                self._index.add(b_vectors, b_doc_ids, b_psg_ids)

            del self._buf_vectors
            del self._buf_doc_ids
            del self._buf_psg_ids

    def _encode(self, texts: "Sequence[str]") -> np.ndarray:
        """Encode a list of strings (respecting the encoder batch size).

        :param texts: The pieces of text to encode.
        :raises RuntimeError: When no encoder exists.
        :return: The vector representations.
        """
        if self._encoder is None:
            raise RuntimeError("An encoder is required.")

        result = []
        for i in range(0, len(texts), self._encoder_batch_size):
            batch = texts[i : i + self._encoder_batch_size]
            result.append(self._encoder(batch))
        return np.concatenate(result)

    def from_dicts(self, data: "Iterable[IndexingDict]") -> None:
        """Index data from dictionaries.

        :param data: An iterable of the dictionaries.
        """
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

    def from_index(self, index: "Index") -> None:
        """Transfer vectors and IDs from another index.

        If the source index uses quantized representations, the vectors are
        reconstructed first.

        :param index: The source index.
        """
        for vectors, doc_ids, psg_ids in tqdm(index.batch_iter(self._batch_size)):
            self._index_batch(vectors, doc_ids, psg_ids)
