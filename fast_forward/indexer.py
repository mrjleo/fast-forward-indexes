"""
.. include:: docs/indexer.md
"""

from typing import Dict, Iterable

from tqdm import tqdm

from fast_forward.encoder import Encoder
from fast_forward.index import Index


class Indexer(object):
    """Utility class for indexing collections."""

    def __init__(self, index: Index, encoder: Encoder, batch_size: int = 32) -> None:
        """Constructor.

        Args:
            index (Index): The index to add the collection to.
            encoder (Encoder): Document/passage encoder.
            batch_size (int, optional): Batch size for encoding. Defaults to 32.
        """
        self._index = index
        self._encoder = encoder
        self._batch_size = batch_size

    def index_dicts(self, data: Iterable[Dict[str, str]]) -> None:
        """Index data from dictionaries.

        The dictionaries should have the key "text" and at least one of "doc_id" and "psg_id".

        Args:
            data (Iterable[Dict[str, str]]): An iterable of the dictionaries.
        """
        texts, doc_ids, psg_ids = [], [], []
        for d in tqdm(data):
            texts.append(d["text"])
            doc_ids.append(d.get("doc_id"))
            psg_ids.append(d.get("psg_id"))

            if len(texts) == self._batch_size:
                self._index.add(self._encoder(texts), doc_ids=doc_ids, psg_ids=psg_ids)
                texts, doc_ids, psg_ids = [], [], []

        if len(texts) > 0:
            self._index.add(self._encoder(texts), doc_ids=doc_ids, psg_ids=psg_ids)
