import unittest

import numpy as np

from fast_forward.encoder import LambdaEncoder
from fast_forward.index.memory import InMemoryIndex
from fast_forward.indexer import Indexer


class TestIndexer(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.index = InMemoryIndex()
        self.indexer = Indexer(
            self.index, LambdaEncoder(lambda q: np.zeros(shape=(16,))), batch_size=2
        )

    def test_index_dicts(self):
        self.indexer.index_dicts(
            [
                {"text": "123", "doc_id": "d1", "psg_id": "d1_p1"},
                {"text": "234", "doc_id": "d1", "psg_id": "d1_p2"},
                {"text": "456", "doc_id": "d1", "psg_id": "d1_p3"},
                {"text": "567", "doc_id": "d2", "psg_id": "d2_p1"},
                {"text": "678", "doc_id": "d3", "psg_id": "d3_p1"},
                {"text": "890", "doc_id": "d4"},
                {"text": "901", "psg_id": "d5_p1"},
            ]
        )
        self.assertEqual(7, len(self.index))
        self.assertEqual(set(("d1", "d2", "d3", "d4")), self.index.doc_ids)
        self.assertEqual(
            set(("d1_p1", "d1_p2", "d1_p3", "d2_p1", "d3_p1", "d5_p1")),
            self.index.psg_ids,
        )


if __name__ == "__main__":
    unittest.main()
