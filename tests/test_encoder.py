import unittest

import numpy as np

from fast_forward.encoder import (
    LambdaEncoder,
    TCTColBERTDocumentEncoder,
    TCTColBERTQueryEncoder,
)


class TestTCTColBERTEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.query_encoder = TCTColBERTQueryEncoder(
            "castorini/tct_colbert-msmarco", device="cpu"
        )
        self.doc_encoder = TCTColBERTDocumentEncoder(
            "castorini/tct_colbert-msmarco", device="cpu"
        )

    def test_query_encoder(self):
        out = self.query_encoder(["test query 1", "test query 2"])
        self.assertEqual(out.shape, (2, 768))

    def test_doc_encoder(self):
        out = self.doc_encoder(["test doc 1", "test doc 11"])
        self.assertEqual(out.shape, (2, 768))


class TestLambdaEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.encoder = LambdaEncoder(lambda q: np.zeros(shape=(768,)))

    def test_encode(self):
        q_enc = self.encoder(["test query 1", "test query 2"])
        self.assertTrue(np.array_equal(q_enc, np.zeros(shape=(2, 768))))


if __name__ == "__main__":
    unittest.main()
