import unittest

import numpy as np

from fast_forward.encoder import LambdaEncoder
from fast_forward.encoder.transformer import (
    TCTColBERTDocumentEncoder,
    TCTColBERTQueryEncoder,
    TASBEncoder,
    BGEEncoder,
    ContrieverEncoder,
)

from ._constants import (
    TCT_COLBERT_QUERY_EXPECTED,
    TCT_COLBERT_DOCUMENT_EXPECTED,
    TAS_B_EXPECTED,
    CONTRIEVER_EXPECTED,
    BGE_ENCODER_EXPECTED,
)

TEST_INPUTS = ["input 1", "second input", "3rd input " * 100]


class TestLambdaEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = LambdaEncoder(lambda q: np.zeros(shape=(768,)))

    def test_encoder(self):
        np.testing.assert_equal(self.encoder(TEST_INPUTS), np.zeros(shape=(3, 768)))


class TestTCTColBERTEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.query_encoder = TCTColBERTQueryEncoder()
        cls.doc_encoder = TCTColBERTDocumentEncoder()

    def test_query_encoder(self):
        np.testing.assert_almost_equal(
            self.query_encoder(TEST_INPUTS),
            TCT_COLBERT_QUERY_EXPECTED,
            decimal=5,
        )

    def test_doc_encoder(self):
        np.testing.assert_almost_equal(
            self.doc_encoder(TEST_INPUTS),
            TCT_COLBERT_DOCUMENT_EXPECTED,
            decimal=5,
        )


class TestTASBEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = TASBEncoder()

    def test_encoder(self):
        np.testing.assert_almost_equal(
            self.encoder(TEST_INPUTS),
            TAS_B_EXPECTED,
            decimal=5,
        )


class TestContrieverEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = ContrieverEncoder()

    def test_encoder(self):
        np.testing.assert_almost_equal(
            self.encoder(TEST_INPUTS),
            CONTRIEVER_EXPECTED,
            decimal=5,
        )


class TestBGEEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.encoder = BGEEncoder()

    def test_encoder(self):
        np.testing.assert_almost_equal(
            self.encoder(TEST_INPUTS),
            BGE_ENCODER_EXPECTED,
            decimal=5,
        )


if __name__ == "__main__":
    unittest.main()
