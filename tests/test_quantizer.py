import unittest

import numpy as np

from fast_forward.quantizer import Quantizer
from fast_forward.quantizer.nanopq import NanoPQ


class TestNanoPQ(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.quantizer = NanoPQ(8, 256)
        self.quantizer_trained = NanoPQ(8, 256)
        self.quantizer_trained.fit(
            np.random.normal(size=(2**10, 768)).astype(np.float32)
        )

    def test_properties(self):
        self.assertEqual((None, 8), self.quantizer.dims)
        self.assertEqual(np.uint8, self.quantizer.dtype)
        self.assertFalse(self.quantizer._trained)

        self.assertEqual((768, 8), self.quantizer_trained.dims)
        self.assertEqual(np.uint8, self.quantizer_trained.dtype)
        self.assertTrue(self.quantizer_trained._trained)

    def test_encoding_decoding(self):
        inputs = np.random.normal(size=(8, 768)).astype(np.float32)
        encoded = self.quantizer_trained.encode(inputs)
        self.assertEqual(encoded.shape, (8, 8))
        self.assertEqual(encoded.dtype, np.uint8)
        decoded = self.quantizer_trained.decode(encoded)
        self.assertEqual(decoded.shape, inputs.shape)

    def test_serialization(self):
        for q in (self.quantizer, self.quantizer_trained):
            q_loaded = Quantizer.deserialize(*q.serialize())

            # nanopq implements __eq__
            self.assertEqual(q._pq, q_loaded._pq)
            self.assertEqual(q._trained, q_loaded._trained)

    def test_errors(self):
        # encoding before the quantizer is trained
        with self.assertRaises(RuntimeError):
            self.quantizer.encode(np.random.normal(size=(8, 768)).astype(np.float32))

        # attaching to index before the quantizer is trained
        with self.assertRaises(RuntimeError):
            self.quantizer.set_attached()


if __name__ == "__main__":
    unittest.main()
