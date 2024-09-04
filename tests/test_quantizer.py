import unittest

import numpy as np

from fast_forward.quantizer import Quantizer
from fast_forward.quantizer.nanopq import NanoOPQ, NanoPQ


class TestQuantizer(unittest.TestCase):
    __test__ = False

    def test_eq(self):
        self.assertEqual(self.quantizer, self.quantizer)
        self.assertEqual(self.quantizer_trained, self.quantizer_trained)
        self.assertNotEqual(self.quantizer, self.quantizer_trained)

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
        inputs = np.random.normal(size=(8, 768)).astype(np.float32)
        quantizer_loaded = Quantizer.deserialize(*self.quantizer.serialize())
        self.assertEqual(self.quantizer, quantizer_loaded)

        quantizer_trained_loaded = Quantizer.deserialize(
            *self.quantizer_trained.serialize()
        )
        self.assertEqual(self.quantizer_trained, quantizer_trained_loaded)
        np.testing.assert_array_equal(
            self.quantizer_trained.encode(inputs),
            quantizer_trained_loaded.encode(inputs),
        )

    def test_errors(self):
        # encoding before the quantizer is trained
        with self.assertRaises(RuntimeError):
            self.quantizer.encode(np.random.normal(size=(8, 768)).astype(np.float32))

        # attaching to index before the quantizer is trained
        with self.assertRaises(RuntimeError):
            self.quantizer.set_attached()


class TestNanoPQ(TestQuantizer):
    __test__ = True

    @classmethod
    def setUpClass(self):
        self.quantizer = NanoPQ(8, 256)
        self.quantizer_trained = NanoPQ(8, 256)
        self.quantizer_trained.fit(
            np.random.normal(size=(2**10, 768)).astype(np.float32)
        )


class TestNanoOPQ(TestQuantizer):
    __test__ = True

    @classmethod
    def setUpClass(self):
        self.quantizer = NanoOPQ(8, 256)
        self.quantizer_trained = NanoOPQ(8, 256)
        self.quantizer_trained.fit(
            np.random.normal(size=(2**10, 768)).astype(np.float32)
        )


if __name__ == "__main__":
    unittest.main()
