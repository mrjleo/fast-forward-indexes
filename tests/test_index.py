import logging
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fast_forward import InMemoryIndex, Mode, OnDiskIndex, Ranking
from fast_forward.encoder import LambdaQueryEncoder
from fast_forward.index import create_coalesced_index

DUMMY_QUERIES = {"q1": "query 1", "q2": "query 2"}
DUMMY_DOC_IDS = ["d0", "d0", "d1", "d2", "d3"]
DUMMY_PSG_IDS = ["p0", "p1", "p2", "p3", "p4"]
DUMMY_VECTORS = np.array(
    [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 1, 1, 0, 0],
        [1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
    ]
)
DUMMY_DIM = DUMMY_VECTORS.shape[1]
DUMMY_DOC_RUN = {
    "q1": {"d0": 100, "d1": 2, "d2": 3, "d3": 200},
    "q2": {"d0": 400, "d1": 5, "d2": 6, "d3": 800, "dx": 7},
}
DUMMY_DOC_RANKING = Ranking(DUMMY_DOC_RUN)
DUMMY_PSG_RUN = {
    "q1": {"p0": 100, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
    "q2": {"p0": 500, "p1": 6, "p2": 7, "p3": 8, "p4": 9},
}
DUMMY_PSG_RANKING = Ranking(DUMMY_PSG_RUN)
DUMMY_ENCODER = LambdaQueryEncoder(lambda _: np.array([1, 1, 1, 1, 1]))


class TestIndexCommon(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        logging.basicConfig(level=logging.DEBUG)
        tempdir = Path(tempfile.gettempdir())

        self.doc_psg_indexes = [
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER),
            OnDiskIndex(
                tempdir / "doc_psg_index.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
                overwrite=True,
            ),
        ]
        for index in self.doc_psg_indexes:
            index.add(
                vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
            )

        self.doc_indexes = [
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER),
            OnDiskIndex(
                tempdir / "doc_index.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
                overwrite=True,
            ),
        ]
        for index in self.doc_indexes:
            index.add(vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS)

        self.psg_indexes = [
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER),
            OnDiskIndex(
                tempdir / "psg_index.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
                overwrite=True,
            ),
        ]
        for index in self.psg_indexes:
            index.add(vectors=DUMMY_VECTORS, psg_ids=DUMMY_PSG_IDS)

    def test_ids(self):
        for index in self.doc_psg_indexes:
            self.assertEqual(set(DUMMY_DOC_IDS), index.doc_ids)
            self.assertEqual(set(DUMMY_PSG_IDS), index.psg_ids)

        for index in self.doc_indexes:
            self.assertEqual(set(DUMMY_DOC_IDS), index.doc_ids)
            self.assertEqual(0, len(index.psg_ids))

        for index in self.psg_indexes:
            self.assertEqual(set(DUMMY_PSG_IDS), index.psg_ids)
            self.assertEqual(0, len(index.doc_ids))

    def test_interpolation(self):
        for index in self.doc_psg_indexes:
            index.mode = Mode.MAXP
            result = index.get_scores(
                DUMMY_DOC_RANKING,
                DUMMY_QUERIES,
                alpha=[0.0, 0.5, 1.0],
                cutoff=None,
                early_stopping=False,
            )
            self.assertEqual(
                result[0.0],
                Ranking(
                    {
                        "q1": {"d0": 2, "d1": 3, "d2": 4, "d3": 5},
                        "q2": {"d0": 2, "d1": 3, "d2": 4, "d3": 5},
                    }
                ),
            )
            self.assertEqual(
                result[0.5],
                Ranking(
                    {
                        "q1": {"d0": 51, "d1": 2.5, "d2": 3.5, "d3": 102.5},
                        "q2": {"d0": 201, "d1": 4, "d2": 5, "d3": 402.5},
                    }
                ),
            )
            self.assertEqual(
                result[1.0],
                Ranking(
                    {
                        "q1": {"d0": 100, "d1": 2, "d2": 3, "d3": 200},
                        "q2": {"d0": 400, "d1": 5, "d2": 6, "d3": 800},
                    }
                ),
            )

    def test_early_stopping(self):
        for index in self.doc_psg_indexes:
            index.mode = Mode.MAXP

            scores_1 = index.get_scores(
                DUMMY_DOC_RANKING,
                DUMMY_QUERIES,
                alpha=0.5,
                cutoff=2,
                early_stopping=True,
            )

            # test unsorted ranking
            scores_2 = index.get_scores(
                Ranking(DUMMY_DOC_RUN, sort=False),
                DUMMY_QUERIES,
                alpha=0.5,
                cutoff=2,
                early_stopping=True,
            )

            self.assertEqual(
                scores_1[0.5],
                Ranking(
                    {"q1": {"d3": 102.5, "d0": 51}, "q2": {"d3": 402.5, "d0": 201}}
                ),
            )
            self.assertEqual(scores_1[0.5], scores_2[0.5])

            with self.assertRaises(ValueError):
                index.get_scores(
                    DUMMY_DOC_RANKING,
                    DUMMY_QUERIES,
                    alpha=0.5,
                    cutoff=None,
                    early_stopping=True,
                )

    def test_firstp(self):
        for index in self.doc_psg_indexes:
            index.mode = Mode.FIRSTP
            self.assertEqual(
                index.get_scores(DUMMY_DOC_RANKING, DUMMY_QUERIES)[0.0],
                Ranking(
                    {
                        "q1": {"d0": 1, "d1": 3, "d2": 4, "d3": 5},
                        "q2": {"d0": 1, "d1": 3, "d2": 4, "d3": 5},
                    }
                ),
            )

    def test_avep(self):
        for index in self.doc_psg_indexes:
            index.mode = Mode.AVEP
            self.assertEqual(
                index.get_scores(DUMMY_DOC_RANKING, DUMMY_QUERIES)[0.0],
                Ranking(
                    {
                        "q1": {"d0": 1.5, "d1": 3, "d2": 4, "d3": 5},
                        "q2": {"d0": 1.5, "d1": 3, "d2": 4, "d3": 5},
                    }
                ),
            )

    def test_passage(self):
        for index in self.doc_psg_indexes:
            index.mode = Mode.PASSAGE
            self.assertEqual(
                index.get_scores(DUMMY_PSG_RANKING, DUMMY_QUERIES)[0.0],
                Ranking(
                    {
                        "q1": {"p0": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
                        "q2": {"p0": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
                    }
                ),
            )

    def test_errors(self):
        for index in self.doc_psg_indexes:
            with self.assertRaises(ValueError):
                index.add(DUMMY_VECTORS, doc_ids=None, psg_ids=None)

        with self.assertRaises(RuntimeError):
            InMemoryIndex(DUMMY_DIM, encoder=None).encode(["test"])

    def test_coalescing(self):
        # delta = 0.3: vectors of d0 should be averaged
        coalesced_index = InMemoryIndex(DUMMY_DIM, mode=Mode.MAXP)
        create_coalesced_index(self.doc_indexes[0], coalesced_index, 0.3)
        self.assertEqual(self.doc_indexes[0].doc_ids, coalesced_index.doc_ids)
        d0_vector_expected = np.average([DUMMY_VECTORS[0], DUMMY_VECTORS[1]], axis=0)
        d0_vectors, _ = coalesced_index._get_vectors(["d0"])
        self.assertEqual(1, len(d0_vectors))
        self.assertTrue(np.array_equal(d0_vector_expected, d0_vectors[0]))

        # delta = 0.2: nothing should change
        coalesced_index = InMemoryIndex(DUMMY_DIM, mode=Mode.MAXP)
        create_coalesced_index(self.doc_indexes[0], coalesced_index, 0.2, buffer_size=2)
        self.assertEqual(self.doc_indexes[0].doc_ids, coalesced_index.doc_ids)
        for doc_id in self.doc_indexes[0].doc_ids:
            vectors_1, _ = self.doc_indexes[0]._get_vectors([doc_id])
            vectors_2, _ = coalesced_index._get_vectors([doc_id])
            self.assertEqual(len(vectors_1), len(vectors_2))
            for v1, v2 in zip(vectors_1, vectors_2):
                print(v1, v2)
                self.assertTrue(np.array_equal(v1, v2))


if __name__ == "__main__":
    unittest.main()
