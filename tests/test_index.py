import logging
import unittest

import numpy as np

from fast_forward import InMemoryIndex, Mode, Ranking
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
DUMMY_DOC_RUN = {
    "q1": {"d0": 100, "d1": 2, "d2": 3, "d3": 200},
    "q2": {"d0": 400, "d1": 5, "d2": 6, "d3": 800, "dx": 7},
}
DUMMY_PSG_RUN = {
    "q1": {"p0": 100, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
    "q2": {"p0": 500, "p1": 6, "p2": 7, "p3": 8, "p4": 9},
}


class TestInMemoryIndex(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        logging.basicConfig(level=logging.DEBUG)

        self.dummy_encoder = LambdaQueryEncoder(lambda q: np.array([1, 1, 1, 1, 1]))

        self.doc_psg_index = InMemoryIndex(self.dummy_encoder)
        self.doc_psg_index.add(
            vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
        )
        self.doc_psg_index.mode = Mode.MAXP

        self.doc_psg_index_chunked = InMemoryIndex(
            self.dummy_encoder, init_size=2, alloc_size=2
        )
        self.doc_psg_index_chunked.add(
            vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
        )
        self.doc_psg_index_chunked.mode = Mode.MAXP

        self.doc_index = InMemoryIndex(self.dummy_encoder)
        self.doc_index.add(vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS)
        self.doc_psg_index.mode = Mode.MAXP

        self.psg_index = InMemoryIndex(self.dummy_encoder)
        self.psg_index.add(vectors=DUMMY_VECTORS, psg_ids=DUMMY_PSG_IDS)
        self.psg_index.mode = Mode.PASSAGE

        self.doc_ranking = Ranking(DUMMY_DOC_RUN)
        self.psg_ranking = Ranking(DUMMY_PSG_RUN)

    def test_ids(self):
        self.assertEqual(set(DUMMY_DOC_IDS), self.doc_psg_index.doc_ids)
        self.assertEqual(set(DUMMY_PSG_IDS), self.doc_psg_index.psg_ids)

        self.assertEqual(set(DUMMY_DOC_IDS), self.doc_index.doc_ids)
        self.assertEqual(0, len(self.doc_index.psg_ids))

        self.assertEqual(set(DUMMY_PSG_IDS), self.psg_index.psg_ids)
        self.assertEqual(0, len(self.psg_index.doc_ids))

    def test_interpolation(self):
        self.doc_psg_index.mode = Mode.MAXP
        results = [
            self.doc_psg_index.get_scores(
                self.doc_ranking,
                DUMMY_QUERIES,
                alpha=[0.0, 0.5, 1.0],
                cutoff=None,
                early_stopping=False,
            ),
            self.doc_psg_index_chunked.get_scores(
                self.doc_ranking,
                DUMMY_QUERIES,
                alpha=[0.0, 0.5, 1.0],
                cutoff=None,
                early_stopping=False,
            ),
        ]

        r00 = Ranking(
            {
                "q1": {"d0": 2, "d1": 3, "d2": 4, "d3": 5},
                "q2": {"d0": 2, "d1": 3, "d2": 4, "d3": 5},
            }
        )
        r05 = Ranking(
            {
                "q1": {"d0": 51, "d1": 2.5, "d2": 3.5, "d3": 102.5},
                "q2": {"d0": 201, "d1": 4, "d2": 5, "d3": 402.5},
            }
        )
        r10 = Ranking(
            {
                "q1": {"d0": 100, "d1": 2, "d2": 3, "d3": 200},
                "q2": {"d0": 400, "d1": 5, "d2": 6, "d3": 800},
            }
        )

        for result in results:
            self.assertEqual(result[1.0], r10)
            self.assertEqual(result[0.5], r05)
            self.assertEqual(result[0.0], r00)

    def test_early_stopping(self):
        self.doc_psg_index.mode = Mode.MAXP
        scores_1 = self.doc_psg_index.get_scores(
            self.doc_ranking, DUMMY_QUERIES, alpha=0.5, cutoff=2, early_stopping=True
        )
        # test unsorted ranking
        scores_2 = self.doc_psg_index.get_scores(
            Ranking(DUMMY_DOC_RUN, sort=False),
            DUMMY_QUERIES,
            alpha=0.5,
            cutoff=2,
            early_stopping=True,
        )
        r_expected = Ranking(
            {"q1": {"d3": 102.5, "d0": 51}, "q2": {"d3": 402.5, "d0": 201}}
        )
        self.assertEqual(scores_1[0.5], r_expected)
        self.assertEqual(scores_2[0.5], r_expected)

        with self.assertRaises(ValueError):
            self.doc_psg_index.get_scores(
                self.doc_ranking,
                DUMMY_QUERIES,
                alpha=0.5,
                cutoff=None,
                early_stopping=True,
            )

    def test_firstp(self):
        self.doc_psg_index.mode = Mode.FIRSTP
        self.assertEqual(
            self.doc_psg_index.get_scores(self.doc_ranking, DUMMY_QUERIES)[0.0],
            Ranking(
                {
                    "q1": {"d0": 1, "d1": 3, "d2": 4, "d3": 5},
                    "q2": {"d0": 1, "d1": 3, "d2": 4, "d3": 5},
                }
            ),
        )

    def test_avep(self):
        self.doc_psg_index.mode = Mode.AVEP
        self.assertEqual(
            self.doc_psg_index.get_scores(self.doc_ranking, DUMMY_QUERIES)[0.0],
            Ranking(
                {
                    "q1": {"d0": 1.5, "d1": 3, "d2": 4, "d3": 5},
                    "q2": {"d0": 1.5, "d1": 3, "d2": 4, "d3": 5},
                }
            ),
        )

    def test_passage(self):
        self.doc_psg_index.mode = Mode.PASSAGE
        self.assertEqual(
            self.doc_psg_index.get_scores(self.psg_ranking, DUMMY_QUERIES)[0.0],
            Ranking(
                {
                    "q1": {"p0": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
                    "q2": {"p0": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
                }
            ),
        )

    def test_errors(self):
        with self.assertRaises(ValueError):
            InMemoryIndex(self.dummy_encoder).add(
                DUMMY_VECTORS, doc_ids=None, psg_ids=None
            )
        with self.assertRaises(RuntimeError):
            InMemoryIndex(encoder=None).encode(["test"])

    def test_coalescing(self):
        # delta = 0.3: vectors of d0 should be averaged
        coalesced_index = InMemoryIndex(mode=Mode.MAXP)
        create_coalesced_index(self.doc_index, coalesced_index, 0.3)
        self.assertEqual(self.doc_index.doc_ids, coalesced_index.doc_ids)
        d0_vector_expected = np.average([DUMMY_VECTORS[0], DUMMY_VECTORS[1]], axis=0)
        d0_vectors, _ = coalesced_index._get_vectors(["d0"])
        self.assertEqual(1, len(d0_vectors))
        self.assertTrue(np.array_equal(d0_vector_expected, d0_vectors[0]))

        # delta = 0.2: nothing should change
        coalesced_index = InMemoryIndex(mode=Mode.MAXP)
        create_coalesced_index(self.doc_index, coalesced_index, 0.2, buffer_size=2)
        self.assertEqual(self.doc_index.doc_ids, coalesced_index.doc_ids)
        for doc_id in self.doc_index.doc_ids:
            vectors_1, _ = self.doc_index._get_vectors([doc_id])
            vectors_2, _ = coalesced_index._get_vectors([doc_id])
            self.assertEqual(len(vectors_1), len(vectors_2))
            for v1, v2 in zip(vectors_1, vectors_2):
                print(v1, v2)
                self.assertTrue(np.array_equal(v1, v2))


if __name__ == "__main__":
    unittest.main()
