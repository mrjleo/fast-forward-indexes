import shutil
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
DUMMY_NUM, DUMMY_DIM = DUMMY_VECTORS.shape
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


class TestIndex(unittest.TestCase):
    __test__ = False

    def setUp(self):
        for index in self.doc_psg_indexes:
            index.add(
                vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
            )

        for index in self.doc_indexes:
            index.add(vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS)

        for index in self.psg_indexes:
            index.add(vectors=DUMMY_VECTORS, psg_ids=DUMMY_PSG_IDS)

    def test_properties(self):
        for index in self.doc_psg_indexes:
            self.assertEqual(set(DUMMY_DOC_IDS), index.doc_ids)
            self.assertEqual(set(DUMMY_PSG_IDS), index.psg_ids)
            self.assertEqual(DUMMY_NUM, len(index))
            self.assertEqual(DUMMY_DIM, index.dim)

        for index in self.doc_indexes:
            self.assertEqual(set(DUMMY_DOC_IDS), index.doc_ids)
            self.assertEqual(0, len(index.psg_ids))
            self.assertEqual(DUMMY_NUM, len(index))
            self.assertEqual(DUMMY_DIM, index.dim)

        for index in self.psg_indexes:
            self.assertEqual(set(DUMMY_PSG_IDS), index.psg_ids)
            self.assertEqual(0, len(index.doc_ids))
            self.assertEqual(DUMMY_NUM, len(index))
            self.assertEqual(DUMMY_DIM, index.dim)

    def test_add_retrieve(self):
        self.assertEqual(0, len(self.index))

        data = np.random.normal(size=(80, 16))
        doc_ids = [f"doc_{int(i/2)}" for i in range(data.shape[0])]
        psg_ids = [f"psg_{i}" for i in range(data.shape[0])]

        # successively add parts of the data and make sure we still get the correct vectors back
        for lower, upper in [(0, 8), (8, 24), (24, 80)]:
            self.index.add(
                data[lower:upper],
                doc_ids=doc_ids[lower:upper],
                psg_ids=psg_ids[lower:upper],
            )
            self.assertEqual(upper, len(self.index))

            self.index.mode = Mode.PASSAGE
            vecs, idxs = self.index._get_vectors(psg_ids[lower:upper])
            np.testing.assert_almost_equal(vecs, data[lower:upper], decimal=6)
            self.assertEqual([[idx] for idx in range(upper - lower)], idxs)

            self.index.mode = Mode.MAXP
            vecs, idxs = self.index._get_vectors(
                [f"doc_{i}" for i in range(int(lower / 2), int(upper / 2))]
            )
            np.testing.assert_almost_equal(vecs, data[lower:upper], decimal=6)
            self.assertEqual(
                [[2 * idx, 2 * idx + 1] for idx in range(int((upper - lower) / 2))],
                idxs,
            )

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
        with self.assertRaises(ValueError):
            self.index_no_enc.add(DUMMY_VECTORS, doc_ids=None, psg_ids=None)
        with self.assertRaises(RuntimeError):
            self.index_no_enc.encode(["test"])
        with self.assertRaises(ValueError):
            self.index_wrong_dim.add(
                DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
            )

    def test_coalescing(self):
        # delta = 0.3: vectors of d0 should be averaged
        create_coalesced_index(self.doc_indexes[0], self.coalesced_indexes[0], 0.3)
        self.assertEqual(self.doc_indexes[0].doc_ids, self.coalesced_indexes[0].doc_ids)
        d0_vector_expected = np.average([DUMMY_VECTORS[0], DUMMY_VECTORS[1]], axis=0)
        d0_vectors, _ = self.coalesced_indexes[0]._get_vectors(["d0"])
        self.assertEqual(1, len(d0_vectors))
        self.assertTrue(np.array_equal(d0_vector_expected, d0_vectors[0]))

        # delta = 0.2: nothing should change
        create_coalesced_index(
            self.doc_indexes[0], self.coalesced_indexes[1], 0.2, buffer_size=2
        )
        self.assertEqual(self.doc_indexes[0].doc_ids, self.coalesced_indexes[1].doc_ids)
        for doc_id in self.doc_indexes[0].doc_ids:
            vectors_1, _ = self.doc_indexes[0]._get_vectors([doc_id])
            vectors_2, _ = self.coalesced_indexes[1]._get_vectors([doc_id])
            self.assertEqual(len(vectors_1), len(vectors_2))
            for v1, v2 in zip(vectors_1, vectors_2):
                print(v1, v2)
                self.assertTrue(np.array_equal(v1, v2))


class TestInMemoryIndex(TestIndex):
    __test__ = True

    def setUp(self):
        self.index = InMemoryIndex(16, init_size=32, alloc_size=32)
        self.doc_psg_indexes = [
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER),
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER, init_size=2, alloc_size=2),
        ]
        self.doc_indexes = [
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER),
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER, init_size=2, alloc_size=2),
        ]
        self.psg_indexes = [
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER),
            InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER, init_size=2, alloc_size=2),
        ]
        self.index_no_enc = InMemoryIndex(DUMMY_DIM, encoder=None)
        self.index_wrong_dim = InMemoryIndex(DUMMY_DIM + 1, encoder=None)
        self.coalesced_indexes = [
            InMemoryIndex(DUMMY_DIM, mode=Mode.MAXP),
            InMemoryIndex(DUMMY_DIM, mode=Mode.MAXP),
        ]
        super().setUp()


class TestOnDiskIndex(TestIndex):
    __test__ = True

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.index = OnDiskIndex(
            self.temp_dir / "index.h5", 16, init_size=32, resize_min_val=32
        )
        self.doc_psg_indexes = [
            OnDiskIndex(
                self.temp_dir / "doc_psg_index.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
            ),
            OnDiskIndex(
                self.temp_dir / "doc_psg_index_2.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
                init_size=2,
                resize_min_val=2,
            ),
        ]
        self.doc_indexes = [
            OnDiskIndex(
                self.temp_dir / "doc_index.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
            ),
            OnDiskIndex(
                self.temp_dir / "doc_index_2.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
                init_size=2,
                resize_min_val=2,
            ),
        ]
        self.psg_indexes = [
            OnDiskIndex(
                self.temp_dir / "psg_index.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
            ),
            OnDiskIndex(
                self.temp_dir / "psg_index_2.h5",
                DUMMY_DIM,
                DUMMY_ENCODER,
                init_size=2,
                resize_min_val=2,
            ),
        ]
        self.index_no_enc = OnDiskIndex(
            self.temp_dir / "index_no_enc.h5", DUMMY_DIM, encoder=None
        )
        self.index_wrong_dim = OnDiskIndex(
            self.temp_dir / "index_wrong_dim.h5", DUMMY_DIM + 1, encoder=None
        )
        self.coalesced_indexes = [
            OnDiskIndex(
                self.temp_dir / "coalesced_index_1.h5", DUMMY_DIM, mode=Mode.MAXP
            ),
            OnDiskIndex(
                self.temp_dir / "coalesced_index_2.h5", DUMMY_DIM, mode=Mode.MAXP
            ),
        ]
        super().setUp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


if __name__ == "__main__":
    unittest.main()
