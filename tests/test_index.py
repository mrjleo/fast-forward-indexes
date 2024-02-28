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
        self.doc_psg_index.add(
            vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
        )
        self.doc_index.add(vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS)
        self.psg_index.add(vectors=DUMMY_VECTORS, psg_ids=DUMMY_PSG_IDS)

    def test_properties(self):
        self.assertEqual(set(DUMMY_DOC_IDS), self.doc_psg_index.doc_ids)
        self.assertEqual(set(DUMMY_PSG_IDS), self.doc_psg_index.psg_ids)
        self.assertEqual(DUMMY_NUM, len(self.doc_psg_index))
        self.assertEqual(DUMMY_DIM, self.doc_psg_index.dim)

        self.assertEqual(set(DUMMY_DOC_IDS), self.doc_index.doc_ids)
        self.assertEqual(0, len(self.doc_index.psg_ids))
        self.assertEqual(DUMMY_NUM, len(self.doc_index))
        self.assertEqual(DUMMY_DIM, self.doc_index.dim)

        self.assertEqual(set(DUMMY_PSG_IDS), self.psg_index.psg_ids)
        self.assertEqual(0, len(self.psg_index.doc_ids))
        self.assertEqual(DUMMY_NUM, len(self.psg_index))
        self.assertEqual(DUMMY_DIM, self.psg_index.dim)

    def test_add_retrieve(self):
        self.assertEqual(0, len(self.index))

        data = np.random.normal(size=(80, 16))
        doc_ids = [f"doc_{int(i/2)}" for i in range(data.shape[0])]
        psg_ids = [f"psg_{i}" for i in range(data.shape[0])]

        # successively add parts of the data and make sure we still get the correct vectors and indices back as the index grows
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
        self.doc_psg_index.mode = Mode.MAXP
        result = self.doc_psg_index.get_scores(
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
        self.doc_psg_index.mode = Mode.MAXP

        scores_1 = self.doc_psg_index.get_scores(
            DUMMY_DOC_RANKING,
            DUMMY_QUERIES,
            alpha=0.5,
            cutoff=2,
            early_stopping=True,
        )

        # test unsorted ranking
        scores_2 = self.doc_psg_index.get_scores(
            Ranking(DUMMY_DOC_RUN, sort=False),
            DUMMY_QUERIES,
            alpha=0.5,
            cutoff=2,
            early_stopping=True,
        )

        self.assertEqual(
            scores_1[0.5],
            Ranking({"q1": {"d3": 102.5, "d0": 51}, "q2": {"d3": 402.5, "d0": 201}}),
        )
        self.assertEqual(scores_1[0.5], scores_2[0.5])

        with self.assertRaises(ValueError):
            self.doc_psg_index.get_scores(
                DUMMY_DOC_RANKING,
                DUMMY_QUERIES,
                alpha=0.5,
                cutoff=None,
                early_stopping=True,
            )

    def test_firstp(self):
        self.doc_psg_index.mode = Mode.FIRSTP
        self.assertEqual(
            self.doc_psg_index.get_scores(DUMMY_DOC_RANKING, DUMMY_QUERIES)[0.0],
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
            self.doc_psg_index.get_scores(DUMMY_DOC_RANKING, DUMMY_QUERIES)[0.0],
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
            self.doc_psg_index.get_scores(DUMMY_PSG_RANKING, DUMMY_QUERIES)[0.0],
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
        create_coalesced_index(self.doc_index, self.coalesced_indexes[0], 0.3)
        self.assertEqual(self.doc_index.doc_ids, self.coalesced_indexes[0].doc_ids)
        d0_vector_expected = np.average([DUMMY_VECTORS[0], DUMMY_VECTORS[1]], axis=0)
        d0_vectors, _ = self.coalesced_indexes[0]._get_vectors(["d0"])
        self.assertEqual(1, len(d0_vectors))
        self.assertTrue(np.array_equal(d0_vector_expected, d0_vectors[0]))

        # delta = 0.2: nothing should change
        create_coalesced_index(
            self.doc_index, self.coalesced_indexes[1], 0.2, buffer_size=2
        )
        self.assertEqual(self.doc_index.doc_ids, self.coalesced_indexes[1].doc_ids)
        for doc_id in self.doc_index.doc_ids:
            vectors_1, _ = self.doc_index._get_vectors([doc_id])
            vectors_2, _ = self.coalesced_indexes[1]._get_vectors([doc_id])
            self.assertEqual(len(vectors_1), len(vectors_2))
            for v1, v2 in zip(vectors_1, vectors_2):
                print(v1, v2)
                self.assertTrue(np.array_equal(v1, v2))


class TestInMemoryIndex(TestIndex):
    __test__ = True

    def setUp(self):
        self.index = InMemoryIndex(16, init_size=32, alloc_size=32)
        self.doc_psg_index = InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER)
        self.doc_index = InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER)
        self.psg_index = InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER)
        self.index_no_enc = InMemoryIndex(DUMMY_DIM, encoder=None)
        self.index_wrong_dim = InMemoryIndex(DUMMY_DIM + 1, encoder=None)
        self.coalesced_indexes = [
            InMemoryIndex(DUMMY_DIM, mode=Mode.MAXP),
            InMemoryIndex(DUMMY_DIM, mode=Mode.MAXP),
        ]
        super().setUp()

    def test_consolidate(self):
        index = InMemoryIndex(16, init_size=8, alloc_size=4)
        data = data = np.random.normal(size=(32, 16))
        psg_ids = [f"psg_{i}" for i in range(32)]

        index.add(data[:14], psg_ids=psg_ids[:14])
        index.consolidate()
        vecs, idxs = index._get_vectors(psg_ids[:14])
        np.testing.assert_almost_equal(vecs, data[:14], decimal=6)
        self.assertEqual([[idx] for idx in range(14)], idxs)

        index.add(data[14:32], psg_ids=psg_ids[14:32])
        index.consolidate()
        vecs, idxs = index._get_vectors(psg_ids)
        np.testing.assert_almost_equal(vecs, data, decimal=6)
        self.assertEqual([[idx] for idx in range(32)], idxs)


class TestOnDiskIndex(TestIndex):
    __test__ = True

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.index = OnDiskIndex(
            self.temp_dir / "index.h5", 16, init_size=32, resize_min_val=32
        )
        self.doc_psg_index = OnDiskIndex(
            self.temp_dir / "doc_psg_index.h5",
            DUMMY_DIM,
            DUMMY_ENCODER,
        )
        self.doc_index = OnDiskIndex(
            self.temp_dir / "doc_index.h5",
            DUMMY_DIM,
            DUMMY_ENCODER,
        )
        self.psg_index = OnDiskIndex(
            self.temp_dir / "psg_index.h5",
            DUMMY_DIM,
            DUMMY_ENCODER,
        )
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

    def test_load(self):
        shutil.copy(
            self.temp_dir / "doc_psg_index.h5", self.temp_dir / "doc_psg_index_copy.h5"
        )
        index_copied = OnDiskIndex.load(self.temp_dir / "doc_psg_index_copy.h5")
        self.assertEqual(index_copied._get_doc_ids(), self.doc_psg_index._get_doc_ids())
        self.assertEqual(index_copied._get_psg_ids(), self.doc_psg_index._get_psg_ids())
        self.doc_psg_index.mode = Mode.PASSAGE
        index_copied.mode = Mode.PASSAGE
        vecs_1, idxs_1 = self.doc_psg_index._get_vectors(DUMMY_PSG_IDS)
        vecs_2, idxs_2 = index_copied._get_vectors(DUMMY_PSG_IDS)
        np.testing.assert_almost_equal(vecs_1, vecs_2, decimal=6)
        self.assertEqual(idxs_1, idxs_2)

        shutil.copy(self.temp_dir / "doc_index.h5", self.temp_dir / "doc_index_copy.h5")
        index_copied = OnDiskIndex.load(self.temp_dir / "doc_index_copy.h5")
        self.assertEqual(index_copied._get_doc_ids(), self.doc_index._get_doc_ids())
        self.assertEqual(index_copied._get_psg_ids(), self.doc_index._get_psg_ids())

        shutil.copy(self.temp_dir / "psg_index.h5", self.temp_dir / "psg_index_copy.h5")
        index_copied = OnDiskIndex.load(self.temp_dir / "psg_index_copy.h5")
        self.assertEqual(index_copied._get_doc_ids(), self.psg_index._get_doc_ids())
        self.assertEqual(index_copied._get_psg_ids(), self.psg_index._get_psg_ids())

    def test_to_memory(self):
        unique_dummy_doc_ids = list(set(DUMMY_DOC_IDS))
        for index, params in [
            (self.doc_index, [(Mode.MAXP, unique_dummy_doc_ids)]),
            (self.psg_index, [(Mode.PASSAGE, DUMMY_PSG_IDS)]),
            (
                self.doc_psg_index,
                [(Mode.MAXP, unique_dummy_doc_ids), (Mode.PASSAGE, DUMMY_PSG_IDS)],
            ),
        ]:
            mem_index = index.to_memory()
            mem_index_buffered = index.to_memory(buffer_size=2)

            for mode, ids in params:
                index.mode = mode
                mem_index.mode = mode
                mem_index_buffered.mode = mode

                self.assertEqual(mem_index._get_doc_ids(), index._get_doc_ids())
                self.assertEqual(mem_index._get_psg_ids(), index._get_psg_ids())
                self.assertEqual(
                    mem_index_buffered._get_doc_ids(), index._get_doc_ids()
                )
                self.assertEqual(
                    mem_index_buffered._get_psg_ids(), index._get_psg_ids()
                )

                _test_get_vectors(mem_index, index, ids)
                _test_get_vectors(mem_index_buffered, index, ids)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


def _test_get_vectors(index_1, index_2, ids):
    vecs_1, idxs_1 = index_1._get_vectors(ids)
    vecs_2, idxs_2 = index_2._get_vectors(ids)
    for i, j in zip(idxs_1, idxs_2):
        print(i, j)
        np.testing.assert_almost_equal(vecs_1[i], vecs_2[j], decimal=6)


if __name__ == "__main__":
    unittest.main()
