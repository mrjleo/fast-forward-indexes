import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fast_forward import InMemoryIndex, Mode, OnDiskIndex, Ranking
from fast_forward.encoder import LambdaEncoder
from fast_forward.util import create_coalesced_index

DUMMY_QUERIES = {"q1": "query 1", "q2": "query 2"}
DUMMY_DOC_IDS = ["d0", "d0", "d1", "d2", "d3"]
UNIQUE_DUMMY_DOC_IDS = list(set(DUMMY_DOC_IDS))
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
DUMMY_DOC_RANKING = Ranking.from_run(DUMMY_DOC_RUN, queries=DUMMY_QUERIES)
DUMMY_PSG_RUN = {
    "q1": {"p0": 100, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
    "q2": {"p0": 500, "p1": 6, "p2": 7, "p3": 8, "p4": 9},
}
DUMMY_PSG_RANKING = Ranking.from_run(DUMMY_PSG_RUN, queries=DUMMY_QUERIES)
DUMMY_ENCODER = LambdaEncoder(lambda _: np.array([1, 1, 1, 1, 1]))


class TestIndex(unittest.TestCase):
    __test__ = False

    def setUp(self):
        self.doc_psg_index.add(
            vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
        )
        self.index_partial_ids.add(vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS)
        self.index_partial_ids.add(vectors=DUMMY_VECTORS, psg_ids=DUMMY_PSG_IDS)
        self.doc_index.add(vectors=DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS)
        self.psg_index.add(vectors=DUMMY_VECTORS, psg_ids=DUMMY_PSG_IDS)

    def test_properties(self):
        self.assertEqual(set(DUMMY_DOC_IDS), self.doc_psg_index.doc_ids)
        self.assertEqual(set(DUMMY_PSG_IDS), self.doc_psg_index.psg_ids)
        self.assertEqual(DUMMY_NUM, len(self.doc_psg_index))
        self.assertEqual(DUMMY_DIM, self.doc_psg_index.dim)

        self.assertEqual(set(DUMMY_DOC_IDS), self.index_partial_ids.doc_ids)
        self.assertEqual(set(DUMMY_PSG_IDS), self.index_partial_ids.psg_ids)
        self.assertEqual(DUMMY_NUM * 2, len(self.index_partial_ids))
        self.assertEqual(DUMMY_DIM, self.index_partial_ids.dim)

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
            _test_vectors(
                vecs, idxs, data[lower:upper], [[idx] for idx in range(upper - lower)]
            )

            self.index.mode = Mode.MAXP
            vecs, idxs = self.index._get_vectors(
                [f"doc_{i}" for i in range(int(lower / 2), int(upper / 2))]
            )
            np.testing.assert_almost_equal(vecs, data[lower:upper], decimal=6)
            self.assertEqual(
                [[2 * idx, 2 * idx + 1] for idx in range(int((upper - lower) / 2))],
                idxs,
            )

    def test_maxp(self):
        self.doc_psg_index.mode = Mode.MAXP
        result = self.doc_psg_index(DUMMY_DOC_RANKING)
        self.assertEqual(
            result,
            Ranking.from_run(
                {
                    "q1": {"d0": 2, "d1": 3, "d2": 4, "d3": 5},
                    "q2": {"d0": 2, "d1": 3, "d2": 4, "d3": 5},
                }
            ),
        )

    def test_firstp(self):
        expected = Ranking.from_run(
            {
                "q1": {"d0": 1, "d1": 3, "d2": 4, "d3": 5},
                "q2": {"d0": 1, "d1": 3, "d2": 4, "d3": 5},
            }
        )
        self.doc_psg_index.mode = Mode.FIRSTP
        self.assertEqual(
            self.doc_psg_index(DUMMY_DOC_RANKING),
            expected,
        )
        self.index_partial_ids.mode = Mode.FIRSTP
        self.assertEqual(
            self.doc_psg_index(DUMMY_DOC_RANKING),
            expected,
        )

    def test_avep(self):
        expected = Ranking.from_run(
            {
                "q1": {"d0": 1.5, "d1": 3, "d2": 4, "d3": 5},
                "q2": {"d0": 1.5, "d1": 3, "d2": 4, "d3": 5},
            }
        )

        self.doc_psg_index.mode = Mode.AVEP
        self.assertEqual(
            self.doc_psg_index(DUMMY_DOC_RANKING),
            expected,
        )
        self.index_partial_ids.mode = Mode.AVEP
        self.assertEqual(
            self.index_partial_ids(DUMMY_DOC_RANKING),
            expected,
        )

    def test_passage(self):
        expected = Ranking.from_run(
            {
                "q1": {"p0": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
                "q2": {"p0": 1, "p1": 2, "p2": 3, "p3": 4, "p4": 5},
            }
        )
        self.doc_psg_index.mode = Mode.PASSAGE
        self.assertEqual(
            self.doc_psg_index(DUMMY_PSG_RANKING),
            expected,
        )
        self.index_partial_ids.mode = Mode.PASSAGE
        self.assertEqual(
            self.index_partial_ids(DUMMY_PSG_RANKING),
            expected,
        )

    def test_errors(self):
        with self.assertRaises(ValueError):
            self.index_no_enc.add(DUMMY_VECTORS, doc_ids=None, psg_ids=None)
        with self.assertRaises(RuntimeError):
            self.index_no_enc.encode_queries(["test"])
        with self.assertRaises(ValueError):
            self.index_wrong_dim.add(
                DUMMY_VECTORS, doc_ids=DUMMY_DOC_IDS, psg_ids=DUMMY_PSG_IDS
            )
        ranking_no_queries = Ranking.from_run(DUMMY_DOC_RUN)
        with self.assertRaises(ValueError):
            self.doc_psg_index(ranking_no_queries)

        with self.assertRaises(ValueError):
            self.doc_psg_index(
                DUMMY_DOC_RANKING, early_stopping=10, early_stopping_alpha=None
            )
        with self.assertRaises(ValueError):
            self.doc_psg_index(
                DUMMY_DOC_RANKING, early_stopping=10, early_stopping_intervals=None
            )

    def test_early_stopping(self):
        vectors = np.stack([[1, 0], [1, 1]] * 10)
        psg_ids = [f"p{i}" for i in range(20)]
        index = InMemoryIndex(
            2, LambdaEncoder(lambda q: np.array([10, 10])), mode=Mode.PASSAGE
        )
        index.add(vectors, psg_ids=psg_ids)
        r = Ranking(
            pd.DataFrame(
                [
                    {"q_id": q, "query": q, "id": f"p{i}", "score": i}
                    for i in range(20)
                    for q in ("q1", "q2")
                ]
            )
        )

        result_expected = Ranking(
            pd.DataFrame(
                [
                    {"q_id": "q2", "id": "p19", "score": 20.0},
                    {"q_id": "q2", "id": "p17", "score": 20.0},
                    {"q_id": "q2", "id": "p15", "score": 20.0},
                    {"q_id": "q2", "id": "p13", "score": 20.0},
                    {"q_id": "q2", "id": "p11", "score": 20.0},
                    {"q_id": "q2", "id": "p18", "score": 10.0},
                    {"q_id": "q2", "id": "p16", "score": 10.0},
                    {"q_id": "q2", "id": "p14", "score": 10.0},
                    {"q_id": "q2", "id": "p12", "score": 10.0},
                    {"q_id": "q2", "id": "p10", "score": 10.0},
                    {"q_id": "q1", "id": "p19", "score": 20.0},
                    {"q_id": "q1", "id": "p17", "score": 20.0},
                    {"q_id": "q1", "id": "p15", "score": 20.0},
                    {"q_id": "q1", "id": "p13", "score": 20.0},
                    {"q_id": "q1", "id": "p11", "score": 20.0},
                    {"q_id": "q1", "id": "p18", "score": 10.0},
                    {"q_id": "q1", "id": "p16", "score": 10.0},
                    {"q_id": "q1", "id": "p14", "score": 10.0},
                    {"q_id": "q1", "id": "p12", "score": 10.0},
                    {"q_id": "q1", "id": "p10", "score": 10.0},
                ]
            )
        )

        self.assertEqual(
            index(
                r,
                early_stopping=5,
                early_stopping_alpha=0.5,
                early_stopping_intervals=(2, 5, 10, 20),
            ),
            result_expected,
        )

        # order of intervals should make no difference
        self.assertEqual(
            index(
                r,
                early_stopping=5,
                early_stopping_alpha=0.5,
                early_stopping_intervals=(5, 2, 20, 10),
            ),
            result_expected,
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
                self.assertTrue(np.array_equal(v1, v2))


class TestInMemoryIndex(TestIndex):
    __test__ = True

    def setUp(self):
        self.index = InMemoryIndex(16, init_size=32, alloc_size=32)
        self.doc_psg_index = InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER)
        self.index_partial_ids = InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER)
        self.doc_index = InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER)
        self.psg_index = InMemoryIndex(DUMMY_DIM, DUMMY_ENCODER)
        self.index_no_enc = InMemoryIndex(DUMMY_DIM, query_encoder=None)
        self.index_wrong_dim = InMemoryIndex(DUMMY_DIM + 1, query_encoder=None)
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
        _test_vectors(vecs, idxs, data[:14], [[idx] for idx in range(14)])

        index.add(data[14:32], psg_ids=psg_ids[14:32])
        index.consolidate()
        vecs, idxs = index._get_vectors(psg_ids)
        _test_vectors(vecs, idxs, data, [[idx] for idx in range(32)])


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
        self.index_partial_ids = OnDiskIndex(
            self.temp_dir / "index_partial_ids.h5",
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
            self.temp_dir / "index_no_enc.h5", DUMMY_DIM, query_encoder=None
        )
        self.index_wrong_dim = OnDiskIndex(
            self.temp_dir / "index_wrong_dim.h5", DUMMY_DIM + 1, query_encoder=None
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
        _test_get_vectors(index_copied, self.doc_psg_index, DUMMY_PSG_IDS)
        self.doc_psg_index.mode = Mode.MAXP
        index_copied.mode = Mode.MAXP
        _test_get_vectors(index_copied, self.doc_psg_index, UNIQUE_DUMMY_DOC_IDS)

        shutil.copy(self.temp_dir / "doc_index.h5", self.temp_dir / "doc_index_copy.h5")
        index_copied = OnDiskIndex.load(self.temp_dir / "doc_index_copy.h5")
        self.assertEqual(index_copied._get_doc_ids(), self.doc_index._get_doc_ids())
        self.assertEqual(index_copied._get_psg_ids(), self.doc_index._get_psg_ids())
        self.doc_index.mode = Mode.MAXP
        index_copied.mode = Mode.MAXP
        _test_get_vectors(index_copied, self.doc_index, UNIQUE_DUMMY_DOC_IDS)

        shutil.copy(self.temp_dir / "psg_index.h5", self.temp_dir / "psg_index_copy.h5")
        index_copied = OnDiskIndex.load(self.temp_dir / "psg_index_copy.h5")
        self.assertEqual(index_copied._get_doc_ids(), self.psg_index._get_doc_ids())
        self.assertEqual(index_copied._get_psg_ids(), self.psg_index._get_psg_ids())
        self.psg_index.mode = Mode.PASSAGE
        index_copied.mode = Mode.PASSAGE
        _test_get_vectors(index_copied, self.psg_index, DUMMY_PSG_IDS)

    def test_to_memory(self):
        for index, params in [
            (self.doc_index, [(Mode.MAXP, UNIQUE_DUMMY_DOC_IDS)]),
            (self.psg_index, [(Mode.PASSAGE, DUMMY_PSG_IDS)]),
            (
                self.doc_psg_index,
                [(Mode.MAXP, UNIQUE_DUMMY_DOC_IDS), (Mode.PASSAGE, DUMMY_PSG_IDS)],
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

    def test_max_id_length(self):
        index = OnDiskIndex(
            self.temp_dir / "max_id_length_index.h5", 16, max_id_length=3
        )
        vectors = np.zeros(shape=(16, 16))
        doc_ids_ok = ["d1"] * 16
        psg_ids_ok = [f"p{i}" for i in range(16)]
        index.add(vectors, doc_ids=doc_ids_ok, psg_ids=psg_ids_ok)

        doc_ids_long = [doc_id + "-long" for doc_id in doc_ids_ok]
        psg_ids_long = [psg_id + "-long" for psg_id in psg_ids_ok]

        with self.assertRaises(RuntimeError):
            index.add(vectors, doc_ids=doc_ids_long)
        with self.assertRaises(RuntimeError):
            index.add(vectors, psg_ids=psg_ids_long)

        # make sure the index remains unchanged in case of the error
        self.assertEqual(index._get_doc_ids(), set(doc_ids_ok))
        self.assertEqual(index._get_psg_ids(), set(psg_ids_ok))
        self.assertEqual(16, len(index))

    def test_ds_buffer_size(self):
        index = OnDiskIndex(
            self.temp_dir / "ds_buffer_size_index.h5",
            16,
            mode=Mode.PASSAGE,
            ds_buffer_size=5,
        )
        psg_reps = np.random.normal(size=(16, 16))
        psg_ids = [f"p{i}" for i in range(16)]
        index.add(psg_reps, psg_ids=psg_ids)
        vecs, id_idxs = index._get_vectors(psg_ids)
        np.testing.assert_almost_equal(
            vecs[id_idxs], psg_reps.reshape((16, 1, 16)), decimal=6
        )

    def tearDown(self):
        shutil.rmtree(self.temp_dir)


def _test_get_vectors(index_1, index_2, ids):
    vecs_1, idxs_1 = index_1._get_vectors(ids)
    vecs_2, idxs_2 = index_2._get_vectors(ids)
    _test_vectors(vecs_1, idxs_1, vecs_2, idxs_2)


def _test_vectors(vecs_1, idxs_1, vecs_2, idxs_2):
    # this accounts for the fact that the indices returned by _get_vectors may be different,
    # but the overall result is still the same
    for i, j in zip(idxs_1, idxs_2):
        np.testing.assert_almost_equal(vecs_1[i], vecs_2[j], decimal=6)


if __name__ == "__main__":
    unittest.main()
