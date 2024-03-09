import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from fast_forward import Ranking

RUN = {
    "q1": {"d0": 1, "d1": 2, "d2": 300},
    "q2": {"d0": 4, "d1": 5, "d2": 600, "d3": 7},
}
DUMMY_QUERIES = {"q1": "query 1", "q2": "query 2"}


class TestRanking(unittest.TestCase):
    def test_ranking(self):
        r = Ranking.from_run(RUN)
        self.assertEqual({"q1", "q2"}, r.q_ids)
        self.assertEqual(len(r), 2)
        self.assertIn("q1", r)
        self.assertIn("q2", r)
        self.assertNotIn("q3", r)

    def test_attach_queries(self):
        r = Ranking.from_run(RUN)
        r_with_queries = r.attach_queries(DUMMY_QUERIES)
        self.assertFalse(r.has_queries)
        self.assertTrue(r_with_queries.has_queries)
        self.assertEqual(
            pd.unique(r_with_queries._df.loc[r._df["q_id"].eq("q1"), "query"]).tolist(),
            ["query 1"],
        )
        self.assertEqual(
            pd.unique(r_with_queries._df.loc[r._df["q_id"].eq("q2"), "query"]).tolist(),
            ["query 2"],
        )

        more_queries = {"qx": "other query"}
        with self.assertRaises(ValueError):
            Ranking.from_run(RUN, queries=more_queries)

        more_queries.update(DUMMY_QUERIES)
        r2_with_queries = Ranking.from_run(RUN, queries=more_queries)
        self.assertEqual(r2_with_queries, r_with_queries)

    def test_eq(self):
        r1 = Ranking.from_run({"q1": {"d1": 1, "d2": 2}})
        r2 = Ranking.from_run({"q1": {"d2": 2, "d1": 1}})
        r3 = Ranking.from_run({"q1": {"d1": 2, "d2": 3}})
        r4 = Ranking.from_run({"q1": {"d1": 1, "d2": 2}, "q2": {}})
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)
        self.assertEqual(r1, r4)

        self.assertNotEqual(r1, "string")
        self.assertNotEqual(r1, 0)

    def test_cut(self):
        self.assertEqual(
            Ranking.from_run(RUN).cut(2),
            Ranking.from_run({"q1": {"d2": 300, "d1": 2}, "q2": {"d2": 600, "d3": 7}}),
        )

    def test_save_load(self):
        r = Ranking.from_run(RUN, name="Dummy")
        fd, f = tempfile.mkstemp()
        f = Path(f)
        r.save(f)
        r_from_file = Ranking.from_file(f)
        self.assertEqual(r, r_from_file)
        self.assertEqual(r.name, r_from_file.name)
        os.close(fd)
        os.remove(f)

    def test_interpolate(self):
        r = Ranking.from_run(RUN)
        df = r._df.copy()
        df["score"] = list(map(np.float32, range(len(r._df))))
        r2 = Ranking(df)
        r_int = r.interpolate(r2, 0.5)
        self.assertNotEqual(r, r_int)
        self.assertEqual(r_int["q1"], {"d2": 152.0, "d1": 3.5, "d0": 3.5})
        self.assertEqual(r_int["q2"], {"d2": 300.0, "d3": 4.0, "d1": 3.5, "d0": 3.5})


if __name__ == "__main__":
    unittest.main()
