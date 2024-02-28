import os
import tempfile
import unittest
from pathlib import Path

from fast_forward import Ranking

RUN = {
    "q1": {"d0": 1, "d1": 2, "d2": 300},
    "q2": {"d0": 4, "d1": 5, "d2": 600, "d3": 7},
}


class TestRanking(unittest.TestCase):
    def test_ranking(self):
        r = Ranking(RUN)
        self.assertEqual({"q1", "q2"}, r.q_ids)
        self.assertEqual(len(r), 2)
        self.assertIn("q1", r)
        self.assertIn("q2", r)
        self.assertNotIn("q3", r)

    def test_sorting(self):
        r = Ranking(RUN, sort=False)
        self.assertFalse(r.is_sorted)
        r.sort()
        self.assertTrue(r.is_sorted)
        self.assertEqual(r["q1"], {"d2": 300, "d1": 2, "d0": 1})
        self.assertEqual(r["q2"], {"d2": 600, "d3": 7, "d1": 5, "d0": 4})

    def test_eq(self):
        r1 = Ranking({"q1": {"d1": 1, "d2": 2}})
        r2 = Ranking({"q1": {"d2": 2, "d1": 1}})
        r3 = Ranking({"q1": {"d1": 2, "d2": 3}})
        r4 = Ranking({"q1": {"d1": 1, "d2": 2}, "q2": {}})
        self.assertEqual(r1, r2)
        self.assertNotEqual(r1, r3)
        self.assertNotEqual(r1, r4)

    def test_cut(self):
        r = Ranking(RUN)
        r.cut(2)
        r_expected = Ranking({"q1": {"d2": 300, "d1": 2}, "q2": {"d2": 600, "d3": 7}})
        self.assertEqual(r, r_expected)

    def test_save_load(self):
        r = Ranking(RUN, name="Dummy")
        fd, f = tempfile.mkstemp()
        f = Path(f)
        r.save(f)
        r_from_file = Ranking.from_file(f)
        self.assertEqual(r, r_from_file)
        self.assertEqual(r.name, r_from_file.name)
        os.close(fd)
        os.remove(f)


if __name__ == "__main__":
    unittest.main()
