import unittest

from fast_forward import Ranking
from fast_forward.util import to_ir_measures

from .test_ranking import DUMMY_QUERIES, RUN


class TestUtil(unittest.TestCase):
    def test_ir_measures_df(self):
        r = Ranking.from_run(RUN, queries=DUMMY_QUERIES)
        df = to_ir_measures(r)
        self.assertTrue(df["query_id"].equals(r._df["q_id"]))
        self.assertTrue(df["doc_id"].equals(r._df["id"]))
        self.assertTrue(df["score"].equals(r._df["score"]))
        self.assertEqual(set(df.columns), set(("query_id", "doc_id", "score")))


if __name__ == "__main__":
    unittest.main()
