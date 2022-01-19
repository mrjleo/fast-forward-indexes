#!/usr/bin/env python3


import logging
import argparse
from pathlib import Path

import numpy as np
from pyserini.dsearch import SimpleDenseSearcher

from fast_forward.index import InMemoryIndex, Mode


LOGGER = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("INDEX", help="Pyserini index name (pre-built) or path.")
    ap.add_argument("--out_file", default="out", help="Output file.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    dsearcher = SimpleDenseSearcher.from_prebuilt_index(args.INDEX, None)
    vectors = list(map(dsearcher.index.reconstruct, range(len(dsearcher.docids))))
    psg_ids, doc_ids = [], []
    for id in dsearcher.docids:
        psg_ids.append(id)
        id_split = id.split("#")
        assert 1 <= len(id_split) <= 2
        doc_ids.append(id_split[0])
    LOGGER.info(f"index size: {len(vectors)}")
    arr = np.array(vectors)
    del vectors
    del dsearcher

    index = InMemoryIndex(mode=Mode.MAXP)
    index.add(arr, doc_ids=doc_ids, psg_ids=psg_ids)
    del arr

    index.save(Path(args.out_file))


if __name__ == "__main__":
    main()
