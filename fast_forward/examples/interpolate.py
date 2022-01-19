#!/usr/bin/env python3


import csv
import logging
import argparse
from pathlib import Path

from fast_forward.ranking import Ranking
from fast_forward.index import InMemoryIndex, Mode
from fast_forward.encoder import TransformerQueryEncoder, TCTColBERTQueryEncoder


LOGGER = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("INDEX", help="Fast-Forward index file.")
    ap.add_argument(
        "MODE", choices=["maxp", "avep", "firstp", "passage"], help="Retrieval mode."
    )
    ap.add_argument("ENCODER", help="Pre-trained transformer encoder.")
    ap.add_argument(
        "SPARSE_SCORES",
        help="TREC runfile containing the scores of the sparse retriever.",
    )
    ap.add_argument("QUERY_FILE", help="Queries (tsv).")
    ap.add_argument(
        "--cutoff", type=int, help="Maximum number of sparse documents per query."
    )
    ap.add_argument(
        "--cutoff_result",
        type=int,
        default=1000,
        help="Maximum number of documents per query in the final ranking.",
    )
    ap.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        help="Interpolation weight.",
    )
    ap.add_argument(
        "--early_stopping", action="store_true", help="Use approximated early stopping."
    )
    ap.add_argument("--target", default="out", help="Output directory.")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO)

    LOGGER.info(f"reading {args.SPARSE_SCORES}")
    sparse_scores = Ranking.from_file(Path(args.SPARSE_SCORES))
    if args.cutoff is not None:
        sparse_scores.cut(args.cutoff)

    LOGGER.info(f"reading {args.QUERY_FILE}")
    with open(args.QUERY_FILE, encoding="utf-8", newline="") as fp:
        queries = {q_id: q for q_id, q in csv.reader(fp, delimiter="\t")}

    if "tct_colbert" in args.ENCODER:
        encoder = TCTColBERTQueryEncoder(args.ENCODER)
    else:
        encoder = TransformerQueryEncoder(args.ENCODER)

    mode = {
        "maxp": Mode.MAXP,
        "avep": Mode.AVEP,
        "firstp": Mode.FIRSTP,
        "passage": Mode.PASSAGE,
    }[args.MODE]

    LOGGER.info(f"reading {args.INDEX}")
    index = InMemoryIndex.from_disk(Path(args.INDEX), encoder, mode=mode)
    result = index.get_scores(
        sparse_scores, queries, args.alpha, args.cutoff_result, args.early_stopping
    )
    for alpha, ranking in result.items():
        name = f"interpolation-{alpha}"
        if args.early_stopping:
            name += "-es"
        ranking.name = name
        target = Path(args.target) / f"{name}.tsv"
        LOGGER.info(f"writing {target}")
        ranking.save(target)


if __name__ == "__main__":
    main()
