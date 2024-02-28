# Fast-Forward Indexes

This is the reference implementation of [Fast-Forward indexes](https://arxiv.org/abs/2110.06051).

⚠ **Important**: As this library is still in its early stages, the API is subject to change!

## Installation

Install the package via `pip`:

```bash
pip install fast-forward-indexes
```

## Getting Started

Using a Fast-Forward index is as simple as providing a TREC run with sparse scores:

```python
from pathlib import Path
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward import OnDiskIndex, Mode, Ranking

# choose a pre-trained query encoder
encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")

# load an index from disk into memory
index = OnDiskIndex.load(Path("/path/to/index.h5"), encoder, Mode.MAXP)

# load a run (TREC format)
first_stage_ranking = Ranking.from_file(Path("/path/to/sparse/run.tsv"))

# load all required queries
queries = {
    "q1": "query 1",
    "q2": "query 2",
    # ...
    "qn": "query n",
}

# compute the corresponding semantic scores and interpolate
alpha = 0.2
result = index.get_scores(
    first_stage_ranking,
    queries,
    alpha=alpha,
    cutoff=10,
    early_stopping=True,
)

# create a new TREC runfile with the interpolated ranking
result[alpha].save(Path("/path/to/interpolated/run.tsv"))
```

## Documentation

A more detailed documentation is available [here](https://mrjleo.github.io/fast-forward-indexes/docs).

## Examples

- [Creating a Fast-Forward index from a prebuilt Pyserini index](fast_forward/examples/create_index_from_pyserini.py)
- [Computing dense scores for a sparse run and interpolating](fast_forward/examples/interpolate.py)
