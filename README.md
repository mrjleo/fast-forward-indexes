# Fast-Forward Indexes

This is the implementation of [Fast-Forward indexes](https://dl.acm.org/doi/abs/10.1145/3485447.3511955).

**Important**: As this library is still in its early stages, the API is subject to change!

## Installation

Install the package via `pip`:

```bash
pip install fast-forward-indexes
```

## Getting Started

Using a Fast-Forward index is as simple as providing a TREC run with sparse scores:

```python
from pathlib import Path
from fast_forward import OnDiskIndex, Mode, Ranking
from fast_forward.encoder import TCTColBERTQueryEncoder

# choose a pre-trained query encoder
encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")

# load an index on disk
ff_index = OnDiskIndex.load(Path("/path/to/index.h5"), encoder, Mode.MAXP)

# load a run (TREC format)
first_stage_ranking = Ranking.from_file(Path("/path/to/input/run.tsv")).cut(5000)

# attach all required queries
first_stage_ranking.attach_queries(
    {
        "q1": "query 1",
        "q2": "query 2",
        # ...
        "qn": "query n",
    }
)

# compute the corresponding semantic scores and interpolate
result = ff_index(first_stage_ranking).interpolate(0.1)

# create a new TREC runfile with the interpolated ranking
result.save(Path("/path/to/output/run.tsv"))
```

## Documentation

A more detailed documentation is available [here](https://mrjleo.github.io/fast-forward-indexes/docs).
