# FastForward Indexes
This is the reference implementation of [FastForward indexes](https://arxiv.org/abs/2110.06051).

⚠ **Important**: As this library is still in its early stages, the API is subject to change! ⚠

## Installation
Clone this repository and run:
```bash
python -m pip install .
```

## Getting Started
Using a FastForward index is as simple as providing a TREC run with sparse scores:
```python
from pathlib import Path
from fastforward.encoder import TCTColBERTQueryEncoder
from fastforward.index import InMemoryIndex, Mode
from fastforward.ranking import Ranking

# choose a pre-trained query encoder
encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")

# load an index from disk into memory
index = InMemoryIndex.from_disk(Path("/path/to/index"), encoder, Mode.MAXP)

# load a sparse run (TREC format)
sparse_ranking = Ranking.from_file(Path("/path/to/sparse/run.tsv"))

# load all required queries
queries = {
    "q1": "query 1",
    "q2": "query 2",
    # ...
    "qn": "query n"
}

# compute the corresponding dense scores and interpolate
alpha = 0.2
result = index.get_scores(
    sparse_ranking,
    queries,
    alpha=alpha,
    cutoff=10,
    early_stopping=True
)

# create a new TREC runfile with the interpolated ranking
result[alpha].save(Path("/path/to/interpolated/run.tsv"))
```

## Examples
* [Creating a FastForward index from a prebuilt Pyserini index](src/fastforward/examples/create_index_from_pyserini.py)
* [Computing dense scores for a sparse run and interpolating](src/fastforward/examples/interpolate.py)
