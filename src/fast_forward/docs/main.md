This is the implementation of [Fast-Forward indexes](https://dl.acm.org/doi/abs/10.1145/3485447.3511955).

**Important**: As this library is still in its early stages, the API is subject to change!

# Features

- Efficient look-up-based computation of semantic ranking scores
- Interpolation of lexical (retrieval) and semantic (re-ranking) scores
- Passage- and document-level ranking, including MaxP, FirstP, and AverageP
- Early stopping for limiting index look-ups
- Index compression via sequential coalescing

# Installation

Install the package via `pip`:

```bash
pip install fast-forward-indexes
```

Alternatively, the package can be installed from source:

```bash
git clone https://github.com/mrjleo/fast-forward-indexes.git
cd fast-forward-indexes
python -m pip install .
```

After installing the package, simply import the library:

```python
import fast_forward
```

# Getting started

Using a Fast-Forward index is as simple as providing a TREC run with sparse scores:

```python
from pathlib import Path
from fast_forward import OnDiskIndex, Mode, Ranking
from fast_forward.encoder import TCTColBERTQueryEncoder

# choose a pre-trained query encoder
encoder = TCTColBERTQueryEncoder("castorini/tct_colbert-msmarco")

# load an index on disk
ff_index = OnDiskIndex.load(Path("/path/to/index.h5"), encoder, mode=Mode.MAXP)

# load a run (TREC format) and attach all required queries
first_stage_ranking = (
    Ranking.from_file(Path("/path/to/input/run.tsv"))
    .attach_queries(
        {
            "q1": "query 1",
            "q2": "query 2",
            # ...
            "qn": "query n",
        }
    )
    .cut(5000)
)

# compute the corresponding semantic scores
out = ff_index(first_stage_ranking)

# interpolate scores and create a new TREC runfile
first_stage_ranking.interpolate(out, 0.1).save(Path("/path/to/output/run.tsv"))
```

## How to...

- [create and use Fast-Forward indexes?](fast_forward/index.html)
- [index a collection?](fast_forward/indexer.html)
- [use quantization to reduce index size?](fast_forward/quantizer.html)
- [create custom encoders?](fast_forward/encoder.html#custom-encoders)
- [read, manipulate, and save rankings?](fast_forward/ranking.html)
- [use Fast-Forward indexes with PyTerrier?](fast_forward/util.html#pyterrier-transformers)
