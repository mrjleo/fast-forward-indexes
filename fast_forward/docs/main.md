This is the implementation of [Fast-Forward indexes](https://dl.acm.org/doi/abs/10.1145/3485447.3511955).

**Important**: As this library is still in its early stages, the API is subject to change!

# Features

- Efficient look-up-based computation of semantic ranking scores
- Interpolation of lexical (sparse) and semantic (dense) scores
- Passage- and document-level ranking, including MaxP, FirstP, and AverageP
- _Early stopping_ of interpolation based on approximated scores
- Index compression via _sequential coalescing_

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

# Guides

This section contains guides for the most important features.

## Creating an index

Creating a Fast-Forward index is simple. The following snippet illustrates how to create a `fast_forward.index.disk.OnDiskIndex` object (given a `fast_forward.encoder.QueryEncoder`, `my_query_encoder`) and add some representations to it:

```python
my_index = OnDiskIndex(Path("my_index.h5"), 768, my_query_encoder)
my_index.add(
    my_vectors,  # shape (3, 768)
    doc_ids=["d1", "d1", "d2"],
    psg_ids=["d1_p1", "d1_p2", "d2_p1"]
)
```

Here, `my_vectors` is a Numpy array of shape `(3, dim)`. The first two vectors correspond to two passages of the document `d1`, the third vector corresponds to `d2`, which has only a single passage. It is also possible to provide either only document IDs or only passage IDs.

The index can then be subsequently loaded back using `fast_forward.index.disk.OnDiskIndex.load`.

## Using an index

The usage of a Fast-Forward index (i.e., computing semantic scores and interpolation) is handled by `fast_forward.index.Index.__call__`. It requires a ranking (typically from a sparse retriever) with the corresponding queries:

```python
ranking = Ranking.from_file(Path("/path/to/sparse/run.tsv"), queries)
result = my_index(ranking)
```

Here, `queries` is a simple dictionary mapping query IDs to actual queries to be encoded. The resulting ranking, `result`, has semantic scores for each query-document (or query-passage) pair attached. The individual scores (i.e., retrieval and re-ranking) can be interpolated using `Ranking.interpolate` in order to compute the final scores:

```python
result = result.interpolate(0.1)
```

## Retrieval modes

Each index has a retrieval mode (`fast_forward.index.Mode`). The active mode influences the way scores are computed (`fast_forward.index.Index.__call__`). For example, consider the [example index from earlier](#creating-an-index). Setting the mode to `fast_forward.index.Mode.PASSAGE` will cause the index to compute scores on the passage level (and expect passage IDs in the input ranking):

```python
my_index.mode = Mode.PASSAGE
```

Similarly, the index can return document IDs, where the score of a document computes as

- the highest score of its passages (`fast_forward.index.Mode.MAXP`),
- the score of the its first passage (`fast_forward.index.Mode.FIRSTP`) or
- the average score of all its passages (`fast_forward.index.Mode.AVEP`).

## Custom query encoders

Custom query encoders can easily be implemented. The preferred way to do this is by subclassing `fast_forward.encoder.QueryEncoder` and overriding the `fast_forward.encoder.QueryEncoder.encode` method. This allows the new query encoder to make use of batch encoding.

Alternatively, one can use the `fast_forward.encoder.LambdaQueryEncoder` class, which wraps a function that encodes a single query. The following example shows how to do this with a [Pyserini](https://github.com/castorini/pyserini) query encoder:

```python
pyserini_encoder = pyserini.dsearch.AnceQueryEncoder("castorini/ance-msmarco-passage")
my_encoder = LambdaQueryEncoder(pyserini_encoder.encode)
```

Note that this method is usually less efficient, as the queries are encoded one by one rather than in batches.

## Sequential coalescing

The sequential coalescing algorithm is a compression technique for indexes with multiple representations per document. It is implemented as `fast_forward.index.create_coalesced_index`. Example usage:

```python
my_index = OnDiskIndex.load(Path("/path/to/index.h5"))
coalesced_index = InMemoryIndex(mode=Mode.MAXP)
create_coalesced_index(my_index, coalesced_index, 0.3)
```

# Examples

Example usage scripts can be found in the `examples` module.

## Creating a Fast-Forward index from Pyserini

You can use this script to convert an existing dense [Pyserini](https://github.com/castorini/pyserini) index to the Fast-Forward format.

```bash
$ python -m fast_forward.examples.create_index_from_pyserini -h
```

```
usage: create_index_from_pyserini.py [-h] [--out_file OUT_FILE] INDEX

positional arguments:
  INDEX                Pyserini index name (pre-built) or path.

optional arguments:
  -h, --help           show this help message and exit
  --out_file OUT_FILE  Output file. (default: out)
```

For example, to convert the pre-built index `msmarco-doc-ance-maxp-bf`:

```bash
python -m fast_forward.examples.create_index_from_pyserini \\
    msmarco-doc-ance-maxp-bf \\
    --out_file my_index
```

## Computing and interpolating scores

This script loads an index into memory and performs re-ranking based on a sparse run.
The query encoder is loaded from the [Hugging Face Hub](https://huggingface.co/models).

```bash
$ python -m fast_forward.examples.interpolate -h
```

```
usage: interpolate.py [-h] [--cutoff CUTOFF] [--cutoff_result CUTOFF_RESULT]
                      [--alpha ALPHA [ALPHA ...]] [--early_stopping] [--target TARGET]
                      INDEX {maxp,avep,firstp,passage} ENCODER SPARSE_SCORES
                      QUERY_FILE

positional arguments:
  INDEX                 Fast-Forward index file.
  {maxp,avep,firstp,passage}
                        Retrieval mode.
  ENCODER               Pre-trained transformer encoder.
  SPARSE_SCORES         TREC runfile containing the scores of the sparse retriever.
  QUERY_FILE            Queries (tsv).

optional arguments:
  -h, --help            show this help message and exit
  --cutoff CUTOFF       Maximum number of sparse documents per query. (default: None)
  --cutoff_result CUTOFF_RESULT
                        Maximum number of documents per query in the final ranking.
                        (default: 1000)
  --alpha ALPHA [ALPHA ...]
                        Interpolation weight. (default: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5,
                        0.6, 0.7, 0.8, 0.9, 1.0])
  --early_stopping      Use approximated early stopping. (default: False)
  --target TARGET       Output directory. (default: out)
```

The following example uses the pre-trained `castorini/tct_colbert-msmarco` query encoder.
We re-rank the top-`5000` documents for each query using an interpolation weight of `0.2`.
Finally, the top-`1000` documents are kept.

```bash
python -m fast_forward.examples.interpolate \\
    my_index \\
    maxp \\
    castorini/tct_colbert-msmarco \\
    my_sparse_run.tsv \\
    queries.tsv \\
    --cutoff 5000 \\
    --cutoff_result 1000 \\
    --target results \\
    --alpha 0.2
```
