"""
# Fast-Forward Indexes
This is the reference implementation of [Fast-Forward indexes](https://arxiv.org/abs/2110.06051).

⚠ **Important**: As this library is still in its early stages, the API is subject to change! ⚠

# Features
* Efficient look-up-based computation of dense retrieval scores.
* Interpolation of sparse and dense scores.
* Passage- and document-level retrieval, including MaxP, FirstP and AverageP.
* *Early stopping* of interpolation based on approximated scores.
* Index compression via *sequential coalescing*.

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

# Getting Started
Using a Fast-Forward index is as simple as providing a TREC run with sparse scores:
```python
from pathlib import Path
from fast_forward.encoder import TCTColBERTQueryEncoder
from fast_forward.index import InMemoryIndex, Mode
from fast_forward.ranking import Ranking

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

# Guides
This section contains guides for the most important features.

## Creating an Index
Creating a Fast-Forward index is simple. The following snippet illustrates how to create a `fast_forward.index.InMemoryIndex` object and add some representations to it:
```python
my_index = InMemoryIndex(my_encoder)
my_index.add(
    my_vectors, # three vectors in total
    doc_ids=["d1", "d1", "d2"],
    psg_ids=["d1_p1", "d1_p2", "d2_p1"]
)
```
Here, `my_vectors` is a Numpy array of shape `(3, dim)`. The first two vectors correspond to two passages of the document `d1`, the third vector corresponds to `d2`, which has only a single passage. It is also possible to provide either only document IDs or only passage IDs.

The index can then be saved to disk using `fast_forward.index.InMemoryIndex.save` and loaded using `fast_forward.index.InMemoryIndex.from_disk`.

## Using an Index
The usage of a Fast-Forward index (i.e. computing dense scores and interpolation) is currently handled by `fast_forward.index.Index.get_scores`. It requires a ranking (typically from a sparse retriever) and the corresponding queries and handles query encoding, scoring, interpolation and so on:
```python
sparse_ranking = Ranking.from_file(Path("/path/to/sparse/run.tsv"))
result = index.get_scores(
    sparse_ranking,
    queries,
    alpha=[0.0, 0.5, 1.0],
    cutoff=1000
)
```
Here, `queries` is a simple dictionary mapping query IDs to actual queries to be encoded. In order to use *early stopping*, use `early_stopping=True`. Note that early stopping is usually only useful for small `cutoff` values.

## Retrieval Modes
Each index has a retrieval mode (`fast_forward.index.Mode`). The active mode influences the way scores are computed (`fast_forward.index.Index.get_scores`). For example, consider the [example index from earlier](#creating-an-index). Setting the mode to `fast_forward.index.Mode.PASSAGE` will cause the index to compute scores on the passage level and return passage IDs:
```python
my_index.mode = Mode.PASSAGE
```
Similarly, the index can return document IDs, where the score of a document computes as
* the highest score of its passages (`fast_forward.index.Mode.MAXP`),
* the score of the its first passage (`fast_forward.index.Mode.FIRSTP`) or
* the average score of all its passages (`fast_forward.index.Mode.AVEP`).

## Custom Query Encoders
Custom query encoders can easily be implemented. The preferred way to do this is by subclassing `fast_forward.encoder.QueryEncoder` and overriding the `fast_forward.encoder.QueryEncoder.encode` method. This allows the new query encoder to make use of batch encoding.

Alternatively, one can use the `fast_forward.encoder.LambdaQueryEncoder` class, which wraps a function that encodes a single query. The following example shows how to do this with a [Pyserini](https://github.com/castorini/pyserini) query encoder:
```python
pyserini_encoder = pyserini.dsearch.AnceQueryEncoder("castorini/ance-msmarco-passage")
my_encoder = LambdaQueryEncoder(pyserini_encoder.encode)
```
Note that this method is usually less efficient, as the queries are encoded one by one rather than in batches.

## Sequential Coalescing
The sequential coalescing algorithm is a compression technique for indexes with multiple representations per document. It is implemented as `fast_forward.index.create_coalesced_index`. Example usage:
```python
my_index = InMemoryIndex.from_disk(Path("/path/to/index"))
coalesced_index = InMemoryIndex(mode=Mode.MAXP)
create_coalesced_index(my_index, coalesced_index, 0.3)
```

# Examples
Example usage scripts can be found in the `examples` module.

## Creating a Fast-Forward Index from Pyserini
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

## Computing and Interpolating Scores
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
"""
