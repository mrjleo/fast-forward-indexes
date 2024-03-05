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

The sequential coalescing algorithm is a compression technique for indexes with multiple representations per document. It is implemented as `fast_forward.util.create_coalesced_index`. Example usage:

```python
my_index = OnDiskIndex.load(Path("/path/to/index.h5"))
coalesced_index = InMemoryIndex(768, mode=Mode.MAXP)
create_coalesced_index(my_index, coalesced_index, 0.3)
```
