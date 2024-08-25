Indexes are the core of the Fast-Forward library. In the following, you'll find some snippets how to create and use indexes.

# Index types

Currently, two types of indexes are available:

- `fast_forward.index.memory.InMemoryIndex`: Everything is held in memory entirely.
- `fast_forward.index.disk.OnDiskIndex`: Vectors are stored on disk and accessed on demand.

OnDiskIndexes can be loaded into memory using `fast_forward.index.disk.OnDiskIndex.to_memory`.

# Creating an index

The following snippet illustrates how to create a `fast_forward.index.disk.OnDiskIndex` object (given a `fast_forward.encoder.Encoder`, `my_query_encoder`) and add some vector representations to it:

```python
my_index = OnDiskIndex(Path("my_index.h5"), my_query_encoder)
my_index.add(
    my_vectors,  # shape (3, 768)
    doc_ids=["d1", "d1", "d2"],
    psg_ids=["d1_p1", "d1_p2", "d2_p1"]
)
```

Here, `my_vectors` is a Numpy array of shape `(3, 768)`, `768` being the dimensionality of the vector representations. The first two vectors correspond to two passages of the document `d1`, the third vector corresponds to `d2`, which has only a single passage. It is also possible to provide either only document IDs or only passage IDs.

The index can then be subsequently loaded back using `fast_forward.index.disk.OnDiskIndex.load`.

# Using an index

Index can be used to compute semantic re-ranking scores by calling them directly. It requires a `fast_forward.ranking.Ranking` (typically, this comes from a sparse retriever) with the corresponding queries:

```python
ranking = Ranking.from_file(Path("/path/to/sparse/run.tsv"), queries)
result = my_index(ranking)
```

Here, `queries` is a simple dictionary mapping query IDs to actual queries to be encoded. The resulting ranking, `result`, has the semantic scores for the query-document (or query-passage) pairs. Afterwards, retrieval and re-ranking scores may be interpolated (see `fast_forward.ranking`).

## Ranking mode

Each index has a ranking mode (`fast_forward.index.Mode`). The active mode determines the way scores are computed. For example, consider the [example index from earlier](#creating-an-index). Setting the mode to `fast_forward.index.Mode.PASSAGE` instructs the index to compute scores on the passage level (and expect passage IDs in the input ranking):

```python
my_index.mode = Mode.PASSAGE
```

Similarly, the index can return document IDs, where the score of a document computes as

- the highest score of its passages (`fast_forward.index.Mode.MAXP`),
- the score of the its first passage (`fast_forward.index.Mode.FIRSTP`) or
- the average score of all its passages (`fast_forward.index.Mode.AVEP`).

## Early stopping

Early stopping is a technique to limit the number of index look-ups. This can be beneficial for OnDiskIndexes, especially when the disk is slow. For early stopping, a relatively small cut-off depth (e.g., `10`) is required, and it is mostly helpful when a large number of candidates are to be re-ranked. More information can be found [in the paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511955).

The following snippet demonstrates early stopping with

- a cut-off depth of `5`,
- interpolation parameter `0.2`,
- intervals `(0, 400)` and `(401, 5000)`.

```python
result = my_index(
    ranking,
    early_stopping=5,
    early_stopping_alpha=0.2,
    early_stopping_intervals=(400, 5000),
)
```

Specifically, in the first step, the top-`400` documents/passages for each query are scored. Afterwards, the top-`5000` documents/passages are scored only for those queries that do not meet the early stopping criterion yet.
