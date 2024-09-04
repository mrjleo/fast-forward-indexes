# Sequential coalescing

The sequential coalescing algorithm is a compression technique for indexes with multiple representations per document. More information can be found [in the paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511955). It is implemented as `fast_forward.util.create_coalesced_index`. Example usage:

```python
my_index = OnDiskIndex.load(Path("/path/to/index.h5"))
coalesced_index = InMemoryIndex(mode=Mode.MAXP)
create_coalesced_index(my_index, coalesced_index, 0.3)
```

# PyTerrier transformers

Fast-Forward indexes can seamlessly be integrated into [PyTerrier](https://pyterrier.readthedocs.io/en/latest/) pipelines using the transformers provided in `fast_forward.util.pyterrier`. Specifically, a re-ranking pipeline might look like this, given that `my_index` is a Fast-Forward index of the MS MARCO passage corpus:

```python
bm25 = pt.BatchRetrieve.from_dataset(
    "msmarco_passage", variant="terrier_stemmed", wmodel="BM25"
)

ff_pl = bm25 % 5000 >> FFScore(my_index) >> FFInterpolate(0.2)
```
