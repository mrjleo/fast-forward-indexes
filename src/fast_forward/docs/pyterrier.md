# PyTerrier transformers

Fast-Forward indexes can seamlessly be integrated into [PyTerrier](https://pyterrier.readthedocs.io/en/latest/) pipelines using the transformers provided in `fast_forward.util.pyterrier`. Specifically, a re-ranking pipeline might look like this, given that `my_index` is a Fast-Forward index of the MS MARCO passage corpus:

```python
bm25 = pt.terrier.Retriever.from_dataset(
    "msmarco_passage", variant="terrier_stemmed", wmodel="BM25"
)

ff_pl = bm25 % 5000 >> FFScore(my_index) >> FFInterpolate(0.2)
```
