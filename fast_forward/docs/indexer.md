The `fast_forward.indexer.Indexer` class is a utility for indexing collections or adding pre-computed vectors to an index.

If the size of the collection is known in advance, it can be specified when the index is created in order to avoid subsequent resizing operations:

```python
my_index = OnDiskIndex(Path("my_index.h5"), init_size=1000000)
```

In order to index a corpus, a document/passage encoder is required, for example:

```python
doc_encoder = TCTColBERTDocumentEncoder(
    "castorini/tct_colbert-msmarco",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
```

The indexer can be created as follows:

```python
indexer = Indexer(my_index, doc_encoder, encoder_batch_size=8)
```

`fast_forward.indexer.Indexer.from_dicts` consumes an iterator that yields dictionaries:

```python
def docs_iter():
    for doc in my_corpus:
        yield {"doc_id": doc.get_doc_id(), "text": doc.get_text()}

indexer.from_dicts(docs_iter())
```

Additionally, indexers can be used to automatically fit and attach a quantizer during indexing. In this example, a quantized version (`target_index`) of an existing index (`source_index`) is created:

```python
from fast_forward.quantizer.nanopq import NanoPQ

Indexer(
    target_index,
    quantizer=NanoPQ(8, 256),
    batch_size=2**16,
    quantizer_fit_batches=2,
).from_index(source_index)
```

Here, the first two batches (of size $2^{16}$) are buffered and used to fit the quantizer.
