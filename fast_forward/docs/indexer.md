The `fast_forward.indexer.Indexer` class is a utility for indexing a collection. If the size of the collection is known in advace, it can be specified when the index is created in order to avoid subsequent resizing operations:

```python
my_index = OnDiskIndex(Path("my_index.h5"), init_size=1000000)
```

For indexing, a document/passage encoder is required, for example:

```python
doc_encoder = TCTColBERTDocumentEncoder(
    "castorini/tct_colbert-msmarco",
    device="cuda:0" if torch.cuda.is_available() else "cpu",
)
```

The indexer can be created as follows:

```python
indexer = Indexer(my_index, doc_encoder, batch_size=8)
```

`fast_forward.indexer.Indexer.index_dicts` consumes an iterator that yields dictionaries:

```python
def docs_iter():
    for doc in my_corpus:
        yield {"doc_id": doc.get_doc_id(), "text": doc.get_text()}

indexer.index_dicts(docs_iter())
```
