# Custom encoders

Custom encoders can easily be implemented. The preferred way to do this is by subclassing `fast_forward.encoder.base.Encoder` and overriding the `_encode` method. This allows the new encoder to make use of batch encoding.

Alternatively, one can use the `LambdaEncoder` class, which wraps a function that encodes a single piece of text. The following example shows how to do this with a [Pyserini](https://github.com/castorini/pyserini) query encoder:

```python
pyserini_encoder = pyserini.encode.AnceQueryEncoder("castorini/ance-msmarco-passage")
ance_encoder = LambdaEncoder(pyserini_encoder.encode)
```

Note that this method is usually less efficient, as the texts are encoded one by one rather than in batches.
