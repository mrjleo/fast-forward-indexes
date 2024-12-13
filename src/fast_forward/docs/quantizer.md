# Quantization

Fast-Forward indexes support (product) quantization as a means of compressing an index. The `Quantizer` class defines the interface for quantizers to implement. Currently, the following quantizers are available:

- `NanoPQ`: Product quantization based on the [nanopq library](https://nanopq.readthedocs.io/en/stable/index.html).
- `NanoOPQ`: Optimized product quantization based on the [nanopq library](https://nanopq.readthedocs.io/en/stable/index.html).

Quantizers must be trained **before** they are used with the corresponding Fast-Forward index. The typical workflow is as follows:

```python
from pathlib import Path

import numpy as np

from fast_forward.index import OnDiskIndex
from fast_forward.quantizer import NanoPQ

# in practice, a subset of the encoded corpus should be used as training vectors
training_vectors = np.random.normal(size=(2**10, 768)).astype(np.float32)

quantizer = NanoPQ(M=8, Ks=256)
quantizer.fit(training_vectors)

index = OnDiskIndex(Path("my_index.h5"), quantizer=quantizer)
```
