# Sequential coalescing

The sequential coalescing algorithm is a compression technique for indexes with multiple representations per document. More information can be found [in the paper](https://dl.acm.org/doi/abs/10.1145/3485447.3511955). It is implemented as `fast_forward.util.create_coalesced_index`. Example usage:

```python
my_index = OnDiskIndex.load(Path("/path/to/index.h5"))
coalesced_index = InMemoryIndex(mode=Mode.MAXP)
create_coalesced_index(my_index, coalesced_index, 0.3)
```
