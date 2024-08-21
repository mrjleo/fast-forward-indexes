from typing import Any, Tuple

import nanopq
import numpy as np

from fast_forward.quantizer import Quantizer, QuantizerAttributes, QuantizerData


class NanoPQ(Quantizer):
    """Product quantizer that uses the nanopq library.

    More information is available [here](https://nanopq.readthedocs.io/en/stable/source/api.html#nanopq.PQ).
    """

    def __init__(
        self, M: int, Ks: int, metric: str = "dot", verbose: bool = False
    ) -> None:
        """Instantiate a nanopq quantizer.

        Args:
            M (int): The number of subspaces.
            Ks (int): The number of codewords per subspace.
            metric (str, optional): The metric to use. Defaults to "dot".
            verbose (bool, optional): Enable verbosity. Defaults to False.
        """
        self._pq = nanopq.PQ(M=M, Ks=Ks, metric=metric, verbose=verbose)
        super().__init__()

    def _fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        self._pq.fit(vecs=vectors, **kwargs)

    @property
    def dtype(self) -> np.dtype:
        return self._pq.code_dtype

    @property
    def dim(self) -> int:
        return self._pq.M

    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        return self._pq.encode(vecs=vectors)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return self._pq.decode(codes=codes)

    def _get_state(self) -> Tuple[QuantizerAttributes, QuantizerData]:
        attributes, data = {}, {}
        for a in ("M", "Ks", "Ds", "metric", "verbose"):
            if hasattr(self._pq, a):
                attributes[a] = getattr(self._pq, a)
        if hasattr(self._pq, "codewords"):
            data["codewords"] = self._pq.codewords
        return attributes, data

    @classmethod
    def _from_state(
        cls, attributes: QuantizerAttributes, data: QuantizerData
    ) -> "NanoPQ":
        quantizer = cls.__new__(cls)
        super(NanoPQ, quantizer).__init__()
        quantizer._pq = nanopq.PQ(
            M=attributes["M"],
            Ks=attributes["Ks"],
            metric=attributes["metric"],
            verbose=attributes["verbose"],
        )
        quantizer._pq.Ds = attributes["Ds"]
        quantizer._pq.codewords = data["codewords"]
        return quantizer
