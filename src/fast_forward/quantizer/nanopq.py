from typing import Any

import nanopq
import numpy as np

from fast_forward.quantizer.base import Quantizer, QuantizerAttributes, QuantizerData


class NanoPQ(Quantizer):
    """Product quantizer that uses the nanopq library.

    More information is available
    [here](https://nanopq.readthedocs.io/en/stable/source/api.html#nanopq.PQ).
    """

    def __init__(
        self, M: int, Ks: int, metric: str = "dot", verbose: bool = False
    ) -> None:
        """Instantiate a nanopq product quantizer.

        :param M: The number of subspaces.
        :param Ks: The number of codewords per subspace.
        :param metric: The metric to use.
        :param verbose: Enable verbosity.
        """
        self._pq = nanopq.PQ(M=M, Ks=Ks, metric=metric, verbose=verbose)
        super().__init__()

    def _fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        self._pq.fit(vecs=vectors, **kwargs)

    def _get_dtype(self) -> np.dtype:
        return np.dtype(self._pq.code_dtype)

    def _get_dims(self) -> tuple[int | None, int | None]:
        if self._pq.Ds is None:
            return None, self._pq.M
        return self._pq.Ds * self._pq.M, self._pq.M

    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        return self._pq.encode(vecs=vectors)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return self._pq.decode(codes=codes)

    def _get_state(self) -> tuple[QuantizerAttributes, QuantizerData]:
        attributes, data = {}, {}

        for a in ("M", "Ks", "Ds", "metric", "verbose"):
            attributes[a] = getattr(self._pq, a)

        if (
            hasattr(self._pq, "codewords")
            and getattr(self._pq, "codewords") is not None
        ):
            data["codewords"] = self._pq.codewords

        return attributes, data

    @classmethod
    def _from_state(
        cls, attributes: QuantizerAttributes, data: QuantizerData
    ) -> "NanoPQ":
        quantizer = cls(
            M=int(attributes["M"]),
            Ks=int(attributes["Ks"]),
            metric=str(attributes["metric"]),
            verbose=bool(attributes["verbose"]),
        )
        if attributes.get("Ds") is not None:
            quantizer._pq.Ds = int(attributes["Ds"])
        if "codewords" in data:
            quantizer._pq.codewords = data["codewords"]
        return quantizer


class NanoOPQ(Quantizer):
    """Optimized product quantizer that uses the nanopq library.

    More information is available
    [here](https://nanopq.readthedocs.io/en/stable/source/api.html#nanopq.OPQ).
    """

    def __init__(
        self, M: int, Ks: int, metric: str = "dot", verbose: bool = False
    ) -> None:
        """Instantiate a nanopq optimized product quantizer.

        :param M: The number of subspaces.
        :param Ks: The number of codewords per subspace.
        :param metric: The metric to use.
        :param verbose: Enable verbosity.
        """
        self._opq = nanopq.OPQ(M=M, Ks=Ks, metric=metric, verbose=verbose)
        super().__init__()

    def _fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        self._opq.fit(vecs=vectors, **kwargs)

    def _get_dtype(self) -> np.dtype:
        return np.dtype(self._opq.code_dtype)

    def _get_dims(self) -> tuple[int | None, int | None]:
        if self._opq.pq.Ds is None:
            return None, self._opq.pq.M
        return self._opq.pq.Ds * self._opq.pq.M, self._opq.pq.M

    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        return self._opq.encode(vecs=vectors)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return self._opq.decode(codes=codes)

    def _get_state(self) -> tuple[QuantizerAttributes, QuantizerData]:
        attributes, data = {}, {}

        for a in ("M", "Ks", "Ds", "metric"):
            attributes[a] = getattr(self._opq.pq, a)

        attributes["verbose"] = self._opq.verbose

        if (
            hasattr(self._opq.pq, "codewords")
            and getattr(self._opq.pq, "codewords") is not None
        ):
            data["codewords"] = self._opq.pq.codewords

        if hasattr(self._opq, "R") and getattr(self._opq, "R") is not None:
            data["R"] = self._opq.R

        return attributes, data

    @classmethod
    def _from_state(
        cls, attributes: QuantizerAttributes, data: QuantizerData
    ) -> "NanoOPQ":
        quantizer = cls(
            M=int(attributes["M"]),
            Ks=int(attributes["Ks"]),
            metric=str(attributes["metric"]),
            verbose=bool(attributes["verbose"]),
        )
        if attributes.get("Ds") is not None:
            quantizer._opq.pq.Ds = int(attributes["Ds"])
        if "codewords" in data:
            quantizer._opq.pq.codewords = data["codewords"]
        if "R" in data:
            quantizer._opq.R = data["R"]
        return quantizer
