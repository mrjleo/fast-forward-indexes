import pickle
from typing import Any, Dict, Union

import nanopq
import numpy as np

from fast_forward.quantizer import Quantizer


class NanoPQQuantizer(Quantizer):
    """Quantizer that uses the nanopq library.

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

    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        return self._pq.encode(vecs=vectors)

    def _decode(self, codes: np.ndarray) -> np.ndarray:
        return self._pq.decode(codes=codes)

    def get_state(self) -> Dict[str, Union[bytes, np.ndarray]]:
        return {"obj_pkl": pickle.dumps(self._pq)}

    @classmethod
    def load(cls, state: Dict[str, Union[bytes, np.ndarray]]) -> "NanoPQQuantizer":
        quantizer = cls.__new__(cls)
        super(NanoPQQuantizer, quantizer).__init__()
        quantizer._pq = pickle.loads(state["obj_pkl"])
        return quantizer
