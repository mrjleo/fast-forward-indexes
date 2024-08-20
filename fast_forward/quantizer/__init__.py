import abc
import logging
from typing import Any, Dict, Union

import numpy as np

LOGGER = logging.getLogger(__name__)


class Quantizer(abc.ABC):
    """Base class for quantizers."""

    _attached: bool = False
    _trained: bool = False

    def set_attached(self) -> None:
        """Set the quantizer as attached, preventing calls to `Quantizer.fit`."""
        if not self._trained:
            raise RuntimeError(
                f"Call {self.__class__.__name__}.fit before attaching the quantizer to an index."
            )
        self._attached = True

    @abc.abstractmethod
    def _fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        """Fit the quantizer (internal method).

        Args:
            vectors (np.ndarray): The training vectors.
            **kwargs (Any): Arguments specific to the quantizer.
        """
        pass

    def fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        """Fit (train) the quantizer.

        Quantizers can only be trained before being attached to an index to avoid inconsistencies.

        Args:
            vectors (np.ndarray): The training vectors.
            **kwargs (Any): Arguments specific to the quantizer.

        Raises:
            RuntimeError: When the quantizer is aready attached to an index.
        """
        if self._attached:
            raise RuntimeError(
                "Quantizers can only be fitted before they are attached to an index."
            )
        self._fit(vectors, **kwargs)
        self._trained = True

    @property
    @abc.abstractmethod
    def dtype(self) -> np.dtype:
        """The data type of the codes produced by this quantizer.

        Returns:
            np.dtype: The data type.
        """
        pass

    @abc.abstractmethod
    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode vectors (internal method).

        Args:
            vectors (np.ndarray): The vectors to be encoded.

        Returns:
            np.ndarray: The codes corresponding to the vectors.
        """
        pass

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode a batch of vectors.

        Args:
            vectors (np.ndarray): The vectors to be encoded.

        Returns:
            np.ndarray: The codes corresponding to the vectors.
        """
        if not self._trained:
            raise RuntimeError(f"Call {self.__class__.__name__}.fit first.")
        return self._encode(vectors)

    @abc.abstractmethod
    def _decode(self, codes: np.ndarray) -> np.ndarray:
        """Reconstruct vectors (internal method).

        Args:
            codes (np.ndarray): The codes to be decoded.

        Returns:
            np.ndarray: The reconstructed vectors.
        """
        pass

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode a batch of codes to obtain approximate vector representations.

        Args:
            codes (np.ndarray): The codes to be decoded.

        Returns:
            np.ndarray: The approximated vectors.
        """
        if not self._trained:
            raise RuntimeError(f"Call {self.__class__.__name__}.fit first.")
        return self._decode(codes)

    @abc.abstractmethod
    def get_state(self) -> Dict[str, Union[str, np.ndarray]]:
        """Return a serialized representation of the quantizer that can be stored in the index.

        Returns:
            Dict[str, Union[str, np.ndarray]]: Key-value pairs representing the state of the quantizer.
        """
        pass

    @classmethod
    @abc.abstractmethod
    def load(state: Dict[str, Union[str, np.ndarray]]) -> "Quantizer":
        """Load a (trained) quantizer based on its serialized representation.

        Returns:
            Quantizer: The loaded qunatizer.
        """
        pass
