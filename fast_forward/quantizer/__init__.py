import abc
import importlib
import logging
from typing import Any, Dict, Tuple, Union

import numpy as np

LOGGER = logging.getLogger(__name__)


QuantizerAttributes = Dict[str, Union[str, bool, int, float]]
QuantizerData = Dict[str, np.ndarray]


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

    @property
    @abc.abstractmethod
    def dim(self) -> int:
        """The dimension of the codes produced by this quantizer.

        Returns:
            np.dtype: The dimension.
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
    def _get_state(self) -> Tuple[QuantizerAttributes, QuantizerData]:
        """Return key-value pairs that represent the state of the quantizer (internal method).

        This method returns a tuple of quantizer attributes (values) and quantizer data (numpy arrays).

        Returns:
            Tuple[QuantizerAttributes, QuantizerData]: Attributes and data of the quantizer.
        """
        pass

    def serialize(
        self,
    ) -> Tuple[QuantizerAttributes, QuantizerAttributes, QuantizerData]:
        """Return a serialized representation of the quantizer that can be stored in the index.

        Returns:
            Tuple[QuantizerAttributes, QuantizerAttributes, QuantizerData]: The serialized quantizer.
        """
        meta = {
            "__module__": self.__class__.__module__,
            "__name__": self.__class__.__name__,
            "_trained": self._trained,
        }
        attributes, data = self._get_state()
        return meta, attributes, data

    @classmethod
    @abc.abstractmethod
    def _from_state(
        cls, attributes: QuantizerAttributes, data: QuantizerData
    ) -> "Quantizer":
        """Instantiate a quantizer based on its state.

        Args:
            attributes (QuantizerAttributes): The quantizer attributes.
            data (QuantizerData): The quantizer attributes.

        Returns:
            Quantizer: The resulting quantizer.
        """
        pass

    @classmethod
    def deserialize(
        cls,
        rep: Tuple[QuantizerAttributes, QuantizerAttributes, QuantizerData],
    ) -> "Quantizer":
        """Reconstruct a serialized quantizer.

        Args:
            rep (Tuple[QuantizerAttributes, QuantizerAttributes, QuantizerData]): The serialized quantizer.

        Returns:
            Quantizer: The loaded quantizer.
        """
        meta, attributes, data = rep
        LOGGER.debug("importing %s.%s", meta["__module__"], meta["__name__"])
        quantizer_mod = importlib.import_module(meta["__module__"])
        quantizer_cls = getattr(quantizer_mod, meta["__name__"])
        quantizer = quantizer_cls._from_state(attributes, data)
        quantizer._trained = meta["_trained"]
        return quantizer