import abc
import importlib
import logging
from collections.abc import Mapping
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


QuantizerAttributes = Mapping[str, str | bool | float]
QuantizerData = Mapping[str, np.ndarray]


class Quantizer(abc.ABC):
    """Base class for quantizers."""

    _attached: bool = False
    _trained: bool = False

    def __eq__(self, o: object) -> bool:
        """Check whether this quantizer is identical to another one.

        :param o: The other quantizer.
        :return: Whether the two quantizers are identical.
        """
        if not isinstance(o, Quantizer):
            return False

        self_meta, self_attributes, self_data = self.serialize()
        o_meta, o_attributes, o_data = o.serialize()

        if self_meta != o_meta or self_attributes != o_attributes:
            return False

        if self_data.keys() != o_data.keys():
            return False

        for k, v in self_data.items():
            if (v != o_data[k]).any():
                return False

        return True

    def set_attached(self) -> None:
        """Set the quantizer as attached, preventing calls to `Quantizer.fit`.

        :raises RuntimeError: When the quantizer has not been fit.
        """
        if not self._trained:
            raise RuntimeError(
                f"Call {self.__class__.__name__}.fit before attaching the quantizer to "
                "an index."
            )
        self._attached = True

    @abc.abstractmethod
    def _fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        pass

    def fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        """Fit (train) the quantizer.

        Quantizers can only be trained before being attached to an index in order to
        avoid inconsistencies.

        :param vectors: The training vectors.
        :param **kwargs: Arguments specific to the quantizer.
        :raises RuntimeError: When the quantizer is aready attached to an index.
        """
        if self._attached:
            raise RuntimeError(
                "Quantizers can only be fitted before they are attached to an index."
            )
        self._fit(vectors, **kwargs)
        self._trained = True

    @abc.abstractmethod
    def _get_dtype(self) -> np.dtype:
        pass

    @property
    def dtype(self) -> np.dtype:
        """The data type of the codes produced by this quantizer.

        :return: The data type of the codes.
        """
        return self._get_dtype()

    @abc.abstractmethod
    def _get_dims(self) -> tuple[int | None, int | None]:
        pass

    @property
    def dims(self) -> tuple[int | None, int | None]:
        """The dimensions before and after quantization.

        May return `None` values before the quantizer is trained.

        :return: Dimension of the original vectors and dimension of the codes.
        """
        return self._get_dims()

    @abc.abstractmethod
    def _encode(self, vectors: np.ndarray) -> np.ndarray:
        pass

    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode a batch of vectors.

        :param vectors: The vectors to be encoded.
        :return: The codes corresponding to the vectors.
        """
        if not self._trained:
            raise RuntimeError(f"Call {self.__class__.__name__}.fit first.")
        return self._encode(vectors)

    @abc.abstractmethod
    def _decode(self, codes: np.ndarray) -> np.ndarray:
        pass

    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode a batch of codes to obtain approximate vector representations.

        :param codes: The codes to be decoded.
        :raises RuntimeError: When the quantizer has not been fit.
        :return: The approximated vectors.
        """
        if not self._trained:
            raise RuntimeError(f"Call {self.__class__.__name__}.fit first.")
        return self._decode(codes)

    @abc.abstractmethod
    def _get_state(self) -> tuple[QuantizerAttributes, QuantizerData]:
        """Return key-value pairs that represent the state of the quantizer.

        This method returns a tuple of quantizer attributes (values) and quantizer data
        (numpy arrays).

        Specific to quantizer implementation.

        :return: Attributes and data of the quantizer.
        """
        pass

    def serialize(
        self,
    ) -> tuple[QuantizerAttributes, QuantizerAttributes, QuantizerData]:
        """Return a serialized representation of the quantizer.

        This representations is used to be stored in the index.

        :return: The serialized quantizer.
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

        :param attributes: The quantizer attributes.
        :param data: The quantizer data.
        :return: The resulting quantizer.
        """
        pass

    @classmethod
    def deserialize(
        cls,
        meta: QuantizerAttributes,
        attributes: QuantizerAttributes,
        data: QuantizerData,
    ) -> "Quantizer":
        """Reconstruct a serialized quantizer.

        :param meta: The quantizer metadata.
        :param attributes: The quantizer attributes.
        :param data: The quantizer data.
        :return: The loaded quantizer.
        """
        LOGGER.debug("reconstructing %s.%s", meta["__module__"], meta["__name__"])
        quantizer_mod = importlib.import_module(str(meta["__module__"]))
        quantizer_cls = getattr(quantizer_mod, str(meta["__name__"]))
        quantizer = quantizer_cls._from_state(attributes, data)
        quantizer._trained = meta["_trained"]
        return quantizer
