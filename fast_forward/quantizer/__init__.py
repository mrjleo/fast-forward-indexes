import abc
import logging
from typing import Any, Dict, Optional, Union

import numpy as np

LOGGER = logging.getLogger(__name__)


class Quantizer(abc.ABC):
    _attached: bool = False

    @abc.abstractmethod
    def _fit(self, vectors: np.ndarray, **kwargs: Any) -> None:
        """Fit the quantizer.

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

    @property
    @abc.abstractmethod
    def dtype(self) -> Optional[np.dtype]:
        """The data type of the codes produced by this quantizer or None if the quantizer is not trained.

        Returns:
            Optional[np.dtype]: The data type (if any).
        """
        pass

    @abc.abstractmethod
    def encode(self, vectors: np.ndarray) -> np.ndarray:
        """Encode a batch of vectors.

        Args:
            vectors (np.ndarray): The vectors to be encoded.

        Returns:
            np.ndarray: The codes corresponding to the vectors.
        """
        pass

    @abc.abstractmethod
    def decode(self, codes: np.ndarray) -> np.ndarray:
        """Decode a batch of codes to obtain approximate vector representations.

        Args:
            codes (np.ndarray): The codes to be decoded.

        Returns:
            np.ndarray: The approximated vectors.
        """
        pass

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
