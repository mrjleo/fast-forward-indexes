import abc
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    import numpy as np


class Encoder(abc.ABC):
    """Base class for encoders."""

    @abc.abstractmethod
    def _encode(self, texts: "Sequence[str]") -> "np.ndarray":
        pass

    def __call__(self, texts: "Sequence[str]") -> "np.ndarray":
        """Encode a list of texts.

        :param texts: The texts to encode.
        :return: The resulting vector representations.
        """
        return self._encode(texts)
