""".. include:: ../docs/encoder.md"""  # noqa: D400, D415

from typing import TYPE_CHECKING

import numpy as np

from fast_forward.encoder.base import Encoder
from fast_forward.encoder.transformer import (
    BGEEncoder,
    ContrieverEncoder,
    TASBEncoder,
    TCTColBERTDocumentEncoder,
    TCTColBERTQueryEncoder,
    TransformerEncoder,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "Encoder",
    "LambdaEncoder",
    "TransformerEncoder",
    "TCTColBERTQueryEncoder",
    "TCTColBERTDocumentEncoder",
    "TASBEncoder",
    "ContrieverEncoder",
    "BGEEncoder",
]


class LambdaEncoder(Encoder):
    """Encoder adapter class for arbitrary encoding functions."""

    def __init__(self, f: "Callable[[str], np.ndarray]") -> None:
        """Create a lambda encoder.

        :param f: Function to encode a single piece of text.
        """
        super().__init__()
        self._f = f

    def _encode(self, texts: "Sequence[str]") -> np.ndarray:
        return np.array(list(map(self._f, texts)))
