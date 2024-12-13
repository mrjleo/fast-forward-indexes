""".. include:: ../docs/quantizer.md"""  # noqa: D400, D415

from fast_forward.quantizer.base import Quantizer
from fast_forward.quantizer.nanopq import NanoOPQ, NanoPQ

__all__ = ["Quantizer", "NanoPQ", "NanoOPQ"]
