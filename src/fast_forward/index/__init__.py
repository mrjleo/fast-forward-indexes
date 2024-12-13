""".. include:: ../docs/index.md"""  # noqa: D400, D415

from fast_forward.index.base import Index, Mode
from fast_forward.index.disk import OnDiskIndex
from fast_forward.index.memory import InMemoryIndex

__all__ = ["Index", "Mode", "OnDiskIndex", "InMemoryIndex"]
