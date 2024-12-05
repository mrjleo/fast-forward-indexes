"""
.. include:: docs/main.md
"""

import importlib.metadata

from fast_forward.index import Mode
from fast_forward.index.disk import OnDiskIndex
from fast_forward.index.memory import InMemoryIndex
from fast_forward.indexer import Indexer
from fast_forward.ranking import Ranking

__version__ = importlib.metadata.version("fast-forward-indexes")
