""".. include:: docs/main.md"""  # noqa: D400, D415

import importlib.metadata

from fast_forward.index import Mode as Mode
from fast_forward.index.disk import OnDiskIndex as OnDiskIndex
from fast_forward.index.memory import InMemoryIndex as InMemoryIndex
from fast_forward.indexer import Indexer as Indexer
from fast_forward.quantizer.nanopq import NanoOPQ as NanoOPQ
from fast_forward.quantizer.nanopq import NanoPQ as NanoPQ
from fast_forward.ranking import Ranking as Ranking

__version__ = importlib.metadata.version("fast-forward-indexes")
