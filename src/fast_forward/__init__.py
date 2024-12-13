""".. include:: docs/main.md
.. include:: docs/ranking.md
"""  # noqa: D205, D400, D415

import importlib.metadata

from fast_forward import encoder, index, quantizer, util
from fast_forward.ranking import Ranking

__all__ = ["encoder", "index", "quantizer", "util", "Ranking"]
__version__ = importlib.metadata.version("fast-forward-indexes")
