"""Put the repository root on ``sys.path``.

The project is a flat set of top-level modules (``detector.py``, ``audio_detect.py``,
...) rather than an installed package, so ``pytest`` needs the root prepended for
``import detector`` to resolve. Without this, bare ``pytest`` fails to collect
while ``python3 -m pytest`` works, which is a confusing difference to hit.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
