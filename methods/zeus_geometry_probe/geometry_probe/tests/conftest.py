"""Make ZEUS and the local probe importable when pytest runs from repository root."""

import sys
from pathlib import Path

ZEUS_ROOT = Path(__file__).resolve().parents[2]
if str(ZEUS_ROOT) not in sys.path:
    sys.path.insert(0, str(ZEUS_ROOT))
