"""Make `import wildfire` and the app's `shared` work from a fresh clone, and keep tests offline."""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for path in (ROOT / "src", ROOT / "app"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# The app reads the scheduled forecast from a GitHub release when it can; tests use
# the snapshot committed with the repository instead.
os.environ.setdefault("WILDFIRE_OFFLINE", "1")
