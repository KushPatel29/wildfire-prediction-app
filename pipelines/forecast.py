"""
Build the seven-day fire-risk forecast from live public data.

    python pipelines/forecast.py [--out data/live]

The work is in `wildfire.forecast`, shared with the app's "rebuild now" button.
GitHub Actions runs this twice a day and attaches the output to the
`live-forecast` release, which the hosted app reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildfire.forecast import LIVE, build  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the seven-day fire-risk forecast.")
    parser.add_argument("--out", type=Path, default=LIVE, help="output directory (default data/live)")
    args = parser.parse_args(argv)
    meta = build(out=args.out, progress=lambda message: print(message, flush=True))
    print(json.dumps(meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
