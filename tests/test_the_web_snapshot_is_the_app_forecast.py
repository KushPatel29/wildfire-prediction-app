"""
The static forecast page must show the forecast the app shows.

It reads `forecast.json`, a compact copy of `forecast.parquet` that the forecast
workflow publishes to the `live-data` branch. A copy is a second place for the
numbers to drift: a reordered cell list, a day dropped at a month boundary, or a
scale applied twice would all draw a plausible map of the wrong forecast.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "pipelines"))

import web_snapshot  # noqa: E402

LIVE = ROOT / "data" / "live"


def test_the_web_snapshot_carries_the_same_forecast():
    snap = web_snapshot.snapshot(LIVE)
    forecast = pd.read_parquet(LIVE / "forecast.parquet")
    forecast["date"] = pd.to_datetime(forecast["date"])

    cells = [cell[0] for cell in snap["cells"]]
    assert len(cells) == len(set(cells)) == forecast["cell_id"].nunique()
    assert snap["days"] == sorted(forecast["date"].dt.strftime("%Y-%m-%d").unique())
    assert snap["lead_days"] == list(range(len(snap["days"])))

    # Every value round-trips to within half a stored unit, cell by cell and day
    # by day - which fails on a reordered cell list, not just a wrong total.
    wide = forecast.pivot(index="cell_id", columns="date", values="risk").reindex(cells)
    stored = np.asarray(snap["risk"], dtype=float) / snap["scale"]
    assert np.abs(stored - wide.to_numpy()).max() <= 0.5 / snap["scale"] + 1e-12
    large = forecast.pivot(index="cell_id", columns="date", values="large_risk").reindex(cells)
    assert np.abs(np.asarray(snap["large"], dtype=float) / snap["large_scale"]
                  - large.to_numpy()).max() <= 0.5 / snap["large_scale"] + 1e-12

    # The page's headline: cells expected to report a fire is the sum of the
    # day's probabilities, the same figure the app's first card shows.
    day0 = forecast[forecast["lead_days"] == 0]
    assert abs(stored[:, 0].sum() - day0["risk"].sum()) <= len(cells) * 0.5 / snap["scale"]

    for cell, row in zip(snap["cells"], forecast.drop_duplicates("cell_id").set_index("cell_id").loc[cells].itertuples()):
        assert abs(cell[1] - row.lat) < 0.01 and abs(cell[2] - row.lon) < 0.01

    assert len(json.dumps(snap, separators=(",", ":"))) < 150_000, "the page loads this on every visit"
