"""Write the compact forecast the static web page reads.

    python pipelines/web_snapshot.py build/live build/web/forecast.json

The Streamlit app reads `forecast.parquet` from the `live-forecast` release. A
static page in a browser cannot: parquet needs a decoder, and GitHub serves
release files without a CORS header, so another site's page is not allowed to
read them. This writes the same forecast as one small JSON file - every cell's
calibrated risk for each day, its normal rate for the month, the day's hotspots
and when and from what the forecast was built - and the forecast workflow
publishes it to the `live-data` branch, which raw.githubusercontent.com serves to
any origin.

Probabilities are stored as integers in units of 1/SCALE (0.01 percentage
points), which keeps the grid's ~750 cells x 8 days under 100 KB and loses nothing the page
shows.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCALE = 10_000
LARGE_SCALE = 1_000_000

PROVINCE_SHORT = {
    "British Columbia": "BC", "Alberta": "AB", "Saskatchewan": "SK", "Manitoba": "MB", "Ontario": "ON",
    "Quebec": "QC", "New Brunswick": "NB", "Nova Scotia": "NS", "Prince Edward Island": "PE",
    "Newfoundland and Labrador": "NL", "Yukon": "YT", "Northwest Territories": "NT", "Nunavut": "NU",
}


def _grid(frame: pd.DataFrame, cells: pd.Index, days: list[pd.Timestamp], column: str, scale: int) -> list:
    wide = frame.pivot(index="cell_id", columns="date", values=column).reindex(index=cells, columns=days)
    values = np.rint(wide.to_numpy(dtype=float) * scale)
    # -1 marks a cell-day with no value; the page draws it as "no data".
    return np.where(np.isnan(values), -1, values).astype(int).tolist()


def snapshot(live: Path) -> dict:
    forecast = pd.read_parquet(live / "forecast.parquet")
    forecast["date"] = pd.to_datetime(forecast["date"])
    meta = json.loads((live / "meta.json").read_text(encoding="utf-8"))
    hotspots_file = live / "hotspots.parquet"
    hotspots = pd.read_parquet(hotspots_file) if hotspots_file.exists() else pd.DataFrame(columns=["lat", "lon"])

    days = sorted(forecast["date"].unique())
    first = forecast.drop_duplicates("cell_id").sort_values(["lat", "lon"]).set_index("cell_id")
    cells = first.index
    return {
        "version": 1,
        "scale": SCALE,
        "large_scale": LARGE_SCALE,
        "generated_at": meta.get("generated_at"),
        "as_of": meta.get("as_of"),
        "stations_used": meta.get("stations_used"),
        "days": [pd.Timestamp(d).strftime("%Y-%m-%d") for d in days],
        "lead_days": [int(forecast.loc[forecast["date"] == d, "lead_days"].iloc[0]) for d in days],
        "cells": [[cell, round(float(row.lat), 2), round(float(row.lon), 2),
                   PROVINCE_SHORT.get(row.province, row.province)]
                  for cell, row in first.iterrows()],
        "normal": np.rint(first["clim_month_rate"].to_numpy(dtype=float) * SCALE).astype(int).tolist(),
        "risk": _grid(forecast, cells, days, "risk", SCALE),
        "large": _grid(forecast, cells, days, "large_risk", LARGE_SCALE),
        "hotspots": [[round(float(lat), 3), round(float(lon), 3)]
                     for lat, lon in hotspots[["lat", "lon"]].itertuples(index=False)],
    }


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__.strip().splitlines()[2], file=sys.stderr)
        return 2
    live, out = Path(argv[0]), Path(argv[1])
    data = snapshot(live)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {out}: {len(data['cells'])} cells x {len(data['days'])} days, "
          f"{out.stat().st_size / 1024:.0f} KB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
