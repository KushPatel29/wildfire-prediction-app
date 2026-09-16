"""
Build the seven-day fire-risk forecast from live public data.

Runs on a schedule in GitHub Actions (`pipelines/forecast.py`) and on demand from
the app. Writes, to `data/live/` by default:

    forecast.parquet   cell x day, lead 0 (latest observed station day) to lead 7:
                       noon weather, FWI System codes, risk of a new fire, risk of a
                       large one, and the cell's normal rate for the month
    hotspots.parquet   satellite hotspots inside Canada over the last two days
    meta.json          when it ran, which station day it starts from, the counts

Outside the fire season (the model is trained on April to October) it writes the
metadata and an empty forecast rather than extrapolating the model to a month it
has never seen.
"""

from __future__ import annotations

import json
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Callable

import pandas as pd

from wildfire import features as F
from wildfire import live
from wildfire.model import FEATURES, MODELS, ROOT, load_all

LIVE = ROOT / "data" / "live"
KEEP = ["cell_id", "date", "lead_days", "province", "lat", "lon", "temp", "rh", "ws", "precip",
        "ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr", "station_km", "clim_month_rate"]


def build(today: date | None = None, out: Path = LIVE, pace: bool = True,
          progress: Callable[[str], None] | None = None) -> dict:
    note = progress or (lambda message: None)
    started = time.time()
    today = today or date.today()
    out.mkdir(parents=True, exist_ok=True)
    cells = pd.read_parquet(MODELS / "cells.parquet")
    meta = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "requested_for": today.isoformat(), "cells": len(cells),
            "sources": {"stations": f"{live.CWFIS}/fwi_obs/current/", "forecast": live.OPEN_METEO,
                        "hotspots": f"{live.CWFIS}/hotspots/"}}

    note("Reading the last two days of satellite hotspots")
    spots = live.hotspots(2, today)
    meta["hotspots"] = len(spots)

    if today.month not in F.SEASON_MONTHS:
        meta.update(in_season=False, rows=0)
        forecast = pd.DataFrame(columns=[*KEEP, "risk", "large_risk"])
    else:
        locations = pd.read_parquet(MODELS / "stations.parquet")
        climatology = pd.read_parquet(MODELS / "climatology.parquet")
        note(f"Reading {live.HISTORY_DAYS} days of CWFIS station observations")
        history, as_of, used = live.observed_history(cells, locations, today)
        note(f"Fetching Open-Meteo's forecast for {len(cells)} cells (about two minutes)")
        weather = live.forecast_weather(cells, as_of, pace=pace)
        note("Stepping the FWI System codes forward and scoring")
        codes = live.carry_codes_forward(history, weather, as_of)
        rows = live.feature_rows(cells, climatology, history, codes, FEATURES)
        rows = rows[rows["date"].dt.month.isin(F.SEASON_MONTHS)]
        models = load_all()
        forecast = rows[KEEP].copy()
        forecast["risk"] = models["has_fire"].predict(rows)
        forecast["large_risk"] = models["has_large_fire"].predict(rows)
        meta.update(in_season=True, rows=len(forecast), as_of=as_of.isoformat(), stations_used=used,
                    first_day=forecast["date"].min().date().isoformat(),
                    last_day=forecast["date"].max().date().isoformat())

    # Write the data before the metadata that describes it, so a reader never finds
    # a meta.json pointing at a forecast that is not there yet.
    spots.to_parquet(out / "hotspots.parquet", index=False)
    forecast.to_parquet(out / "forecast.parquet", index=False)
    meta["seconds"] = round(time.time() - started, 1)
    (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta
