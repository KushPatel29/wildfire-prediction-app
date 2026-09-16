"""
Score a fire season no part of the model has seen against what satellites saw burning.

    python pipelines/season_check.py [--year 2026]

The National Fire Database, which the backtest is scored against, is published a
season or more after the fact. For the current season the independent record is
CWFIS's satellite hotspot archive. This scores every grid cell on every day of the
season from that day's observed station weather - the inputs the live forecast uses
at lead 0 - and asks how well the risk ranked the cells where satellites detected
**new fire activity**: hotspots in a cell that had none in the preceding 14 days.

That label is not the NFDB's. It misses fires too small or short-lived for a
satellite pass, and it counts a large fire spreading into a neighbouring cell as new
activity there. So the comparison that matters is relative: the same label scores
both models, the cell's normal rate for the month, and the FWI on its own.

Downloads (cached under data/raw/, skipped when already present):

    fwi_obs/current/cwfis_fwi_YYYYMMDD.csv   one per day from April 1
    hotspots/YYYYMMDD.csv                    one per day from 14 days before April 1

Writes data/published/season_<year>_daily.parquet and season_<year>.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildfire import features as F  # noqa: E402
from wildfire import live  # noqa: E402
from wildfire.evaluation import SEED, TOP_SHARE, rank_within, ranking  # noqa: E402
from wildfire.model import MODELS, load_all  # noqa: E402

RAW = ROOT / "data" / "raw"
PUBLISHED = ROOT / "data" / "published"
LOOKBACK_DAYS = 14


def fetch(url: str, path: Path) -> Path | None:
    if path.exists() and path.stat().st_size > 0:
        return path
    try:
        response = live._get(url)
    except requests.RequestException:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    return path


def download(year: int, last_day: date) -> tuple[dict[date, Path | None], dict[date, Path | None]]:
    season = [d.date() for d in pd.date_range(date(year, 4, 1), last_day)]
    spot_days = [d.date() for d in pd.date_range(date(year, 4, 1) - timedelta(days=LOOKBACK_DAYS), last_day)]
    station_jobs = [(f"{live.CWFIS}/fwi_obs/current/cwfis_fwi_{d:%Y%m%d}.csv",
                     RAW / "cwfis_current" / f"cwfis_fwi_{d:%Y%m%d}.csv") for d in season]
    spot_jobs = [(f"{live.CWFIS}/hotspots/{d:%Y%m%d}.csv", RAW / "hotspots" / f"{d:%Y%m%d}.csv") for d in spot_days]
    with ThreadPoolExecutor(max_workers=4) as pool:
        stations = dict(zip(season, pool.map(lambda job: fetch(*job), station_jobs)))
        spots = dict(zip(spot_days, pool.map(lambda job: fetch(*job), spot_jobs)))
    return stations, spots


def new_detections(spot_files: dict[date, Path | None], cells: pd.DataFrame, first_day: date) -> pd.DataFrame:
    """Cell-days with hotspots where the previous LOOKBACK_DAYS had none."""
    present = sorted(day for day, path in spot_files.items() if path is not None)
    frames = []
    for day in present:
        frame = pd.read_csv(spot_files[day])
        frame.columns = [c.strip().lower() for c in frame.columns]
        frame = frame.loc[live.in_canada(frame), ["lat", "lon"]]
        frame["date"] = pd.Timestamp(day)
        frames.append(frame)
    spots = pd.concat(frames, ignore_index=True)
    spots["cell_id"] = F.cell_ids(spots["lat"], spots["lon"]).to_numpy()
    counts = spots[spots["cell_id"].isin(cells["cell_id"])].groupby(["cell_id", "date"]).size()
    grid = pd.MultiIndex.from_product([cells["cell_id"], pd.to_datetime(present)], names=["cell_id", "date"])
    table = counts.rename("hotspots").reindex(grid, fill_value=0).reset_index().sort_values(["cell_id", "date"])
    prior = (table.groupby("cell_id")["hotspots"]
             .transform(lambda s: s.shift(1).rolling(LOOKBACK_DAYS, min_periods=1).sum()).fillna(0))
    table["new_detection"] = ((table["hotspots"] > 0) & (prior == 0)).astype("int8")
    return table[table["date"] >= pd.Timestamp(first_day)].reset_index(drop=True)


def cell_days(station_files: dict[date, Path | None], cells: pd.DataFrame, climatology: pd.DataFrame,
              locations: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for day, path in sorted(station_files.items()):
        if path is None:
            continue
        raw = live.parse_station_csv(path.read_text(encoding="utf-8", errors="replace"), day, locations)
        cleaned = F.clean_stations(raw)
        if cleaned.empty:
            continue
        frame = F.interpolate_day(cells, cleaned)
        frame.insert(1, "date", pd.Timestamp(day))
        frames.append(frame)
    table = F.add_recent_weather(pd.concat(frames, ignore_index=True))
    table = table.merge(cells[["cell_id", "lat", "lon", "province", "lightning_share"]], on="cell_id", how="left")
    return table.merge(climatology, on=["cell_id", "month"], how="left")


def same_day_capture(frame: pd.DataFrame, rank_column: str) -> float:
    """Of all new detections, the share inside exactly that day's riskiest tenth of cells."""
    cut = np.ceil(TOP_SHARE * frame.groupby("date")["date"].transform("size"))
    return float(frame.loc[frame[rank_column] <= cut, "new_detection"].sum() / max(frame["new_detection"].sum(), 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score a season against satellite hotspots.")
    parser.add_argument("--year", type=int, default=date.today().year)
    args = parser.parse_args(argv)
    last_day = min(date(args.year, 10, 31), date.today() - timedelta(days=1))

    cells = pd.read_parquet(MODELS / "cells.parquet")
    climatology = pd.read_parquet(MODELS / "climatology.parquet")
    locations = pd.read_parquet(MODELS / "stations.parquet")
    stations, spots = download(args.year, last_day)
    print(f"station days {sum(p is not None for p in stations.values())}/{len(stations)}, "
          f"hotspot days {sum(p is not None for p in spots.values())}/{len(spots)}", flush=True)

    table = cell_days(stations, cells, climatology, locations)
    models = load_all()
    for target, column in (("has_fire", "risk"), ("has_large_fire", "large_risk")):
        raw = models[target].raw(table)
        table[column] = models[target].calibration(raw).astype("float32")
        table[f"{column}_raw"] = raw
    labels = new_detections(spots, cells, date(args.year, 4, 1))
    scored = table.merge(labels, on=["cell_id", "date"], how="inner").reset_index(drop=True)
    scored["fwi_score"] = scored["fwi"].fillna(scored["fwi"].median())
    scored["clim_score"] = scored["clim_month_rate"].fillna(0.0)
    scored["random_tiebreak"] = np.random.default_rng(SEED).random(len(scored))
    scored["rank_in_day"] = rank_within(scored, "date", "risk", "risk_raw")
    scored["clim_rank_in_day"] = rank_within(scored, "date", "clim_score", "random_tiebreak")
    scored["fwi_rank_in_day"] = rank_within(scored, "date", "fwi_score", "random_tiebreak")
    scored["large_rank_in_day"] = rank_within(scored, "date", "large_risk", "large_risk_raw")

    y = scored["new_detection"].to_numpy()
    # Satellites see fires big and hot enough to detect from orbit, so the large-fire
    # model - trained on the fires that grow past 200 ha - is scored as well.
    candidates = {"model": ("risk", "risk_raw", "rank_in_day"),
                  "large_fire_model": ("large_risk", "large_risk_raw", "large_rank_in_day"),
                  "climatology": ("clim_score", None, "clim_rank_in_day"),
                  "fwi_alone": ("fwi_score", None, "fwi_rank_in_day")}
    by_month = []
    for month, part in scored.groupby(scored["date"].dt.month):
        if part["new_detection"].nunique() < 2:
            continue
        entry = {"month": int(month), "cell_days": len(part), "new_detections": int(part["new_detection"].sum())}
        for name, (column, tiebreak, _) in candidates.items():
            entry[f"{name}_roc_auc"] = ranking(part["new_detection"].to_numpy(), part[column].to_numpy())["roc_auc"]
        by_month.append(entry)
    report = {
        "year": args.year, "first_day": scored["date"].min().date().isoformat(),
        "last_day": scored["date"].max().date().isoformat(),
        "station_days": int(sum(p is not None for p in stations.values())),
        "hotspot_days": int(sum(p is not None for p in spots.values())),
        "hotspot_days_missing": int(sum(p is None for p in spots.values())),
        "cell_days": int(len(scored)), "new_detections": int(y.sum()), "positive_rate": float(y.mean()),
        "lookback_days": LOOKBACK_DAYS,
        "pooled": {name: ranking(y, scored[column].to_numpy(),
                                 tiebreak=None if tiebreak is None else scored[tiebreak].to_numpy())
                   for name, (column, tiebreak, _) in candidates.items()},
        "same_day_top_decile": {name: same_day_capture(scored, rank) for name, (_, _, rank) in candidates.items()},
        "by_month": by_month,
    }
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    daily = scored[["cell_id", "date", "province", "lat", "lon", "fwi", "risk", "large_risk", "clim_month_rate",
                    "rank_in_day", "hotspots", "new_detection"]].copy()
    daily["in_top_decile"] = daily["rank_in_day"] <= np.ceil(TOP_SHARE * daily.groupby("date")["date"].transform("size"))
    for column in ("lat", "lon", "fwi", "clim_month_rate"):
        daily[column] = daily[column].astype("float32")
    daily = daily.astype({"rank_in_day": "int16", "hotspots": "int32"})
    daily.to_parquet(PUBLISHED / f"season_{args.year}_daily.parquet", index=False)
    (PUBLISHED / f"season_{args.year}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pooled = report["pooled"]
    print(f"{args.year}: {report['cell_days']:,} cell-days to {report['last_day']}, {report['new_detections']:,} new "
          f"detections; AUC model {pooled['model']['roc_auc']:.3f} | climatology {pooled['climatology']['roc_auc']:.3f} | "
          f"FWI alone {pooled['fwi_alone']['roc_auc']:.3f}; same-day top decile model "
          f"{report['same_day_top_decile']['model']:.1%} | climatology {report['same_day_top_decile']['climatology']:.1%} "
          f"| FWI alone {report['same_day_top_decile']['fwi_alone']:.1%} | large-fire model "
          f"{report['same_day_top_decile']['large_fire_model']:.1%} (AUC {pooled['large_fire_model']['roc_auc']:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
