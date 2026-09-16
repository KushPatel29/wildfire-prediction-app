"""
Build the modelling table: one row per grid cell per fire-season day.

    python pipelines/build_table.py

Reads the raw NRCan downloads in data/raw, writes data/processed/:

    cells.parquet            the grid: cells with >= 10 fire starts in the training years
    fires.parquet            usable fire starts, 2000 onward
    cell_days.parquet        features + targets, April-October, every year on disk

Station decades are processed one at a time so the whole 7-million-row station
record never has to sit in memory at once.
"""

from __future__ import annotations

import argparse
import sys
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildfire import features as F  # noqa: E402

RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
PROCESSED = ROOT / "data" / "processed"
TRAIN_YEARS = (2000, 2016)
FIRST_YEAR = 2000
STATION_COLUMNS = ["rep_date", "aes", "name", "lat", "lon", "temp", "rh", "ws", "precip",
                   "ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr", "calcstatus"]


def load_nfdb() -> pd.DataFrame:
    cached = INTERIM / "nfdb_points.parquet"
    if cached.exists():
        return pd.read_parquet(cached)
    with zipfile.ZipFile(RAW / "NFDB_point_txt.zip") as archive:
        name = next(n for n in archive.namelist() if n.endswith(".txt"))
        with archive.open(name) as handle:
            frame = pd.read_csv(handle, encoding="latin-1", low_memory=False)
    frame.columns = [c.lstrip("﻿").lstrip("ï»¿") for c in frame.columns]
    return frame


def station_decades() -> list[tuple[str, callable]]:
    """(label, loader) for every decade of station data on disk, oldest first."""
    decades = []
    for label in ("2000s", "2010s"):
        parquet = INTERIM / f"cwfis_fwi_{label}.parquet"
        if parquet.exists():
            decades.append((label, lambda p=parquet: pd.read_parquet(p)))
    raw_2020s = RAW / "cwfis_fwi2020s.csv"
    parquet_2020s = INTERIM / "cwfis_fwi_2020s.parquet"
    if parquet_2020s.exists():
        decades.append(("2020s", lambda: pd.read_parquet(parquet_2020s)))
    elif raw_2020s.exists():
        decades.append(("2020s", lambda: _read_2020s(raw_2020s, parquet_2020s)))
    return decades


def station_locations() -> pd.DataFrame:
    """aes -> name, lat, lon. The 2020s file carries only the station's AES id, so
    its locations come from the CWFIS station lists, backed up by the coordinates
    the older decades carried inline for the same stations."""
    frames = []
    for listing in sorted(RAW.glob("cwfis_allstn*.csv"), reverse=True):
        stations = pd.read_csv(listing, encoding="latin-1", low_memory=False)
        stations.columns = [c.strip().lower() for c in stations.columns]
        frames.append(stations[["aes", "name", "lat", "lon"]])
    for label in ("2010s", "2000s"):
        parquet = INTERIM / f"cwfis_fwi_{label}.parquet"
        if parquet.exists():
            older = pd.read_parquet(parquet, columns=["aes", "name", "lat", "lon"])
            frames.append(older.drop_duplicates("aes", keep="last"))
    locations = pd.concat(frames, ignore_index=True)
    locations["aes"] = locations["aes"].astype(str).str.strip()
    locations = locations.dropna(subset=["lat", "lon"]).drop_duplicates("aes", keep="first")
    return locations


def _read_2020s(path: Path, cache: Path) -> pd.DataFrame:
    """The 2020s file is published uncompressed, without station names or
    coordinates; attach them by AES id and cache the result."""
    locations = station_locations()
    frames = []
    for chunk in pd.read_csv(path, chunksize=500_000, low_memory=False):
        chunk.columns = [c.strip().lower() for c in chunk.columns]
        chunk["aes"] = chunk["aes"].astype(str).str.strip()
        chunk = chunk.drop(columns=[c for c in ("name", "lat", "lon") if c in chunk.columns])
        chunk = chunk.merge(locations, on="aes", how="left")
        for column in [c for c in STATION_COLUMNS if c not in chunk.columns]:
            chunk[column] = pd.NA
        chunk = chunk[STATION_COLUMNS]
        chunk["rep_date"] = pd.to_datetime(chunk["rep_date"], errors="coerce").dt.normalize()
        frames.append(chunk)
    frame = pd.concat(frames, ignore_index=True)
    located = frame["lat"].notna().mean()
    print(f"  2020s: {len(frame):,} rows, {located:.1%} located via station lists")
    frame.to_parquet(cache, index=False)
    return frame


#: Everything `derive` computes. Stripped before a re-derive so a renamed or
#: dropped feature cannot survive in the table as a stale column.
INTERPOLATED = ["cell_id", "date", "station_km", *F.WEATHER, *F.CODES,
                "fires", "lightning_fires", "human_fires", "large_fires",
                "has_fire", "has_large_fire"]


def derive(table: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Everything the model reads that is not a station reading: rolling windows,
    what the neighbours saw, what is normal here, and the cell's own history."""
    table = F.add_recent_weather(table)
    table = F.add_neighbour_weather(table)
    table = table.merge(cells[["cell_id", "lat", "lon", "province", "lightning_share"]], on="cell_id", how="left")
    table = table.merge(F.cell_climatology(table, TRAIN_YEARS), on=["cell_id", "month"], how="left")
    table = F.add_anomalies(table)
    floats = table.select_dtypes("float64").columns
    table[floats] = table[floats].astype("float32")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the modelling table.")
    parser.add_argument(
        "--features-only", action="store_true",
        help="re-derive the features from the interpolated columns already in "
             "cell_days.parquet, without re-reading the station archives")
    args = parser.parse_args(argv)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    started = time.time()

    if args.features_only:
        cells = pd.read_parquet(PROCESSED / "cells.parquet")
        table = pd.read_parquet(PROCESSED / "cell_days.parquet", columns=INTERPOLATED)
        print(f"re-deriving features for {len(table):,} interpolated cell-days")
    else:
        fires = F.clean_fires(load_nfdb())
        fires = fires[fires["date"].dt.year >= FIRST_YEAR].reset_index(drop=True)
        cells = F.build_cells(fires, TRAIN_YEARS)
        fires.to_parquet(PROCESSED / "fires.parquet", index=False)
        cells.to_parquet(PROCESSED / "cells.parquet", index=False)
        print(f"fires {len(fires):,}; cells {len(cells)} (>= 10 starts in {TRAIN_YEARS[0]}-{TRAIN_YEARS[1]})")

        frames = []
        for label, load in station_decades():
            t0 = time.time()
            stations = F.clean_stations(load())
            daily = F.interpolate(cells, stations)
            frames.append(daily)
            print(f"  {label}: {stations['rep_date'].dt.year.min()}-{stations['rep_date'].dt.year.max()}, "
                  f"{len(daily):,} cell-days, {time.time() - t0:.0f}s")
        table = pd.concat(frames, ignore_index=True)

        counts = F.targets(fires, cells)
        table = table.merge(counts, on=["cell_id", "date"], how="left")
        for column in ("fires", "lightning_fires", "human_fires", "large_fires"):
            table[column] = table[column].fillna(0).astype("int16")
        table["has_fire"] = (table["fires"] > 0).astype("int8")
        table["has_large_fire"] = (table["large_fires"] > 0).astype("int8")

    table = derive(table, cells)
    table.to_parquet(PROCESSED / "cell_days.parquet", index=False)
    years = table["date"].dt.year
    print(f"cell_days {len(table):,} rows, {years.min()}-{years.max()}, "
          f"fire-day rate {table['has_fire'].mean():.2%}, station within 200 km {table['station_km'].notna().mean():.1%}, "
          f"{time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
