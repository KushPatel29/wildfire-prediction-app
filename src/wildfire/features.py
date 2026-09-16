"""
From national fire records and fire-weather stations to one row per grid cell per day.

The unit of prediction is a 1-degree cell on one day of the fire season: will at
least one new fire be reported there? A cell is about 110 km north-south and 70 km
east-west at Canadian latitudes - coarse enough that the weather stations can
describe it, fine enough that a fire agency can put crews and aircraft against it.

Three sources meet here:

* the Canadian National Fire Database (NFDB) gives every recorded fire's location,
  report date, cause and final size - the targets;
* CWFIS station observations give noon weather and the official FWI System codes
  - the features, carried from the stations to each cell by inverse-distance
  weighting over the nearest stations that reported that day;
* a cell's own fire history gives its baseline - computed from training years only,
  so a test season never leaks into the rate it is compared with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

EARTH_KM = 6371.0
CELL_DEG = 1.0
SEASON_MONTHS = (4, 5, 6, 7, 8, 9, 10)          # 98% of fires start April to October
WEATHER = ["temp", "rh", "ws", "precip"]
CODES = ["ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr"]
LARGE_FIRE_HA = 200.0                             # 5.6% of fires, 99% of the area burned

# NFDB agency codes -> province or territory. Parks Canada reports fires inside the
# national parks of every province, so its fires are assigned by location instead.
AGENCIES = {
    "BC": "British Columbia", "AB": "Alberta", "SK": "Saskatchewan", "MB": "Manitoba",
    "ON": "Ontario", "QC": "Quebec", "NB": "New Brunswick", "NS": "Nova Scotia",
    "PE": "Prince Edward Island", "NL": "Newfoundland and Labrador", "YT": "Yukon",
    "NT": "Northwest Territories", "NU": "Nunavut",
}


# --------------------------------------------------------------------------
# Fires and cells
# --------------------------------------------------------------------------

def clean_fires(nfdb: pd.DataFrame) -> pd.DataFrame:
    """Usable fire starts: a valid report date and coordinates inside Canada."""
    fires = nfdb.copy()
    fires["date"] = pd.to_datetime(
        dict(year=fires["YEAR"], month=fires["MONTH"], day=fires["DAY"]), errors="coerce")
    keep = (fires["date"].notna()
            & fires["LATITUDE"].between(41.0, 84.0)
            & fires["LONGITUDE"].between(-142.0, -52.0)
            & (fires["PRESCRIBED"].fillna("").str.upper().isin(["", "0", "NO", "N"])))
    fires = fires.loc[keep, ["NFDBFIREID", "SRC_AGENCY", "LATITUDE", "LONGITUDE", "date",
                             "SIZE_HA", "CAUSE"]].rename(columns=str.lower)
    fires["cell_id"] = cell_ids(fires["latitude"], fires["longitude"])
    return fires.reset_index(drop=True)


def cell_ids(lat, lon) -> pd.Series:
    lat0 = np.floor(np.asarray(lat, dtype=float) / CELL_DEG) * CELL_DEG
    lon0 = np.floor(np.asarray(lon, dtype=float) / CELL_DEG) * CELL_DEG
    return pd.Series([f"{a:.0f}_{b:.0f}" for a, b in zip(lat0, lon0)])


def build_cells(fires: pd.DataFrame, years: tuple[int, int], min_fires: int = 10) -> pd.DataFrame:
    """Cells that saw at least `min_fires` starts in the given years.

    A cell with a handful of fires in twenty years is mostly tundra, prairie crop
    or ocean edge; keeping it adds hundreds of thousands of all-negative rows and
    teaches the model nothing a zero-risk default would not."""
    window = fires[fires["date"].dt.year.between(*years)]
    counts = window.groupby("cell_id").size()
    kept = counts[counts >= min_fires].index
    cells = window[window["cell_id"].isin(kept)].groupby("cell_id").agg(
        fires=("nfdbfireid", "size"),
        lightning_share=("cause", lambda c: float((c == "N").mean())),
        agency=("src_agency", lambda a: _majority_agency(a)),
    ).reset_index()
    parts = cells["cell_id"].str.split("_", expand=True).astype(float)
    cells["lat"] = parts[0] + CELL_DEG / 2
    cells["lon"] = parts[1] + CELL_DEG / 2
    cells["province"] = cells["agency"].map(AGENCIES).fillna("Other")
    return cells.drop(columns="agency")


def _majority_agency(agencies: pd.Series) -> str:
    provincial = agencies[agencies != "PC"]
    return (provincial if len(provincial) else agencies).mode().iloc[0]


def targets(fires: pd.DataFrame, cells: pd.DataFrame) -> pd.DataFrame:
    """Fire starts per cell per day: any cause, lightning, human, and large."""
    f = fires[fires["cell_id"].isin(cells["cell_id"])]
    return f.groupby(["cell_id", "date"]).agg(
        fires=("nfdbfireid", "size"),
        lightning_fires=("cause", lambda c: int((c == "N").sum())),
        human_fires=("cause", lambda c: int((c == "H").sum())),
        large_fires=("size_ha", lambda s: int((s >= LARGE_FIRE_HA).sum())),
    ).reset_index()


# --------------------------------------------------------------------------
# Stations to cells
# --------------------------------------------------------------------------

def clean_stations(obs: pd.DataFrame) -> pd.DataFrame:
    """One row per station per day, with weather, and codes only where CWFIS
    calculated them from that station's own observations (calcstatus 1)."""
    frame = obs.dropna(subset=["lat", "lon", "rep_date"]).copy()
    frame = frame[frame["rep_date"].dt.month.isin(SEASON_MONTHS)]
    # The decade archives arrive typed; CWFIS's daily current-conditions file does
    # not, and a blank or flagged reading turns a whole column to text. Coerce, so a
    # live day and a training day are cleaned identically.
    for column in ["lat", "lon", *WEATHER, *CODES, "calcstatus"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    invalid = frame["calcstatus"] != 1
    frame.loc[invalid, CODES] = np.nan
    for column, low, high in (("temp", -40, 50), ("rh", 0, 100), ("ws", 0, 150), ("precip", 0, 300)):
        frame.loc[~frame[column].between(low, high), column] = np.nan
    frame = frame.sort_values(["rep_date", "aes"]).drop_duplicates(["rep_date", "aes"])
    return frame[["rep_date", "aes", "lat", "lon", *WEATHER, *CODES]].reset_index(drop=True)


def interpolate_day(cells: pd.DataFrame, stations_today: pd.DataFrame,
                    k: int = 4, radius_km: float = 200.0, power: float = 2.0) -> pd.DataFrame:
    """Inverse-distance weighted station values at every cell centre for one day.

    Each variable is weighted over the stations that actually have it, so a
    station reporting weather but no codes still informs temperature. Beyond
    `radius_km` a station does not count; a cell with none in range gets NaN,
    which the model reads as "unobserved" rather than as zero."""
    out = pd.DataFrame({"cell_id": cells["cell_id"].to_numpy()})
    columns = [*WEATHER, *CODES]
    if stations_today.empty:
        for column in columns:
            out[column] = np.nan
        out["station_km"] = np.nan
        return out
    tree = BallTree(np.radians(stations_today[["lat", "lon"]].to_numpy()), metric="haversine")
    query = np.radians(cells[["lat", "lon"]].to_numpy())
    kk = min(k, len(stations_today))
    dist, idx = tree.query(query, k=kk)
    km = dist * EARTH_KM
    in_range = km <= radius_km
    weights = np.where(in_range, 1.0 / np.maximum(km, 1.0) ** power, 0.0)
    out["station_km"] = np.where(in_range[:, 0], km[:, 0], np.nan)
    values = stations_today[columns].to_numpy(dtype=float)
    for j, column in enumerate(columns):
        v = values[idx, j]
        w = np.where(np.isnan(v), 0.0, weights)
        total = w.sum(axis=1)
        out[column] = np.where(total > 0, np.nansum(np.where(np.isnan(v), 0.0, v) * w, axis=1) / np.where(total > 0, total, 1.0), np.nan)
    return out


def interpolate(cells: pd.DataFrame, stations: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """interpolate_day for every day in `stations`."""
    frames = []
    for day, group in stations.groupby("rep_date", sort=True):
        frame = interpolate_day(cells, group, **kwargs)
        frame.insert(1, "date", day)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


# --------------------------------------------------------------------------
# Per-cell history
# --------------------------------------------------------------------------

def add_recent_weather(table: pd.DataFrame) -> pd.DataFrame:
    """Rolling context a single day cannot carry: a week of drying, the last rain."""
    table = table.sort_values(["cell_id", "date"]).reset_index(drop=True)
    grouped = table.groupby("cell_id", sort=False)
    for column, window, how in (("fwi", 3, "mean"), ("fwi", 7, "mean"), ("fwi", 7, "max"),
                                 ("isi", 3, "mean"), ("temp", 3, "mean"), ("rh", 3, "mean"),
                                 ("precip", 3, "sum"), ("precip", 7, "sum"), ("precip", 14, "sum")):
        rolled = grouped[column].rolling(window, min_periods=1)
        table[f"{column}_{how}_{window}d"] = getattr(rolled, how)().reset_index(level=0, drop=True)
    wet = (table["precip"].fillna(0.0) >= 2.0).astype(int)
    # A run restarts at every change between wet and dry and at the start of each
    # cell's series, so one cell's dry spell never carries into the next cell's.
    new_run = (wet != wet.shift()) | (table["cell_id"] != table["cell_id"].shift())
    run = wet.groupby(new_run.cumsum()).cumcount() + 1
    table["days_since_rain"] = np.where(wet == 1, 0, run).clip(0, 60)
    table["dc_change_7d"] = table["dc"] - grouped["dc"].shift(7)
    doy = table["date"].dt.dayofyear
    table["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    table["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    table["month"] = table["date"].dt.month
    return table


def cell_climatology(table: pd.DataFrame, train_years: tuple[int, int]) -> pd.DataFrame:
    """Each cell's historical probability of a fire start, by month, from training
    years only - the "normal for this place and time of year" the model improves on."""
    train = table[table["date"].dt.year.between(*train_years)]
    by_month = train.groupby(["cell_id", "month"])["has_fire"].mean().rename("clim_month_rate")
    overall = train.groupby("cell_id")["has_fire"].mean().rename("clim_cell_rate")
    return by_month.reset_index().merge(overall.reset_index(), on="cell_id", how="left")
