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
                             "SIZE_HA", "CAUSE"]].rename(columns=str.lower).reset_index(drop=True)
    fires["cell_id"] = cell_ids(fires["latitude"], fires["longitude"])
    return fires


def cell_ids(lat, lon) -> pd.Series:
    """The 1° cell of each point, indexed like `lat` when it is a Series.

    It used to return a fresh 0..n index. clean_fires assigned it to a filtered
    frame that still carried the NFDB's own row labels, pandas aligned the two
    by label, and from the first dropped row onward every fire took the cell of
    a fire further down the file - 97.7% of them, which put Nova Scotia's
    fires in the Yukon."""
    lat0 = np.floor(np.asarray(lat, dtype=float) / CELL_DEG) * CELL_DEG
    lon0 = np.floor(np.asarray(lon, dtype=float) / CELL_DEG) * CELL_DEG
    index = lat.index if isinstance(lat, pd.Series) else None
    return pd.Series([f"{a:.0f}_{b:.0f}" for a, b in zip(lat0, lon0)], index=index)


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
    # A cell only Parks Canada reports from is inside a national park: give it the
    # province of the nearest cell a provincial agency reports from.
    parks = cells["province"] == "Other"
    if parks.any() and (~parks).any():
        known = cells[~parks]
        scale = np.cos(np.radians(cells["lat"].mean()))
        for i in cells.index[parks]:
            d2 = (known["lat"] - cells.at[i, "lat"]) ** 2 + ((known["lon"] - cells.at[i, "lon"]) * scale) ** 2
            cells.at[i, "province"] = known.at[d2.idxmin(), "province"]
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

def vapour_pressure_deficit(temp, rh) -> np.ndarray:
    """Drying power of the air, in kPa (Tetens over water).

    The FWI System reads temperature and humidity separately; VPD is the quantity
    the two combine into physically - how hard the air pulls moisture out of fuel.
    30 C at 30% RH and 20 C at 10% RH are both "dry" on a humidity reading and are
    not remotely the same fire day: 3.0 kPa against 2.1.
    """
    temp = np.asarray(temp, dtype=float)
    rh = np.asarray(rh, dtype=float)
    saturated = 0.6108 * np.exp(17.27 * temp / (temp + 237.3))
    return np.clip(saturated * (1.0 - rh / 100.0), 0.0, None)


def add_recent_weather(table: pd.DataFrame) -> pd.DataFrame:
    """Rolling context a single day cannot carry: a week of drying, the last rain."""
    table = table.sort_values(["cell_id", "date"]).reset_index(drop=True)
    table["vpd"] = vapour_pressure_deficit(table["temp"], table["rh"])
    grouped = table.groupby("cell_id", sort=False)
    for column, window, how in (("fwi", 3, "mean"), ("fwi", 7, "mean"), ("fwi", 7, "max"),
                                 ("isi", 3, "mean"), ("temp", 3, "mean"), ("rh", 3, "mean"),
                                 ("precip", 3, "sum"), ("precip", 7, "sum"), ("precip", 14, "sum"),
                                 # Drought carries further back than a week: DC moves over a
                                 # month, and a month of rain is what ends a drought code's run.
                                 ("vpd", 3, "mean"), ("vpd", 7, "max"), ("bui", 7, "mean"),
                                 ("dc", 30, "mean"), ("precip", 30, "sum")):
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
    # Half of Canada's fire starts are people, and people keep a week: Saturday and
    # Sunday carry measurably more human ignitions than a Wednesday does.
    table["is_weekend"] = (table["date"].dt.dayofweek >= 5).astype("int8")
    return table


# --------------------------------------------------------------------------
# What the neighbours are doing
# --------------------------------------------------------------------------

NEIGHBOUR_OFFSETS = [(dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0)]


def adjacency(cell_ids: pd.Series | list[str]) -> np.ndarray:
    """C x C matrix, 1 where two cells share an edge or a corner.

    Cell ids are the floored degree coordinates, so the eight neighbours of a cell
    are its id with one degree added or removed on either axis."""
    ids = list(dict.fromkeys(cell_ids))
    index = {cell: i for i, cell in enumerate(ids)}
    matrix = np.zeros((len(ids), len(ids)), dtype="float32")
    for cell, i in index.items():
        lat, lon = (int(part) for part in cell.split("_"))
        for dy, dx in NEIGHBOUR_OFFSETS:
            j = index.get(f"{lat + dy}_{lon + dx}")
            if j is not None:
                matrix[i, j] = 1.0
    return matrix


def add_neighbour_weather(table: pd.DataFrame, columns=("fwi", "dc")) -> pd.DataFrame:
    """Each cell's neighbours' weather the same day, averaged over those that reported.

    Fire weather is regional and lightning outbreaks are regional; a cell whose own
    stations were silent, or which sits at the edge of a drying airmass, is described
    by its surroundings better than by its own interpolated reading. Computed as a
    matrix product over the day-by-cell grid rather than a self-join, which on four
    million rows would be an eight-fold row explosion.
    """
    ids = sorted(table["cell_id"].unique())
    neighbours = adjacency(ids)
    for column in columns:
        grid = table.pivot_table(index="date", columns="cell_id", values=column, aggfunc="mean")
        grid = grid.reindex(columns=ids)
        values = grid.to_numpy(dtype="float32")
        present = (~np.isnan(values)).astype("float32")
        total = np.nan_to_num(values) @ neighbours
        count = present @ neighbours
        mean = np.divide(total, count, out=np.full_like(total, np.nan), where=count > 0)
        flat = pd.DataFrame(mean, index=grid.index, columns=ids).stack(future_stack=True)
        flat.index.names = ["date", "cell_id"]
        table = table.merge(flat.rename(f"{column}_neighbour").reset_index(),
                            on=["date", "cell_id"], how="left")
    return table


# --------------------------------------------------------------------------
# What is normal here
# --------------------------------------------------------------------------

NORMAL_COLUMNS = ["fwi", "temp", "dc"]


def cell_climatology(table: pd.DataFrame, train_years: tuple[int, int]) -> pd.DataFrame:
    """What is normal for this cell in this month, from training years only.

    Two kinds of normal, and the model uses both. `clim_month_rate` is the cell's
    historical probability of a fire start - the baseline any useful model has to
    beat. The weather normals are what the anomaly features are measured against:
    an FWI of 20 is an ordinary July in the southern interior and the driest week
    of the decade on the Labrador coast, and a model reading absolute FWI across a
    continent cannot tell those apart.
    """
    train = table[table["date"].dt.year.between(*train_years)]
    by_month = train.groupby(["cell_id", "month"])["has_fire"].mean().rename("clim_month_rate")
    overall = train.groupby("cell_id")["has_fire"].mean().rename("clim_cell_rate")
    normals = (train.groupby(["cell_id", "month"])[NORMAL_COLUMNS].mean()
               .rename(columns={c: f"clim_{c}" for c in NORMAL_COLUMNS}))
    return (by_month.reset_index()
            .merge(overall.reset_index(), on="cell_id", how="left")
            .merge(normals.reset_index(), on=["cell_id", "month"], how="left"))


def add_anomalies(table: pd.DataFrame) -> pd.DataFrame:
    """Today against this cell's own normal for the month.

    Runs after the climatology merge, in the table build and in the live forecast
    alike, so the two produce the same columns from the same normals."""
    for column in NORMAL_COLUMNS:
        table[f"{column}_anom"] = table[column] - table[f"clim_{column}"]
    return table
