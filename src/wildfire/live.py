"""
The live forecast: today's official fire-weather codes carried forward with forecast weather.

Nothing in here is trained. It assembles, for every grid cell and each of the next
seven days, exactly the feature row the model was trained on - from three live
public sources:

1. **CWFIS station observations** for the last thirty days
   (`fwi_obs/current/cwfis_fwi_YYYYMMDD.csv`). The most recent complete day supplies
   the starting FFMC, DMC and DC; the month supplies the rolling features (a week
   of FWI, two weeks of rain, the days since it last rained) the model reads.
2. **Open-Meteo's forecast** for each cell centre: local-noon temperature, humidity
   and wind, and noon-to-noon precipitation - the four inputs the FWI System takes.
3. **CWFIS satellite hotspots** for the map, as context. They are not a model input.

Stepping the codes forward uses `wildfire.fwi`, the same implementation that is
tested against CWFIS's published values, so a forecast day and a training day are
built by the same arithmetic.
"""

from __future__ import annotations

import io
import time
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
import requests

from wildfire import features as F
from wildfire import fwi

CWFIS = "https://cwfis.cfs.nrcan.gc.ca/downloads"
OPEN_METEO = "https://api.open-meteo.com/v1/forecast"
HISTORY_DAYS = 30                 # long enough to count a month without rain
FORECAST_DAYS = 7
BATCH = 90
TIMEOUT = 45
# Open-Meteo's free tier allows 600 calls a minute and counts every location in a
# multi-location request as a call. 771 cells in one burst trips it; 400 a minute
# does not, and costs about two minutes per forecast.
LOCATIONS_PER_MINUTE = 400
RATE_LIMIT_WAIT = 61
# CWFIS publishes a day's file while stations are still reporting. A day with fewer
# than this share of the month's median station count is treated as partial.
COMPLETE_SHARE = 0.85

# Canada's outline as (lon, lat), coarse - within tens of kilometres along the US
# border - but enough to drop the US and Alaska hotspots in CWFIS's North American
# file. The ocean sides are drawn well offshore.
CANADA_OUTLINE = [
    (-141.0, 84.0), (-52.0, 84.0), (-52.0, 43.0), (-66.0, 43.0), (-66.9, 44.75), (-67.2, 45.2),
    (-67.8, 45.7), (-67.78, 47.06), (-68.3, 47.35), (-69.2, 47.45), (-70.0, 46.7), (-70.3, 45.9),
    (-71.1, 45.3), (-71.5, 45.01), (-74.7, 45.0), (-75.8, 44.4), (-76.5, 44.0), (-77.5, 43.6),
    (-79.05, 43.3), (-79.05, 42.9), (-80.5, 42.4), (-82.5, 41.7), (-83.1, 42.05), (-83.1, 42.3),
    (-82.5, 42.6), (-82.42, 43.0), (-82.5, 45.34), (-83.6, 45.9), (-84.5, 46.5), (-84.8, 46.9),
    (-88.4, 48.3), (-89.6, 48.0), (-91.5, 48.1), (-93.2, 48.6), (-94.6, 48.7), (-95.15, 49.38),
    (-95.15, 49.0), (-123.0, 49.0), (-123.3, 48.3), (-134.0, 48.3), (-134.0, 54.5), (-130.7, 54.7),
    (-130.0, 56.0), (-131.8, 56.6), (-133.4, 58.4), (-135.5, 59.8), (-137.5, 59.2), (-139.0, 60.0),
    (-141.0, 60.3),
]


@dataclass
class LiveInputs:
    history: pd.DataFrame       # cell x day, last HISTORY_DAYS complete observed days
    forecast: pd.DataFrame      # cell x day, next FORECAST_DAYS days of noon weather
    as_of: date                 # latest complete observed station day
    stations_used: int


def _get(url: str, attempts: int = 3, **params) -> requests.Response:
    """GET with retries for transient failures.

    A 404 is an answer, not a failure - CWFIS has not published that day yet - so it
    is raised at once. A 429 is Open-Meteo saying this minute's allowance is spent
    (every location in a batch counts as a call), so the retry waits the minute out."""
    error: requests.RequestException = requests.RequestException(f"no response from {url}")
    for attempt in range(attempts):
        try:
            response = requests.get(url, params=params or None, timeout=TIMEOUT)
        except requests.RequestException as exc:
            error = exc
            time.sleep(1.5 * (attempt + 1))
            continue
        if response.status_code == 200:
            return response
        if response.status_code in (400, 403, 404):
            response.raise_for_status()
        error = requests.HTTPError(f"{response.status_code} from {url}", response=response)
        if attempt < attempts - 1:
            time.sleep(RATE_LIMIT_WAIT if response.status_code == 429 else 1.5 * (attempt + 1))
    raise error


def parse_station_csv(text: str, day: date, locations: pd.DataFrame) -> pd.DataFrame:
    """A CWFIS current-conditions file, located by station id, in the training layout.

    The daily file carries no coordinates, so each station is placed from the
    station list built with the archives; one it has never seen is dropped."""
    frame = pd.read_csv(io.StringIO(text), low_memory=False)
    frame.columns = [c.strip().lower() for c in frame.columns]
    frame["aes"] = frame["aes"].astype(str).str.strip()
    frame = frame.drop(columns=[c for c in ("lat", "lon") if c in frame])
    frame = frame.merge(locations[["aes", "lat", "lon"]], on="aes", how="inner")
    frame["rep_date"] = pd.Timestamp(day)
    if "calcstatus" not in frame:
        frame["calcstatus"] = 1
    return frame


def station_day(day: date, locations: pd.DataFrame) -> pd.DataFrame | None:
    """One day of CWFIS station observations, or None if it is not published."""
    try:
        response = _get(f"{CWFIS}/fwi_obs/current/cwfis_fwi_{day:%Y%m%d}.csv")
    except requests.RequestException:
        return None
    return parse_station_csv(response.text, day, locations)


def complete_days(days: list[tuple[date, pd.DataFrame]]) -> list[tuple[date, pd.DataFrame]]:
    """Drop the newest days while they are partial.

    `days` runs newest first. The file for the newest day can hold half the network
    - on 2026-09-15 it held 1,110 of about 2,100 stations, leaving 30% of cells with
    no station in range - and a forecast started from it would restart those cells'
    drought codes from spring values. Older partial days stay: the rolling features
    tolerate a thin day, the starting codes do not."""
    if not days:
        return days
    typical = float(np.median([len(frame) for _, frame in days]))
    start = 0
    while start < len(days) and len(days[start][1]) < COMPLETE_SHARE * typical:
        start += 1
    return days[start:]


def observed_history(cells: pd.DataFrame, locations: pd.DataFrame, today: date) -> tuple[pd.DataFrame, date, int]:
    """The last HISTORY_DAYS station days CWFIS has published, starting from the
    latest complete one, interpolated to cells."""
    days: list[tuple[date, pd.DataFrame]] = []
    day = today
    while len(days) < HISTORY_DAYS + 2 and (today - day).days < HISTORY_DAYS + 12:
        raw = station_day(day, locations)
        if raw is not None and len(raw):
            cleaned = F.clean_stations(raw)
            if len(cleaned):
                days.append((day, cleaned))
        day -= timedelta(days=1)
    days = complete_days(days)[:HISTORY_DAYS]
    if not days:
        raise RuntimeError("CWFIS published no complete station observations in the last four weeks")
    frames = []
    for observed, cleaned in days:
        interpolated = F.interpolate_day(cells, cleaned)
        interpolated.insert(1, "date", pd.Timestamp(observed))
        frames.append(interpolated)
    history = pd.concat(frames, ignore_index=True).sort_values(["cell_id", "date"], ignore_index=True)
    return history, days[0][0], max(len(frame) for _, frame in days)


def noon_weather(place: dict, start: date, days: int = FORECAST_DAYS) -> list[dict]:
    """Local-noon weather for the `days` days after `start` from one Open-Meteo location.

    Precipitation is summed over the 24 hours ending at local noon, which is the
    window the FWI System's rain input is defined on."""
    hourly = pd.DataFrame(place["hourly"])
    hourly["time"] = pd.to_datetime(hourly["time"])
    hourly["rain24"] = hourly["precipitation"].rolling(24, min_periods=1).sum()
    noon = hourly[hourly["time"].dt.hour == 12]
    rows = []
    for hour in noon.itertuples(index=False):
        day = hour.time.date()
        if start < day <= start + timedelta(days=days):
            rows.append({"date": pd.Timestamp(day), "temp": hour.temperature_2m, "rh": hour.relative_humidity_2m,
                         "ws": hour.wind_speed_10m, "precip": hour.rain24})
    return rows


def forecast_weather(cells: pd.DataFrame, start: date, pace: bool = True) -> pd.DataFrame:
    """Local-noon forecast weather at each cell centre for FORECAST_DAYS days after `start`."""
    rows = []
    for offset in range(0, len(cells), BATCH):
        batch = cells.iloc[offset:offset + BATCH]
        response = _get(
            OPEN_METEO, attempts=4,
            latitude=",".join(f"{v:.2f}" for v in batch["lat"]),
            longitude=",".join(f"{v:.2f}" for v in batch["lon"]),
            hourly="temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
            timezone="auto", past_days=1, forecast_days=FORECAST_DAYS + 1,
        )
        payload = response.json()
        payload = payload if isinstance(payload, list) else [payload]
        for cell_id, place in zip(batch["cell_id"], payload):
            rows.extend({"cell_id": cell_id, **row} for row in noon_weather(place, start))
        if pace and offset + BATCH < len(cells):
            time.sleep(60.0 * len(batch) / LOCATIONS_PER_MINUTE)
    return pd.DataFrame(rows)


def carry_codes_forward(history: pd.DataFrame, weather: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Step every cell's moisture codes through the forecast days with forecast noon weather.

    Each cell starts from its latest interpolated codes on or before `as_of`: usually
    that day's, and for a cell whose stations were silent that day, the last day
    they reported. Only a cell with no codes in the whole month falls back to
    the FWI System's season-start values."""
    defaults = {"ffmc": fwi.FFMC_START, "dmc": fwi.DMC_START, "dc": fwi.DC_START}
    observed = history[history["date"] <= pd.Timestamp(as_of)].sort_values("date")
    latest = observed.groupby("cell_id")[list(defaults)].last()
    start = history[history["date"] == pd.Timestamp(as_of)].set_index("cell_id")
    state = {k: latest[k].reindex(start.index).fillna(defaults[k]) for k in defaults}
    days = []
    for day, group in weather.sort_values("date").groupby("date"):
        group = group.set_index("cell_id").reindex(start.index)
        current = {k: state[k].reindex(group.index).to_numpy(dtype=float) for k in state}
        out = fwi.step(current, group["temp"].to_numpy(dtype=float), group["rh"].to_numpy(dtype=float),
                       group["ws"].to_numpy(dtype=float), group["precip"].fillna(0).to_numpy(dtype=float), day.month)
        frame = group[["temp", "rh", "ws", "precip"]].copy()
        for key, values in out.items():
            frame[key] = values
        frame["date"] = day
        days.append(frame.reset_index())
        state = {k: pd.Series(out[k], index=group.index) for k in defaults}
    forecast = pd.concat(days, ignore_index=True)
    forecast["station_km"] = forecast["cell_id"].map(start["station_km"])
    return forecast


def feature_rows(cells: pd.DataFrame, climatology: pd.DataFrame, history: pd.DataFrame,
                 forecast: pd.DataFrame, features: list[str], include_latest_observed: bool = True) -> pd.DataFrame:
    """History and forecast stitched into one series per cell, so the rolling
    windows on a forecast day reach back into observed days exactly as they did
    in training. Returns the forecast days, and by default the latest observed day
    as lead 0 - the day's risk from measured weather."""
    as_of = history["date"].max()
    combined = pd.concat([history.assign(is_forecast=False), forecast.assign(is_forecast=True)],
                         ignore_index=True)
    combined = F.add_recent_weather(combined)
    combined = combined.merge(cells[["cell_id", "lat", "lon", "province", "lightning_share"]], on="cell_id", how="left")
    combined = combined.merge(climatology, on=["cell_id", "month"], how="left")
    keep = combined["is_forecast"] | (include_latest_observed & (combined["date"] == as_of))
    rows = combined[keep].copy()
    rows["lead_days"] = (rows["date"] - as_of).dt.days
    missing = [f for f in features if f not in rows]
    if missing:
        raise KeyError(f"live feature rows lack {missing}")
    return rows.reset_index(drop=True)


def in_canada(frame: pd.DataFrame) -> pd.Series:
    """Points inside CANADA_OUTLINE, by ray casting."""
    x = frame["lon"].to_numpy(dtype=float)
    y = frame["lat"].to_numpy(dtype=float)
    ring = np.asarray(CANADA_OUTLINE)
    inside = np.zeros(len(x), dtype=bool)
    with np.errstate(divide="ignore", invalid="ignore"):
        for (x1, y1), (x2, y2) in zip(ring, np.roll(ring, 1, axis=0)):
            straddles = (y1 > y) != (y2 > y)
            inside ^= straddles & (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1)
    return pd.Series(inside, index=frame.index)


def hotspots(days: int = 2, today: date | None = None) -> pd.DataFrame:
    """Satellite hotspots CWFIS published over the last `days` days, inside Canada
    (the daily file covers North America)."""
    today = today or date.today()
    frames = []
    for back in range(days + 2):
        day = today - timedelta(days=back)
        try:
            response = _get(f"{CWFIS}/hotspots/{day:%Y%m%d}.csv")
        except requests.RequestException:
            continue
        frame = pd.read_csv(io.StringIO(response.text))
        frame.columns = [c.strip().lower() for c in frame.columns]
        frames.append(frame)
        if len(frames) == days:
            break
    if not frames:
        return pd.DataFrame(columns=["lat", "lon", "rep_date", "sensor", "fwi", "hfi", "estarea"])
    spots = pd.concat(frames, ignore_index=True)
    spots["rep_date"] = pd.to_datetime(spots["rep_date"], errors="coerce")
    return spots[in_canada(spots)].reset_index(drop=True)


def build_live_inputs(cells: pd.DataFrame, locations: pd.DataFrame, today: date | None = None,
                      pace: bool = True) -> LiveInputs:
    today = today or date.today()
    history, as_of, used = observed_history(cells, locations, today)
    weather = forecast_weather(cells, as_of, pace=pace)
    forecast = carry_codes_forward(history, weather, as_of)
    return LiveInputs(history=history, forecast=forecast, as_of=as_of, stations_used=used)
