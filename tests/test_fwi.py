"""
The FWI System code is held to the numbers the Canadian Wildland Fire Information
System publishes, not to itself.

Every feature the model reads and every forecast the app draws comes out of
`wildfire.fwi`. An error in one of Van Wagner's constants would not crash
anything: it would shift every drought code a little, the model would learn the
shifted scale, and the live forecast - initialised from CWFIS's own codes - would
start every day on a different scale from the one it was trained on. So the test
replays a real station season through the implementation, starting from the
station's official codes on its first day and feeding it only that station's noon
weather, and requires the result to track CWFIS's published codes.

Where the station file is not on disk (a fresh clone before `make data`), the
replay skips and the unit checks below still run.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from wildfire import fwi

ROOT = Path(__file__).resolve().parents[1]
STATIONS_2010S = ROOT / "data" / "interim" / "cwfis_fwi_2010s.parquet"
CANADIAN_STATIONS = 8


def test_start_codes_are_van_wagners():
    state = fwi.start_state(3)
    assert np.allclose(state["ffmc"], 85.0)
    assert np.allclose(state["dmc"], 6.0)
    assert np.allclose(state["dc"], 15.0)


def test_rain_wets_and_sun_dries():
    """Direction, not magnitude: a heavy rain lowers every moisture code and a hot,
    dry, windy day raises them. A sign error anywhere fails this."""
    state = {"ffmc": np.array([90.0]), "dmc": np.array([40.0]), "dc": np.array([300.0])}
    wet = fwi.step(state, temp=12.0, rh=95.0, wind=5.0, rain=25.0, month=7)
    dry = fwi.step(state, temp=32.0, rh=15.0, wind=30.0, rain=0.0, month=7)
    for code in ("ffmc", "dmc", "dc"):
        assert wet[code][0] < state[code][0] < dry[code][0], code
    assert dry["fwi"][0] > wet["fwi"][0]


def test_codes_stay_in_range_under_extremes():
    temp = np.array([-40.0, 50.0, 25.0, 0.0])
    rh = np.array([0.0, 100.0, 5.0, 60.0])
    wind = np.array([0.0, 120.0, 60.0, 10.0])
    rain = np.array([0.0, 200.0, 0.0, 3.0])
    out = fwi.step(fwi.start_state(4), temp, rh, wind, rain, month=np.array([1, 7, 8, 10]))
    assert np.all((out["ffmc"] >= 0) & (out["ffmc"] <= 101))
    for code in ("dmc", "dc", "isi", "bui", "fwi", "dsr"):
        assert np.all(np.isfinite(out[code])) and np.all(out[code] >= 0), code


@pytest.fixture(scope="module")
def season_2019() -> pd.DataFrame:
    if not STATIONS_2010S.exists():
        pytest.skip("station observations not built; run the data pipeline first")
    obs = pd.read_parquet(STATIONS_2010S)
    season = obs[(obs["rep_date"].dt.year == 2019) & obs["rep_date"].dt.month.between(4, 9)
                 & (obs["calcstatus"] == 1)
                 & obs["lat"].between(48.0, 70.0) & obs["lon"].between(-141.0, -52.0)]
    return season.dropna(subset=["temp", "rh", "ws", "precip", "ffmc", "dmc", "dc", "isi", "bui", "fwi"])


def _replay(station: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    station = station.sort_values("rep_date").drop_duplicates("rep_date")
    consecutive = station["rep_date"].diff().dt.days.fillna(1).eq(1).cumprod().astype(bool)
    station = station[consecutive]
    state = {k: station[k].iloc[:1].to_numpy() for k in ("ffmc", "dmc", "dc")}
    rows = []
    for _, day in station.iloc[1:].iterrows():
        out = fwi.step(state, day["temp"], day["rh"], day["ws"], day["precip"], day["rep_date"].month)
        rows.append({k: float(np.ravel(v)[0]) for k, v in out.items()})
        state = {k: np.ravel(out[k]) for k in ("ffmc", "dmc", "dc")}
    return pd.DataFrame(rows, index=station.index[1:]), station.iloc[1:]


def test_a_canadian_season_replays_to_the_published_codes(season_2019):
    """Eight Canadian stations with an unbroken April-September record. FFMC, DC,
    ISI and FWI track CWFIS almost exactly; DMC carries CWFIS's daily rounding and
    drifts by about two points over a season, so its bound is looser."""
    counts = season_2019.groupby("aes").size().sort_values(ascending=False)
    checked = 0
    for aes in counts.index:
        mine, official = _replay(season_2019[season_2019["aes"] == aes])
        if len(mine) < 150:
            continue
        mae = {k: float(np.mean(np.abs(mine[k] - official[k]))) for k in ("ffmc", "dmc", "dc", "isi", "fwi")}
        name = official["name"].iloc[0]
        assert mae["ffmc"] < 0.5, (name, mae)
        assert mae["dc"] < 2.0, (name, mae)
        assert mae["isi"] < 0.3, (name, mae)
        assert mae["fwi"] < 1.0, (name, mae)
        assert mae["dmc"] < 5.0, (name, mae)
        checked += 1
        if checked == CANADIAN_STATIONS:
            break
    assert checked == CANADIAN_STATIONS, f"only {checked} stations had a full season to replay"
