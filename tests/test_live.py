"""The live forecast's assembly: partial days, silent stations, noon weather, the border."""

from datetime import date

import numpy as np
import pandas as pd
import pytest
import requests

from wildfire import fwi, live
from wildfire.model import FEATURES


def test_a_partial_newest_day_is_dropped_but_an_older_thin_day_is_kept():
    sizes = [1110, 2104, 2105, 1500, 2102, 2100]                    # newest first, as on 2026-09-15
    days = [(date(2026, 9, 15 - i), pd.DataFrame(index=range(n))) for i, n in enumerate(sizes)]
    assert [day.day for day, _ in live.complete_days(days)] == [14, 13, 12, 11, 10]


def test_a_cell_whose_stations_were_silent_starts_from_its_last_report():
    as_of = pd.Timestamp("2026-09-14")
    history = pd.DataFrame({
        "cell_id": ["a", "a", "b", "b"], "date": [as_of - pd.Timedelta(days=1), as_of] * 2,
        "ffmc": [85.0, 88.0, 80.0, np.nan], "dmc": [30.0, 31.0, 50.0, np.nan], "dc": [300.0, 305.0, 400.0, np.nan],
        "station_km": [10.0, 10.0, np.nan, np.nan],
    })
    weather = pd.DataFrame({"cell_id": ["a", "b"], "date": [as_of + pd.Timedelta(days=1)] * 2,
                            "temp": 20.0, "rh": 40.0, "ws": 10.0, "precip": 0.0})
    out = live.carry_codes_forward(history, weather, as_of.date()).set_index("cell_id")
    expected = fwi.step({"ffmc": np.array([88.0, 80.0]), "dmc": np.array([31.0, 50.0]), "dc": np.array([305.0, 400.0])},
                        np.full(2, 20.0), np.full(2, 40.0), np.full(2, 10.0), np.zeros(2), 9)
    for code in ("ffmc", "dmc", "dc"):
        assert out.loc[["a", "b"], code].to_numpy() == pytest.approx(expected[code]), code


def test_noon_weather_reads_local_noon_and_the_24_hours_of_rain_before_it():
    hours = pd.date_range("2026-07-01T00:00", periods=72, freq="h")
    place = {"hourly": {"time": hours.strftime("%Y-%m-%dT%H:%M").tolist(),
                        "temperature_2m": [float(h.hour) for h in hours], "relative_humidity_2m": [50.0] * 72,
                        "wind_speed_10m": [5.0] * 72, "precipitation": [1.0] * 72}}
    rows = live.noon_weather(place, start=date(2026, 7, 1), days=2)
    assert [row["date"].day for row in rows] == [2, 3]
    assert all(row["temp"] == 12.0 and row["precip"] == 24.0 for row in rows)


def test_feature_rows_carry_every_model_feature_from_lead_0_to_the_last_forecast_day():
    cells = pd.DataFrame({"cell_id": ["50_-121", "51_-121"], "lat": [50.5, 51.5], "lon": [-120.5, -120.5],
                          "province": ["British Columbia"] * 2, "lightning_share": [0.6, 0.7]})
    weather = {"temp": 22.0, "rh": 35.0, "ws": 12.0, "precip": 0.0, "ffmc": 89.0, "dmc": 40.0, "dc": 350.0,
               "isi": 7.0, "bui": 60.0, "fwi": 18.0, "dsr": 4.0, "station_km": 30.0}
    history = pd.DataFrame([{"cell_id": c, "date": d, **weather}
                            for c in cells["cell_id"] for d in pd.date_range("2026-07-01", periods=14)])
    forecast = pd.DataFrame([{"cell_id": c, "date": d, **weather}
                             for c in cells["cell_id"] for d in pd.date_range("2026-07-15", periods=7)])
    climatology = pd.DataFrame({"cell_id": cells["cell_id"], "month": 7, "clim_month_rate": [0.05, 0.04],
                                "clim_cell_rate": [0.03, 0.02]})
    rows = live.feature_rows(cells, climatology, history, forecast, FEATURES)
    assert sorted(rows["lead_days"].unique()) == list(range(8))
    assert len(rows) == 2 * 8
    assert rows[FEATURES].notna().all().all()


def test_the_daily_station_file_is_placed_by_station_id():
    text = ("NAME,AGENCY,AES,WMO,REPDATE,TEMP,RH,WS,PRECIP,FFMC,DMC,DC,BUI,ISI,FWI,DSR,CALCSTATUS\n"
            "Alpha,BC, A1 ,1,2026-07-01 12:00,21.0,30,12,0,90,40,350,60,8,20,5,1\n"
            "Unknown,BC,ZZ,2,2026-07-01 12:00,19.0,35,10,0,88,35,340,55,6,15,3,1\n")
    locations = pd.DataFrame({"aes": ["A1"], "lat": [50.1], "lon": [-120.2]})
    frame = live.parse_station_csv(text, date(2026, 7, 1), locations)
    assert frame["aes"].tolist() == ["A1"]
    assert (frame.loc[0, "lat"], frame.loc[0, "lon"]) == (50.1, -120.2)
    assert frame.loc[0, "rep_date"] == pd.Timestamp("2026-07-01")


@pytest.mark.parametrize(("place", "lat", "lon", "inside"), [
    ("Vancouver", 49.28, -123.12, True), ("Victoria", 48.43, -123.37, True), ("Seattle", 47.61, -122.33, False),
    ("Winnipeg", 49.90, -97.14, True), ("Grand Forks ND", 47.93, -97.03, False), ("Idaho Falls", 43.49, -112.04, False),
    ("Toronto", 43.65, -79.38, True), ("Windsor", 42.31, -83.04, True), ("Buffalo", 42.89, -78.88, False),
    ("Montreal", 45.50, -73.57, True), ("Bangor ME", 44.80, -68.77, False), ("Fredericton", 45.96, -66.64, True),
    ("Whitehorse", 60.72, -135.06, True), ("Juneau", 58.30, -134.42, False), ("Fairbanks", 64.84, -147.72, False),
    ("Iqaluit", 63.75, -68.52, True),
])
def test_the_outline_keeps_canada_and_drops_its_neighbours(place, lat, lon, inside):
    assert bool(live.in_canada(pd.DataFrame({"lat": [lat], "lon": [lon]})).iloc[0]) is inside, place


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code

    def raise_for_status(self):
        raise requests.HTTPError(str(self.status_code), response=self)


def test_a_404_is_raised_at_once_and_a_429_waits_out_the_minute(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(live.time, "sleep", sleeps.append)
    monkeypatch.setattr(live.requests, "get", lambda *args, **kwargs: calls.append(1) or _Response(404))
    with pytest.raises(requests.HTTPError):
        live._get("https://example.invalid/cwfis_fwi_20260916.csv")
    assert len(calls) == 1 and sleeps == []

    responses = iter([_Response(429), _Response(200)])
    monkeypatch.setattr(live.requests, "get", lambda *args, **kwargs: next(responses))
    assert live._get("https://example.invalid/forecast").status_code == 200
    assert sleeps == [live.RATE_LIMIT_WAIT]
