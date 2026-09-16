"""The table-building steps the model's features come out of."""

import numpy as np
import pandas as pd
import pytest

from wildfire import features as F


def station(**overrides) -> dict:
    row = {"rep_date": pd.Timestamp("2026-07-01"), "aes": "A", "lat": 50.0, "lon": -120.0, "temp": 20.0,
           "rh": 40.0, "ws": 10.0, "precip": 0.0, "ffmc": 88.0, "dmc": 30.0, "dc": 300.0, "isi": 5.0,
           "bui": 45.0, "fwi": 12.0, "dsr": 2.0, "calcstatus": 1}
    row.update(overrides)
    return row


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return float(2 * F.EARTH_KM * np.arcsin(np.sqrt(a)))


def test_cell_ids_floor_to_the_whole_degree():
    assert F.cell_ids([49.99, 50.0, 60.4], [-123.01, -123.0, -100.5]).tolist() == ["49_-124", "50_-123", "60_-101"]


def test_station_files_read_as_text_are_cleaned_like_the_typed_archives():
    """CWFIS's daily file arrives untyped; one blank reading turns a column to text.
    Cleaning has to coerce it, or a live day fails where a training day passed."""
    rows = [station(aes="A", temp="21.5", rh="35", ffmc="90.1", calcstatus="1"),
            station(aes="B", temp="", rh="150", dc="NA", calcstatus="0"),
            station(aes="C", rep_date=pd.Timestamp("2026-11-02"))]
    frame = pd.DataFrame(rows)
    for column in frame:
        if column != "rep_date":
            frame[column] = frame[column].astype(str)
    cleaned = F.clean_stations(frame).set_index("aes")
    assert list(cleaned.index) == ["A", "B"]                      # November is outside the season
    assert cleaned.loc["A", "temp"] == 21.5 and cleaned.loc["A", "ffmc"] == pytest.approx(90.1)
    assert np.isnan(cleaned.loc["B", "temp"]) and np.isnan(cleaned.loc["B", "rh"])  # blank, and out of range
    assert cleaned.loc["B", F.CODES].isna().all()                  # codes CWFIS did not calculate
    assert cleaned.loc["B", "ws"] == 10.0


def test_interpolation_is_exact_at_a_station_and_blind_beyond_200_km():
    cells = pd.DataFrame({"cell_id": ["near", "far"], "lat": [50.0, 70.0], "lon": [-120.0, -120.0]})
    stations = pd.DataFrame([station(aes="A", temp=20.0), station(aes="B", lat=52.0, temp=10.0)])
    out = F.interpolate_day(cells, stations).set_index("cell_id")
    assert out.loc["near", "temp"] == pytest.approx(20.0)          # B is 222 km away and does not count
    assert out.loc["near", "station_km"] == pytest.approx(0.0, abs=1e-6)
    assert out.loc["far", ["temp", "dc", "station_km"]].isna().all()


def test_each_variable_is_weighted_over_the_stations_that_report_it():
    cells = pd.DataFrame({"cell_id": ["c"], "lat": [50.0], "lon": [-120.0]})
    a = station(aes="A", lat=50.3, temp=10.0, dc=np.nan)
    b = station(aes="B", lon=-121.0, temp=20.0, dc=400.0)
    out = F.interpolate_day(cells, pd.DataFrame([a, b]))
    wa, wb = (1 / haversine_km(50.0, -120.0, s["lat"], s["lon"]) ** 2 for s in (a, b))
    assert out.loc[0, "temp"] == pytest.approx((10.0 * wa + 20.0 * wb) / (wa + wb), rel=1e-6)
    assert out.loc[0, "dc"] == pytest.approx(400.0)                # A has no drought code to dilute it


def test_recent_weather_rolls_within_a_cell_and_rain_resets_the_dry_count():
    days = list(pd.date_range("2026-07-01", periods=5))
    table = pd.DataFrame({
        "cell_id": ["a"] * 5 + ["b"] * 5, "date": days * 2,
        "fwi": [1.0, 2.0, 3.0, 4.0, 5.0] + [10.0] * 5, "precip": [0.0, 0.0, 5.0, 0.0, 0.0] + [0.0] * 5,
        "isi": 1.0, "temp": 20.0, "rh": 40.0, "dc": 300.0,
    })
    out = F.add_recent_weather(table)
    a, b = out[out["cell_id"] == "a"], out[out["cell_id"] == "b"]
    assert a["fwi_mean_3d"].tolist() == pytest.approx([1.0, 1.5, 2.0, 3.0, 4.0])
    assert a["days_since_rain"].tolist() == [1, 2, 0, 1, 2]
    assert b["days_since_rain"].tolist() == [1, 2, 3, 4, 5]        # a's dry spell does not carry into b
    assert b["fwi_mean_3d"].iloc[0] == 10.0
