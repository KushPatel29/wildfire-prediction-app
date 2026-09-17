"""
The dashboard and the app have to be reading the same evidence.

`pipelines/powerbi_export.py` writes one CSV per dashboard table, and every one
is an aggregate of something this repository already computed: the published
backtest, `reports/metrics.json`, the live forecast, the rebuilt 2024 model. That
is only worth anything if the aggregate still equals its source - otherwise the
Power BI page says 36.7% and the Streamlit page says something else, and a reader
has no way to tell which is the number.

So each export is reconciled here against what it was made from. The tolerances
are for float formatting in CSV, nothing else.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "powerbi" / "data"
PUBLISHED = ROOT / "data" / "published"
LIVE = ROOT / "data" / "live"
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


def exported(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA / f"{name}.csv")


def test_the_backtest_days_are_the_published_backtest_days():
    published = pd.read_parquet(PUBLISHED / "backtest_summary.parquet")
    days = exported("fact_backtest_day")
    assert len(days) == len(published)
    assert days["fires"].sum() == published["fires"].sum()
    assert days["fires_in_top_decile"].sum() == published["fires_in_top_decile"].sum()


def test_the_capture_rate_on_the_dashboard_is_the_one_the_readme_quotes():
    """36.7% is on the README, the app's replay page and this dashboard's first
    card. One of them moving on its own is the thing to catch."""
    days = exported("fact_backtest_day")
    capture = days["fires_in_top_decile"].sum() / days["fires"].sum()
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert f"{capture:.1%} of the {days['fires'].sum():,} fires reported in" in readme


def test_the_cells_are_aggregated_from_the_same_days():
    """The map reads by cell, the trend reads by day; both come from the same
    825,000 cell-days and have to add up to the same fires."""
    by_cell = exported("fact_backtest_cell")
    by_day = exported("fact_backtest_day")
    assert by_cell["fires"].sum() == by_day["fires"].sum()
    assert by_cell["fires_caught"].sum() == by_day["fires_in_top_decile"].sum()
    assert by_cell["large_fires"].sum() == by_day["large_fires"].sum()


def test_every_score_on_the_dashboard_is_a_score_from_the_metrics_file():
    scores = exported("fact_model_scores").set_index(["target", "split", "method"])
    for target, info in METRICS["targets"].items():
        for split, methods in info["splits"].items():
            for method, source in methods.items():
                row = scores.loc[(target, split.title(), method)]
                assert row["roc_auc"] == pytest.approx(source["roc_auc"], abs=5e-7)
                assert row["share_of_fires_in_top_decile"] == pytest.approx(
                    source["share_of_fires_in_top_decile"], abs=5e-7)


def test_the_reliability_curve_is_the_published_one():
    curve = exported("fact_reliability")
    for target, info in METRICS["targets"].items():
        published = info["reliability"]
        rows = curve[curve["target"] == target].sort_values("decile")
        assert len(rows) == len(published)
        assert rows["predicted"].tolist() == pytest.approx([r["predicted"] for r in published], abs=5e-7)


def test_the_forecast_on_the_dashboard_is_the_forecast_the_app_shows():
    live = pd.read_parquet(LIVE / "forecast.parquet")
    forecast = exported("fact_forecast")
    assert len(forecast) == len(live)
    assert forecast["risk"].sum() == pytest.approx(live["risk"].sum(), rel=1e-6)
    assert sorted(forecast["lead_days"].unique()) == sorted(live["lead_days"].unique())


def test_the_riskiest_tenth_on_the_dashboard_is_exactly_a_tenth():
    """The same rule the evidence counts by: ceil(10%) of that day's cells, not
    everything above the 90th percentile."""
    forecast = exported("fact_forecast")
    for day, rows in forecast.groupby("date"):
        expected = -(-len(rows) // 10)              # ceil
        assert rows["in_top_decile"].sum() == expected, day


def test_the_season_check_on_the_dashboard_is_the_published_one():
    report = json.loads((PUBLISHED / "season_2026.json").read_text(encoding="utf-8"))
    scores = exported("fact_season_scores").set_index("method")
    for method, source in report["pooled"].items():
        assert scores.loc[method, "roc_auc"] == pytest.approx(source["roc_auc"], abs=5e-7)
        assert scores.loc[method, "same_day_top_decile"] == pytest.approx(
            report["same_day_top_decile"][method], abs=5e-7)
    days = exported("fact_season_day")
    assert days["detections"].sum() == report["new_detections"]


def test_the_2024_model_on_the_dashboard_is_the_rebuilt_one():
    report = json.loads((ROOT / "reports" / "hackathon_2024.json").read_text(encoding="utf-8"))
    scores = exported("fact_hackathon_scores").set_index(["panel", "scoring"])
    for panel, info in report["panels"].items():
        for scoring, source in info["scorings"].items():
            assert scores.loc[(panel, scoring), "r2"] == pytest.approx(source["r2"], abs=5e-7)
    months = exported("fact_hackathon_month")
    published = pd.read_parquet(PUBLISHED / "hackathon_2024.parquet")
    assert months["actual_ha"].sum() == pytest.approx(published["area_ha"].sum(), rel=1e-9)


def test_the_fire_record_on_the_dashboard_is_the_published_record():
    by_year = exported("fact_fires_year")
    published = pd.read_parquet(PUBLISHED / "fires_by_year.parquet")
    assert by_year["fires"].sum() == published["fires"].sum()
    assert by_year["area_ha"].sum() == pytest.approx(published["area_ha"].sum(), rel=1e-9)


def test_the_grid_is_the_grid_the_model_scores():
    cells = exported("dim_cell")
    published = pd.read_parquet(ROOT / "models" / "cells.parquet")
    assert len(cells) == len(published)
    assert set(cells["cell_id"]) == set(published["cell_id"])
