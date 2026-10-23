"""
The published forecasts are graded by rules fixed in advance, and these tests hold
the grader to them.

The track record is the one number here that nobody chose: it is what the model
said, in public, before the outcome existed. That only holds if the grader reads
the snapshots the way the page drew them, keeps the first forecast of each day,
never lets a cell's own fire count as "next door", and grades only days the
satellite archive has covered.
"""
from __future__ import annotations

import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

import track_record as pipeline  # noqa: E402
import web_snapshot  # noqa: E402
from wildfire import track_record as T  # noqa: E402

LIVE = ROOT / "data" / "live"
CELLS = [f"{50 + i // 5}_{-120 + i % 5}" for i in range(20)]


def payload(as_of: date, built: datetime, risk: np.ndarray, normal: np.ndarray,
            cells: list[str] = CELLS, hotspots: list[list[float]] | None = None) -> dict:
    """A snapshot in the exact format web_snapshot publishes."""
    days = [as_of + timedelta(days=k) for k in range(risk.shape[1])]
    grid = np.where(np.isnan(risk), -1, np.rint(risk * web_snapshot.SCALE)).astype(int)
    return {
        "version": 1, "scale": web_snapshot.SCALE, "large_scale": web_snapshot.LARGE_SCALE,
        "generated_at": built.isoformat(), "as_of": as_of.isoformat(), "stations_used": 100,
        "days": [d.isoformat() for d in days], "lead_days": list(range(len(days))),
        "cells": [[c, 0.0, 0.0, "BC"] for c in cells],
        "normal": np.rint(normal * web_snapshot.SCALE).astype(int).tolist(),
        "risk": grid.tolist(),
        "large": np.where(np.isnan(risk), -1, np.rint(risk / 10 * web_snapshot.LARGE_SCALE)).astype(int).tolist(),
        "hotspots": hotspots or [],
    }


def built(day: date, hour: int) -> datetime:
    return datetime(day.year, day.month, day.day, hour, 15, tzinfo=timezone.utc)


def test_the_decoder_reads_what_the_publisher_writes():
    """Decoding the real published format gives back the app's own forecast."""
    snap = T.Snapshot.from_json("live", web_snapshot.snapshot(LIVE))
    forecast = pd.read_parquet(LIVE / "forecast.parquet")
    forecast["date"] = pd.to_datetime(forecast["date"])
    assert len(snap.cell_ids) == forecast["cell_id"].nunique()
    assert snap.lead_days == list(range(len(snap.days)))
    wide = forecast.pivot(index="cell_id", columns="date", values="risk").reindex(snap.cell_ids)
    assert np.nanmax(np.abs(snap.risk - wide.to_numpy())) <= 0.5 / web_snapshot.SCALE + 1e-12


def test_unpublished_cell_days_stay_missing_instead_of_becoming_zero_risk():
    risk = np.full((20, 3), 0.01)
    risk[4, 2] = np.nan
    snap = T.Snapshot.from_json("x", payload(date(2026, 10, 1), built(date(2026, 10, 2), 0), risk, np.full(20, 0.001)))
    assert np.isnan(snap.risk[4, 2]) and snap.risk[4, 1] == pytest.approx(0.01)


def test_only_the_first_forecast_of_each_station_day_on_the_current_grid_is_graded():
    day = date(2026, 10, 1)
    risk, normal = np.full((20, 3), 0.01), np.full(20, 0.001)
    morning = T.Snapshot.from_json("a", payload(day, built(day + timedelta(1), 0), risk, normal))
    afternoon = T.Snapshot.from_json("b", payload(day, built(day + timedelta(1), 13), risk, normal))
    old_grid = T.Snapshot.from_json("c", payload(day + timedelta(1), built(day + timedelta(2), 0), risk[:19], normal[:19],
                                                 cells=CELLS[:19]))
    kept, excluded = T.first_per_station_day([afternoon, old_grid, morning], set(CELLS))
    assert [s.sha for s in kept] == ["a"]
    assert excluded == {"published": 3, "off_grid": 1, "republished_same_day": 1}


def test_fire_next_door_counts_the_eight_neighbours_and_never_the_cell_itself():
    # One hotspot inside cell 50_-119.
    snap = T.Snapshot.from_json("x", payload(date(2026, 10, 1), built(date(2026, 10, 2), 0), np.full((20, 1), 0.01),
                                             np.full(20, 0.001), hotspots=[[50.5, -118.5]]))
    counts = dict(zip(snap.cell_ids, T.fire_next_door(snap)))
    assert counts["50_-119"] == 0          # its own cell: not "next door"
    assert counts["50_-120"] == 1          # west
    assert counts["51_-118"] == 1          # north-east diagonal
    assert counts["53_-116"] == 0          # far away


def test_a_target_day_counts_as_a_forecast_only_if_it_starts_after_the_build():
    day = date(2026, 10, 1)
    snap = T.Snapshot.from_json("x", payload(day, built(day + timedelta(1), 0), np.full((20, 4), 0.01), np.full(20, 0.001)))
    rows = T.forecast_rows([snap])
    flags = rows.drop_duplicates("lead_days").set_index("lead_days")["prospective"].to_dict()
    assert flags == {0: False, 1: False, 2: True, 3: True}


def _graded_case(model_is_right: bool):
    """Two new detections per target day; the model ranks them first or last."""
    day = date(2026, 10, 1)
    risk = np.tile(np.linspace(0.02, 0.001, 20)[:, None], (1, 6))
    if not model_is_right:
        risk = risk[::-1].copy()
    normal = np.linspace(0.0001, 0.002, 20)            # normal ranks the same two cells last
    snap = T.Snapshot.from_json("x", payload(day, built(day + timedelta(1), 0), risk, normal))
    rows = T.forecast_rows([snap])
    labels = pd.DataFrame([(c, pd.Timestamp(day + timedelta(k)), int(c in CELLS[:2]))
                           for c in CELLS for k in range(5)],        # day 5 not yet in the archive
                          columns=["cell_id", "date", "new_detection"])
    return T.grade(rows, labels, draws=200)


def test_a_model_that_ranks_the_new_fires_first_catches_all_of_them_and_its_edge_is_measured():
    report, graded = _graded_case(model_is_right=True)
    p = report["prospective"]
    assert p["target_days"] == 3                     # leads 2, 3, 4; lead 5 has no label yet
    assert p["model"]["roc_auc"] == pytest.approx(1.0)
    assert p["model"]["top_decile_capture"] == pytest.approx(1.0)
    assert p["normal_for_the_month"]["top_decile_capture"] == pytest.approx(0.0)
    ci = report["prospective_intervals"]
    assert ci["model_capture"] == [pytest.approx(1.0), pytest.approx(1.0)]
    assert ci["edge_on_normal_capture"][0] == pytest.approx(1.0)
    assert graded["date"].max() == pd.Timestamp(date(2026, 10, 5))


def test_a_model_that_ranks_them_last_is_reported_as_losing():
    report, _ = _graded_case(model_is_right=False)
    p = report["prospective"]
    assert p["model"]["top_decile_capture"] == pytest.approx(0.0)
    assert report["prospective_intervals"]["edge_on_normal_capture"][1] <= 0


def test_the_readme_block_is_written_from_the_report_and_nothing_else_changes(tmp_path):
    report, graded = _graded_case(model_is_right=True)
    report |= {"forecasts_graded": 1}
    readme = tmp_path / "README.md"
    readme.write_text(f"# Title\n\nbefore\n\n{pipeline.BEGIN}\nold\n{pipeline.END}\n\nafter\n", encoding="utf-8")
    assert pipeline.write_readme(report, readme)
    text = readme.read_text(encoding="utf-8")
    assert text.startswith("# Title\n\nbefore\n\n") and text.endswith("\n\nafter\n")
    assert "old" not in text and "| **This model** | 1.000 | 100.0% |" in text
    daily = pipeline.daily_table(graded)
    assert int(daily["new_detections"].sum()) == int(graded["new_detection"].sum())
    assert int(daily["caught_model"].sum()) == int(daily["new_detections"].sum())


def test_the_written_verdict_follows_the_interval_not_the_point_estimate():
    assert pipeline.verdict([0.01, 0.05], "normal", "ROC-AUC") == "ahead of normal on ROC-AUC"
    assert pipeline.verdict([-0.05, -0.01], "normal", "ROC-AUC") == "behind normal on ROC-AUC"
    # A point estimate of +4 points with an interval through zero is not a win.
    assert pipeline.verdict([-0.05, 0.13], "normal", "the top 10%") == "not yet distinguishable from normal on the top 10%"
    assert "too few" in pipeline.verdict([None, None], "normal", "ROC-AUC")


def test_an_empty_record_says_so_rather_than_printing_zeros():
    assert "No forecast has a graded target day yet" in pipeline.readme_block({"graded_cell_days": 0})


def test_the_readme_quotes_the_committed_track_record():
    """The weekly workflow writes both; a hand edit to either breaks this."""
    import json
    report_file = ROOT / "data" / "published" / "track_record.json"
    if not report_file.exists():
        pytest.skip("no track record has been graded yet")
    report = json.loads(report_file.read_text(encoding="utf-8"))
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert pipeline.readme_block(report) in readme
