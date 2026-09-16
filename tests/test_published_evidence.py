"""The evidence the app shows, checked against itself and against the claims made about it."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "data" / "published"
LIVE = ROOT / "data" / "live"
METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
SEASON = PUBLISHED / "season_2026.json"


def test_the_model_beats_both_baselines_on_every_held_out_split():
    for target, info in METRICS["targets"].items():
        for split, methods in info["splits"].items():
            for metric in ("roc_auc", "pr_auc", "share_of_fires_in_top_decile"):
                model = methods["model"][metric]
                assert model > methods["climatology"][metric], (target, split, metric)
                assert model > methods["fwi_logistic"][metric], (target, split, metric)


def test_the_published_replay_reproduces_the_reported_test_scores():
    """The replay the app draws is scored by the deployed code path; if it was built
    with different trees or a different calibration, its AUC drifts from the report."""
    daily = pd.read_parquet(PUBLISHED / "backtest_daily.parquet", columns=["risk", "large_risk", "fires", "large_fires"])
    test = {target: info["splits"]["test"]["model"]["roc_auc"] for target, info in METRICS["targets"].items()}
    assert roc_auc_score(daily["fires"] > 0, daily["risk"]) == pytest.approx(test["has_fire"], abs=2e-4)
    assert roc_auc_score(daily["large_fires"] > 0, daily["large_risk"]) == pytest.approx(test["has_large_fire"], abs=2e-4)


def test_every_replay_day_ranks_each_cell_once_and_counts_exactly_a_tenth():
    daily = pd.read_parquet(PUBLISHED / "backtest_daily.parquet", columns=["date", "rank_in_day", "fires"])
    summary = pd.read_parquet(PUBLISHED / "backtest_summary.parquet").set_index("date")
    per_day = daily.groupby("date")
    assert (per_day["rank_in_day"].nunique() == per_day.size()).all()
    assert (per_day["rank_in_day"].max() == per_day.size()).all()
    cut = np.ceil(0.10 * per_day["date"].transform("size"))
    captured = daily["fires"].where(daily["rank_in_day"] <= cut, 0).groupby(daily["date"]).sum()
    assert (captured == summary["fires_in_top_decile"]).all()
    assert (summary["fires_in_top_decile"] <= summary["fires"]).all()


@pytest.mark.skipif(not SEASON.exists(), reason="season check not published")
def test_the_season_check_agrees_with_its_daily_file():
    report = json.loads(SEASON.read_text(encoding="utf-8"))
    daily = pd.read_parquet(PUBLISHED / "season_2026_daily.parquet")
    assert report["cell_days"] == len(daily)
    assert report["new_detections"] == int(daily["new_detection"].sum())
    assert (daily.groupby("date")["in_top_decile"].sum() == np.ceil(0.10 * daily.groupby("date").size())).all()
    share = daily.loc[daily["in_top_decile"], "new_detection"].sum() / daily["new_detection"].sum()
    assert share == pytest.approx(report["same_day_top_decile"]["model"])


def test_the_bundled_live_snapshot_matches_its_metadata():
    meta = json.loads((LIVE / "meta.json").read_text(encoding="utf-8"))
    forecast = pd.read_parquet(LIVE / "forecast.parquet")
    assert len(forecast) == meta["rows"]
    if meta["in_season"]:
        assert forecast["cell_id"].nunique() == meta["cells"]
        assert forecast["risk"].between(0, 1).all() and forecast["large_risk"].between(0, 1).all()
        assert forecast["lead_days"].min() == 0
        assert str(forecast["date"].min().date()) == meta["first_day"] == meta["as_of"]
