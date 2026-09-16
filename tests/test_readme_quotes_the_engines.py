"""
Every figure the README states must still be the figure the pipelines produce.

The tests next door pin the evidence; nothing pinned the front page. A metric can
move, its assertion be updated, and the README go on quoting last month's run with a
green build the whole way - and while writing this one, the README quoted a fire
count that was never produced at all (143,229 against an actual 164,707). So each
assertion below reads the value from the published artefact and looks for it in the
README **as a reader sees it**, thousands separators and all.

When a figure legitimately changes, this fails and names the number to edit.
"""

import json
import math
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "data" / "published"


@pytest.fixture(scope="module")
def readme() -> str:
    return " ".join((ROOT / "README.md").read_text(encoding="utf-8").split())


@pytest.fixture(scope="module")
def metrics() -> dict:
    return json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def season() -> dict:
    return json.loads((PUBLISHED / "season_2026.json").read_text(encoding="utf-8"))


def quoted(readme: str, needle: str) -> bool:
    return " ".join(needle.split()) in readme


def test_the_readme_is_the_real_one(readme):
    """Guards against every assertion below passing on a stub."""
    assert len(readme) > 8000
    assert "Four things that would have been silently wrong" in readme


@pytest.mark.parametrize("target", ["has_fire", "has_large_fire"])
def test_the_test_season_table_is_the_scored_one(readme, metrics, target):
    for method, scores in metrics["targets"][target]["splits"]["test"].items():
        assert quoted(readme, f"{scores['roc_auc']:.3f}"), (target, method, "ROC-AUC")
        assert quoted(readme, f"{scores['pr_auc']:.3f}"), (target, method, "PR-AUC")
        assert quoted(readme, f"{scores['share_of_fires_in_top_decile']:.1%}"), (target, method, "top decile")
        assert quoted(readme, f"{scores['lift_top_decile']:.1f}×"), (target, method, "lift")
    assert quoted(readme, f"Brier {metrics['targets']['has_fire']['splits']['test']['model']['brier']:.4f}")


def test_the_same_day_capture_headline_is_the_published_one(readme):
    summary = pd.read_parquet(PUBLISHED / "backtest_summary.parquet")
    fires = int(summary["fires"].sum())
    capture = summary["fires_in_top_decile"].sum() / fires
    cells = int(summary["cells"].max())
    assert quoted(readme, f"rank the {cells} cells fresh")
    assert quoted(readme, f"take the riskiest {math.ceil(0.10 * cells)}")
    assert quoted(readme, f"{capture:.1%} of the {fires:,} fires reported in")


def test_the_2026_season_table_is_the_scored_one(readme, season):
    for method, scores in season["pooled"].items():
        assert quoted(readme, f"{scores['roc_auc']:.3f}"), (method, "ROC-AUC")
    for method, share in season["same_day_top_decile"].items():
        assert quoted(readme, f"{share:.1%}"), (method, "same-day capture")
    assert quoted(readme, f"{season['new_detections']:,} such cell-days out of {season['cell_days']:,}")


def test_the_calibration_sentence_quotes_the_reliability_table(readme, metrics):
    """The README says where calibration holds and where it drifts; both are read
    off the same deciles the model card plots."""
    deciles = metrics["targets"]["has_fire"]["reliability"]
    top = deciles[-1]
    assert quoted(readme, f"{top['predicted']:.1%} predicted against {top['observed']:.1%} observed")
    middle = [row for row in deciles if 0.02 < row["predicted"] < 0.06]
    assert middle, "no middle deciles to quote"
    for row in middle:
        assert quoted(readme, f"the model calls {row['predicted']:.1%}") or \
               quoted(readme, f"the {row['predicted']:.1%} decile on {row['observed']:.1%}"), row


def test_the_dataset_table_counts_the_rows_that_exist(readme):
    fires = int(pd.read_parquet(PUBLISHED / "fires_by_year.parquet")["fires"].sum())
    cells = len(pd.read_parquet(ROOT / "models" / "cells.parquet"))
    stations = len(pd.read_parquet(ROOT / "models" / "stations.parquet"))
    assert quoted(readme, f"{fires:,} in the National Fire Database")
    assert quoted(readme, f"{cells} one-degree cells")
    assert quoted(readme, f"{stations:,} CWFIS stations")


def test_the_feature_count_and_gain_claims_hold(readme, metrics):
    features = metrics["features"]
    gain = metrics["targets"]["has_fire"]["feature_gain"]
    assert quoted(readme, f"**Features ({len(features)}).**")
    assert quoted(readme, f"{gain['clim_month_rate']:.0%} of the trees' gain is")
    assert quoted(readme, f"{gain['clim_cell_rate']:.0%} its rate over the season")


def test_the_badge_counts_the_tests_that_exist(readme):
    """The badge is a claim like any other, so it is collected rather than trusted."""
    collect = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
                             cwd=ROOT, capture_output=True, text=True)
    assert collect.returncode == 0, collect.stdout[-2000:]
    collected = sum(1 for line in collect.stdout.splitlines() if "::" in line)
    stated = int(re.search(r"tests-(\d+)%20passing", readme).group(1))
    assert collected == stated, f"badge says {stated}, pytest collects {collected}"
