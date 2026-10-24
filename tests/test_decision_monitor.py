import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
from decision_monitor import summarize  # noqa: E402


def sample():
    return pd.DataFrame(
        [
            {
                "issued": "2026-10-01",
                "date": "2026-10-03",
                "lead_days": 2,
                "prospective": True,
                "cells": 100,
                "new_detections": 5,
                "caught_model": 2,
            },
            {
                "issued": "2026-10-02",
                "date": "2026-10-03",
                "lead_days": 1,
                "prospective": False,
                "cells": 100,
                "new_detections": 99,
                "caught_model": 9,
            },
        ]
    )


def test_costs_reconcile_at_the_forecast_target_grain_and_exclude_nonprospective():
    result = summarize(sample(), review_cost=2, missed_detection_cost=10)
    row = result["slices"][0]
    assert row["selected_cells"] == 10
    assert row["missed"] == 3
    assert row["selected_without_detection"] == 8
    assert row["illustrative_cost_per_forecast_target"] == 50
    assert row["capture"] == 0.4
    assert result["automatic_retraining_or_promotion"] is False


def test_zero_positives_are_unavailable_instead_of_perfect_capture():
    frame = sample().iloc[:1].copy()
    frame["new_detections"] = frame["caught_model"] = 0
    result = summarize(frame)["slices"][0]
    assert result["capture"] is None
    assert result["capture_interval_95"] is None


@pytest.mark.parametrize("problem", ["duplicates", "impossible_count", "negative_cost"])
def test_invalid_monitoring_evidence_is_rejected(problem):
    frame = sample()
    if problem == "duplicates":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif problem == "impossible_count":
        frame.loc[0, "caught_model"] = 11
    with pytest.raises(ValueError):
        summarize(frame, review_cost=-1 if problem == "negative_cost" else 1)
