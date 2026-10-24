"""Decision-cost sensitivity by lead on the existing prospective daily grade."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/published/track_record_daily.parquet"


def summarize(daily, review_cost=1.0, missed_detection_cost=100.0):
    if any(not math.isfinite(value) or value < 0 for value in (review_cost, missed_detection_cost)):
        raise ValueError("Costs must be finite, nonnegative assumptions")
    required = {"issued", "date", "lead_days", "prospective", "cells", "new_detections", "caught_model"}
    if not required <= set(daily.columns) or daily.empty:
        raise ValueError("Nonempty daily evaluation with all required fields is needed")
    if daily[list(required)].isna().any().any() or daily.duplicated(["issued", "date", "lead_days"]).any():
        raise ValueError("Missing or duplicate evaluation groups")
    if not daily.prospective.isin([True, False]).all():
        raise ValueError("Prospective eligibility must be boolean")
    frame = daily.loc[daily.prospective.eq(True) & daily.lead_days.between(2, 7)].copy()
    if frame.empty:
        raise ValueError("No eligible prospective leads 2–7")
    for field in ("cells", "new_detections", "caught_model"):
        values = pd.to_numeric(frame[field], errors="raise")
        if not np.isfinite(values).all() or (values < 0).any() or (values % 1 != 0).any():
            raise ValueError("Evaluation counts must be nonnegative integers")
    frame["selected_cells"] = np.ceil(frame.cells * 0.1).astype(int)
    if (
        (frame.cells <= 0)
        | (frame.new_detections > frame.cells)
        | (frame.caught_model > frame.new_detections)
        | (frame.caught_model > frame.selected_cells)
    ).any():
        raise ValueError("Evaluation counts do not reconcile")
    frame["missed"] = frame.new_detections - frame.caught_model
    frame["selected_without_detection"] = frame.selected_cells - frame.caught_model
    slices = []
    rng = np.random.default_rng(29)
    for lead, group in frame.groupby("lead_days", sort=True):
        # Whole target-day clusters retain dependence among forecasts for a day.
        per_day = group.groupby("date")[["new_detections", "caught_model"]].sum().to_numpy()
        ratios = []
        for _ in range(1000):
            positives, caught = per_day[rng.integers(0, len(per_day), len(per_day))].sum(axis=0)
            if positives:
                ratios.append(caught / positives)
        positives, caught = int(group.new_detections.sum()), int(group.caught_model.sum())
        selected, missed = int(group.selected_cells.sum()), int(group.missed.sum())
        slices.append(
            {
                "lead_days": int(lead),
                "forecast_target_groups": len(group),
                "target_days": int(group.date.nunique()),
                "cell_days": int(group.cells.sum()),
                "new_detections": positives,
                "selected_cells": selected,
                "caught": caught,
                "missed": missed,
                "selected_without_detection": selected - caught,
                "capture": caught / positives if positives else None,
                "capture_interval_95": np.percentile(ratios, [2.5, 97.5]).tolist() if ratios else None,
                "illustrative_cost_per_forecast_target": (selected * review_cost + missed * missed_detection_cost) / len(group),
            }
        )
    return {
        "schema_version": 1,
        "classification": "observed_prospective_counts_with_assumed_costs",
        "first_target_day": str(pd.to_datetime(frame.date).min().date()),
        "last_target_day": str(pd.to_datetime(frame.date).max().date()),
        "cost_assumptions": {
            "per_selected_cell_review": review_cost,
            "per_missed_detection": missed_detection_cost,
            "unit": "illustrative units, not observed agency dollars",
        },
        "slices": slices,
        "automatic_retraining_or_promotion": False,
        "limits": [
            "Do not sum lead slices as unique fires or unique target days; horizons overlap",
            "New satellite activity is the existing evaluation label, not all wildfire starts",
            "1000 target-day bootstrap draws describe uncertainty; no automatic drift conclusion",
            "Selection remains the fixed top 10% rule; costs do not select a model or threshold",
        ],
    }


if __name__ == "__main__":
    result = summarize(pd.read_parquet(SOURCE))
    result["source_sha256"] = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    (ROOT / "reports/decision_monitor.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"lead_slices": len(result["slices"]), "last_target_day": result["last_target_day"]}))
