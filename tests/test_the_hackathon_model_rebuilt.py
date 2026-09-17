"""
The 2024 model has to stay the 2024 model.

`pipelines/hackathon_2024.py` exists to be compared against, which only works if
nobody quietly improves it. A random forest with the notebook's own hyperparameters
on the notebook's own eight features is the comparison; the same forest with better
features is a different model wearing its name, and the comparison would then say
nothing at all.

So this pins the rebuild to the notebook, and pins the two findings that come out of
re-scoring it: a random split over months scores far better than the same model
scored out of time, and the feature it leans on hardest is one that is not known
until the month it is predicting is over.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines"))

import hackathon_2024 as legacy  # noqa: E402

REPORT = json.loads((ROOT / "reports" / "hackathon_2024.json").read_text(encoding="utf-8"))
PUBLISHED = pd.read_parquet(ROOT / "data" / "published" / "hackathon_2024.parquet")
PANEL = REPORT["panels"]["nfdb"]


def test_the_rebuild_keeps_the_notebook_hyperparameters():
    """RandomizedSearchCV chose these in February 2024. They are the baseline."""
    assert legacy.FOREST == {"n_estimators": 1200, "min_samples_split": 10,
                             "min_samples_leaf": 2, "max_features": "sqrt",
                             "max_depth": 20, "random_state": 42}
    assert REPORT["model"]["kind"] == "RandomForestRegressor"


def test_the_rebuild_keeps_the_notebook_feature_set():
    """`fs1` in Wildfire_predictive_analysis.ipynb, in its own order."""
    assert legacy.FEATURES_2024 == ["Number_of_Fires", "tempmin", "tempmax", "temp",
                                    "precip", "humidity", "Month", "windspeed"]
    assert REPORT["features"] == legacy.FEATURES_2024


def test_a_random_split_over_months_flatters_the_model():
    """The finding. Splitting months at random puts June 2015 in training and June
    2016 in test, and a fire season is strongly autocorrelated - so the score
    measures how well the model interpolates a year it has already seen."""
    random_split = PANEL["scorings"]["as_scored_in_2024"]["r2"]
    out_of_time = PANEL["scorings"]["out_of_time"]["r2"]
    assert random_split > out_of_time + 0.1, (random_split, out_of_time)


def test_the_model_leans_on_a_number_it_would_not_have_in_advance():
    """`Number_of_Fires` counts the fires in the month whose area is being predicted.
    A forecast cannot have it. Dropping it is what makes the comparison a forecast,
    and it costs the model measurably - which is the point."""
    with_count = PANEL["scorings"]["out_of_time"]["r2"]
    without_count = PANEL["scorings"]["out_of_time_without_the_fire_count"]["r2"]
    assert without_count < with_count
    assert "Number_of_Fires" in legacy.FEATURES_2024


def test_both_panels_are_scored_and_neither_is_presented_as_the_other():
    """The tables the 2024 project read stop in 2021, leaving twelve months to score
    out of time; the point layer this repository builds runs to 2024. Both are
    reported, with their windows stated."""
    assert set(REPORT["panels"]) == {"legacy", "nfdb"}
    assert REPORT["panels"]["legacy"]["years"][1] <= 2021
    assert REPORT["panels"]["nfdb"]["years"] == [2000, 2024]
    assert REPORT["panels"]["legacy"]["scorings"]["out_of_time"]["rows_test"] <= 12
    assert REPORT["panels"]["nfdb"]["scorings"]["out_of_time"]["rows_test"] == 60


def test_the_published_months_are_the_scored_months():
    assert len(PUBLISHED) == PANEL["scorings"]["out_of_time"]["rows_test"]
    assert PUBLISHED["Year"].between(2020, 2024).all()
    assert PUBLISHED.groupby("Year")["Month"].nunique().eq(12).all()
    assert (PUBLISHED["predicted_ha"] >= 0).all()


def test_the_panel_holds_whole_years_only():
    """A January with a year of missing months behind it would sit in the test set
    as if it were a year."""
    assert PANEL["rows"] == 299                     # 2000-01 is dropped: no 30-day history
    assert PANEL["years"] == [2000, 2024]


def test_the_error_is_stated_in_hectares_as_well_as_r2():
    """R2 on a right-skewed target is easy to misread; the mean absolute error says
    how wrong the answer is in the unit the fire agencies use."""
    for scoring in PANEL["scorings"].values():
        assert scoring["mae"] > 0
        assert scoring["rmse"] >= scoring["mae"]
        assert scoring["mean_observed_ha"] > 0


def test_todays_model_is_not_scored_on_this_target():
    """Guardrail for the comparison page: the two models answer different questions -
    hectares burned in a month nationally, against which cell reports a fire today -
    so nothing may quietly compare their R2 with an ROC-AUC."""
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    assert "r2" not in json.dumps(metrics["targets"]["has_fire"]["splits"]["test"]["model"])
    assert REPORT["target"].startswith("hectares burned")
