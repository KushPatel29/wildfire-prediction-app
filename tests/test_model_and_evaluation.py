"""The saved models as the app loads them, and the scoring rules the evidence uses."""

import numpy as np
import pandas as pd
import pytest
from sklearn.isotonic import IsotonicRegression

from wildfire.evaluation import rank_within, ranking, top_mask
from wildfire.model import FEATURES, Calibration, load_all


def test_the_threshold_table_reproduces_isotonic_regression(tmp_path):
    rng = np.random.default_rng(0)
    raw = rng.random(5000)
    y = (rng.random(5000) < raw ** 2).astype(int)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw, y)
    Calibration.from_isotonic(isotonic).save(tmp_path / "calibration.json")
    probe = np.concatenate([rng.random(2000), [-0.5, 0.0, 1.0, 1.5]])
    assert Calibration.load(tmp_path / "calibration.json")(probe) == pytest.approx(isotonic.predict(probe), abs=1e-12)


@pytest.fixture(scope="module")
def models():
    return load_all()


def test_both_models_predict_with_the_trees_up_to_their_best_round(models):
    """Early stopping keeps 100 rounds past the best; the calibration never saw them."""
    for target, model in models.items():
        start, stop = model.trees
        assert start == 0 and 0 < stop < model.booster.num_boosted_rounds(), target


def test_scores_are_probabilities_that_rise_with_fire_weather(models):
    mild = {"temp": 18, "rh": 55, "ws": 10, "precip": 2, "ffmc": 80, "dmc": 20, "dc": 200, "isi": 3, "bui": 30,
            "fwi": 6, "dsr": 1, "fwi_mean_3d": 6, "fwi_mean_7d": 6, "fwi_max_7d": 8, "isi_mean_3d": 3,
            "temp_mean_3d": 18, "rh_mean_3d": 55, "precip_sum_3d": 4, "precip_sum_7d": 10, "precip_sum_14d": 20,
            "days_since_rain": 1, "dc_change_7d": 10, "doy_sin": np.sin(2 * np.pi * 200 / 365.25),
            "doy_cos": np.cos(2 * np.pi * 200 / 365.25), "lat": 50.5, "lon": -120.5, "lightning_share": 0.6,
            "clim_month_rate": 0.05, "clim_cell_rate": 0.03, "station_km": 30}
    severe = dict(mild, temp=33, rh=15, ws=25, precip=0, ffmc=94, dmc=80, dc=550, isi=18, bui=110, fwi=45, dsr=30,
                  fwi_mean_3d=40, fwi_mean_7d=35, fwi_max_7d=45, isi_mean_3d=16, temp_mean_3d=32, rh_mean_3d=18,
                  precip_sum_3d=0, precip_sum_7d=0, precip_sum_14d=0, days_since_rain=20, dc_change_7d=40)
    rows = pd.DataFrame([mild, severe])[FEATURES]
    for target, model in models.items():
        p = model.predict(rows)
        assert np.all((p >= 0) & (p <= 1)), target
        assert p[1] > p[0], target


def test_the_top_tenth_is_exactly_a_tenth_when_every_score_ties():
    assert top_mask(np.ones(1000)).sum() == 100
    assert top_mask(np.ones(771)).sum() == 78


def test_ties_break_on_the_tiebreak_and_missing_scores_rank_last():
    score = np.array([0.5, 0.5, 0.5, np.nan, 0.1] + [0.0] * 5)
    tiebreak = np.array([1, 3, 2, 9, 0] + [0] * 5)
    assert np.flatnonzero(top_mask(score, tiebreak)).tolist() == [1]


def test_ranks_restart_for_each_day():
    frame = pd.DataFrame({"day": [1, 1, 1, 2, 2], "risk": [0.2, 0.9, 0.2, 0.1, 0.3], "raw": [0.3, 0.9, 0.1, 0.1, 0.3]})
    assert rank_within(frame, "day", "risk", "raw").tolist() == [2, 1, 3, 2, 1]


def test_the_top_decile_share_counts_fires_not_rows():
    y = np.array([1] + [0] * 8 + [1])
    fires = np.array([5] + [0] * 8 + [1])
    out = ranking(y, np.arange(10, dtype=float)[::-1], fires)
    assert out["share_of_fires_in_top_decile"] == pytest.approx(5 / 6)
    assert out["lift_top_decile"] == pytest.approx(5.0)
