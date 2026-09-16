"""
The trained model as everything downstream uses it: the feature list, the two
targets, and scoring.

Training writes each target as an XGBoost booster (`models/<target>.json`) and an
isotonic calibration (`models/<target>_calibration.json`). The calibration is kept
as its threshold table rather than a pickled scikit-learn object: isotonic
regression is a monotone piecewise-linear map, so `np.interp` over the thresholds
reproduces it (to within 3e-5 - the thresholds were fitted in float32), and the
hosted app loads it without scikit-learn or a pickle tied to one library version.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "models"

FEATURES = [
    "temp", "rh", "ws", "precip", "ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr",
    "fwi_mean_3d", "fwi_mean_7d", "fwi_max_7d", "isi_mean_3d", "temp_mean_3d", "rh_mean_3d",
    "precip_sum_3d", "precip_sum_7d", "precip_sum_14d", "days_since_rain", "dc_change_7d",
    "doy_sin", "doy_cos", "lat", "lon", "lightning_share", "clim_month_rate", "clim_cell_rate",
    "station_km",
]
TARGETS = {"has_fire": "any new fire", "has_large_fire": "a fire that grows past 200 ha"}


@dataclass(frozen=True)
class Calibration:
    """Raw booster score -> calibrated probability, clipped at both ends."""
    x: np.ndarray
    y: np.ndarray

    def __call__(self, raw: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(raw, dtype=float), self.x, self.y)

    @classmethod
    def from_isotonic(cls, isotonic) -> "Calibration":
        return cls(np.asarray(isotonic.X_thresholds_, dtype=float), np.asarray(isotonic.y_thresholds_, dtype=float))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps({"x": self.x.tolist(), "y": self.y.tolist()}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Calibration":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(np.asarray(payload["x"], dtype=float), np.asarray(payload["y"], dtype=float))


@dataclass(frozen=True)
class TrainedTarget:
    name: str
    booster: xgb.Booster
    calibration: Calibration

    @property
    def trees(self) -> tuple[int, int]:
        """The trees to predict with.

        Early stopping trains 100 rounds past the best validation score and keeps
        them. XGBoost's scikit-learn wrapper - which scored the validation seasons the
        calibration was fitted on - predicts with the trees up to the best round; a
        bare `Booster.predict` uses all of them. Scoring with every tree shifted
        probabilities by up to 0.15 against a calibration that never saw those trees,
        so this reads the best round the booster stores and stops there."""
        best = self.booster.attr("best_iteration")
        return (0, int(best) + 1) if best is not None else (0, 0)

    def raw(self, rows: pd.DataFrame) -> np.ndarray:
        """Uncalibrated probability: the booster's own output, before isotonic calibration."""
        return self.booster.predict(xgb.DMatrix(rows[FEATURES]), iteration_range=self.trees)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.calibration(self.raw(rows)).astype("float32")


def load(target: str, models: Path = MODELS) -> TrainedTarget:
    booster = xgb.Booster()
    booster.load_model(models / f"{target}.json")
    return TrainedTarget(target, booster, Calibration.load(models / f"{target}_calibration.json"))


def load_all(models: Path = MODELS) -> dict[str, TrainedTarget]:
    return {target: load(target, models) for target in TARGETS}
