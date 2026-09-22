"""
The trained model as everything downstream uses it: the feature list, the two
targets, and scoring.

Each target is two boosters read as one number. The first asks the binary
question - will this cell report a fire today - and the second models the count of
starts as a Poisson intensity, which answers the same question through
`P(at least one) = 1 - exp(-lambda)`. They disagree about different rows: the
classifier is better calibrated in the middle of the distribution, and the count
model separates a cell-day with four starts from one with one. On the validation
seasons the average puts 55.45% of fires inside the day's riskiest tenth, the count
model alone 55.41% and the classifier alone 54.86%: it is kept because it is never
worse, not because it is much better. The average is then calibrated once, so the
published number is still a probability.

Training writes, per target, `models/<target>.json` (classifier),
`models/<target>_counts.json` (Poisson) and `models/<target>_calibration.json`. The
calibration is kept as its threshold table rather than a pickled scikit-learn
object: isotonic regression is a monotone piecewise-linear map, so `np.interp` over
the thresholds reproduces it (to within 3e-5 - the thresholds were fitted in
float32), and the hosted app loads it without scikit-learn or a pickle tied to one
library version.
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
    "temp", "rh", "ws", "precip", "vpd", "ffmc", "dmc", "dc", "isi", "bui", "fwi", "dsr",
    "fwi_mean_3d", "fwi_mean_7d", "fwi_max_7d", "isi_mean_3d", "temp_mean_3d", "rh_mean_3d",
    "vpd_mean_3d", "vpd_max_7d", "bui_mean_7d", "dc_mean_30d",
    "precip_sum_3d", "precip_sum_7d", "precip_sum_14d", "precip_sum_30d",
    "days_since_rain", "dc_change_7d",
    # Today against this cell's own normal for the month, and against its neighbours
    # the same day. Absolute fire weather does not mean the same thing in two places.
    "fwi_anom", "temp_anom", "dc_anom", "fwi_neighbour", "dc_neighbour",
    "doy_sin", "doy_cos", "is_weekend", "lat", "lon", "lightning_share",
    "clim_month_rate", "clim_cell_rate", "station_km",
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


def trees_of(booster: xgb.Booster) -> tuple[int, int]:
    """The trees to predict with.

    Early stopping trains 100 rounds past the best validation score and keeps
    them. XGBoost's scikit-learn wrapper - which scored the validation seasons the
    calibration was fitted on - predicts with the trees up to the best round; a
    bare `Booster.predict` uses all of them. Scoring with every tree shifted
    probabilities by up to 0.15 against a calibration that never saw those trees,
    so this reads the best round the booster stores and stops there."""
    best = booster.attr("best_iteration")
    return (0, int(best) + 1) if best is not None else (0, 0)


@dataclass(frozen=True)
class Member:
    """One booster and how to read its output as a probability."""
    booster: xgb.Booster
    kind: str                                    # "classifier" or "counts"

    def probability(self, matrix: xgb.DMatrix) -> np.ndarray:
        raw = self.booster.predict(matrix, iteration_range=trees_of(self.booster))
        if self.kind == "counts":
            # A Poisson intensity is not a probability; the chance of at least one
            # start is what the intensity implies.
            return 1.0 - np.exp(-np.clip(raw, 0.0, None))
        return raw


@dataclass(frozen=True)
class TrainedTarget:
    name: str
    members: tuple[Member, ...]
    calibration: Calibration

    @property
    def booster(self) -> xgb.Booster:
        """The classifier, for anything that reads one model's trees (feature gain)."""
        return self.members[0].booster

    @property
    def trees(self) -> tuple[int, int]:
        return trees_of(self.booster)

    def raw(self, rows: pd.DataFrame) -> np.ndarray:
        """Uncalibrated probability: the members averaged, before isotonic calibration."""
        matrix = xgb.DMatrix(rows[FEATURES])
        return np.mean([member.probability(matrix) for member in self.members], axis=0)

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return self.calibration(self.raw(rows)).astype("float32")


def load(target: str, models: Path = MODELS) -> TrainedTarget:
    members = []
    for suffix, kind in (("", "classifier"), ("_counts", "counts")):
        path = models / f"{target}{suffix}.json"
        if not path.is_file():
            continue                              # a target trained before the count model
        booster = xgb.Booster()
        booster.load_model(path)
        members.append(Member(booster, kind))
    return TrainedTarget(target, tuple(members), Calibration.load(models / f"{target}_calibration.json"))


def load_all(models: Path = MODELS) -> dict[str, TrainedTarget]:
    return {target: load(target, models) for target in TARGETS}
