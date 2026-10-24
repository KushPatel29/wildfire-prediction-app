"""Feature failure paths are independent of an expensive registry replay."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
from release_drill import validate_features

from wildfire.model import FEATURES


def test_finite_numeric_feature_contract_accepts_reordered_columns():
    validate_features(pd.DataFrame({name: [1.0, 2.0] for name in reversed(FEATURES)}))


@pytest.mark.parametrize("problem", ["missing", "extra", "empty", "nan", "string", "duplicate"])
def test_feature_contract_rejects_unsafe_batches(problem):
    frame = pd.DataFrame({name: [1.0, 2.0] for name in FEATURES})
    if problem == "missing":
        frame = frame.drop(columns=[FEATURES[0]])
    elif problem == "extra":
        frame["unexpected"] = 1
    elif problem == "empty":
        frame = frame.iloc[:0]
    elif problem == "nan":
        frame.loc[0, FEATURES[0]] = np.nan
    elif problem == "string":
        frame[FEATURES[0]] = "invalid"
    else:
        frame = pd.concat([frame, frame[[FEATURES[0]]]], axis=1)
    with pytest.raises(ValueError):
        validate_features(frame)
