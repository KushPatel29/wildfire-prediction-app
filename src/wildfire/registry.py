"""
The model as MLflow sees it: a registered, loadable version per trained model,
and the rule that decides which version is live.

A registered version is a pyfunc wrapper, not a second copy of the trees. The
boosters are 80 MB and already live in Git; the wrapper records the commit they
were trained at and the SHA-256 of every file, and on load it uses models/ when
the hashes match or reads that commit's files out of Git when they do not. Either
way a version loads only the exact files it was registered with.

The promotion rule (`gate`) is written once here and tested in
tests/test_mlflow_registry.py. It compares like with like: when a candidate was
trained and validated on the same rows as the champion it must be no worse on
the validation seasons; when the data itself changed (a label fix, new seasons),
the old champion's numbers no longer measure anything comparable, so the
candidate must instead beat both baselines on its own data. Test seasons are
never used to choose: they are scored once, after the choice.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MODEL_FILES = ("{t}.json", "{t}_counts.json", "{t}_calibration.json")

# How much worse than the champion a like-for-like candidate may be before the
# gate refuses it: nothing on the operational number, a hair on ranking and
# calibration, which move a little between otherwise identical fits.
ROC_AUC_SLACK = 0.002
BRIER_SLACK = 0.02  # relative


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def files_at(commit: str | None, target: str, root: Path = ROOT) -> dict[str, bytes]:
    """The target's model files, from the working tree (commit=None) or from Git."""
    out = {}
    for pattern in MODEL_FILES:
        name = pattern.format(t=target)
        if commit is None:
            path = root / "models" / name
            if path.is_file():
                out[name] = path.read_bytes()
        else:
            shown = subprocess.run(["git", "show", f"{commit}:models/{name}"], cwd=root,
                                   capture_output=True, check=False)
            if shown.returncode == 0:
                out[name] = shown.stdout
    return out


class DeployedModel(mlflow.pyfunc.PythonModel):
    """Loads exactly the files a version was registered with, then scores like the app."""

    def __init__(self, target: str, commit: str, hashes: dict[str, str], features: list[str]):
        self.target = target
        self.commit = commit
        self.hashes = dict(hashes)
        # The feature list this version was trained on. Earlier versions used fewer
        # features than the current wildfire.model.FEATURES, so a version scores
        # with its own list, never the current one.
        self.features = list(features)

    def load_context(self, context):  # noqa: ARG002 - the files live in Git, not in the artifact
        from wildfire.model import load

        root = Path(os.environ.get("WILDFIRE_ROOT", ROOT))
        current = files_at(None, self.target, root)
        if {name: sha256(data) for name, data in current.items()} == self.hashes:
            self._model = load(self.target, root / "models")
            return
        stored = files_at(self.commit, self.target, root)
        found = {name: sha256(data) for name, data in stored.items()}
        if found != self.hashes:
            raise RuntimeError(f"{self.target} at {self.commit}: the files in Git do not match the registered hashes")
        self._tmp = tempfile.TemporaryDirectory()
        for name, data in stored.items():
            (Path(self._tmp.name) / name).write_bytes(data)
        self._model = load(self.target, Path(self._tmp.name))

    def predict(self, context, model_input: pd.DataFrame, params=None):  # noqa: ARG002
        import numpy as np
        import xgboost as xgb

        matrix = xgb.DMatrix(model_input[self.features])
        raw = np.mean([member.probability(matrix) for member in self._model.members], axis=0)
        return pd.Series(self._model.calibration(raw).astype("float32"), name=f"p_{self.target}")


@dataclass(frozen=True)
class Decision:
    promote: bool
    comparison: str  # "like-for-like" or "data changed"
    reason: str


def _validation(metrics: dict) -> dict:
    return metrics["splits"]["validation"]


def same_data(a: dict, b: dict) -> bool:
    """Trained and validated on the same rows: the validation set has the same size
    and the same positives."""
    va, vb = _validation(a)["model"], _validation(b)["model"]
    return va["rows"] == vb["rows"] and round(va["positive_rate"] * va["rows"]) == round(vb["positive_rate"] * vb["rows"])


def beats_baselines(metrics: dict) -> tuple[bool, str]:
    v = _validation(metrics)
    model = v["model"]
    failures = [
        f"{name} {key}"
        for name in ("climatology", "fwi_logistic")
        for key in ("roc_auc", "share_of_fires_in_top_decile")
        if model[key] <= v[name][key]
    ]
    return not failures, ("beats both baselines on validation" if not failures
                          else "does not beat " + ", ".join(failures))


def gate(candidate: dict, champion: dict | None) -> Decision:
    ok, why = beats_baselines(candidate)
    if not ok:
        return Decision(False, "baselines", why)
    if champion is None:
        return Decision(True, "first version", why)
    if not same_data(candidate, champion):
        return Decision(True, "data changed",
                        "validation rows differ from the champion's, so its scores are not comparable; " + why)
    c, k = _validation(candidate)["model"], _validation(champion)["model"]
    problems = []
    if c["share_of_fires_in_top_decile"] < k["share_of_fires_in_top_decile"]:
        problems.append("fewer fires in the riskiest tenth")
    if c["roc_auc"] < k["roc_auc"] - ROC_AUC_SLACK:
        problems.append("lower ROC-AUC")
    if c["brier"] > k["brier"] * (1 + BRIER_SLACK):
        problems.append("worse Brier score")
    if problems:
        return Decision(False, "like-for-like", "worse than the champion on validation: " + ", ".join(problems))
    return Decision(True, "like-for-like",
                    f"top-decile capture {k['share_of_fires_in_top_decile']:.4f} -> {c['share_of_fires_in_top_decile']:.4f}, "
                    f"ROC-AUC {k['roc_auc']:.4f} -> {c['roc_auc']:.4f} on the same validation rows")
