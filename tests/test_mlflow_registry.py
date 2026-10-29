"""The MLflow store committed in mlflow/ against the files it describes.

Every trained version of the model is a training run and a registered version;
the version aliased `champion` must be the files the app serves and must score
exactly as the app does; the promotion gate must say what it decided and why;
and the monitoring run must be the track record the weekly workflow graded.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "pipelines"))

import mlflow  # noqa: E402
import mlflow_tracking as mt  # noqa: E402

from wildfire.model import FEATURES, TARGETS, load  # noqa: E402
from wildfire.registry import files_at, gate, sha256  # noqa: E402

METRICS = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
SUMMARY = json.loads((ROOT / "mlflow" / "summary.json").read_text(encoding="utf-8"))
TRACK = json.loads((ROOT / "data" / "published" / "track_record.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def client():
    return mt.client()


def _champion(client, target):
    return client.get_model_version_by_alias(mt.registered_name(target), "champion")


def test_every_target_has_a_champion_and_three_versions():
    for target in TARGETS:
        model = SUMMARY["registered_models"][mt.registered_name(target)]
        assert [v["version"] for v in model["versions"]] == [1, 2, 3]
        assert model["champion"] == 3


def test_the_champion_is_the_files_the_app_serves(client):
    for target in TARGETS:
        version = _champion(client, target)
        run = client.get_run(version.tags["run_id"])
        recorded = json.loads((mt.local_artifacts(run.info.artifact_uri) / "model_files_sha256.json").read_text())
        assert recorded == {name: sha256(data) for name, data in files_at(None, target).items()}


def test_the_champion_scores_exactly_like_the_app():
    rows = pd.DataFrame({f: np.random.default_rng(11).uniform(0, 30, 8) for f in FEATURES})
    for target in TARGETS:
        served = mt.load_version(target, "champion").predict(rows).to_numpy()
        assert np.array_equal(served, load(target).predict(rows)), target


def test_the_champion_training_run_holds_the_published_evaluation(client):
    for target in TARGETS:
        run = client.get_run(_champion(client, target).tags["run_id"])
        for split in ("validation", "test"):
            for who, values in METRICS["targets"][target]["splits"][split].items():
                for key, value in values.items():
                    if isinstance(value, (int, float)):
                        assert run.data.metrics[f"{split}_{who}_{key}"] == pytest.approx(value), (split, who, key)
        assert run.data.params["best_iteration"] == str(METRICS["targets"][target]["best_iteration"])


def test_the_gate_recorded_each_promotion_and_its_reason():
    for target in TARGETS:
        versions = SUMMARY["registered_models"][mt.registered_name(target)]["versions"]
        assert [v["comparison"] for v in versions] == ["first version", "like-for-like", "data changed"]
        assert {v["gate"] for v in versions} == {"promoted"}
        assert "same validation rows" in versions[1]["reason"]
        assert "not comparable" in versions[2]["reason"]


def _metrics(rows=1000, positives=25, roc=0.85, top=0.6, brier=0.02, clim=(0.8, 0.5), fwi=(0.7, 0.35)):
    model = {"rows": rows, "positive_rate": positives / rows, "roc_auc": roc,
             "share_of_fires_in_top_decile": top, "brier": brier}
    return {"splits": {"validation": {
        "model": model,
        "climatology": {"roc_auc": clim[0], "share_of_fires_in_top_decile": clim[1]},
        "fwi_logistic": {"roc_auc": fwi[0], "share_of_fires_in_top_decile": fwi[1]},
    }}}


def test_the_gate_rules():
    champion = _metrics()
    assert gate(_metrics(), None).comparison == "first version"
    assert gate(_metrics(top=0.61, roc=0.851), champion).promote
    assert not gate(_metrics(top=0.59), champion).promote, "fewer fires in the riskiest tenth"
    assert not gate(_metrics(roc=0.84), champion).promote, "ROC-AUC beyond the slack"
    assert gate(_metrics(roc=0.849), champion).promote, "inside the ROC-AUC slack"
    assert not gate(_metrics(brier=0.0205), champion).promote, "calibration more than 2% worse"
    changed = gate(_metrics(rows=900, top=0.55, roc=0.82), champion)
    assert changed.promote and changed.comparison == "data changed"
    assert not gate(_metrics(top=0.45), None).promote, "must beat the climatology baseline"


def test_the_monitoring_run_is_the_published_track_record(client):
    experiment = client.get_experiment_by_name(mt.PROSPECTIVE)
    runs = client.search_runs([experiment.experiment_id], f"tags.last_target_day = '{TRACK['last_target_day']}'")
    assert len(runs) == 1
    m = runs[0].data.metrics
    p = TRACK["prospective"]
    assert m["model_roc_auc"] == pytest.approx(p["model"]["roc_auc"])
    assert m["model_top_decile_capture"] == pytest.approx(p["model"]["top_decile_capture"])
    assert m["normal_for_the_month_roc_auc"] == pytest.approx(p["normal_for_the_month"]["roc_auc"])
    assert (m["model_auc_low"], m["model_auc_high"]) == pytest.approx(tuple(TRACK["prospective_intervals"]["model_auc"]))
    champion = _champion(client, "has_fire")
    assert runs[0].data.tags["champion_has_fire"] == f"v{champion.version} @ {champion.tags['git_commit'][:7]}"


def test_the_summary_is_what_the_store_holds():
    assert mt.summary() == SUMMARY


def _full_history() -> bool:
    shallow = subprocess.run(["git", "rev-parse", "--is-shallow-repository"], cwd=ROOT,
                             capture_output=True, text=True, check=False).stdout.strip()
    return shallow == "false"


@pytest.mark.skipif(not _full_history(), reason="older versions load their files from Git history")
def test_an_older_version_loads_its_own_files_from_git():
    rows = pd.DataFrame({f: np.random.default_rng(3).uniform(0, 30, 8) for f in FEATURES})
    first = mt.load_version("has_fire", 1).predict(rows).to_numpy()
    assert not np.allclose(first, load("has_fire").predict(rows))


def _locations(db: Path) -> list[str]:
    with sqlite3.connect(db) as con:
        return [value for table, column in mt.LOCATION_COLUMNS
                for (value,) in con.execute(f"SELECT {column} FROM {table}") if value]


def test_a_store_opened_in_another_checkout_writes_its_artifacts_there(tmp_path):
    """The weekly grade failed on 2026-10-26: the committed store named the folders of the
    machine that wrote it, and the runner was denied when it tried to create them."""
    checkout = tmp_path / "another-checkout"
    shutil.copytree(mt.STORE, checkout / "mlflow")
    db = checkout / "mlflow" / "mlflow.db"
    assert mt.relocate(db, root=checkout) > 0
    assert all(str(checkout.resolve().as_posix()) in value for value in _locations(db))
    assert mt.relocate(db, root=checkout) == 0, "relocating twice changes nothing"

    c = mlflow.MlflowClient(tracking_uri=f"sqlite:///{db}")
    try:
        experiment = c.get_experiment_by_name(mt.PROSPECTIVE).experiment_id
        run = c.create_run(experiment)
        c.log_dict(run.info.run_id, {"ok": True}, "probe.json")
        assert run.info.artifact_uri.startswith(checkout.resolve().as_uri())
        assert list((checkout / "mlflow" / "artifacts").rglob("probe.json"))
    finally:
        c._tracking_client.store.engine.dispose()
