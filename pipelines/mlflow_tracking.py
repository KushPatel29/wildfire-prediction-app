"""
The model's lifecycle in MLflow: every trained version, which one is live, and how
the live one has done since.

    python pipelines/mlflow_tracking.py register   one training run per trained model in Git history,
                                                   registered, and promoted only through the gate
    python pipelines/mlflow_tracking.py monitor    the current track record as a monitoring run
    python pipelines/mlflow_tracking.py summary    write mlflow/summary.json from the store

All three are idempotent: a commit or a grading already in the store is skipped.
The store is committed (mlflow/mlflow.db, artifacts in mlflow/artifacts), so

    mlflow ui --backend-store-uri sqlite:///mlflow/mlflow.db

browses the same history the tests read. The Track record workflow runs
`monitor` and `summary` after each weekly grading and commits the result.

Training runs are recorded from Git, not re-fitted: every version of
reports/metrics.json is the evaluation train.py wrote when that model was fitted,
and the model files at the same commit are the ones it fitted. Re-fitting needs
the 4.0M-row cell-day table built from the National Fire Database, which is not
in the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402
from mlflow.models import ModelSignature  # noqa: E402
from mlflow.types import ColSpec, Schema  # noqa: E402

from wildfire.model import TARGETS  # noqa: E402
from wildfire.registry import DeployedModel, files_at, gate, sha256  # noqa: E402

STORE = ROOT / "mlflow"
TRACKING_URI = f"sqlite:///{STORE / 'mlflow.db'}"
ARTIFACTS = (STORE / "artifacts").as_uri()
TRAINING = "wildfire-training"
PROSPECTIVE = "wildfire-prospective"
REQUIREMENTS = ["xgboost==2.0.3", "numpy==2.5.3", "pandas==3.0.3", "scikit-learn==1.9.0"]


def registered_name(target: str) -> str:
    return f"wildfire-{target.replace('_', '-')}"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()


def client() -> MlflowClient:
    STORE.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_registry_uri(TRACKING_URI)
    return MlflowClient()


def experiment(c: MlflowClient, name: str) -> str:
    found = c.get_experiment_by_name(name)
    return found.experiment_id if found else c.create_experiment(name, artifact_location=f"{ARTIFACTS}/{name}")


def metric_commits() -> list[str]:
    """Every commit that changed reports/metrics.json, oldest first."""
    return git("log", "--reverse", "--format=%H", "--", "reports/metrics.json").split()


def flat(prefix: str, metrics: dict) -> dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in metrics.items() if isinstance(v, (int, float))}


def register() -> None:
    c = client()
    exp = experiment(c, TRAINING)
    for commit in metric_commits():
        metrics = json.loads(git("show", f"{commit}:reports/metrics.json"))
        subject = git("show", "-s", "--format=%s", commit)
        date = git("show", "-s", "--format=%cI", commit)
        for target in TARGETS:
            name = registered_name(target)
            done = c.search_runs([exp], f"tags.git_commit = '{commit}' and tags.target = '{target}'")
            if done:
                continue
            m = metrics["targets"][target]
            hashes = {fname: sha256(data) for fname, data in files_at(commit, target).items()}
            with mlflow.start_run(experiment_id=exp, run_name=f"{target} @ {commit[:7]}") as run:
                mlflow.set_tags({"git_commit": commit, "commit_subject": subject, "committed_at": date,
                                 "target": target, "meaning": m["meaning"],
                                 "recorded_from": "reports/metrics.json written by train.py at this commit"})
                mlflow.log_params({"train_years": "-".join(map(str, metrics["train_years"])),
                                   "validation_years": "-".join(map(str, metrics["validation_years"])),
                                   "test_from": metrics["test_from"], "features": len(metrics["features"]),
                                   "best_iteration": m["best_iteration"],
                                   "members": ",".join(f"{k}:{v}" for k, v in m.get("members", {"classifier": m["best_iteration"]}).items())})
                for split in ("validation", "test"):
                    for who, values in m["splits"][split].items():
                        mlflow.log_metrics(flat(f"{split}_{who}", values))
                mlflow.log_dict(m["splits"], "evaluation.json")
                mlflow.log_dict(hashes, "model_files_sha256.json")
                signature = ModelSignature(inputs=Schema([ColSpec("double", f) for f in metrics["features"]]),
                                           outputs=Schema([ColSpec("float")]))
                info = mlflow.pyfunc.log_model(name="model",
                                               python_model=DeployedModel(target, commit, hashes, metrics["features"]),
                                               signature=signature, pip_requirements=REQUIREMENTS)
            version = mlflow.register_model(info.model_uri, name, tags={"git_commit": commit})
            champion = champion_metrics(c, name)
            decision = gate(m, champion)
            c.set_model_version_tag(name, version.version, "gate", "promoted" if decision.promote else "rejected")
            c.set_model_version_tag(name, version.version, "gate_comparison", decision.comparison)
            c.set_model_version_tag(name, version.version, "gate_reason", decision.reason)
            c.set_model_version_tag(name, version.version, "run_id", run.info.run_id)
            if decision.promote:
                c.set_registered_model_alias(name, "champion", version.version)
            print(f"{name} v{version.version} ({commit[:7]}): {'promoted' if decision.promote else 'rejected'} - {decision.reason}")


def champion_metrics(c: MlflowClient, name: str) -> dict | None:
    try:
        version = c.get_model_version_by_alias(name, "champion")
    except mlflow.exceptions.MlflowException:
        return None
    commit = version.tags["git_commit"]
    target = name.removeprefix("wildfire-").replace("-", "_")
    return json.loads(git("show", f"{commit}:reports/metrics.json"))["targets"][target]


def local_artifacts(uri: str) -> Path:
    """The store records absolute artifact paths from the machine that wrote them;
    read them relative to this checkout instead, so a fresh clone can load a model."""
    path = uri.removeprefix("file://")
    return STORE / "artifacts" / path.split("/mlflow/artifacts/", 1)[1]


def load_version(target: str, alias_or_version: str | int = "champion"):
    """A registered version as a pyfunc, loaded from this checkout's store."""
    c = client()
    name = registered_name(target)
    if isinstance(alias_or_version, str):
        version = c.get_model_version_by_alias(name, alias_or_version)
    else:
        version = c.get_model_version(name, str(alias_or_version))
    logged = c.get_logged_model(version.source.removeprefix("models:/"))
    return mlflow.pyfunc.load_model(str(local_artifacts(logged.artifact_location)))


def monitor() -> None:
    c = client()
    exp = experiment(c, PROSPECTIVE)
    record = json.loads((ROOT / "data" / "published" / "track_record.json").read_text(encoding="utf-8"))
    graded = record["last_target_day"]
    if c.search_runs([exp], f"tags.last_target_day = '{graded}'"):
        print(f"track record through {graded} already logged")
        return
    champions = {t: c.get_model_version_by_alias(registered_name(t), "champion") for t in TARGETS}
    p, ci = record["prospective"], record["prospective_intervals"]
    with mlflow.start_run(experiment_id=exp, run_name=f"graded through {graded}"):
        mlflow.set_tags({"last_target_day": graded, "first_target_day": record["first_target_day"],
                         "generated_on": record["generated_on"], "rule": json.dumps(record["rule"]),
                         **{f"champion_{t}": f"v{v.version} @ {v.tags['git_commit'][:7]}" for t, v in champions.items()}})
        mlflow.log_metrics({"target_days": p["target_days"], "cell_days": p["cell_days"],
                            "new_detections": p["new_detections"], "forecasts_graded": record["forecasts_graded"]})
        for who in ("model", "large_fire_model", "normal_for_the_month", "fire_next_door"):
            mlflow.log_metrics(flat(who, p[who]))
        for key, (low, high) in ((k, v) for k, v in ci.items() if isinstance(v, list)):
            mlflow.log_metrics({f"{key}_low": low, f"{key}_high": high})
        mlflow.log_dict(record, "track_record.json")
    print(f"logged the track record through {graded}")


def summary() -> dict:
    c = client()
    out = {"tracking_uri": "sqlite:///mlflow/mlflow.db", "registered_models": {}, "training_runs": [],
           "monitoring_runs": []}
    for target in TARGETS:
        name = registered_name(target)
        champion = c.get_model_version_by_alias(name, "champion")
        versions = sorted(c.search_model_versions(f"name = '{name}'"), key=lambda v: int(v.version))
        out["registered_models"][name] = {
            "champion": int(champion.version),
            "versions": [{"version": int(v.version), "git_commit": v.tags["git_commit"], "gate": v.tags["gate"],
                          "comparison": v.tags["gate_comparison"], "reason": v.tags["gate_reason"],
                          "run_id": v.tags["run_id"]} for v in versions],
        }
    for run in c.search_runs([c.get_experiment_by_name(TRAINING).experiment_id], order_by=["attributes.start_time ASC"]):
        out["training_runs"].append({
            "run_id": run.info.run_id, "target": run.data.tags["target"], "git_commit": run.data.tags["git_commit"],
            "commit_subject": run.data.tags["commit_subject"],
            **{k: run.data.metrics[k] for k in ("validation_model_roc_auc", "validation_model_share_of_fires_in_top_decile",
                                                 "test_model_roc_auc", "test_model_share_of_fires_in_top_decile")},
        })
    mon = c.get_experiment_by_name(PROSPECTIVE)
    for run in c.search_runs([mon.experiment_id], order_by=["attributes.start_time ASC"]) if mon else []:
        out["monitoring_runs"].append({
            "run_id": run.info.run_id, "last_target_day": run.data.tags["last_target_day"],
            "champion_has_fire": run.data.tags["champion_has_fire"],
            **{k: run.data.metrics[k] for k in ("model_roc_auc", "model_top_decile_capture", "model_auc_low", "model_auc_high",
                                                 "normal_for_the_month_roc_auc", "target_days")},
        })
    (STORE / "summary.json").write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["register", "monitor", "summary"])
    args = parser.parse_args(argv)
    {"register": register, "monitor": monitor, "summary": summary}[args.command]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
