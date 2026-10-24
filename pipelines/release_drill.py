"""Exercise MLflow promotion/rejection/rollback in a disposable registry copy.

The registered candidate reuses the current artifact: this is a release replay,
not retraining, a new accuracy result or a change to the live forecast.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

import mlflow_tracking as tracking  # noqa: E402

from mlflow import MlflowClient  # noqa: E402
from wildfire.model import FEATURES, load  # noqa: E402
from wildfire.registry import files_at, gate, sha256  # noqa: E402


@contextmanager
def copied_registry(source):
    with tempfile.TemporaryDirectory(prefix="wildfire-registry-drill-") as temp:
        registry = Path(temp) / "registry.db"
        shutil.copy2(source, registry)
        client = MlflowClient(tracking_uri=f"sqlite:///{registry.as_posix()}", registry_uri=f"sqlite:///{registry.as_posix()}")
        try:
            yield client
        finally:
            # Explicitly close SQLAlchemy pools before Windows removes the copy.
            client._tracking_client.store.engine.dispose()
            client._get_registry_client().store.engine.dispose()


def validate_features(frame: pd.DataFrame) -> None:
    if frame.empty or len(set(frame.columns)) != len(frame.columns):
        raise ValueError("A nonempty, unambiguous feature frame is required")
    if set(frame.columns) != set(FEATURES):
        raise ValueError("Feature contract mismatch")
    try:
        values = frame[FEATURES].to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Features must be numeric") from exc
    if not np.isfinite(values).all():
        raise ValueError("Features must be finite")


def guarded_switch(client, name, expected, destination, candidate, champion, hashes, observed):
    """Single-operator lab check; MLflow alias writes are not cross-client CAS."""
    if str(client.get_model_version_by_alias(name, "champion").version) != str(expected):
        raise RuntimeError("Champion changed; re-review before switching")
    if not hashes or hashes != observed:
        raise RuntimeError("Artifact integrity check failed")
    decision = gate(candidate, champion)
    if not decision.promote:
        return {"changed": False, "reason": decision.reason}
    client.set_registered_model_alias(name, "champion", str(destination))
    return {"changed": True, "reason": decision.reason}


def run_drill() -> dict:
    source = ROOT / "mlflow/mlflow.db"
    source_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    metrics = json.loads((ROOT / "reports/metrics.json").read_text(encoding="utf-8"))["targets"]
    feature_frame = pd.DataFrame(
        {feature: np.random.default_rng(index + 29).uniform(0, 30, 12) for index, feature in enumerate(FEATURES)}
    )
    validate_features(feature_frame)
    events, checks = [], {}
    with copied_registry(source) as client:
        for target, target_metrics in metrics.items():
            name = tracking.registered_name(target)
            original = client.get_model_version_by_alias(name, "champion")
            model = load(target)
            before = model.predict(feature_frame)
            if not np.isfinite(before).all() or ((before < 0) | (before > 1)).any():
                raise AssertionError("Canary predictions are invalid")
            run = client.get_run(original.tags["run_id"])
            expected = json.loads((tracking.local_artifacts(run.info.artifact_uri) / "model_files_sha256.json").read_text())
            observed = {name_: sha256(data) for name_, data in files_at(None, target).items()}
            logged = client.get_logged_model(original.source.removeprefix("models:/"))
            artifact = tracking.local_artifacts(logged.artifact_location)
            candidate = client.create_model_version(
                name,
                artifact.as_uri(),
                run_id=original.tags["run_id"],
                tags={"release_drill": "artifact replay; no new training", "run_id": original.tags["run_id"]},
            )
            degraded = deepcopy(target_metrics)
            degraded["splits"]["validation"]["model"]["share_of_fires_in_top_decile"] = 0
            rejected = guarded_switch(
                client, name, original.version, candidate.version, degraded, target_metrics, expected, observed
            )
            checks[f"{target}_degradation_rejected"] = (
                not rejected["changed"] and client.get_model_version_by_alias(name, "champion").version == original.version
            )
            try:
                guarded_switch(
                    client,
                    name,
                    original.version,
                    candidate.version,
                    target_metrics,
                    target_metrics,
                    expected,
                    {"tampered": "hash"},
                )
            except RuntimeError:
                checks[f"{target}_tampering_rejected"] = True
            else:
                checks[f"{target}_tampering_rejected"] = False
            promoted = guarded_switch(
                client, name, original.version, candidate.version, target_metrics, target_metrics, expected, observed
            )
            checks[f"{target}_replay_promoted"] = (
                promoted["changed"] and client.get_model_version_by_alias(name, "champion").version == candidate.version
            )
            rolled_back = guarded_switch(
                client, name, candidate.version, original.version, target_metrics, target_metrics, expected, observed
            )
            checks[f"{target}_rollback_restored_version"] = (
                rolled_back["changed"] and client.get_model_version_by_alias(name, "champion").version == original.version
            )
            checks[f"{target}_actual_model_canary_unchanged"] = bool(np.array_equal(before, load(target).predict(feature_frame)))
            events.append(
                {
                    "target": target,
                    "original_version": original.version,
                    "replayed_version": candidate.version,
                    "rejection": rejected,
                    "promotion": promoted,
                    "rollback": rolled_back,
                    "canary_rows": len(feature_frame),
                    "artifact_hashes": observed,
                }
            )
    checks["source_registry_unchanged"] = hashlib.sha256(source.read_bytes()).hexdigest() == source_digest
    if not all(checks.values()):
        raise AssertionError(checks)
    return {
        "schema_version": 1,
        "executed_at_utc": datetime.now(UTC).isoformat(),
        "environment": "disposable copy of local MLflow SQLite registry",
        "boundary": "Release replay of existing artifacts; no retraining, live alias change, new forecast grade or enterprise deployment. Numeric canaries test execution, not model accuracy.",
        "checks": checks,
        "events": events,
        "source_registry_sha256": source_digest,
    }


if __name__ == "__main__":
    receipt = run_drill()
    target = ROOT / "reports/release_drill.json"
    target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt["checks"], indent=2))
