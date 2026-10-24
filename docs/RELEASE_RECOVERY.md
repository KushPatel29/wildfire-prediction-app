# Model release replay and recovery

`python pipelines/release_drill.py` exercises the actual MLflow alias APIs against
a disposable copy of the committed registry. It rejects degraded validation
capture and incorrect artifact hashes, promotes a replay of the current artifact,
rolls back to the original registered version and verifies unchanged numeric
canaries through the actual XGBoost models. The source registry is hashed before
and after. [Execution receipt](../reports/release_drill.json).

This is a release/recovery exercise, not new training or a different live model.
The existing validation-only promotion policy remains the selection rule; no test
season or prospective outcome chooses the replay. Feature contract checks reject
missing, duplicate, extra, nonnumeric and nonfinite inputs before a batch is scored.
The numeric canary rows test execution only; they are not observed wildfire cases.

MLflow alias updates are not atomic compare-and-swap across operators. The helper
detects an already-changed champion but assumes a single release operator. A real
shared service would need an external deployment lock, access controls, monitored
serving and a separately authorized rollback policy. No production on-call or
automatic retraining is claimed. The live forecast, grading history and original
registry remain unchanged.

## Decisions and uncertainty by lead

`python pipelines/decision_monitor.py` reads the existing prospective daily
grade, excludes nonprospective and leads 0–1, reconciles selected/caught/missed
cell counts and reports leads 2–7 separately. The
[decision monitor](../reports/decision_monitor.json) attaches whole-target-day
bootstrap intervals and explicit review/miss cost assumptions. Those units are
illustrative, not agency dollars. Overlapping horizons are not unique fires.
The fixed selection rule, model and published grade do not change; neither cost
sensitivity nor a short slice automatically triggers retraining or promotion.
