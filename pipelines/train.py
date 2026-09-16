"""
Train and evaluate the daily ignition-risk model, out of time.

    python pipelines/train.py                   fit both targets, save them, evaluate
    python pipelines/train.py --evaluate-only   re-score the saved models

Splits by fire season, never by row: a random split would put a July 2018 day in
training and the July 2018 day next to it in test, and every weather feature the two
share would flatter the score. Seasons are:

    train       2000-2016   fit the model
    validation  2017-2019   stop boosting, fit the probability calibration
    test        2020-      scored once, reported as-is (when the 2020s file is present)

Two baselines stand next to it, because a model that cannot beat them is not worth
deploying:

    climatology   the cell's own historical fire rate for that month (train years only)
    FWI logistic  a logistic regression on FWI, ISI, BUI and month - roughly what a
                  fire-danger class table gives an agency today

Evaluation scores the models exactly as the app and the published evidence do -
through `wildfire.model` and `wildfire.evaluation` - so a number in metrics.json is
a number the deployed code reproduces.

Writes models/ (unless --evaluate-only) and reports/metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildfire.evaluation import probability, ranking  # noqa: E402
from wildfire.model import FEATURES, MODELS, TARGETS, Calibration, TrainedTarget, load  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"

TRAIN_YEARS = (2000, 2016)
VALID_YEARS = (2017, 2019)
TEST_FROM = 2020
BASELINE_FWI = ["fwi", "isi", "bui", "month"]


def split(table: pd.DataFrame) -> dict[str, pd.DataFrame]:
    years = table["date"].dt.year
    parts = {
        "train": table[years.between(*TRAIN_YEARS)],
        "validation": table[years.between(*VALID_YEARS)],
        "test": table[years >= TEST_FROM],
    }
    return {name: part.reset_index(drop=True) for name, part in parts.items() if len(part)}


def score(y: np.ndarray, p: np.ndarray, fires: np.ndarray | None = None, tiebreak: np.ndarray | None = None) -> dict:
    """Ranking, calibration and the number an agency acts on: of all the fires that
    started, how many fell on exactly the 10% of cell-days ranked riskiest."""
    return {"rows": int(len(y)), "positive_rate": float(np.mean(y)), **ranking(y, p, fires, tiebreak),
            **probability(y, p)}


def fit_target(parts: dict[str, pd.DataFrame], target: str) -> TrainedTarget:
    train, valid = parts["train"], parts["validation"]
    model = xgb.XGBClassifier(
        n_estimators=2000, learning_rate=0.05, max_depth=7, min_child_weight=20,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, tree_method="hist",
        eval_metric="logloss", early_stopping_rounds=100, n_jobs=-1, random_state=1904,
    )
    model.fit(train[FEATURES], train[target], eval_set=[(valid[FEATURES], valid[target])], verbose=False)
    booster = model.get_booster()
    booster.set_attr(best_iteration=str(model.best_iteration))
    raw_valid = model.predict_proba(valid[FEATURES])[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(raw_valid, valid[target])
    return TrainedTarget(target, booster, Calibration.from_isotonic(isotonic))


def evaluate_target(parts: dict[str, pd.DataFrame], target: str, trained: TrainedTarget) -> dict:
    train = parts["train"]
    medians = train[BASELINE_FWI].median()
    baseline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    baseline.fit(train[BASELINE_FWI].fillna(medians), train[target])

    report = {"best_iteration": int(trained.trees[1] - 1), "splits": {}}
    counts_column = "fires" if target == "has_fire" else "large_fires"
    scored = {}
    for name, part in parts.items():
        if name == "train":
            continue
        y, counts = part[target].to_numpy(), part[counts_column].to_numpy()
        raw = trained.raw(part)
        p = trained.calibration(raw)
        scored[name] = (raw, p)
        report["splits"][name] = {
            # Calibration flattens the booster's output into plateaus; its raw score
            # breaks the ties inside them. The baselines' ties break at random.
            "model": score(y, p, counts, tiebreak=raw),
            "climatology": score(y, part["clim_month_rate"].fillna(0).to_numpy(), counts),
            "fwi_logistic": score(y, baseline.predict_proba(part[BASELINE_FWI].fillna(medians))[:, 1], counts),
        }

    last = "test" if "test" in parts else "validation"
    held = parts[last]
    raw, p = scored[last]
    by_year = []
    for year, idx in held.groupby(held["date"].dt.year).indices.items():
        rows = held.iloc[idx]
        if rows[target].nunique() < 2:
            continue
        by_year.append({"year": int(year),
                        **score(rows[target].to_numpy(), p[idx], rows[counts_column].to_numpy(), raw[idx])})
    report["by_year"] = by_year
    by_province = []
    for province, idx in held.groupby("province").indices.items():
        rows = held.iloc[idx]
        if rows[target].sum() < 30 or rows[target].nunique() < 2:
            continue
        by_province.append({"province": province,
                            **score(rows[target].to_numpy(), p[idx], rows[counts_column].to_numpy(), raw[idx])})
    report["by_province"] = by_province
    bins = pd.qcut(p, 10, duplicates="drop")
    observed = held[target].to_numpy()
    report["reliability"] = [
        {"predicted": float(p[bins == b].mean()), "observed": float(observed[bins == b].mean()),
         "rows": int((bins == b).sum())}
        for b in bins.categories
    ]
    gain = trained.booster.get_score(importance_type="gain")
    total = sum(gain.values()) or 1.0
    report["feature_gain"] = dict(sorted(((k, v / total) for k, v in gain.items()), key=lambda kv: -kv[1]))
    report["held_out_split"] = last
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train and evaluate the fire-risk models.")
    parser.add_argument("--evaluate-only", action="store_true", help="re-score the saved models without refitting")
    args = parser.parse_args(argv)
    started = time.time()
    parts = split(pd.read_parquet(PROCESSED / "cell_days.parquet"))
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    metrics = {"features": FEATURES, "train_years": TRAIN_YEARS, "validation_years": VALID_YEARS,
               "test_from": TEST_FROM, "targets": {}}
    for target, meaning in TARGETS.items():
        if args.evaluate_only:
            trained = load(target)
        else:
            trained = fit_target(parts, target)
            trained.booster.save_model(MODELS / f"{target}.json")
            trained.calibration.save(MODELS / f"{target}_calibration.json")
        report = evaluate_target(parts, target, trained)
        metrics["targets"][target] = {"meaning": meaning, **report}
        held = report["splits"][report["held_out_split"]]
        print(f"{target}: {report['held_out_split']} ROC-AUC model {held['model']['roc_auc']:.3f} "
              f"| climatology {held['climatology']['roc_auc']:.3f} | FWI logistic {held['fwi_logistic']['roc_auc']:.3f}; "
              f"top decile holds {held['model']['share_of_fires_in_top_decile']:.1%} of fires", flush=True)
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"done in {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
