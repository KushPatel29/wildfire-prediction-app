"""
Publish the small, deployable evidence the app reads.

    python pipelines/publish.py

The modelling table is 4 million rows and stays on the machine that built it. The
hosted app needs a few megabytes: the grid, the model, and pre-computed evidence -
fire history, the out-of-time backtest, and daily risk maps for the test seasons so
a reader can replay a real season against the fires that actually started.

Writes data/published/:

    fires_by_year.parquet        starts and area burned by year, province and cause
    fires_by_month.parquet       seasonality by province
    backtest_daily.parquet       every test cell-day: predicted risk, its rank that day, fires started
    backtest_summary.parquet     per day: fires that started inside exactly the riskiest 10% of cells
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

from train import TEST_FROM  # noqa: E402
from wildfire import features as F  # noqa: E402
from wildfire.evaluation import TOP_SHARE, rank_within  # noqa: E402
from wildfire.model import load_all  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
PUBLISHED = ROOT / "data" / "published"
CAUSES = {"N": "Lightning", "H": "Human", "U": "Unknown"}


def daily_summary(daily: pd.DataFrame) -> pd.DataFrame:
    """For each day: rank that day's cells and count the fires that started inside
    exactly the riskiest tenth - the question a duty officer answers when deciding
    where to pre-position crews that morning."""
    cut = np.ceil(TOP_SHARE * daily.groupby("date")["cell_id"].transform("size"))
    flagged = daily.assign(
        top_fires=daily["fires"].where(daily["rank_in_day"] <= cut, 0),
        top_large=daily["large_fires"].where(daily["large_rank_in_day"] <= cut, 0),
    )
    summary = flagged.groupby("date").agg(
        cells=("cell_id", "size"), fires=("fires", "sum"), fires_in_top_decile=("top_fires", "sum"),
        large_fires=("large_fires", "sum"), large_in_top_decile=("top_large", "sum"),
        mean_risk=("risk", "mean"), mean_fwi=("fwi", "mean"),
    ).reset_index()
    counts = ["cells", "fires", "fires_in_top_decile", "large_fires", "large_in_top_decile"]
    return summary.astype({column: "int32" for column in counts})


def main() -> int:
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    fires = pd.read_parquet(PROCESSED / "fires.parquet")
    fires["province"] = fires["src_agency"].map({**F.AGENCIES, "PC": "Parks Canada"}).fillna("Other")
    fires["cause_label"] = fires["cause"].map(CAUSES).fillna("Unknown")
    fires["year"] = fires["date"].dt.year
    fires["month"] = fires["date"].dt.month
    by_year = fires.groupby(["year", "province", "cause_label"]).agg(
        fires=("nfdbfireid", "size"), area_ha=("size_ha", "sum"),
        large_fires=("size_ha", lambda s: int((s >= F.LARGE_FIRE_HA).sum()))).reset_index()
    by_year.to_parquet(PUBLISHED / "fires_by_year.parquet", index=False)
    by_month = fires.groupby(["province", "month"]).agg(
        fires=("nfdbfireid", "size"), area_ha=("size_ha", "sum")).reset_index()
    by_month.to_parquet(PUBLISHED / "fires_by_month.parquet", index=False)

    table = pd.read_parquet(PROCESSED / "cell_days.parquet")
    test = table[table["date"].dt.year >= TEST_FROM].reset_index(drop=True)
    models = load_all()
    for target, column in (("has_fire", "risk"), ("has_large_fire", "large_risk")):
        raw = models[target].raw(test)
        test[column] = models[target].calibration(raw).astype("float32")
        test[f"{column}_raw"] = raw
    test["rank_in_day"] = rank_within(test, "date", "risk", "risk_raw")
    test["large_rank_in_day"] = rank_within(test, "date", "large_risk", "large_risk_raw")

    daily = test[["cell_id", "date", "province", "lat", "lon", "fwi", "risk", "large_risk", "rank_in_day",
                  "large_rank_in_day", "fires", "lightning_fires", "human_fires", "large_fires"]].copy()
    for column in ("lat", "lon", "fwi"):
        daily[column] = daily[column].astype("float32")
    for column in ("rank_in_day", "large_rank_in_day", "fires", "lightning_fires", "human_fires", "large_fires"):
        daily[column] = daily[column].astype("int16")
    daily.to_parquet(PUBLISHED / "backtest_daily.parquet", index=False)
    summary = daily_summary(daily)
    summary.to_parquet(PUBLISHED / "backtest_summary.parquet", index=False)

    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text(encoding="utf-8"))
    total = int(summary["fires"].sum())
    print(f"published: {len(by_year):,} history rows, {len(daily):,} backtest cell-days, {len(summary)} days; "
          f"same-day top-decile capture {summary['fires_in_top_decile'].sum() / max(total, 1):.1%} of {total:,} "
          f"test fires; model test AUC {metrics['targets']['has_fire']['splits']['test']['model']['roc_auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
