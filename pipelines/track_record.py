"""
Grade every forecast the scheduled workflow published against what satellites saw afterwards.

    python pipelines/track_record.py [--ref origin/live-data]

Reads each version of `forecast.json` from the `live-data` branch history (the
forecast exactly as the public page showed it, with its build time), keeps the
first forecast published for each station day, downloads CWFIS's daily hotspot
files for the days those forecasts covered, and scores each lead time against
new fire activity. The rules are in `src/wildfire/track_record.py`.

Writes:
    data/published/track_record.json        the grade, by lead, with bootstrap intervals
    data/published/track_record_daily.parquet  per forecast x target day: cells, new
                                             activity, and how much of it each ranking's
                                             top tenth caught

The weekly workflow (.github/workflows/track-record.yml) reruns this, so the
record grows as the season does and the README block between the track-record
markers is regenerated from the same file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "pipelines"))

import season_check as S  # noqa: E402
from wildfire import live  # noqa: E402
from wildfire import track_record as T  # noqa: E402
from wildfire.evaluation import TOP_SHARE  # noqa: E402
from wildfire.model import MODELS  # noqa: E402

PUBLISHED = ROOT / "data" / "published"
README = ROOT / "README.md"
BEGIN, END = "<!-- track-record:begin -->", "<!-- track-record:end -->"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout


def read_history(ref: str) -> list[T.Snapshot]:
    shas = git("log", "--format=%H", ref, "--", "forecast.json").split()
    snapshots = []
    for sha in shas:
        try:
            snapshots.append(T.Snapshot.from_json(sha[:7], json.loads(git("show", f"{sha}:forecast.json"))))
        except (KeyError, ValueError, subprocess.CalledProcessError) as error:
            print(f"skipped {sha[:7]}: {error}", file=sys.stderr)
    return snapshots


def labels_for(first_target: date, last_day: date, cells: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    days = [d.date() for d in pd.date_range(first_target - timedelta(days=S.LOOKBACK_DAYS), last_day)]
    jobs = [(f"{live.CWFIS}/hotspots/{d:%Y%m%d}.csv", S.RAW / "hotspots" / f"{d:%Y%m%d}.csv") for d in days]
    files = {day: S.fetch(url, path) for day, (url, path) in zip(days, jobs)}
    coverage = {"hotspot_days": sum(p is not None for p in files.values()), "hotspot_days_requested": len(files)}
    return S.new_detections(files, cells, first_target), coverage


def daily_table(graded: pd.DataFrame) -> pd.DataFrame:
    """Per forecast x target day: how much new activity each ranking's top tenth caught."""
    cut = np.ceil(TOP_SHARE * graded.groupby(["issued", "date"])["cell_id"].transform("size"))
    out = graded.assign(
        caught_model=(graded["rank_model"] <= cut) & (graded["new_detection"] == 1),
    ).groupby(["issued", "date", "lead_days", "prospective"], as_index=False).agg(
        cells=("cell_id", "size"), new_detections=("new_detection", "sum"), caught_model=("caught_model", "sum"))
    return out.astype({"lead_days": "int8", "cells": "int16", "new_detections": "int16", "caught_model": "int16"})


verdict = T.verdict  # one rule for the README and the app


def readme_block(report: dict) -> str:
    """The README's track-record section, written from the report so it cannot drift from it."""
    p = report.get("prospective")
    if not p:
        return f"{BEGIN}\n_No forecast has a graded target day yet._\n{END}"
    ci = report["prospective_intervals"]

    def pct(x: float) -> str:
        return f"{x * 100:.1f}%"

    lines = [
        BEGIN,
        f"{report['forecasts_graded']} forecasts, one per station day from {report['first_forecast']} to "
        f"{report['last_forecast']}, graded on {p['target_days']} target days through {report['last_target_day']}: "
        f"{p['cell_days']:,} prospective cell-days (leads 2-7), {p['new_detections']:,} of them new fire activity.",
        "",
        "| Leads 2-7, as published | ROC-AUC | New activity in the forecast's riskiest 10% |",
        "|---|---:|---:|",
    ]
    names = {"model": "**This model**", "large_fire_model": "Large-fire model",
             "normal_for_the_month": "Normal for the month", "fire_next_door": "Fire next door (last 48 h)"}
    for key, label in names.items():
        lines.append(f"| {label} | {p[key]['roc_auc']:.3f} | {pct(p[key]['top_decile_capture'])} |")
    calls = [verdict(ci["edge_on_normal_auc"], "normal for the month", "ranking (ROC-AUC)"),
             verdict(ci["edge_on_fire_next_door_auc"], "fire next door", "ranking (ROC-AUC)"),
             verdict(ci["edge_on_normal_capture"], "normal for the month", "the top 10%"),
             verdict(ci["edge_on_fire_next_door_capture"], "fire next door", "the top 10%")]
    lines += [
        "",
        f"**So far the published model is {calls[0]}, {calls[1]}, {calls[2]}, and {calls[3]}.** "
        f"Each call is made by the 95% interval of the edge, not by the point estimate.",
        "",
        f"95% intervals from resampling whole target days: the model's ROC-AUC "
        f"{ci['model_auc'][0]:.3f} to {ci['model_auc'][1]:.3f}; its top-10% capture "
        f"{pct(ci['model_capture'][0])} to {pct(ci['model_capture'][1])}; its edge on normal for the month "
        f"{ci['edge_on_normal_capture'][0] * 100:+.1f} to {ci['edge_on_normal_capture'][1] * 100:+.1f} points; "
        f"its edge on fire next door {ci['edge_on_fire_next_door_capture'][0] * 100:+.1f} to "
        f"{ci['edge_on_fire_next_door_capture'][1] * 100:+.1f} points.",
        "",
        "| Lead (days) | Cell-days | New activity | Model ROC-AUC | Model top 10% | Normal top 10% | Fire next door top 10% |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["by_lead"]:
        note = " (not a forecast)" if row["lead_days"] < T.PROSPECTIVE_FROM_LEAD else ""
        lines.append(
            f"| {row['lead_days']}{note} | {row['cell_days']:,} | {row['new_detections']:,} | "
            f"{row['model']['roc_auc']:.3f} | {pct(row['model']['top_decile_capture'])} | "
            f"{pct(row['normal_for_the_month']['top_decile_capture'])} | "
            f"{pct(row['fire_next_door']['top_decile_capture'])} |")
    lines.append(END)
    return "\n".join(lines)


def write_readme(report: dict, path: Path = README) -> bool:
    text = path.read_text(encoding="utf-8")
    if BEGIN not in text or END not in text:
        return False
    head, rest = text.split(BEGIN, 1)
    _, tail = rest.split(END, 1)
    updated = head + readme_block(report) + tail
    if updated != text:
        path.write_text(updated, encoding="utf-8")
    return True


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade the published forecasts.")
    parser.add_argument("--ref", default="origin/live-data", help="git ref holding the forecast history")
    parser.add_argument("--last-day", type=date.fromisoformat, default=date.today() - timedelta(days=1),
                        help="last complete hotspot day (default: yesterday)")
    args = parser.parse_args(argv)

    cells = pd.read_parquet(MODELS / "cells.parquet")
    snapshots = read_history(args.ref)
    kept, excluded = T.first_per_station_day(snapshots, set(cells["cell_id"]))
    rows = T.forecast_rows(kept)
    rows = rows[rows["date"] <= pd.Timestamp(args.last_day)]
    labels, coverage = labels_for(rows["date"].min().date(), args.last_day, cells)
    report, graded = T.grade(rows, labels)
    report |= {"generated_on": str(date.today()), "last_hotspot_day": str(args.last_day),
               "forecasts_graded": len(kept), "excluded": excluded} | coverage

    PUBLISHED.mkdir(parents=True, exist_ok=True)
    (PUBLISHED / "track_record.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    daily_table(graded).to_parquet(PUBLISHED / "track_record_daily.parquet", index=False)
    write_readme(report)

    p = report.get("prospective") or {}
    if p:
        print(f"{len(kept)} forecasts, {p['target_days']} target days, {p['new_detections']} new detections "
              f"(leads 2-7): model top-10% {p['model']['top_decile_capture']:.1%} AUC {p['model']['roc_auc']:.3f} | "
              f"normal {p['normal_for_the_month']['top_decile_capture']:.1%} | "
              f"fire next door {p['fire_next_door']['top_decile_capture']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
