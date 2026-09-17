"""
Export the dashboard's tables: one CSV per table, nothing recomputed.

    python pipelines/powerbi_export.py

Power BI reads CSVs from `powerbi/data/`. Every one of them is an aggregate of
something this repository has already computed and tested - the published
backtest, `reports/metrics.json`, the live forecast, the rebuilt 2024 model - so
a number on the dashboard is the same number the app shows and the tests pin. No
DAX measure re-derives a rate from its parts; where a rate is needed, both parts
are exported and the measure divides them.

The one table that is not a straight copy is the backtest: 825,000 cell-days is
not a CSV anybody should commit, so it leaves as two aggregates - by day, which
is how the capture rate is read, and by cell and season, which is how the map
is drawn. The cell-day file stays in data/published for the app.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wildfire.evaluation import TOP_SHARE  # noqa: E402

PUBLISHED = ROOT / "data" / "published"
LIVE = ROOT / "data" / "live"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
OUT = ROOT / "powerbi" / "data"

SPLIT_OF_YEAR = {**{year: "Train" for year in range(2000, 2017)},
                 **{year: "Validation" for year in range(2017, 2020)},
                 **{year: "Test" for year in range(2020, 2031)}}
PROVINCE_SHORT = {
    "British Columbia": "BC", "Alberta": "AB", "Saskatchewan": "SK", "Manitoba": "MB",
    "Ontario": "ON", "Quebec": "QC", "New Brunswick": "NB", "Nova Scotia": "NS",
    "Prince Edward Island": "PE", "Newfoundland and Labrador": "NL", "Yukon": "YT",
    "Northwest Territories": "NT", "Nunavut": "NU", "Parks Canada": "Parks",
    "Other": "Other",
}
TARGET_LABEL = {"has_fire": "Any new fire", "has_large_fire": "A fire past 200 ha"}
METHOD_LABEL = {"model": "This model", "climatology": "Normal for the month",
                "fwi_logistic": "FWI logistic regression"}


def write(frame: pd.DataFrame, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8", lineterminator="\n")
    print(f"  {name:28} {len(frame):>7,} rows x {len(frame.columns)} cols")


# --------------------------------------------------------------------------
# Dimensions
# --------------------------------------------------------------------------

def dimensions(forecast: pd.DataFrame, backtest_days: pd.DataFrame) -> None:
    cells = pd.read_parquet(MODELS / "cells.parquet")
    cells["province_short"] = cells["province"].map(PROVINCE_SHORT).fillna("Other")
    cells = cells.rename(columns={"fires": "fires_2000_2016"})
    write(cells[["cell_id", "lat", "lon", "province", "province_short",
                 "lightning_share", "fires_2000_2016"]], "dim_cell")

    provinces = pd.DataFrame({"province": sorted(set(cells["province"]) |
                                                 set(pd.read_parquet(PUBLISHED / "fires_by_year.parquet")["province"]))})
    provinces["province_short"] = provinces["province"].map(PROVINCE_SHORT).fillna("Other")
    write(provinces, "dim_province")

    first = min(backtest_days["date"].min(), forecast["date"].min())
    last = max(backtest_days["date"].max(), forecast["date"].max())
    days = pd.DataFrame({"date": pd.date_range(first, last, freq="D")})
    days["year"] = days["date"].dt.year
    days["month"] = days["date"].dt.month
    days["month_name"] = days["date"].dt.strftime("%b")
    days["month_index"] = days["month"]
    days["day_of_year"] = days["date"].dt.dayofyear
    days["in_season"] = days["month"].between(4, 10)
    days["season_split"] = days["year"].map(SPLIT_OF_YEAR).fillna("Live")
    write(days, "dim_date")

    seasons = pd.DataFrame({"year": sorted(set(range(2000, int(days["year"].max()) + 1)))})
    seasons["season_split"] = seasons["year"].map(SPLIT_OF_YEAR).fillna("Live")
    seasons["decade"] = (seasons["year"] // 10 * 10).astype(str) + "s"
    write(seasons, "dim_season")


# --------------------------------------------------------------------------
# Facts
# --------------------------------------------------------------------------

def live_forecast() -> pd.DataFrame:
    forecast = pd.read_parquet(LIVE / "forecast.parquet")
    forecast["date"] = pd.to_datetime(forecast["date"])
    forecast["risk_vs_normal"] = np.where(forecast["clim_month_rate"] > 0,
                                          forecast["risk"] / forecast["clim_month_rate"], np.nan)
    # The day's riskiest tenth is what the forecast page acts on, and it has to be
    # exactly a tenth of that day's cells - the same rule the evidence counts by.
    forecast["rank_in_day"] = (forecast.sort_values(["date", "risk"], ascending=[True, False])
                               .groupby("date").cumcount() + 1).reindex(forecast.index)
    cut = np.ceil(TOP_SHARE * forecast.groupby("date")["cell_id"].transform("size"))
    forecast["in_top_decile"] = (forecast["rank_in_day"] <= cut).astype(int)
    columns = ["cell_id", "date", "lead_days", "province", "lat", "lon", "temp", "rh", "ws",
               "precip", "fwi", "risk", "large_risk", "clim_month_rate", "risk_vs_normal",
               "rank_in_day", "in_top_decile"]
    return forecast[columns]


def backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    summary = pd.read_parquet(PUBLISHED / "backtest_summary.parquet")
    summary["date"] = pd.to_datetime(summary["date"])
    summary["year"] = summary["date"].dt.year

    daily = pd.read_parquet(PUBLISHED / "backtest_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    cut = np.ceil(TOP_SHARE * daily.groupby("date")["cell_id"].transform("size"))
    daily["flagged"] = (daily["rank_in_day"] <= cut).astype(int)
    daily["fires_caught"] = daily["fires"].where(daily["flagged"] == 1, 0)
    daily["large_caught"] = daily["large_fires"].where(daily["large_rank_in_day"] <= cut, 0)
    by_cell = daily.groupby(["cell_id", "year"]).agg(
        province=("province", "first"), lat=("lat", "first"), lon=("lon", "first"),
        days=("date", "size"), fires=("fires", "sum"), fires_caught=("fires_caught", "sum"),
        large_fires=("large_fires", "sum"), large_caught=("large_caught", "sum"),
        days_flagged=("flagged", "sum"), mean_risk=("risk", "mean"), mean_fwi=("fwi", "mean"),
        lightning_fires=("lightning_fires", "sum"), human_fires=("human_fires", "sum"),
    ).reset_index()
    return summary, by_cell


def history() -> None:
    by_year = pd.read_parquet(PUBLISHED / "fires_by_year.parquet")
    by_year["province_short"] = by_year["province"].map(PROVINCE_SHORT).fillna("Other")
    write(by_year, "fact_fires_year")
    by_month = pd.read_parquet(PUBLISHED / "fires_by_month.parquet")
    by_month["month_name"] = pd.to_datetime(by_month["month"], format="%m").dt.strftime("%b")
    by_month["province_short"] = by_month["province"].map(PROVINCE_SHORT).fillna("Other")
    write(by_month, "fact_fires_month")


def model_scores() -> None:
    metrics = json.loads((REPORTS / "metrics.json").read_text(encoding="utf-8"))
    rows, years, provinces, reliability, gains = [], [], [], [], []
    for target, info in metrics["targets"].items():
        label = TARGET_LABEL[target]
        for split, methods in info["splits"].items():
            for method, scores in methods.items():
                rows.append({"target": target, "target_label": label, "split": split.title(),
                             "method": method, "method_label": METHOD_LABEL[method],
                             "is_model": int(method == "model"), **scores})
        for row in info["by_year"]:
            years.append({"target": target, "target_label": label, **row})
        for row in info["by_province"]:
            provinces.append({"target": target, "target_label": label,
                              "province_short": PROVINCE_SHORT.get(row["province"], "Other"), **row})
        for decile, row in enumerate(info["reliability"], start=1):
            reliability.append({"target": target, "target_label": label, "decile": decile, **row})
        for feature, gain in list(info["feature_gain"].items())[:20]:
            gains.append({"target": target, "target_label": label, "feature": feature, "gain": gain})
    write(pd.DataFrame(rows), "fact_model_scores")
    write(pd.DataFrame(years), "fact_model_year")
    write(pd.DataFrame(provinces), "fact_model_province")
    write(pd.DataFrame(reliability), "fact_reliability")
    write(pd.DataFrame(gains), "fact_feature_gain")


SEASON_LABEL = {"model": "Any-fire model", "large_fire_model": "Large-fire model",
                "climatology": "Normal for the month", "fwi_alone": "FWI alone"}


def season_check() -> None:
    """The current season, graded against satellite detections rather than the NFDB."""
    path = PUBLISHED / "season_2026.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    rows = [{"year": report["year"], "method": method,
             "method_label": SEASON_LABEL.get(method, method),
             "is_model": int("model" in method),
             "same_day_top_decile": report["same_day_top_decile"][method], **scores}
            for method, scores in report["pooled"].items()]
    write(pd.DataFrame(rows), "fact_season_scores")
    months = pd.DataFrame(report["by_month"]).assign(year=report["year"])
    # A month number on an axis is read as a continuous scale and labelled 4, 6, 8.
    months["month_name"] = pd.to_datetime(months["month"], format="%m").dt.strftime("%b")
    write(months, "fact_season_month")

    # 129,000 cell-days is not a CSV anyone should commit: the dashboard reads the
    # season by day and by cell, which is how both of its visuals cut it.
    daily = pd.read_parquet(PUBLISHED / "season_2026_daily.parquet")
    daily["date"] = pd.to_datetime(daily["date"])
    by_day = daily.groupby("date").agg(
        cells=("cell_id", "size"), detections=("new_detection", "sum"),
        detections_in_top_decile=("new_detection", lambda s: int(s[daily.loc[s.index, "in_top_decile"] == 1].sum())),
        mean_risk=("risk", "mean"), mean_fwi=("fwi", "mean"),
    ).reset_index()
    write(by_day, "fact_season_day")
    by_cell = daily.groupby("cell_id").agg(
        province=("province", "first"), lat=("lat", "first"), lon=("lon", "first"),
        days=("date", "size"), detections=("new_detection", "sum"),
        days_in_top_decile=("in_top_decile", "sum"), mean_risk=("risk", "mean"),
        mean_fwi=("fwi", "mean"),
    ).reset_index()
    write(by_cell, "fact_season_cell")


def hackathon() -> None:
    path = REPORTS / "hackathon_2024.json"
    if not path.is_file():
        return
    report = json.loads(path.read_text(encoding="utf-8"))
    label = {"as_scored_in_2024": "As scored in 2024 (random split)",
             "out_of_time": "Out of time (2020-2024)",
             "out_of_time_without_the_fire_count": "As a forecast (no fire count)"}
    rows = [{"panel": panel, "scoring": name, "scoring_label": label[name],
             "sort_order": order, **scores}
            for panel, info in report["panels"].items()
            for order, (name, scores) in enumerate(info["scorings"].items(), start=1)]
    write(pd.DataFrame(rows), "fact_hackathon_scores")
    months = pd.read_parquet(PUBLISHED / "hackathon_2024.parquet")
    months = months.rename(columns={"Year": "year", "Month": "month",
                                    "Number_of_Fires": "fires", "area_ha": "actual_ha"})
    months["month_name"] = pd.to_datetime(months["month"], format="%m").dt.strftime("%b")
    months["error_ha"] = months["predicted_ha_without_count"] - months["actual_ha"]
    write(months, "fact_hackathon_month")


def main() -> int:
    print("powerbi/data:")
    forecast = live_forecast()
    summary, by_cell = backtest()
    dimensions(forecast, summary)
    write(forecast, "fact_forecast")
    write(summary, "fact_backtest_day")
    write(by_cell, "fact_backtest_cell")
    hotspots = pd.read_parquet(LIVE / "hotspots.parquet")
    keep = [c for c in ("lat", "lon", "rep_date", "sensor", "fwi", "hfi", "estarea") if c in hotspots]
    hotspots = hotspots[keep].rename(columns={"rep_date": "detected_at"})
    hotspots["detected_at"] = pd.to_datetime(hotspots["detected_at"])
    # The day is the key the date table joins on; the time of the satellite pass
    # is a label. Left together, the column types as a datetime and matches no
    # date in dim_date - a relationship that exists and filters nothing.
    hotspots.insert(2, "date", hotspots["detected_at"].dt.normalize().dt.strftime("%Y-%m-%d"))
    hotspots["detected_at"] = hotspots["detected_at"].dt.strftime("%H:%M UTC")
    write(hotspots, "fact_hotspot")
    history()
    model_scores()
    season_check()
    hackathon()
    meta = json.loads((LIVE / "meta.json").read_text(encoding="utf-8"))
    built = pd.Timestamp(meta["generated_at"]).strftime("%Y-%m-%d %H:%M UTC")
    write(pd.DataFrame([{"generated_at": built, "as_of": meta.get("as_of", ""),
                         "cells": meta["cells"], "rows": meta["rows"],
                         "hotspots": meta["hotspots"], "stations_used": meta.get("stations_used", 0)}]),
          "fact_forecast_run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
