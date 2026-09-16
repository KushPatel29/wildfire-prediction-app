"""
The 2024 hackathon model, rebuilt and re-scored.

    python pipelines/hackathon_2024.py

Team 1904 Coders won with a monthly model: join national fire statistics to
monthly weather, and fit a random forest to predict **hectares burned** from
temperature, humidity, wind, rain, the month, and the number of fires. The
notebooks are kept unchanged in `legacy/2024-hackathon/`; this rebuilds that model
from the same published tables and scores it three ways.

It is here for two reasons. The first is that it is the honest ancestor of the
model in `pipelines/train.py`, and a reader deserves to see what changed rather
than a claim that something changed. The second is that the way it was scored -
a random 80/20 split over months - is the single most common way a time-series
model flatters itself, and having both numbers side by side says more than either.

Three scorings:

    as scored in 2024   random 80/20 split over month rows, R2 / MAE / RMSE
    out of time         fit on months up to 2016, scored on 2020 onward
    without the count   the same, with the contemporaneous fire count dropped

The third matters most. `Number_of_Fires` is the count of fires in the month whose
area is being predicted - it is not known in advance, so a model that leans on it
cannot forecast anything. Dropping it is what separates "explains the past" from
"predicts next month", and the gap between those two R2 values is the finding.

Weather is reconstructed from the CWFIS station archives rather than the single
weather series the original notebook read (`Weather_area.csv`, which is not in the
repository): the national mean over every reporting station each month, with
tempmax and tempmin as that month's warmest and coldest station days. Different
source, same shape - and stated here rather than implied.

Writes reports/hackathon_2024.json and data/published/hackathon_2024.parquet.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

LEGACY = ROOT / "legacy" / "2024-hackathon"
INTERIM = ROOT / "data" / "interim"
PUBLISHED = ROOT / "data" / "published"
REPORTS = ROOT / "reports"

#: The hyperparameters the notebook's RandomizedSearchCV settled on, kept exactly.
FOREST = dict(n_estimators=1200, min_samples_split=10, min_samples_leaf=2,
              max_features="sqrt", max_depth=20, random_state=42)
#: `fs1` in Wildfire_predictive_analysis.ipynb, in the notebook's own order.
FEATURES_2024 = ["Number_of_Fires", "tempmin", "tempmax", "temp", "precip", "humidity",
                 "Month", "windspeed"]
TARGET = "area_ha"
SEED = 42
TRAIN_TO = 2016
TEST_FROM = 2020
LAST_YEAR = 2024
MONTHS = {name: number for number, name in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August",
     "September", "October", "November", "December"], start=1)}


def nfd_monthly() -> pd.DataFrame:
    """National fires and hectares burned per calendar month, from the NFD tables
    the 2024 project read."""
    frames = {}
    for label, filename, column in (
            ("fires", "NFD - Number of fires by month - EN FR.csv", "Number"),
            ("area_ha", "NFD - Area burned by month - EN FR.csv", "Area (hectares)")):
        frame = pd.read_csv(LEGACY / filename, encoding="latin-1")
        frame["Month"] = frame["Month"].map(MONTHS)
        frame = frame.dropna(subset=["Month"])          # "Unspecified" carries no month
        frames[label] = (frame.groupby(["Year", "Month"])[column].sum()
                         .rename(label).reset_index())
    monthly = frames["fires"].merge(frames["area_ha"], on=["Year", "Month"], how="inner")
    return monthly.rename(columns={"fires": "Number_of_Fires"})


def station_monthly() -> pd.DataFrame:
    """Monthly national weather from the CWFIS station archives."""
    frames = []
    for path in sorted(INTERIM.glob("cwfis_fwi_*.parquet")):
        stations = pd.read_parquet(path, columns=["rep_date", "temp", "rh", "ws", "precip"])
        stations = stations.dropna(subset=["rep_date"])
        for column in ("temp", "rh", "ws", "precip"):
            stations[column] = pd.to_numeric(stations[column], errors="coerce")
        stations["Year"] = stations["rep_date"].dt.year
        stations["Month"] = stations["rep_date"].dt.month
        frames.append(stations.groupby(["Year", "Month"]).agg(
            temp=("temp", "mean"), tempmax=("temp", "max"), tempmin=("temp", "min"),
            humidity=("rh", "mean"), windspeed=("ws", "mean"), precip=("precip", "mean"),
        ).reset_index())
    weather = pd.concat(frames, ignore_index=True)
    return weather.groupby(["Year", "Month"], as_index=False).mean()


def nfdb_monthly() -> pd.DataFrame:
    """The same monthly shape from the National Fire Database point layer.

    The published tables the 2024 project read stop in 2021, which leaves twelve
    months to score out of time - too few to carry a comparison. The point layer
    this repository already builds runs to 2024 and aggregates to the same two
    columns, so the out-of-time window matches the one the current model is scored
    on (2020-2024). Both panels are scored; neither is presented as the other."""
    fires = pd.read_parquet(ROOT / "data" / "processed" / "fires.parquet",
                            columns=["date", "size_ha"])
    fires["Year"] = fires["date"].dt.year
    fires["Month"] = fires["date"].dt.month
    monthly = fires.groupby(["Year", "Month"]).agg(
        Number_of_Fires=("size_ha", "size"), area_ha=("size_ha", "sum")).reset_index()
    return monthly


def panel(source: str) -> pd.DataFrame:
    """One row per calendar month, fires and weather, whole seasons only.

    Capped at LAST_YEAR: the fire record runs a season ahead of the station
    archives, and a January with weather but a year's worth of missing months
    behind it would sit in the test set as if it were a year."""
    fires = nfd_monthly() if source == "legacy" else nfdb_monthly()
    table = fires.merge(station_monthly(), on=["Year", "Month"], how="inner")
    table = table[table["Year"] <= LAST_YEAR]
    return table.sort_values(["Year", "Month"]).reset_index(drop=True)


def forest(train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict:
    model = RandomForestRegressor(**FOREST)
    model.fit(train[features], train[TARGET])
    predicted = model.predict(test[features])
    observed = test[TARGET].to_numpy()
    return {
        "rows_train": int(len(train)), "rows_test": int(len(test)),
        "r2": float(r2_score(observed, predicted)),
        "mae": float(mean_absolute_error(observed, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(observed, predicted))),
        "mean_observed_ha": float(observed.mean()),
        "predicted": predicted.tolist(),
    }


def score_panel(table: pd.DataFrame) -> dict:
    """The three scorings, on one monthly panel."""
    without_count = [f for f in FEATURES_2024 if f != "Number_of_Fires"]
    random_train, random_test = train_test_split(table, test_size=0.2, random_state=SEED)
    early = table[table["Year"] <= TRAIN_TO]
    late = table[table["Year"] >= TEST_FROM]
    return {
        "as_scored_in_2024": forest(random_train, random_test, FEATURES_2024),
        "out_of_time": forest(early, late, FEATURES_2024),
        "out_of_time_without_the_fire_count": forest(early, late, without_count),
    }


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    report = {
        "weather": "CWFIS station archives, national monthly means",
        "features": FEATURES_2024,
        "model": {"kind": "RandomForestRegressor", **{k: v for k, v in FOREST.items()}},
        "target": "hectares burned in the month, nationally",
        "panels": {},
    }
    for source, described in (("legacy", "the NFD monthly tables the 2024 project read"),
                              ("nfdb", "the NFDB point layer, aggregated to the same two columns")):
        table = panel(source)
        scorings = score_panel(table)
        report["panels"][source] = {
            "source": described,
            "years": [int(table["Year"].min()), int(table["Year"].max())],
            "rows": int(len(table)),
            "scorings": {name: {k: v for k, v in scoring.items() if k != "predicted"}
                         for name, scoring in scorings.items()},
        }
        print(f"--- {source}: {described}, {len(table)} months "
              f"{table['Year'].min()}-{table['Year'].max()}")
        for name, scoring in scorings.items():
            print(f"    {name:36} R2 {scoring['r2']:+.3f}  MAE {scoring['mae']:,.0f} ha  "
                  f"RMSE {scoring['rmse']:,.0f} ha  ({scoring['rows_train']} train / {scoring['rows_test']} test)")
        if source == "nfdb":
            late = table[table["Year"] >= TEST_FROM]
            published = late[["Year", "Month", "Number_of_Fires", TARGET]].copy()
            published["predicted_ha"] = scorings["out_of_time"]["predicted"]
            published["predicted_ha_without_count"] =                 scorings["out_of_time_without_the_fire_count"]["predicted"]
            published.to_parquet(PUBLISHED / "hackathon_2024.parquet", index=False)
    (REPORTS / "hackathon_2024.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
