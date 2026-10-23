"""
The forecasts as they were published, graded against what happened afterwards.

The backtest scores seasons the model never saw, and the season check scores 2026
from each day's observed weather. Neither is the forecast anyone actually read.
Every run of the scheduled forecast publishes a snapshot to the `live-data`
branch, so the branch history is a record of exactly what the model said, and
when, before the outcome existed. This module grades that record.

Rules, fixed before the first grade was computed:

* **One forecast per station day: the first one published.** The workflow runs
  twice a day and often republishes the same station day; taking the earliest
  means no forecast benefits from a later, better-informed rebuild.
* **Only forecasts from the current grid.** Snapshots built before the cell fix of
  22 September 2026 used a 771-cell grid and the model trained on scrambled
  labels. They are counted and excluded, not graded.
* **Prospective means the target day starts after the forecast was built.** Lead
  0 is the station day itself, already over when the forecast is published, and
  lead 1 is usually the publication day; both are graded and labelled as
  not-yet-forecasts. The verdict rests on leads 2 to 7.
* **The label is the season check's**: a cell shows *new fire activity* on a day
  it has satellite hotspots after 14 days with none.
* **Two baselines, both available at publication time.** The cell's normal rate
  for the month (in the snapshot), and *fire next door*: hotspots in the eight
  surrounding cells in the 48 hours before the forecast was built. "Is it burning
  in this cell now" is not a baseline for this label - a cell with recent hotspots
  cannot show *new* activity for two weeks - but spread into a neighbouring cell
  is exactly what the label counts. Most cells have no neighbouring hotspots, so
  fire next door breaks its ties by the normal rate: the baseline is "fire next
  door, then normal for the month", which is the harder of the two to beat.
* **Uncertainty comes from resampling days.** Cell-days on the same date share
  weather, so rows are not independent; the bootstrap draws whole target days.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import numpy as np
import pandas as pd

from wildfire.evaluation import SEED, TOP_SHARE, rank_within
from wildfire.features import cell_ids

PROSPECTIVE_FROM_LEAD = 2
NEIGHBOUR_HOURS = 48
BOOTSTRAP_DRAWS = 1000


@dataclass(frozen=True)
class Snapshot:
    """One published forecast, decoded."""
    sha: str
    built_at: datetime
    as_of: date
    days: list[date]
    lead_days: list[int]
    cell_ids: list[str]
    normal: np.ndarray            # (cells,) probability
    risk: np.ndarray              # (cells, days) probability, NaN where unpublished
    large: np.ndarray             # (cells, days) probability, NaN where unpublished
    hotspots: np.ndarray          # (n, 2) lat, lon in the 48 hours before the build

    @classmethod
    def from_json(cls, sha: str, payload: dict) -> "Snapshot":
        def grid(values, scale):
            array = np.asarray(values, dtype=float)
            return np.where(array < 0, np.nan, array / scale)

        spots = np.asarray(payload.get("hotspots") or [], dtype=float).reshape(-1, 2)
        return cls(
            sha=sha,
            built_at=pd.Timestamp(payload["generated_at"]).tz_convert("UTC").to_pydatetime(),
            as_of=date.fromisoformat(payload["as_of"]),
            days=[date.fromisoformat(d) for d in payload["days"]],
            lead_days=[int(x) for x in payload["lead_days"]],
            cell_ids=[str(cell[0]) for cell in payload["cells"]],
            normal=np.asarray(payload["normal"], dtype=float) / payload["scale"],
            risk=grid(payload["risk"], payload["scale"]),
            large=grid(payload["large"], payload["large_scale"]),
            hotspots=spots,
        )


def first_per_station_day(snapshots: list[Snapshot], grid: set[str]) -> tuple[list[Snapshot], dict]:
    """The earliest forecast published for each station day, on the current grid."""
    on_grid = [s for s in snapshots if set(s.cell_ids) == grid]
    kept: dict[date, Snapshot] = {}
    for snap in sorted(on_grid, key=lambda s: s.built_at):
        kept.setdefault(snap.as_of, snap)
    excluded = {
        "published": len(snapshots),
        "off_grid": len(snapshots) - len(on_grid),
        "republished_same_day": len(on_grid) - len(kept),
    }
    return [kept[d] for d in sorted(kept)], excluded


def fire_next_door(snapshot: Snapshot) -> np.ndarray:
    """Hotspots in the eight cells around each cell, in the 48 hours before the build."""
    if len(snapshot.hotspots) == 0:
        return np.zeros(len(snapshot.cell_ids))
    # The grid's own rule, not a second copy of it: a cell id that disagrees with
    # features.cell_ids by one row is how 97.7% of fires once landed in the wrong cell.
    counts = cell_ids(snapshot.hotspots[:, 0], snapshot.hotspots[:, 1]).value_counts()
    out = np.zeros(len(snapshot.cell_ids))
    for i, cell in enumerate(snapshot.cell_ids):
        lat, lon = (int(part) for part in cell.split("_"))
        out[i] = sum(counts.get(f"{lat + dy}_{lon + dx}", 0)
                     for dy in (-1, 0, 1) for dx in (-1, 0, 1) if (dy, dx) != (0, 0))
    return out


def forecast_rows(snapshots: list[Snapshot]) -> pd.DataFrame:
    """One row per published forecast x cell x lead day."""
    frames = []
    for snap in snapshots:
        neighbours = fire_next_door(snap)
        built_day = snap.built_at.date()
        for j, (target, lead) in enumerate(zip(snap.days, snap.lead_days)):
            frames.append(pd.DataFrame({
                "issued": pd.Timestamp(snap.as_of),
                "built_at": pd.Timestamp(snap.built_at),
                "sha": snap.sha,
                "date": pd.Timestamp(target),
                "lead_days": lead,
                "prospective": target > built_day,
                "cell_id": snap.cell_ids,
                "risk": snap.risk[:, j],
                "large_risk": snap.large[:, j],
                "normal": snap.normal,
                "fire_next_door": neighbours,
            }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


SCORES = {
    "model": ("risk", "large_risk"),
    "large_fire_model": ("large_risk", "risk"),
    "normal_for_the_month": ("normal", None),
    "fire_next_door": ("fire_next_door", "normal"),
}


def _auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if y.min() == y.max():
        return float("nan")
    filled = np.where(np.isnan(score), np.nanmin(score) if np.isfinite(score).any() else 0.0, score)
    return float(roc_auc_score(y, filled))


def _ranked(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank every score within each forecast's target day, exactly as the app ranks cells."""
    frame = frame.copy()
    frame["group"] = frame["issued"].astype(str) + "|" + frame["date"].astype(str)
    frame["random_tiebreak"] = np.random.default_rng(SEED).random(len(frame))
    for name, (column, tiebreak) in SCORES.items():
        filled = frame[column].fillna(-1.0)
        frame[f"_{name}"] = filled
        frame[f"rank_{name}"] = rank_within(frame.assign(**{f"_{name}": filled}), "group", f"_{name}",
                                            tiebreak or "random_tiebreak")
    frame["cells_in_group"] = frame.groupby("group")["group"].transform("size")
    return frame


def _capture(frame: pd.DataFrame, name: str) -> float:
    cut = np.ceil(TOP_SHARE * frame["cells_in_group"])
    hits = frame.loc[frame[f"rank_{name}"] <= cut, "new_detection"].sum()
    return float(hits / max(frame["new_detection"].sum(), 1))


def _summary(frame: pd.DataFrame) -> dict:
    y = frame["new_detection"].to_numpy()
    out = {"cell_days": int(len(frame)), "new_detections": int(y.sum()),
           "target_days": int(frame["date"].nunique()), "forecasts": int(frame["issued"].nunique())}
    for name, (column, _) in SCORES.items():
        out[name] = {"roc_auc": _auc(y, frame[column].to_numpy(dtype=float)),
                     "top_decile_capture": _capture(frame, name)}
    return out


def _bootstrap(frame: pd.DataFrame, draws: int, seed: int) -> dict:
    """95% intervals from resampling whole target days, for the model and its edge on each baseline.

    Resampling days never moves a cell's rank inside its own day, so each ranking's
    top-tenth flag is computed once; only the pooled ROC-AUC needs the rows."""
    from sklearn.metrics import roc_auc_score

    days, day_index = np.unique(frame["date"].to_numpy(), return_inverse=True)
    rows_of = [np.flatnonzero(day_index == k) for k in range(len(days))]
    y = frame["new_detection"].to_numpy()
    cut = np.ceil(TOP_SHARE * frame["cells_in_group"].to_numpy())
    names = {"model": "risk", "normal": "normal", "fire_next_door": "fire_next_door"}
    ranks = {"model": "rank_model", "normal": "rank_normal_for_the_month", "fire_next_door": "rank_fire_next_door"}
    scores = {k: frame[c].fillna(-1.0).to_numpy(dtype=float) for k, c in names.items()}
    top = {k: frame[r].to_numpy() <= cut for k, r in ranks.items()}
    # Per day: positives, and positives caught by each ranking's top tenth.
    pos_day = np.bincount(day_index, weights=y, minlength=len(days))
    hit_day = {k: np.bincount(day_index, weights=y * top[k], minlength=len(days)) for k in top}

    rng = np.random.default_rng(seed)
    stats = {k: [] for k in ("model_capture", "model_auc", "edge_on_normal_capture", "edge_on_normal_auc",
                             "edge_on_fire_next_door_capture", "edge_on_fire_next_door_auc")}
    for _ in range(draws):
        pick = rng.integers(0, len(days), size=len(days))
        positives = pos_day[pick].sum()
        idx = np.concatenate([rows_of[k] for k in pick])
        if positives == 0 or positives == len(idx):
            continue
        capture = {k: hit_day[k][pick].sum() / positives for k in top}
        auc = {k: roc_auc_score(y[idx], scores[k][idx]) for k in scores}
        stats["model_capture"].append(capture["model"])
        stats["model_auc"].append(auc["model"])
        for base in ("normal", "fire_next_door"):
            stats[f"edge_on_{base}_capture"].append(capture["model"] - capture[base])
            stats[f"edge_on_{base}_auc"].append(auc["model"] - auc[base])
    out = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if v else [None, None]
           for k, v in stats.items()}
    return out | {"draws": int(len(stats["model_auc"])), "resampled": "target days"}


def grade(rows: pd.DataFrame, labels: pd.DataFrame, draws: int = BOOTSTRAP_DRAWS, seed: int = SEED) -> tuple[dict, pd.DataFrame]:
    """Join forecasts to the outcome on their target day and score them by lead.

    `labels` holds cell_id, date, new_detection for every cell on every day the
    hotspot archive covers; target days it does not cover yet are left ungraded.
    """
    joined = rows.merge(labels[["cell_id", "date", "new_detection"]], on=["cell_id", "date"], how="inner")
    if joined.empty:
        return {"graded_cell_days": 0}, joined
    ranked = _ranked(joined)
    by_lead = []
    for lead, part in ranked.groupby("lead_days"):
        entry = {"lead_days": int(lead), "prospective_share": float(part["prospective"].mean())} | _summary(part)
        by_lead.append(entry)
    prospective = ranked[ranked["lead_days"] >= PROSPECTIVE_FROM_LEAD]
    report = {
        "rule": {"one_forecast_per_station_day": "first published",
                 "prospective_from_lead": PROSPECTIVE_FROM_LEAD,
                 "label": "new fire activity: satellite hotspots after 14 days with none",
                 "top_share": TOP_SHARE},
        "first_target_day": str(ranked["date"].min().date()),
        "last_target_day": str(ranked["date"].max().date()),
        "first_forecast": str(ranked["issued"].min().date()),
        "last_forecast": str(ranked["issued"].max().date()),
        "strictly_prospective_rows": float(prospective["prospective"].mean()) if len(prospective) else None,
        "by_lead": by_lead,
        "prospective": _summary(prospective) if len(prospective) else None,
        "prospective_intervals": (_bootstrap(prospective, draws, seed)
                                  if len(prospective) and prospective["new_detection"].nunique() > 1 else None),
    }
    keep = ["issued", "built_at", "date", "lead_days", "prospective", "cell_id", "risk", "large_risk", "normal",
            "fire_next_door", "new_detection", "rank_model"]
    return report, ranked[keep]


def verdict(interval: list[float | None], baseline: str, metric: str) -> str:
    """One sentence per comparison, decided by the interval and nothing else.

    The README is rewritten weekly by a workflow, so its wording cannot be chosen
    after looking at the numbers: "ahead" only when the whole 95% interval of the
    edge is above zero, "behind" only when it is all below."""
    lo, hi = interval
    if lo is None or hi is None:
        return f"too few graded days to compare with {baseline} on {metric}"
    if lo > 0:
        return f"ahead of {baseline} on {metric}"
    if hi < 0:
        return f"behind {baseline} on {metric}"
    return f"not yet distinguishable from {baseline} on {metric}"
