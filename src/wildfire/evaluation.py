"""
Scoring rules shared by training, the replay and the season check.

The headline number is operational: of the fires that started, what share fell in
the 10% of cells (or cell-days) ranked riskiest. "10%" has to mean exactly 10%. An
isotonic calibration outputs a few hundred distinct values and a climatology
baseline is tied across whole regions, so `score >= quantile(0.9)` can select more
than a tenth of the rows and flatter whichever ranking has the biggest plateau at
the cut. Here exactly ceil(10%) rows are taken. Ties break on a secondary score
where one exists - the model's uncalibrated probability, which calibration only
flattens - and otherwise at random with a fixed seed, the expected value of an
arbitrary pick.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TOP_SHARE = 0.10
SEED = 1904


def top_mask(score, tiebreak=None, share: float = TOP_SHARE, seed: int = SEED) -> np.ndarray:
    """Exactly ceil(share * n) rows with the highest score; missing scores rank last."""
    score = np.asarray(score, dtype=float)
    n = len(score)
    if tiebreak is None:
        tiebreak = np.random.default_rng(seed).random(n)
    primary = np.where(np.isnan(score), -np.inf, score)
    order = np.lexsort((-np.asarray(tiebreak, dtype=float), -primary))
    mask = np.zeros(n, dtype=bool)
    mask[order[:int(np.ceil(share * n))]] = True
    return mask


def rank_within(frame: pd.DataFrame, by: str, score: str, tiebreak: str | None = None) -> np.ndarray:
    """1 for the riskiest row of each group, with the same tie rule as top_mask."""
    keys = [by, score] + ([tiebreak] if tiebreak else [])
    order = frame.sort_values(keys, ascending=[True] + [False] * (len(keys) - 1), kind="stable")
    ranks = order.groupby(by, sort=False).cumcount() + 1
    return ranks.reindex(frame.index).to_numpy()


def ranking(y, score, weights=None, tiebreak=None) -> dict[str, float]:
    """How well `score` orders rows: ROC-AUC, PR-AUC, and the share of `weights`
    (fires; defaults to the positives) inside exactly the top 10%."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    filled = np.where(np.isnan(score), np.nanmin(score), score)
    top = top_mask(score, tiebreak)
    counts = y if weights is None else np.asarray(weights)
    return {
        "roc_auc": float(roc_auc_score(y, filled)),
        "pr_auc": float(average_precision_score(y, filled)),
        "share_of_fires_in_top_decile": float(counts[top].sum() / max(counts.sum(), 1)),
        "lift_top_decile": float(y[top].mean() / max(y.mean(), 1e-9)),
    }


def probability(y, p) -> dict[str, float]:
    """Calibration-sensitive scores, for outputs that are meant as probabilities."""
    from sklearn.metrics import brier_score_loss, log_loss

    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return {"brier": float(brier_score_loss(y, p)), "log_loss": float(log_loss(y, p))}
