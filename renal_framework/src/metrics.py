"""
Metrics and threshold utilities used across pipelines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    x = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-x))


def binary_metrics_from_logits(y_true: np.ndarray, logits: np.ndarray, threshold: float = 0.5) -> dict:
    """
    Compute binary metrics from logits.
    """
    y_true = np.asarray(y_true).astype(int)
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    f1_macro = f1_score(y_true, preds, average="macro") if len(np.unique(y_true)) > 1 else np.nan
    f1_pos = f1_score(y_true, preds, pos_label=1) if y_true.sum() > 0 else np.nan

    return {
        "f1_macro": f1_macro,
        "f1_pos": f1_pos,
        "probs": probs,
        "preds": preds,
    }


def tune_threshold_for_f1_pos(
    y_true: np.ndarray,
    logits_or_probs: np.ndarray,
    input_is_logits: bool = True,
    grid: np.ndarray | None = None,
) -> tuple[float, float]:
    """
    Tune threshold to maximize positive-class F1.
    """
    y_true = np.asarray(y_true).astype(int)
    vals = np.asarray(logits_or_probs, dtype=float)
    probs = sigmoid(vals) if input_is_logits else vals

    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)

    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (probs >= thr).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1) if y_true.sum() > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def evaluate_per_year_with_threshold(
    y_true: np.ndarray, logits: np.ndarray, years: np.ndarray, threshold: float
) -> pd.DataFrame:
    """
    Per-year F1 summary using a fixed threshold.
    """
    y_true = np.asarray(y_true).astype(int)
    years = np.asarray(years).astype(int)
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    rows = []
    for yr in sorted(np.unique(years)):
        mask = years == yr
        yt = y_true[mask]
        yp = preds[mask]

        f1m = f1_score(yt, yp, average="macro") if len(np.unique(yt)) > 1 else np.nan
        f1p = f1_score(yt, yp, pos_label=1) if yt.sum() > 0 else np.nan

        rows.append(
            {
                "year": int(yr),
                "n_samples": int(mask.sum()),
                "n_deaths": int(yt.sum()),
                "f1_macro": None if np.isnan(f1m) else float(f1m),
                "f1_pos": None if np.isnan(f1p) else float(f1p),
                "threshold": float(threshold),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)