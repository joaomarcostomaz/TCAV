"""Inference utilities mirrored from Célula 6 and 7."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import DEFAULT_BATCH_SIZE_PREDICT
from src.utils.torch_helpers import make_dist_tensor


@dataclass
class InferenceResult:
    probs_pos: np.ndarray
    y_pred: np.ndarray
    metrics: Dict[str, float]
    inference_time: float


def batched_predict_proba(
    model,
    X: np.ndarray,
    dist_shift_domain: np.ndarray,
    *,
    device: torch.device = torch.device("cpu"),
    example_shape: Tuple[int, ...] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE_PREDICT,
) -> np.ndarray:
    preds: list[np.ndarray] = []
    t0 = time.perf_counter()
    for start in range(0, X.shape[0], batch_size):
        end = min(start + batch_size, X.shape[0])
        xb = X[start:end].astype(np.float32)
        dist_np = dist_shift_domain[start:end]
        dist_tensor = make_dist_tensor(dist_np, device=device, target_shape_example=example_shape)
        try:
            proba = model.predict_proba(xb, additional_x={"dist_shift_domain": dist_tensor})
        except Exception:
            proba = model.predict_proba(xb, additional_x={"dist_shift_domain": dist_np})
        if isinstance(proba, torch.Tensor):
            proba = proba.detach().cpu().numpy()
        preds.append(np.asarray(proba))
    elapsed = time.perf_counter() - t0
    probs = np.vstack(preds)
    if probs.ndim == 2 and probs.shape[1] > 1:
        probs = probs[:, 1]
    else:
        probs = probs.ravel()
    return probs


def evaluate_predictions(y_true: np.ndarray, probs_pos: np.ndarray, *, threshold: float = 0.5) -> InferenceResult:
    y_pred = (probs_pos >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_pos": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_pos": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, probs_pos)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    return InferenceResult(probs_pos, y_pred, metrics, inference_time=0.0)


def per_year_metrics(years: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    import pandas as pd

    rows = []
    for yr in sorted(np.unique(years)):
        mask = years == yr
        if not mask.any():
            continue
        yt = y_true[mask]
        yp = y_pred[mask]
        pr = probs[mask]
        entry = {
            "year": int(yr),
            "n_samples": int(mask.sum()),
            "n_deaths": int(yt.sum()),
            "accuracy": accuracy_score(yt, yp),
            "precision_macro": precision_score(yt, yp, average="macro", zero_division=0),
            "recall_macro": recall_score(yt, yp, average="macro", zero_division=0),
            "f1_macro": f1_score(yt, yp, average="macro", zero_division=0),
            "precision_pos": precision_score(yt, yp, pos_label=1, zero_division=0),
            "recall_pos": recall_score(yt, yp, pos_label=1, zero_division=0),
            "f1_pos": f1_score(yt, yp, pos_label=1, zero_division=0),
        }
        if len(np.unique(yt)) > 1:
            try:
                entry["auc"] = roc_auc_score(yt, pr)
            except Exception:
                entry["auc"] = float("nan")
        rows.append(entry)
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax)
    plt.title("Confusion Matrix (test)")
    plt.tight_layout()
    plt.show()
