"""Surrogate regression between embeddings and model probabilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.config import DEFAULT_BATCH_SIZE_PREDICT
from src.modeling.inference import batched_predict_proba
from src.utils import logging as log


@dataclass
class SurrogateArtifacts:
    model: Ridge
    scaler: Optional[StandardScaler]
    r2_val: float
    mse_val: float
    train_size: int
    val_size: int


def compute_model_probabilities(
    model,
    X: np.ndarray,
    years: np.ndarray,
    year_to_domain: Dict[int, int],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE_PREDICT,
    device: torch.device | str = torch.device("cpu"),
    example_shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    dist = np.array([year_to_domain[int(y)] for y in years], dtype=np.int64)
    dev = torch.device(device) if not isinstance(device, torch.device) else device
    probs = batched_predict_proba(
        model,
        X=X.astype(np.float32),
        dist_shift_domain=dist,
        device=dev,
        example_shape=example_shape,
        batch_size=batch_size,
    )
    return probs


def train_surrogate(
    embeddings: np.ndarray,
    target_probs: np.ndarray,
    *,
    alpha: float = 1.0,
    subsample: int | None = None,
    standardize: bool = True,
    holdout_frac: float = 0.2,
    random_state: int = 42,
) -> SurrogateArtifacts:
    if embeddings.shape[0] != target_probs.shape[0]:
        raise ValueError("Embeddings and target probabilities must have matching rows.")
    rng = np.random.default_rng(random_state)
    X = embeddings
    y = target_probs
    if subsample is not None and subsample < X.shape[0]:
        idx = rng.choice(X.shape[0], subsample, replace=False)
        X = X[idx]
        y = y[idx]
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=holdout_frac,
        random_state=random_state,
    )
    scaler = None
    if standardize:
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_val = scaler.transform(X_val)
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    preds_val = model.predict(X_val)
    r2 = r2_score(y_val, preds_val)
    mse = mean_squared_error(y_val, preds_val)
    log.info(f"Surrogate Ridge trained (alpha={alpha}) R2={r2:.4f} MSE={mse:.4f}")
    return SurrogateArtifacts(
        model=model,
        scaler=scaler,
        r2_val=float(r2),
        mse_val=float(mse),
        train_size=int(X_train.shape[0]),
        val_size=int(X_val.shape[0]),
    )
