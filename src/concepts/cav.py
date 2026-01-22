"""Concept Activation Vector (CAV) utilities mirrored from the notebook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.utils import check_random_state

from src.utils import logging as log


@dataclass
class CAV:
    cluster_id: int
    size_pure: int
    vector: np.ndarray
    pure_indices: np.ndarray
    negative_indices: np.ndarray
    classifier: LogisticRegression
    centroid: np.ndarray
    quantile: float


def _select_pure_indices(
    embeddings: np.ndarray,
    member_indices: np.ndarray,
    centroid: np.ndarray,
    quantile: float,
    min_pure: int,
) -> np.ndarray:
    if member_indices.size == 0:
        return np.empty(0, dtype=int)
    distances = np.linalg.norm(embeddings[member_indices] - centroid.reshape(1, -1), axis=1)
    if distances.size == 0:
        return np.empty(0, dtype=int)
    cutoff = np.quantile(distances, quantile)
    pure_idx = member_indices[distances <= cutoff]
    if pure_idx.size < min_pure:
        return np.empty(0, dtype=int)
    return pure_idx


def train_cavs(
    embeddings: np.ndarray,
    labels: np.ndarray,
    centroids: np.ndarray,
    *,
    quantile: float = 0.25,
    min_pure: int = 10,
    negative_multiplier: int = 5,
    min_negative: int = 200,
    logistic_kwargs: Optional[dict] = None,
    random_state: int = 42,
) -> Dict[int, CAV]:
    """Train a logistic regression CAV per cluster label."""

    rng = check_random_state(random_state)
    logistic_kwargs = logistic_kwargs or {
        "C": 0.1,
        "solver": "liblinear",
        "max_iter": 2000,
        "random_state": random_state,
    }
    n_samples = embeddings.shape[0]
    cavs: Dict[int, CAV] = {}
    for cluster_id in np.unique(labels):
        member_idx = np.where(labels == cluster_id)[0]
        centroid = centroids[int(cluster_id)] if centroids is not None else embeddings[member_idx].mean(axis=0)
        pure_idx = _select_pure_indices(embeddings, member_idx, centroid, quantile, min_pure)
        if pure_idx.size == 0:
            log.warn(f"Cluster {cluster_id}: insufficient pure samples; skipping CAV training.")
            continue
        neg_pool = np.setdiff1d(np.arange(n_samples), pure_idx)
        if neg_pool.size == 0:
            log.warn(f"Cluster {cluster_id}: negative pool empty; skipping CAV training.")
            continue
        neg_target = max(min_negative, pure_idx.size * negative_multiplier)
        neg_n = min(neg_target, neg_pool.size)
        neg_idx = rng.choice(neg_pool, size=neg_n, replace=False)
        Xc = np.vstack([embeddings[pure_idx], embeddings[neg_idx]])
        yc = np.hstack([np.ones(pure_idx.size, dtype=int), np.zeros(neg_idx.size, dtype=int)])
        clf = LogisticRegression(**logistic_kwargs)
        clf.fit(Xc, yc)
        w = clf.coef_.ravel()
        norm = np.linalg.norm(w)
        if norm == 0:
            log.warn(f"Cluster {cluster_id}: zero-norm weight vector; skipping.")
            continue
        vector = (w / norm).astype(np.float32)
        cavs[int(cluster_id)] = CAV(
            cluster_id=int(cluster_id),
            size_pure=int(pure_idx.size),
            vector=vector,
            pure_indices=pure_idx,
            negative_indices=neg_idx,
            classifier=clf,
            centroid=centroid,
            quantile=quantile,
        )
        log.info(f"CAV trained for cluster {cluster_id} with {pure_idx.size} pure / {neg_idx.size} negative samples.")
    return cavs
