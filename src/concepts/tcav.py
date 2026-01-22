"""TCAV finite-difference evaluation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from src.concepts.cav import CAV
from src.concepts.surrogate import SurrogateArtifacts


@dataclass
class TCAVResult:
    cluster_id: int
    tcav_positive_fraction: float
    mean_derivative: float
    alpha_used: float
    derivatives: np.ndarray


def compute_tcav(
    cavs: Dict[int, CAV],
    embeddings: np.ndarray,
    surrogate: SurrogateArtifacts,
    *,
    finite_rel: float = 0.05,
    batch_size: int = 1024,
) -> Dict[int, TCAVResult]:
    if not cavs:
        return {}
    n_eval = embeddings.shape[0]
    scaler = surrogate.scaler
    model = surrogate.model
    results: Dict[int, TCAVResult] = {}
    for cluster_id, cav in cavs.items():
        direction = cav.vector.astype(np.float32)
        projection = embeddings.dot(direction)
        proj_std = float(np.std(projection))
        alpha = finite_rel * proj_std if proj_std > 0 else finite_rel
        derivatives = np.zeros(n_eval, dtype=np.float32)
        for start in range(0, n_eval, batch_size):
            end = min(start + batch_size, n_eval)
            emb_batch = embeddings[start:end]
            plus = emb_batch + alpha * direction
            minus = emb_batch - alpha * direction
            if scaler is not None:
                plus = scaler.transform(plus)
                minus = scaler.transform(minus)
            p_plus = model.predict(plus)
            p_minus = model.predict(minus)
            derivatives[start:end] = ((p_plus - p_minus) / (2.0 * alpha)).astype(np.float32)
        tcav_prop = float(np.mean(derivatives > 0))
        mean_derivative = float(np.nanmean(derivatives))
        results[cluster_id] = TCAVResult(
            cluster_id=cluster_id,
            tcav_positive_fraction=tcav_prop,
            mean_derivative=mean_derivative,
            alpha_used=alpha,
            derivatives=derivatives,
        )
    return results
