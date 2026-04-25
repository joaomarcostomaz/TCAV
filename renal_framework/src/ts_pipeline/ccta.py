"""
ccta.py

Constrained Counterfactual Trajectory Adjustment (CCTA)-style utilities:
- feasibility projection
- gradient ranking of lag-feature cells
- sparse edit search
- decoding actions and cohort summaries
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


@dataclass
class CCTAConfig:
    max_rel_change: float = 0.50
    max_abs_delta: float = 2.0
    sparsity_budget: int = 12
    monotonic_nondecreasing_keywords: Optional[List[str]] = None


def is_monotonic_feature(name: str, keywords: Optional[List[str]]) -> bool:
    if not keywords:
        return False
    u = str(name).upper()
    return any(k in u for k in keywords)


def project_feasible(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    feature_names: Optional[List[str]] = None,
    cfg: CCTAConfig = CCTAConfig(),
) -> np.ndarray:
    """
    Feasibility projection:
      1) non-negativity
      2) bounded rel/abs changes
      3) optional monotonic constraints by keyword
    """
    x = x_cf.copy()
    x = np.maximum(x, 0.0)

    delta = x - x_orig
    rel_cap = cfg.max_rel_change * (np.abs(x_orig) + 1e-6)
    cap = np.minimum(rel_cap, cfg.max_abs_delta)
    delta = np.clip(delta, -cap, cap)
    x = x_orig + delta
    x = np.maximum(x, 0.0)

    if feature_names is not None and cfg.monotonic_nondecreasing_keywords:
        for f in range(x.shape[1]):
            fname = feature_names[f] if f < len(feature_names) else f"f_{f}"
            if is_monotonic_feature(fname, cfg.monotonic_nondecreasing_keywords):
                for l in range(1, x.shape[0]):
                    x[l, f] = max(x[l, f], x[l - 1, f])

    return x


@torch.no_grad()
def risk_prob_from_seq_np(
    model: nn.Module,
    X_seq_np: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    probs = []
    for i in range(0, len(X_seq_np), batch_size):
        xb = torch.tensor(X_seq_np[i : i + batch_size], dtype=torch.float32, device=device)
        p = torch.sigmoid(model(xb)).detach().cpu().numpy()
        probs.append(p)
    return np.concatenate(probs, axis=0)


def gradient_rank_cells_for_patient(
    model: nn.Module,
    x_seq: np.ndarray,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      rank_idx: flattened indices sorted descending by |grad*input|
      score_map: [L,F]
    """
    xb = torch.tensor(x_seq[None, ...], dtype=torch.float32, device=device, requires_grad=True)
    model.zero_grad(set_to_none=True)
    logit = model(xb).sum()
    logit.backward()

    grad = xb.grad.detach().cpu().numpy()[0]
    gxi = np.abs(grad * x_seq)
    rank_idx = np.argsort(-gxi.reshape(-1))
    return rank_idx, gxi


def apply_sparse_edits(
    x_orig: np.ndarray,
    rank_idx: np.ndarray,
    n_edits: int,
    step_scale: float = 0.25,
) -> np.ndarray:
    """
    Reduce top-ranked cells by step_scale proportion.
    """
    x_cf = x_orig.copy()
    used = 0
    L, F = x_cf.shape

    for idx in rank_idx:
        if used >= n_edits:
            break
        l = idx // F
        f = idx % F

        cur = x_cf[l, f]
        if cur <= 0:
            continue

        x_cf[l, f] = max(0.0, cur * (1.0 - step_scale))
        used += 1

    return x_cf


def ccta_search_for_patient(
    model: nn.Module,
    x_seq: np.ndarray,
    threshold: float = 0.5,
    max_edits: int = 12,
    edit_grid: List[int] = [1, 2, 3, 4, 6, 8, 10, 12],
    step_scales: List[float] = [0.15, 0.25, 0.35, 0.50],
    feature_names: Optional[List[str]] = None,
    cfg: CCTAConfig = CCTAConfig(),
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Search sparse feasible counterfactual minimizing predicted risk.
    """
    p0 = float(risk_prob_from_seq_np(model, x_seq[None, ...], device=device)[0])
    rank_idx, gxi_map = gradient_rank_cells_for_patient(model, x_seq, device=device)

    best = {
        "x_cf": x_seq.copy(),
        "p_cf": p0,
        "p0": p0,
        "n_edits": 0,
        "step_scale": 0.0,
        "success": (p0 < threshold),
    }

    for n_edits in edit_grid:
        if n_edits > max_edits:
            continue
        for ss in step_scales:
            x_candidate = apply_sparse_edits(x_seq, rank_idx, n_edits=n_edits, step_scale=ss)
            x_candidate = project_feasible(x_seq, x_candidate, feature_names=feature_names, cfg=cfg)
            p_candidate = float(risk_prob_from_seq_np(model, x_candidate[None, ...], device=device)[0])

            if p_candidate < best["p_cf"]:
                best = {
                    "x_cf": x_candidate,
                    "p_cf": p_candidate,
                    "p0": p0,
                    "n_edits": n_edits,
                    "step_scale": ss,
                    "success": (p_candidate < threshold),
                }

            if p_candidate < threshold:
                return {**best, "gxi_map": gxi_map, "rank_idx": rank_idx}

    return {**best, "gxi_map": gxi_map, "rank_idx": rank_idx}


def decode_actions(
    x_orig: np.ndarray,
    x_cf: np.ndarray,
    feature_names: Optional[List[str]] = None,
    topn: int = 10,
) -> List[Dict[str, Any]]:
    """
    Decode top changed lag-feature cells into action table.
    """
    delta = x_cf - x_orig
    flat_idx = np.argsort(np.abs(delta).reshape(-1))[::-1]

    actions = []
    L, F = delta.shape
    k = 0
    for idx in flat_idx:
        l = idx // F
        f = idx % F
        d = float(delta[l, f])
        if np.isclose(d, 0.0):
            continue

        fname = feature_names[f] if (feature_names is not None and f < len(feature_names)) else f"f_{f}"
        actions.append(
            {
                "lag": int(l),
                "feature_idx": int(f),
                "feature_name": fname,
                "delta": d,
                "orig_value": float(x_orig[l, f]),
                "cf_value": float(x_cf[l, f]),
                "direction": "decrease" if d < 0 else "increase",
            }
        )
        k += 1
        if k >= topn:
            break
    return actions