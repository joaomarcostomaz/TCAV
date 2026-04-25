"""
cctsi.py

Clinical-Consistency Temporal SHAP-IG (CCTS-I) utilities:
- temporal coalition groups
- conditional masking references
- grouped SHAP-like contribution estimates
- grouped IG priors
- clinical consistency + stability regularization
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn

from .interpretability import integrated_gradients


@dataclass
class CCTSIConfig:
    lambda_shap: float = 0.55
    lambda_ig: float = 0.45
    lambda_clinical: float = 0.10
    lambda_stability: float = 0.05
    n_group_samples: int = 40
    n_stability_boot: int = 5
    max_groups_local: int = 120
    rng_seed: int = 42


def build_temporal_groups(
    L: int,
    F: int,
    mode: str = "lag_feature_blocks",
    top_feature_idx: Optional[np.ndarray] = None,
) -> List[List[Tuple[int, int]]]:
    groups: List[List[Tuple[int, int]]] = []

    if mode == "lag_feature_blocks":
        for l in range(L):
            for f in range(F):
                groups.append([(l, f)])

    elif mode == "lag_blocks":
        for l in range(L):
            groups.append([(l, f) for f in range(F)])

    elif mode == "feature_blocks":
        for f in range(F):
            groups.append([(l, f) for l in range(L)])

    elif mode == "mixed_top":
        if top_feature_idx is None:
            top_feature_idx = np.arange(min(40, F))
        top_set = set(map(int, top_feature_idx))

        for l in range(L):
            for f in sorted(top_set):
                groups.append([(l, f)])

        rem_feats = [f for f in range(F) if f not in top_set]
        for l in range(L):
            if rem_feats:
                groups.append([(l, f) for f in rem_feats])
    else:
        raise ValueError(mode)

    return groups


def nearest_reference_indices(
    x_query: np.ndarray,
    year_query: int,
    X_ref: np.ndarray,
    Yr_ref: np.ndarray,
    k: int = 64,
    same_year_only: bool = True,
) -> np.ndarray:
    if same_year_only:
        mask = Yr_ref == year_query
        idx_pool = np.where(mask)[0]
        if len(idx_pool) == 0:
            idx_pool = np.arange(len(X_ref))
    else:
        idx_pool = np.arange(len(X_ref))

    Xc = X_ref[idx_pool]
    d = np.sum((Xc.reshape(len(Xc), -1) - x_query.reshape(1, -1)) ** 2, axis=1)
    ord_idx = np.argsort(d)[: min(k, len(d))]
    return idx_pool[ord_idx]


def mask_group_with_reference(
    x: np.ndarray,
    x_ref: np.ndarray,
    group_cells: List[Tuple[int, int]],
) -> np.ndarray:
    z = x.copy()
    for (l, f) in group_cells:
        z[l, f] = x_ref[l, f]
    return z


@torch.no_grad()
def risk_prob_single(model: nn.Module, x_seq: np.ndarray, device: str = "cpu") -> float:
    xb = torch.tensor(x_seq[None, ...], dtype=torch.float32, device=device)
    return float(torch.sigmoid(model(xb)).detach().cpu().numpy()[0])


def local_group_ig_prior(
    model: nn.Module,
    x: np.ndarray,
    groups: List[List[Tuple[int, int]]],
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    xb = torch.tensor(x[None, ...], dtype=torch.float32, device=device)
    ig = integrated_gradients(model, xb, baseline=torch.zeros_like(xb), steps=24)[0].detach().cpu().numpy()

    gvals = []
    for g in groups:
        gvals.append(float(np.mean([ig[l, f] for (l, f) in g])))
    return np.asarray(gvals, dtype=float), ig


def local_shap_like_group_values(
    model: nn.Module,
    x: np.ndarray,
    year: int,
    groups: List[List[Tuple[int, int]]],
    X_ref: np.ndarray,
    Yr_ref: np.ndarray,
    n_samples: int = 40,
    rng_seed: int = 42,
    device: str = "cpu",
) -> np.ndarray:
    """
    SHAP-like grouped marginal estimate under conditional references:
    phi_g ~ E_ref[ f(x_plus) - f(x_minus) ]
    """
    ref_idx = nearest_reference_indices(
        x_query=x,
        year_query=year,
        X_ref=X_ref,
        Yr_ref=Yr_ref,
        k=max(64, n_samples),
        same_year_only=True,
    )

    rng = np.random.default_rng(rng_seed)
    if len(ref_idx) > n_samples:
        ref_idx = rng.choice(ref_idx, size=n_samples, replace=False)

    phi = np.zeros(len(groups), dtype=float)

    for gi, g in enumerate(groups):
        diffs = []
        for ridx in ref_idx:
            xref = X_ref[ridx]

            x_minus = mask_group_with_reference(x, xref, g)

            x_plus = xref.copy()
            for (l, f) in g:
                x_plus[l, f] = x[l, f]

            p_plus = risk_prob_single(model, x_plus, device=device)
            p_minus = risk_prob_single(model, x_minus, device=device)
            diffs.append(p_plus - p_minus)

        phi[gi] = float(np.mean(diffs))

    return phi


def clinical_consistency_penalty_for_group(
    group_cells: List[Tuple[int, int]],
    group_value: float,
    feature_direction_prior: Dict[int, int],
) -> float:
    penalties = []
    for (_, f) in group_cells:
        prior = int(feature_direction_prior.get(f, 0))  # +1 / -1 / 0
        if prior == 0:
            penalties.append(0.0)
        else:
            penalties.append(max(0.0, -prior * group_value))
    return float(np.mean(penalties)) if penalties else 0.0


def cctsi_local_explain(
    model: nn.Module,
    x: np.ndarray,
    year: int,
    groups: List[List[Tuple[int, int]]],
    X_ref: np.ndarray,
    Yr_ref: np.ndarray,
    feature_direction_prior: Optional[Dict[int, int]] = None,
    cfg: CCTSIConfig = CCTSIConfig(),
    device: str = "cpu",
) -> Dict[str, Any]:
    """
    Main local CCTS-I explanation.
    """
    if feature_direction_prior is None:
        feature_direction_prior = defaultdict(lambda: 0)

    phi_ig_group, ig_map = local_group_ig_prior(model, x, groups, device=device)

    phi_shap_boot = []
    for b in range(cfg.n_stability_boot):
        phi_sh = local_shap_like_group_values(
            model=model,
            x=x,
            year=year,
            groups=groups,
            X_ref=X_ref,
            Yr_ref=Yr_ref,
            n_samples=cfg.n_group_samples,
            rng_seed=cfg.rng_seed + b,
            device=device,
        )
        phi_shap_boot.append(phi_sh)

    phi_shap_boot = np.stack(phi_shap_boot, axis=0)
    phi_shap_mean = phi_shap_boot.mean(axis=0)
    phi_shap_std = phi_shap_boot.std(axis=0)

    clinical_pen = np.array(
        [
            clinical_consistency_penalty_for_group(groups[g], phi_shap_mean[g], feature_direction_prior)
            for g in range(len(groups))
        ],
        dtype=float,
    )
    stab_pen = phi_shap_std

    phi_final_group = (
        cfg.lambda_shap * phi_shap_mean
        + cfg.lambda_ig * phi_ig_group
        - cfg.lambda_clinical * clinical_pen
        - cfg.lambda_stability * stab_pen
    )

    L = max(l for g in groups for (l, _) in g) + 1
    F = max(f for g in groups for (_, f) in g) + 1

    map_shap = np.zeros((L, F), dtype=float)
    map_ig = np.zeros((L, F), dtype=float)
    map_final = np.zeros((L, F), dtype=float)

    for gidx, cells in enumerate(groups):
        for (l, f) in cells:
            map_shap[l, f] += phi_shap_mean[gidx]
            map_ig[l, f] += phi_ig_group[gidx]
            map_final[l, f] += phi_final_group[gidx]

    return {
        "phi_final_group": phi_final_group,
        "phi_shap_group": phi_shap_mean,
        "phi_ig_group": phi_ig_group,
        "phi_shap_std_group": phi_shap_std,
        "clinical_pen_group": clinical_pen,
        "stability_pen_group": stab_pen,
        "map_shap": map_shap,
        "map_ig": map_ig,
        "map_final": map_final,
        "ig_raw_map": ig_map,
        "interpretation_scope": "associational_non_causal",
    }