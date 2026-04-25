"""
interpretability.py

Model-agnostic interpretability utilities for sequence classifiers:
- Gradient x Input
- Integrated Gradients
- Embedding extraction helper dispatcher
- Global attribution maps
- Optional projection helpers
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def gradient_x_input_attribution(model: nn.Module, xb: torch.Tensor) -> torch.Tensor:
    """
    Compute Gradient x Input attribution for positive logit target.
    xb: [B, L, F]
    returns: [B, L, F]
    """
    model.eval()
    x = xb.clone().detach().requires_grad_(True)
    logits = model(x)
    score = logits.sum()
    score.backward()
    grad = x.grad
    attr = grad * x
    return attr.detach()


def integrated_gradients(
    model: nn.Module,
    xb: torch.Tensor,
    baseline: Optional[torch.Tensor] = None,
    steps: int = 32,
) -> torch.Tensor:
    """
    Approximate Integrated Gradients for binary logit target.
    xb: [B, L, F]
    returns: [B, L, F]
    """
    model.eval()
    if baseline is None:
        baseline = torch.zeros_like(xb)

    alphas = torch.linspace(0, 1, steps, device=xb.device).view(-1, 1, 1, 1)
    xb_exp = xb.unsqueeze(0)
    base_exp = baseline.unsqueeze(0)
    interp = base_exp + alphas * (xb_exp - base_exp)

    grads = []
    for s in range(steps):
        x_s = interp[s].clone().detach().requires_grad_(True)
        out = model(x_s).sum()
        out.backward()
        grads.append(x_s.grad.detach())

    grads = torch.stack(grads, dim=0)
    avg_grad = grads.mean(dim=0)
    ig = (xb - baseline) * avg_grad
    return ig.detach()


@torch.no_grad()
def predict_probs(model: nn.Module, loader: DataLoader, device: str = "cpu"):
    """
    Predict probabilities with metadata from loader.
    """
    model.eval()
    probs_all, y_all, years_all, pids_all = [], [], [], []
    for xb, yb, yrs, pids in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        probs_all.append(probs)
        y_all.append(yb.numpy())
        years_all.append(yrs.numpy())
        pids_all.extend(list(pids))

    probs = np.concatenate(probs_all)
    y = np.concatenate(y_all).astype(int)
    years = np.concatenate(years_all).astype(int)
    pids = np.asarray(pids_all, dtype=str)
    return probs, y, years, pids


def f1_pos_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float = 0.5) -> float:
    pred = (probs >= threshold).astype(int)
    if y_true.sum() == 0:
        return 0.0
    tp = ((pred == 1) & (y_true == 1)).sum()
    fp = ((pred == 1) & (y_true == 0)).sum()
    fn = ((pred == 0) & (y_true == 1)).sum()
    if (2 * tp + fp + fn) == 0:
        return 0.0
    return float((2 * tp) / (2 * tp + fp + fn))


def compute_global_attribution_maps(
    model: nn.Module,
    loader: DataLoader,
    method: str = "gxi",
    max_batches: int = 20,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      global_abs: [L, F] mean abs attribution
      global_signed: [L, F] mean signed attribution
    """
    abs_maps, signed_maps = [], []

    for i, (xb, _, _, _) in enumerate(loader):
        if i >= max_batches:
            break
        xb = xb.to(device, non_blocking=True)

        if method == "gxi":
            attr = gradient_x_input_attribution(model, xb)
        elif method == "ig":
            attr = integrated_gradients(model, xb, baseline=torch.zeros_like(xb), steps=24)
        else:
            raise ValueError("method must be 'gxi' or 'ig'")

        abs_maps.append(attr.abs().mean(dim=0).detach().cpu().numpy())
        signed_maps.append(attr.mean(dim=0).detach().cpu().numpy())

    global_abs = np.mean(np.stack(abs_maps, axis=0), axis=0)
    global_signed = np.mean(np.stack(signed_maps, axis=0), axis=0)
    return global_abs, global_signed


def top_feature_indices_from_attr(attr_abs_map: np.ndarray, top_k: int = 40) -> np.ndarray:
    """
    attr_abs_map: [L, F]
    return top feature indices by mean absolute attribution over lags.
    """
    feat_importance = attr_abs_map.mean(axis=0)
    return np.argsort(-feat_importance)[:top_k]