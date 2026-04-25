"""
training.py

Unified train/infer/eval utilities for TS classifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    device: str = "cpu"
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    early_stop_patience: int = 4


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, alpha: Optional[float] = None, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = probs * targets + (1 - probs) * (1 - targets)
        focal = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal = at * focal
        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


def _binary_metrics_from_logits(y_true: np.ndarray, logits: np.ndarray):
    probs = 1.0 / (1.0 + np.exp(-logits))
    preds = (probs >= 0.5).astype(int)

    f1_macro = f1_score(y_true, preds, average="macro") if len(np.unique(y_true)) > 1 else np.nan
    f1_pos = f1_score(y_true, preds, pos_label=1) if y_true.sum() > 0 else np.nan
    return probs, preds, f1_macro, f1_pos


def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    y_train_all: np.ndarray,
    cfg: TrainConfig,
    loss_name: str = "bce",
) -> nn.Module:
    model = model.to(cfg.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    y_train_all = y_train_all.astype(int)
    pos = y_train_all.sum()
    neg = len(y_train_all) - pos

    if loss_name == "bce":
        pos_weight = torch.tensor([neg / max(pos, 1)], dtype=torch.float32, device=cfg.device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    elif loss_name == "focal":
        criterion = BinaryFocalLoss(gamma=2.0, alpha=0.75, reduction="mean")
    else:
        raise ValueError(f"Unknown loss_name: {loss_name}")

    best_state = None
    best_loss = float("inf")
    patience = 0

    for ep in range(1, cfg.epochs + 1):
        model.train()
        ep_losses = []

        for xb, yb, _, _ in train_loader:
            xb = xb.to(cfg.device, non_blocking=True)
            yb = yb.to(cfg.device, non_blocking=True)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()

            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            optimizer.step()
            ep_losses.append(loss.item())

        train_loss = float(np.mean(ep_losses)) if ep_losses else np.nan
        print(f"[train] epoch={ep:02d} loss={train_loss:.6f}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= cfg.early_stop_patience:
                print(f"[train] early stop at epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


@torch.no_grad()
def predict_logits(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_logits, all_y, all_years, all_pids = [], [], [], []

    for xb, yb, yrs, pids in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb).detach().cpu().numpy()

        all_logits.append(logits)
        all_y.append(yb.numpy())
        all_years.append(yrs.numpy())
        all_pids.extend(list(pids))

    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0).astype(int)
    years = np.concatenate(all_years, axis=0).astype(int)
    pids = np.asarray(all_pids, dtype=str)
    return logits, y, years, pids


def evaluate_per_year(y_true: np.ndarray, logits: np.ndarray, years: np.ndarray) -> pd.DataFrame:
    rows = []
    for yr in sorted(np.unique(years)):
        mask = years == yr
        yt = y_true[mask]
        lg = logits[mask]
        _, _, f1_macro, f1_pos = _binary_metrics_from_logits(yt, lg)

        rows.append(
            {
                "year": int(yr),
                "n_samples": int(mask.sum()),
                "n_deaths": int(yt.sum()),
                "f1_macro": None if np.isnan(f1_macro) else float(f1_macro),
                "f1_pos": None if np.isnan(f1_pos) else float(f1_pos),
            }
        )
    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def tune_threshold_for_f1_pos(y_true: np.ndarray, logits: np.ndarray, grid=None):
    if grid is None:
        grid = np.arange(0.05, 0.96, 0.01)

    probs = 1.0 / (1.0 + np.exp(-logits))
    best_thr, best_f1 = 0.5, -1.0
    for thr in grid:
        pred = (probs >= thr).astype(int)
        f1 = f1_score(y_true, pred, pos_label=1) if y_true.sum() > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)