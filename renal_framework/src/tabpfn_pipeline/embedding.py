"""
tabpfn_pipeline.embedding

Embedding extraction utilities for TabPFN model internals, plus normalization, caching,
feature-scaler persistence, and test split preparation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
import joblib
from sklearn.preprocessing import StandardScaler


@dataclass
class EmbeddingExtractConfig:
    batch_size: int = 512
    max_extract: Optional[int] = None
    use_cache: bool = True


def flatten_embeddings(arr: np.ndarray) -> np.ndarray:
    """
    Flatten embeddings to [N, D] for downstream concept modules.
    """
    arr = np.asarray(arr)
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3 and arr.shape[1] == 1:
        sq = np.squeeze(arr, axis=1)
        return sq if sq.ndim == 2 else sq.reshape(sq.shape[0], -1)
    return arr.reshape(arr.shape[0], -1)


def make_dist_tensor(
    dist_np: np.ndarray,
    device: torch.device | str = "cpu",
    target_shape_example: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Build dist_shift_domain tensor with notebook-compatible shape behavior.
    """
    t = torch.tensor(dist_np, dtype=torch.long, device=device)
    if target_shape_example is not None and t.ndim == 1:
        if len(target_shape_example) >= 3:
            t = t.reshape(-1, 1, 1)
    else:
        if t.ndim == 1:
            t = t.reshape(-1, 1, 1)
    return t


def batch_get_embeddings_via_get_embeddings(
    model,
    X_all: np.ndarray,
    dist_full: np.ndarray,
    batch_size: int = 512,
    device: torch.device | str = "cpu",
    example_add_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[np.ndarray, list]:
    """
    Extract embeddings in batches using model.get_embeddings.
    Returns embeddings and a list of tensor clones (for gradient workflows if needed).
    """
    out_list = []
    tensors_list = []
    n = X_all.shape[0]

    for s in range(0, n, batch_size):
        e = min(s + batch_size, n)
        xb = X_all[s:e].astype(np.float32)
        dist_b = dist_full[s:e].astype(np.int64)

        dist_t = make_dist_tensor(
            dist_b,
            device=device,
            target_shape_example=example_add_shape,
        )

        try:
            emb_b = model.get_embeddings(xb, additional_x={"dist_shift_domain": dist_t})
        except (TypeError, RuntimeError):
            emb_b = model.get_embeddings(xb)

        if isinstance(emb_b, torch.Tensor):
            emb_b_np = emb_b.detach().cpu().numpy()
            emb_t = emb_b.detach().clone().requires_grad_(True)
        else:
            emb_b_np = np.asarray(emb_b)
            emb_t = torch.tensor(emb_b_np).clone().requires_grad_(True)

        out_list.append(np.asarray(emb_b_np))
        tensors_list.append(emb_t)

    return np.vstack(out_list), tensors_list


def extract_embeddings_robust(
    model,
    X: np.ndarray,
    years: np.ndarray,
    year_to_domain_map: Dict[int, int],
    cfg: EmbeddingExtractConfig = EmbeddingExtractConfig(),
    device: torch.device | str = "cpu",
    is_train: bool = False,
    ctx_idx: Optional[np.ndarray] = None,
    example_add_shape: Optional[Tuple[int, ...]] = None,
) -> np.ndarray:
    """
    Extract embeddings robustly with batch mode and optional sample cap/context exclusion.
    """
    X_all = np.asarray(X, dtype=np.float32)
    years_all = np.asarray(years).astype(int)

    if cfg.max_extract is not None:
        X_all = X_all[: cfg.max_extract]
        years_all = years_all[: cfg.max_extract]

    if is_train and ctx_idx is not None:
        exclude_mask = np.ones(X_all.shape[0], dtype=bool)
        exclude_mask[ctx_idx] = False
        X_all = X_all[exclude_mask]
        years_all = years_all[exclude_mask]

    dist_vec = np.asarray([year_to_domain_map[int(y)] for y in years_all], dtype=np.int64)

    if not hasattr(model, "get_embeddings"):
        raise RuntimeError("Model has no get_embeddings method; cannot extract embeddings.")

    emb, _ = batch_get_embeddings_via_get_embeddings(
        model=model,
        X_all=X_all,
        dist_full=dist_vec,
        batch_size=cfg.batch_size,
        device=device,
        example_add_shape=example_add_shape,
    )
    return np.asarray(emb)


def fit_embedding_scaler(train_emb_flat: np.ndarray) -> StandardScaler:
    """
    Fit StandardScaler on train embeddings only.
    """
    scaler = StandardScaler()
    scaler.fit(train_emb_flat)
    return scaler


def transform_embeddings(scaler: StandardScaler, emb_flat: np.ndarray) -> np.ndarray:
    """
    Transform embeddings with fitted scaler.
    """
    return scaler.transform(emb_flat)


def fit_or_load_feature_scaler(
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    feature_cols: list[str],
    scaler_path: str | Path,
) -> Tuple[StandardScaler, pd.DataFrame, pd.DataFrame]:
    """
    Notebook-aligned feature normalization for raw tabular inputs:
    - load existing scaler_standard.pkl if present
    - otherwise fit on train only
    - transform train/test with identical feature ordering
    """
    scaler_path = Path(scaler_path)

    X_train_df = train_rows[feature_cols].copy()
    X_test_df = test_rows[feature_cols].copy()

    if scaler_path.exists():
        scaler = joblib.load(scaler_path)
        if hasattr(scaler, "feature_names_in_"):
            feature_cols = scaler.feature_names_in_.tolist()
            X_train_df = train_rows[feature_cols].copy()
            X_test_df = test_rows[feature_cols].copy()
    else:
        scaler = StandardScaler()
        scaler.fit(X_train_df)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, scaler_path)

    X_train_norm = pd.DataFrame(
        scaler.transform(X_train_df),
        columns=X_train_df.columns,
        index=X_train_df.index,
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(X_test_df),
        columns=X_test_df.columns,
        index=X_test_df.index,
    )

    return scaler, X_train_norm, X_test_norm


def load_or_extract_embeddings(
    model,
    X_train_np: np.ndarray,
    X_test_np: np.ndarray,
    years_train: np.ndarray,
    years_test: np.ndarray,
    year_to_domain_map: Dict[int, int],
    embeddings_dir: str | Path,
    cfg: EmbeddingExtractConfig = EmbeddingExtractConfig(),
    device: torch.device | str = "cpu",
    example_add_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, np.ndarray]:
    """
    Notebook-aligned cached embedding workflow:
    - if train/test emb + flat exist, load
    - else extract and persist
    """
    embeddings_dir = Path(embeddings_dir)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    p_train = embeddings_dir / "train_emb.npy"
    p_test = embeddings_dir / "test_emb.npy"
    p_train_flat = embeddings_dir / "train_emb_flat.npy"
    p_test_flat = embeddings_dir / "test_emb_flat.npy"

    if cfg.use_cache and p_train.exists() and p_test.exists() and p_train_flat.exists() and p_test_flat.exists():
        train_emb = np.load(p_train)
        test_emb = np.load(p_test)
        train_emb_flat = np.load(p_train_flat)
        test_emb_flat = np.load(p_test_flat)
    else:
        train_emb = extract_embeddings_robust(
            model=model,
            X=X_train_np,
            years=years_train,
            year_to_domain_map=year_to_domain_map,
            cfg=cfg,
            device=device,
            is_train=True,
            ctx_idx=None,
            example_add_shape=example_add_shape,
        )
        test_emb = extract_embeddings_robust(
            model=model,
            X=X_test_np,
            years=years_test,
            year_to_domain_map=year_to_domain_map,
            cfg=cfg,
            device=device,
            is_train=False,
            ctx_idx=None,
            example_add_shape=example_add_shape,
        )

        train_emb_flat = flatten_embeddings(train_emb)
        test_emb_flat = flatten_embeddings(test_emb)

        np.save(p_train, train_emb)
        np.save(p_test, test_emb)
        np.save(p_train_flat, train_emb_flat)
        np.save(p_test_flat, test_emb_flat)

    return {
        "train_emb": train_emb,
        "test_emb": test_emb,
        "train_emb_flat": train_emb_flat,
        "test_emb_flat": test_emb_flat,
    }


def temporal_test_subsplits(y_test: np.ndarray, random_state: int = 42) -> Dict[str, np.ndarray]:
    """
    Build notebook-compatible test splits:
      discovery / cav_train / tcav_eval / held_out
    """
    from sklearn.model_selection import train_test_split

    y_test = np.asarray(y_test).astype(int)
    n_test = len(y_test)
    all_idx = np.arange(n_test)

    idx_discovery, idx_rest = train_test_split(
        all_idx,
        test_size=0.67,
        random_state=random_state,
        stratify=y_test,
    )

    idx_cav_train, idx_eval_hold = train_test_split(
        idx_rest,
        test_size=0.5,
        random_state=random_state,
        stratify=y_test[idx_rest],
    )

    idx_tcav_eval, idx_held_out = train_test_split(
        idx_eval_hold,
        test_size=0.5,
        random_state=random_state,
        stratify=y_test[idx_eval_hold],
    )

    # overlap safety checks (same notebook logic)
    assert len(set(idx_discovery) & set(idx_cav_train)) == 0, "Discovery overlaps with CAV Train!"
    assert len(set(idx_discovery) & set(idx_tcav_eval)) == 0, "Discovery overlaps with TCAV Eval!"
    assert len(set(idx_discovery) & set(idx_held_out)) == 0, "Discovery overlaps with Held-Out!"
    assert len(set(idx_cav_train) & set(idx_tcav_eval)) == 0, "CAV Train overlaps with TCAV Eval!"
    assert len(set(idx_cav_train) & set(idx_held_out)) == 0, "CAV Train overlaps with Held-Out!"
    assert len(set(idx_tcav_eval) & set(idx_held_out)) == 0, "TCAV Eval overlaps with Held-Out!"

    return {
        "idx_test_discover": np.asarray(idx_discovery),
        "idx_test_cav_train": np.asarray(idx_cav_train),
        "idx_test_tcav_eval": np.asarray(idx_tcav_eval),
        "idx_test_held_out": np.asarray(idx_held_out),
    }