"""
tabpfn_pipeline.evaluation

Notebook-aligned evaluation utilities for Drift-Resilient TabPFN:
- model loading/fallback + fit
- notebook-compatible dist tensor construction
- strict per-year walk-forward evaluation
- artifact persistence
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import time
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import tabpfn
from importlib import resources
from tabpfn.best_models import get_best_tabpfn, TabPFNModelPathsConfig
from tabpfn import TabPFNClassifier


@dataclass
class TabPFNEvalConfig:
    rng_seed: int = 42
    tabpfn_model_name: str = "tabpfn_dist_model_1"
    batch_size_predict: int = 512
    run_id: str = "single_fast_balanced_safe"


def infer_model_additional_x_device(model) -> torch.device:
    """
    Infer internal device where model stores additional_x_ tensors.
    Notebook-compatible behavior.
    """
    device = torch.device("cpu")
    try:
        if (
            hasattr(model, "additional_x_")
            and model.additional_x_ is not None
            and "dist_shift_domain" in model.additional_x_
        ):
            v = model.additional_x_["dist_shift_domain"]
            if isinstance(v, torch.Tensor):
                device = v.device
    except Exception:
        device = torch.device("cpu")
    return device


def make_dist_tensor(
    dist_np: np.ndarray,
    device: torch.device | str,
    target_shape_example: Optional[Tuple[int, ...]] = None,
) -> torch.Tensor:
    """
    Notebook-compatible tensor shaper for dist_shift_domain.
    """
    t = torch.tensor(dist_np, dtype=torch.long, device=device)
    if target_shape_example is not None and t.ndim == 1:
        if len(target_shape_example) >= 3:
            t = t.reshape(-1, 1, 1)
    else:
        if t.ndim == 1:
            t = t.reshape(-1, 1, 1)
    return t


def fit_drift_resilient_tabpfn(
    X_train_np: np.ndarray,
    y_train_np: np.ndarray,
    train_years: np.ndarray,
    eval_cfg: TabPFNEvalConfig,
) -> Dict[str, Any]:
    """
    Load drift-resilient TabPFN model (preferred path) with fallback,
    then fit using dist_shift_domain additional_x.

    Returns dict with:
      - model
      - fit_time_sec
      - year_to_domain_train
      - dist_shift_domain_train_np
      - model_add_x_device
      - example_add_shape
    """
    # Ensure writable arrays (TabPFN may mutate y internally)
    X_train_np = np.ascontiguousarray(X_train_np.astype(np.float32, copy=True))
    y_train_np = np.ascontiguousarray(y_train_np.astype(np.float32, copy=True))
    X_train_np.setflags(write=True)
    y_train_np.setflags(write=True)

    years_train = np.asarray(train_years).astype(int)
    year_to_domain_train = {y: i for i, y in enumerate(sorted(set(years_train.tolist())))}
    dist_shift_domain_train_np = np.ascontiguousarray(
        np.array([year_to_domain_train[int(y)] for y in years_train], dtype=np.int64)
    )
    dist_shift_domain_train_np.setflags(write=True)

    # Load model
    try:
        libpath = resources.files(tabpfn)
        model_path_config = TabPFNModelPathsConfig(
            paths=[f"{libpath}/model_cache/{eval_cfg.tabpfn_model_name}.cpkt"],
            task_type="dist_shift_multiclass",
        )
        drift_model = get_best_tabpfn(
            task_type="dist_shift_multiclass",
            model_type="single_fast",
            paths_config=model_path_config,
            debug=False,
            device="auto",
        )
        if hasattr(drift_model, "show_progress"):
            drift_model.show_progress = False
        if hasattr(drift_model, "seed"):
            drift_model.seed = eval_cfg.rng_seed
    except Exception:
        drift_model = TabPFNClassifier(device="auto")

    # Fit
    t0 = time.perf_counter()
    drift_model = drift_model.fit(
        X_train_np,
        y_train_np,
        additional_x={"dist_shift_domain": dist_shift_domain_train_np},
    )
    fit_time_sec = time.perf_counter() - t0

    model_add_x_device = infer_model_additional_x_device(drift_model)

    example_add_shape = None
    try:
        if (
            hasattr(drift_model, "additional_x_")
            and drift_model.additional_x_ is not None
            and "dist_shift_domain" in drift_model.additional_x_
        ):
            v = drift_model.additional_x_["dist_shift_domain"]
            if isinstance(v, torch.Tensor):
                example_add_shape = tuple(v.shape)
    except Exception:
        example_add_shape = None

    return {
        "model": drift_model,
        "fit_time_sec": float(fit_time_sec),
        "year_to_domain_train": year_to_domain_train,
        "dist_shift_domain_train_np": dist_shift_domain_train_np,
        "model_add_x_device": model_add_x_device,
        "example_add_shape": example_add_shape,
    }


def ensure_test_feature_columns(test_rows: pd.DataFrame, top_k_events: List[str]) -> pd.DataFrame:
    """
    Ensure all train-selected features exist in test_rows.
    Missing features are zero-filled.
    """
    out = test_rows.copy()
    missing = [c for c in top_k_events if c not in out.columns]
    if missing:
        zeros = pd.DataFrame(0, index=out.index, columns=missing)
        out = pd.concat([out, zeros], axis=1)
    return out


def walkforward_evaluate_tabpfn(
    drift_model,
    test_rows: pd.DataFrame,
    top_k_events: List[str],
    train_years: List[int],
    model_add_x_device: torch.device | str,
    batch_size_predict: int = 512,
    example_add_shape: Optional[Tuple[int, ...]] = None,
) -> Dict[str, Any]:
    """
    Strict per-year walk-forward evaluation used in notebook:
    - loop over eval_year
    - batch predict_proba with dist_shift_domain
    - compute f1_macro and f1_pos
    """
    test_rows = ensure_test_feature_columns(test_rows, top_k_events)

    combined_years = sorted(set(train_years).union(set(test_rows["year"].astype(int).unique().tolist())))
    year_to_domain_combined = {y: i for i, y in enumerate(combined_years)}

    results_per_year: List[Dict[str, Any]] = []
    t_infer_total = 0.0

    for eval_year in sorted(test_rows["year"].astype(int).unique()):
        df_year = test_rows[test_rows["year"] == eval_year].reset_index(drop=True)
        if df_year.shape[0] == 0:
            continue

        n_samples = len(df_year)
        y_true = (df_year["DEATH"] > 0).astype(int).values
        dist_dom_all = np.array(
            [year_to_domain_combined[int(y)] for y in df_year["year"].astype(int)],
            dtype=np.int64,
        )

        preds_list = []
        t0_year = time.perf_counter()

        for start in range(0, n_samples, batch_size_predict):
            end = min(start + batch_size_predict, n_samples)

            Xb_np = df_year[top_k_events].iloc[start:end].values.astype(np.float32)
            dist_dom_np = dist_dom_all[start:end]

            dist_dom_t = make_dist_tensor(
                dist_np=dist_dom_np,
                device=model_add_x_device,
                target_shape_example=example_add_shape,
            )

            if torch.device(model_add_x_device).type == "cpu":
                preds_proba_batch = drift_model.predict_proba(
                    Xb_np,
                    additional_x={"dist_shift_domain": dist_dom_t},
                )
            else:
                Xb_t = torch.tensor(Xb_np, dtype=torch.float32, device=model_add_x_device)
                with torch.no_grad():
                    preds_proba_batch = drift_model.predict_proba(
                        Xb_t,
                        additional_x={"dist_shift_domain": dist_dom_t},
                    )

            if isinstance(preds_proba_batch, torch.Tensor):
                preds_proba_batch = preds_proba_batch.detach().cpu().numpy()
            preds_list.append(preds_proba_batch)

        t_infer_year = time.perf_counter() - t0_year
        t_infer_total += t_infer_year

        preds_proba = np.vstack(preds_list)
        y_pred = np.argmax(preds_proba, axis=1)

        f1_macro = f1_score(y_true, y_pred, average="macro") if len(np.unique(y_true)) > 1 else float("nan")
        f1_pos = f1_score(y_true, y_pred, pos_label=1) if y_true.sum() > 0 else float("nan")

        results_per_year.append(
            {
                "year": int(eval_year),
                "n_samples": int(n_samples),
                "n_deaths": int(y_true.sum()),
                "f1_macro": float(f1_macro) if not np.isnan(f1_macro) else None,
                "f1_pos": float(f1_pos) if not np.isnan(f1_pos) else None,
                "infer_time_sec": float(t_infer_year),
            }
        )

    return {
        "results_per_year": results_per_year,
        "total_infer_time_sec": float(t_infer_total),
        "year_to_domain_combined": year_to_domain_combined,
        "test_rows_checked": test_rows,
    }


def save_tabpfn_temporal_artifacts(
    out_dir: Path,
    results_per_year: List[Dict[str, Any]],
    train_rows: pd.DataFrame,
    test_rows: pd.DataFrame,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Save notebook-equivalent outputs:
      - evaluation_by_year.csv
      - train_rows_temporal.csv
      - test_rows_temporal.csv
      - optional run_meta.json
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(results_per_year).to_csv(out_dir / "evaluation_by_year.csv", index=False)
    train_rows.to_csv(out_dir / "train_rows_temporal.csv", index=False)
    test_rows.to_csv(out_dir / "test_rows_temporal.csv", index=False)

    if meta is not None:
        import json
        with (out_dir / "run_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)