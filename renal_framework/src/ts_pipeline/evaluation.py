"""
evaluation.py

Evaluation, aggregation, and artifact-writing helpers for temporal experiments.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


@dataclass
class RunArtifacts:
    run_id: str
    model_name: str
    model_class: str
    train_time_sec: float
    infer_time_sec: float
    lookback: int
    n_features: int
    n_train_samples: int
    n_test_samples: int
    threshold_selected: Optional[float] = None
    extra: Optional[Dict] = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def evaluate_per_year_with_threshold(
    y_true: np.ndarray,
    logits: np.ndarray,
    years: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    """
    Per-year metrics with explicit thresholding.
    """
    probs = sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    rows = []
    for yr in sorted(np.unique(years)):
        m = years == yr
        yt = y_true[m]
        yp = preds[m]

        f1m = f1_score(yt, yp, average="macro") if len(np.unique(yt)) > 1 else np.nan
        f1p = f1_score(yt, yp, pos_label=1) if yt.sum() > 0 else np.nan

        rows.append(
            {
                "year": int(yr),
                "n_samples": int(m.sum()),
                "n_deaths": int(yt.sum()),
                "f1_macro": None if np.isnan(f1m) else float(f1m),
                "f1_pos": None if np.isnan(f1p) else float(f1p),
                "threshold": float(threshold),
            }
        )

    return pd.DataFrame(rows).sort_values("year").reset_index(drop=True)


def save_run_outputs(
    run_dir: Path,
    eval_df: pd.DataFrame,
    pids: np.ndarray,
    years: np.ndarray,
    y_true: np.ndarray,
    logits: np.ndarray,
    artifacts: RunArtifacts,
) -> None:
    """
    Save canonical run outputs:
      - evaluation_by_year.csv
      - test_predictions.csv
      - run_config.json
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    eval_df.to_csv(run_dir / "evaluation_by_year.csv", index=False)

    probs = sigmoid(logits)
    pd.DataFrame(
        {
            "patient_id": pids,
            "year": years,
            "y_true": y_true,
            "logit": logits,
            "prob": probs,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)

    cfg = {
        "run_id": artifacts.run_id,
        "model_name": artifacts.model_name,
        "model_class": artifacts.model_class,
        "train_time_sec": float(artifacts.train_time_sec),
        "infer_time_sec": float(artifacts.infer_time_sec),
        "lookback": int(artifacts.lookback),
        "n_features": int(artifacts.n_features),
        "n_train_samples": int(artifacts.n_train_samples),
        "n_test_samples": int(artifacts.n_test_samples),
    }
    if artifacts.threshold_selected is not None:
        cfg["threshold_selected"] = float(artifacts.threshold_selected)
    if artifacts.extra:
        cfg.update(artifacts.extra)

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(cfg, f, indent=2)


def aggregate_all_runs(results_root: Path) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Aggregate all run folders that contain evaluation_by_year.csv.
    Saves:
      - all_runs_evaluation_by_year.csv
      - all_runs_summary.csv
    """
    run_dirs = [p for p in results_root.iterdir() if p.is_dir() and (p / "evaluation_by_year.csv").exists()]
    if not run_dirs:
        return None, None

    yearly_all = []
    summary_all = []

    for rd in sorted(run_dirs):
        run_id = rd.name
        eval_fp = rd / "evaluation_by_year.csv"
        cfg_fp = rd / "run_config.json"

        df_eval = pd.read_csv(eval_fp)
        df_eval["run_id"] = run_id
        df_eval["model_name"] = run_id.split("_")[0]
        yearly_all.append(df_eval)

        avg_f1_pos = df_eval["f1_pos"].dropna().mean() if "f1_pos" in df_eval else np.nan
        avg_f1_macro = df_eval["f1_macro"].dropna().mean() if "f1_macro" in df_eval else np.nan

        cfg = {}
        if cfg_fp.exists():
            with open(cfg_fp, "r") as f:
                cfg = json.load(f)

        summary_all.append(
            {
                "run_id": run_id,
                "model_name": cfg.get("model_name", run_id.split("_")[0]),
                "model_class": cfg.get("model_class", ""),
                "avg_f1_pos": None if np.isnan(avg_f1_pos) else float(avg_f1_pos),
                "avg_f1_macro": None if np.isnan(avg_f1_macro) else float(avg_f1_macro),
                "train_time_sec": cfg.get("train_time_sec", None),
                "infer_time_sec": cfg.get("infer_time_sec", None),
                "lookback": cfg.get("lookback", None),
                "n_features": cfg.get("n_features", None),
                "n_train_samples": cfg.get("n_train_samples", None),
                "n_test_samples": cfg.get("n_test_samples", None),
            }
        )

    yearly_df = pd.concat(yearly_all, ignore_index=True).sort_values(["run_id", "year"]).reset_index(drop=True)
    summary_df = pd.DataFrame(summary_all).sort_values(["avg_f1_pos", "avg_f1_macro"], ascending=False).reset_index(drop=True)

    yearly_df.to_csv(results_root / "all_runs_evaluation_by_year.csv", index=False)
    summary_df.to_csv(results_root / "all_runs_summary.csv", index=False)

    return yearly_df, summary_df