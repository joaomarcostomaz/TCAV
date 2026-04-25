"""
tabpfn_pipeline.rules

Decision-tree rule extraction utilities for concept interpretation.

This module reproduces notebook behavior:
- train per-factor trees over multiple activation percentiles
- extract text rules from trees
- choose best rule by precision/recall constraints
- output rules grouped by percentile and best percentile selection
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text


@dataclass
class RuleExtractionConfig:
    percentiles: List[int]
    max_depth: int = 15
    min_samples_leaf: float = 0.01
    min_positive_samples: int = 50
    min_rule_precision: float = 0.90
    min_rule_recall: float = 0.25
    random_state: int = 42


def tree_rules_to_df(tree: DecisionTreeClassifier, feature_names: List[str]) -> Tuple[pd.DataFrame, str]:
    """
    Convert sklearn tree text export into a path-based rule DataFrame.
    """
    tree_text = export_text(
        tree,
        feature_names=feature_names,
        max_depth=tree.get_depth(),
    )

    rules = []
    path = []

    for line in tree_text.split("\n"):
        if not line.strip():
            continue

        depth = line.count("|   ")

        if "class:" in line:
            cls = line.split("class:")[1].strip()
            rules.append({"Path": " AND ".join(path), "Class": cls})
        else:
            condition = line.split("|--- ")[-1].strip()
            path = path[:depth]
            path.append(condition)

    return pd.DataFrame(rules), tree_text


def _mask_from_path(path: str, X: np.ndarray, feature_cols: List[str]) -> np.ndarray:
    """
    Evaluate a single rule path mask against matrix X.
    """
    mask = np.ones(X.shape[0], dtype=bool)
    conditions = path.split(" AND ")

    for cond in conditions:
        cond = cond.strip()
        if not cond:
            continue

        if "<=" in cond:
            feat, thr = cond.split(" <= ")
            if feat not in feature_cols:
                continue
            feat_idx = feature_cols.index(feat)
            mask &= (X[:, feat_idx] <= float(thr))
        elif ">" in cond:
            feat, thr = cond.split(" > ")
            if feat not in feature_cols:
                continue
            feat_idx = feature_cols.index(feat)
            mask &= (X[:, feat_idx] > float(thr))

    return mask


def extract_rules_per_percentile(
    X_feat_dt_train: np.ndarray,
    factor_scores_dt_train: np.ndarray,
    feature_cols: List[str],
    cfg: RuleExtractionConfig,
) -> Tuple[Dict[int, List[Dict[str, Any]]], Dict[int, Dict[int, List[float]]]]:
    """
    Extract best rule per factor for each percentile.

    Parameters
    ----------
    X_feat_dt_train : np.ndarray
        Input feature matrix used to train trees.
    factor_scores_dt_train : np.ndarray
        Concept activations matrix [N, K].
    feature_cols : list[str]
        Feature names aligned with X columns.
    cfg : RuleExtractionConfig
        Hyperparameters and thresholds.

    Returns
    -------
    rules_per_percentile : dict
        {percentile: [rule_records]}
    thresholds_per_p : dict
        {percentile: {factor_idx: [thresholds_used]}}
    """
    n_factors = factor_scores_dt_train.shape[1]
    rules_per_percentile = {p: [] for p in cfg.percentiles}
    thresholds_per_p = {p: {k: [] for k in range(n_factors)} for p in cfg.percentiles}

    for k in range(n_factors):
        scores = factor_scores_dt_train[:, k]
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            continue

        for perc in cfg.percentiles:
            threshold = np.percentile(non_zero, perc)
            thresholds_per_p[perc][k].append(float(threshold))

            idx_pos = np.where(scores >= threshold)[0]
            y_high = np.zeros_like(scores, dtype=int)
            y_high[idx_pos] = 1

            # notebook fallback
            if y_high.sum() < cfg.min_positive_samples:
                idx_pos = np.where(scores > 0)[0]
                y_high = np.zeros_like(scores, dtype=int)
                y_high[idx_pos] = 1
                if y_high.sum() < cfg.min_positive_samples:
                    continue

            tree = DecisionTreeClassifier(
                max_depth=cfg.max_depth,
                min_samples_leaf=cfg.min_samples_leaf,
                random_state=cfg.random_state,
            )
            tree.fit(X_feat_dt_train, y_high)

            total_positives = int((y_high == 1).sum())
            if total_positives == 0:
                continue

            rules_df, tree_text = tree_rules_to_df(tree, feature_cols)

            importances = tree.feature_importances_
            top_features_idx = np.argsort(importances)[-10:][::-1]
            top_features = [(feature_cols[i], float(importances[i])) for i in top_features_idx]

            best_rule = None
            best_recall = 0.0
            best_prec = None

            for _, row in rules_df.iterrows():
                path = row["Path"]
                mask = _mask_from_path(path, X_feat_dt_train, feature_cols)

                if mask.sum() == 0:
                    continue

                tp_covered = int(((y_high == 1) & mask).sum())
                rule_recall = tp_covered / max(total_positives, 1)
                rule_precision = float(y_high[mask].mean())

                if rule_precision < cfg.min_rule_precision or rule_recall < cfg.min_rule_recall:
                    continue

                if rule_recall > best_recall:
                    best_recall = float(rule_recall)
                    best_prec = float(rule_precision)
                    best_rule = row

            if best_rule is not None:
                rules_per_percentile[perc].append(
                    {
                        "Factor": int(k),
                        "Path": best_rule["Path"],
                        "Class": best_rule["Class"],
                        "Top_features": top_features,
                        "Precision": float(best_prec),
                        "Recall": float(best_recall),
                        "tree_text": tree_text,
                    }
                )

    return rules_per_percentile, thresholds_per_p


def select_best_percentile(rules_per_percentile: Dict[int, List[Dict[str, Any]]]) -> int:
    """
    Select percentile that yields the largest number of extracted rules.
    """
    if len(rules_per_percentile) == 0:
        raise ValueError("rules_per_percentile is empty.")
    return max(rules_per_percentile.keys(), key=lambda p: len(rules_per_percentile[p]))


def rules_to_dataframe(rules_per_percentile: Dict[int, List[Dict[str, Any]]], percentile: int) -> pd.DataFrame:
    """
    Convert selected percentile rules to DataFrame.
    """
    if percentile not in rules_per_percentile:
        return pd.DataFrame()
    return pd.DataFrame(rules_per_percentile[percentile])