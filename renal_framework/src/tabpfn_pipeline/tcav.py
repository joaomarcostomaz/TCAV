"""
tabpfn_pipeline.tcav

TCAV and concept-rule utilities for TabPFN embedding analysis:
- Decision-tree rule parsing/application
- CAV dataclass + training
- Gradient-based directional derivatives
- Robust significance testing with random baselines
- SAE rule-conditioned CAV training (notebook parity)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import re
import warnings
import torch.utils.checkpoint as cp

import numpy as np
import pandas as pd
import torch
from scipy import stats
from sklearn.linear_model import LogisticRegression


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class CAV:
    cluster_id: int
    size_pure: int
    vector: np.ndarray
    pure_indices: np.ndarray
    negative_indices: np.ndarray
    classifier: LogisticRegression
    centroid: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.float32))
    quantile: float = 0.05
    


@dataclass
class TCAVGradientResult:
    cluster_id: int
    tcav_positive_fraction: float
    mean_derivative: float
    derivatives: np.ndarray
    all_grads: np.ndarray


def _is_concept_entry(k: Any, v: Any) -> bool:
    """
    Notebook-parity concept dict guard.
    """
    return isinstance(k, (int, np.integer)) and isinstance(v, dict) and ("v_activ" in v)


def build_cavs_for_gradient(
    cavs_source: Dict[Any, Any],
    quantile: float = 0.05,
) -> Dict[int, CAV]:
    """
    Convert raw cav_dict_* entries into CAV dataclass objects.
    """
    cavs_for_gradient: Dict[int, CAV] = {}
    for k, entry in cavs_source.items():
        if not _is_concept_entry(k, entry):
            continue
        cavs_for_gradient[int(k)] = CAV(
            cluster_id=int(k),
            size_pure=int(entry.get("size_pos", 0)),
            vector=np.asarray(entry["v_activ"], dtype=np.float32),
            pure_indices=np.asarray(entry["pos_idx"]),
            negative_indices=np.asarray(entry["neg_idx"]),
            classifier=entry["clf"],
            centroid=np.array([], dtype=np.float32),
            quantile=float(quantile),
        )
    return cavs_for_gradient


def prepare_model_for_tcav_gradients(model_proc: torch.nn.Module) -> torch.nn.Module:
    """
    Notebook-parity memory/checkpoint prep before gradient extraction.
    """
    model_proc.eval()

    if hasattr(model_proc, "reset_save_peak_mem_factor"):
        model_proc.reset_save_peak_mem_factor(None)

    if hasattr(model_proc, "transformer_encoder") and hasattr(model_proc.transformer_encoder, "recompute_each_layer"):
        model_proc.transformer_encoder.recompute_each_layer = False

    if hasattr(model_proc, "transformer_decoder") and model_proc.transformer_decoder is not None:
        if hasattr(model_proc.transformer_decoder, "recompute_each_layer"):
            model_proc.transformer_decoder.recompute_each_layer = False

    # Disable checkpoint recomputation in this analysis path
    cp.checkpoint = lambda func, *args, **kwargs: func(*args)
    return model_proc


def compute_random_baseline(
    gradients: np.ndarray,
    n_random: int = 100,
    seed: int = 42,
) -> np.ndarray:
    """
    Notebook-parity random-unit-vector TCAV baseline.
    """
    rng = np.random.default_rng(seed)
    emb_dim = gradients.shape[1]
    random_tcav_scores = []

    for _ in range(n_random):
        v_random = rng.standard_normal(emb_dim).astype(np.float32)
        v_random = v_random / (np.linalg.norm(v_random) + 1e-12)
        directional = gradients @ v_random
        score = float((directional > 0).mean())
        random_tcav_scores.append(score)

    return np.asarray(random_tcav_scores, dtype=float)


def summarize_random_baseline(random_baseline: np.ndarray) -> Dict[str, float]:
    """
    Summary stats helper for random baseline distributions.
    """
    rb = np.asarray(random_baseline, dtype=float)
    return {
        "mean": float(rb.mean()),
        "std": float(rb.std()),
        "min": float(rb.min()),
        "max": float(rb.max()),
        "p95": float(np.percentile(rb, 95)),
        "p05": float(np.percentile(rb, 5)),
    }


def get_analysis_data(
    analysis_source: str,
    results_dl: Optional[Dict[int, TCAVGradientResult]],
    cav_dict_activ: Optional[Dict[Any, Any]],
    all_grads_dl: Optional[np.ndarray],
    results_sae: Optional[Dict[int, TCAVGradientResult]],
    cav_dict_sae: Optional[Dict[Any, Any]],
    all_grads_sae: Optional[np.ndarray],
) -> Tuple[Dict[int, TCAVGradientResult], Dict[Any, Any], np.ndarray]:
    """
    Notebook-parity selector:
      if analysis_source == 'dl' -> (results_dl, cav_dict_activ, all_grads_dl)
      else -> (results_sae, cav_dict_sae, all_grads_sae)
    """
    src = analysis_source.lower()
    if src in ("dl", "dictionary_learning", "dictionary"):
        if results_dl is None or cav_dict_activ is None or all_grads_dl is None:
            raise ValueError("DL analysis requested but DL artifacts are missing.")
        return results_dl, cav_dict_activ, all_grads_dl

    if results_sae is None or cav_dict_sae is None or all_grads_sae is None:
        raise ValueError("SAE analysis requested but SAE artifacts are missing.")
    return results_sae, cav_dict_sae, all_grads_sae


def filter_significant_factors_by_tcav(
    robust_tcav_results: Dict[int, Dict[str, Any]],
    min_distance_from_half: float = 0.1,
) -> Tuple[List[int], List[float], List[float]]:
    """
    Notebook-parity filter for significant concepts:
      significant == True and |mean_tcav - 0.5| > threshold
    """
    filtered = []
    tcavs = []
    all_tcavs = []

    for factor, result in robust_tcav_results.items():
        mt = float(result.get("mean_concept_tcav", np.nan))
        all_tcavs.append(mt)

        if bool(result.get("significant", False)) and abs(mt - 0.5) > float(min_distance_from_half):
            filtered.append(int(factor))
            tcavs.append(mt)

    return filtered, tcavs, all_tcavs


def build_matched_rules_tcav_table(
    rules_per_percentile: Dict[int, List[Dict[str, Any]]],
    best_p: int,
    filtered_factors: List[int],
    robust_tcav_results: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Notebook-parity matched table:
    intersection of extracted-rule factors and TCAV-significant factors.
    """
    rules_best_p = rules_per_percentile.get(best_p, [])
    rules_df = pd.DataFrame(rules_best_p)

    if rules_df.empty:
        return rules_df

    factors_with_rules = set(rules_df["Factor"].tolist())
    matched_factors = factors_with_rules.intersection(set(filtered_factors))

    matched_df = rules_df[rules_df["Factor"].isin(matched_factors)].copy()
    matched_df["Mean TCAV"] = matched_df["Factor"].apply(
        lambda f: robust_tcav_results[int(f)]["mean_concept_tcav"]
        if int(f) in robust_tcav_results else np.nan
    )
    return matched_df


# -----------------------------
# Rule parsing / application
# -----------------------------
def parse_decision_tree_rule(rule_string: str) -> List[Tuple[str, str, float]]:
    """
    Parse rule string like:
      'EVENT_X > 0.5 AND DIAG_Y <= 1.5'
    into list of (feature, op, threshold).
    """
    if not rule_string or pd.isna(rule_string):
        return []

    conditions = []
    parts = str(rule_string).split(" AND ")

    for part in parts:
        part = part.strip()
        if not part:
            continue

        match_le = re.match(r"(.+?)\s*<=\s*([-\d.]+)", part)
        match_lt = re.match(r"(.+?)\s*<\s*([-\d.]+)", part)
        match_ge = re.match(r"(.+?)\s*>=\s*([-\d.]+)", part)
        match_gt = re.match(r"(.+?)\s*>\s*([-\d.]+)", part)

        if match_le:
            conditions.append((match_le.group(1).strip(), "<=", float(match_le.group(2))))
        elif match_lt:
            conditions.append((match_lt.group(1).strip(), "<", float(match_lt.group(2))))
        elif match_ge:
            conditions.append((match_ge.group(1).strip(), ">=", float(match_ge.group(2))))
        elif match_gt:
            conditions.append((match_gt.group(1).strip(), ">", float(match_gt.group(2))))

    return conditions


def apply_rule_conditions(
    X_df_or_np: pd.DataFrame | np.ndarray,
    conditions: List[Tuple[str, str, float]],
    feature_cols: List[str],
) -> np.ndarray:
    """
    Apply parsed rule conditions and return boolean mask.
    """
    n_samples = len(X_df_or_np)
    if len(conditions) == 0:
        return np.zeros(n_samples, dtype=bool)

    mask = np.ones(n_samples, dtype=bool)

    for feature, op, thresh in conditions:
        if feature not in feature_cols:
            continue

        if isinstance(X_df_or_np, np.ndarray):
            feat_idx = feature_cols.index(feature)
            vals = X_df_or_np[:, feat_idx]
        else:
            vals = X_df_or_np[feature].values

        if op == "<=":
            cm = vals <= thresh
        elif op == "<":
            cm = vals < thresh
        elif op == ">=":
            cm = vals >= thresh
        elif op == ">":
            cm = vals > thresh
        else:
            continue

        mask &= cm

    return mask


# -----------------------------
# CAV training
# -----------------------------
def _to_original_space_vector(w_std: np.ndarray, scaler_emb) -> np.ndarray:
    """
    Convert linear model weight from standardized embedding space to original embedding space.
    """
    if hasattr(scaler_emb, "scale_"):
        v_original = w_std / scaler_emb.scale_
    else:
        v_original = w_std
    nrm = np.linalg.norm(v_original)
    if nrm == 0:
        return v_original.astype(np.float32)
    return (v_original / nrm).astype(np.float32)


def sample_label_balanced(
    pool_idx: np.ndarray,
    y: np.ndarray,
    target_counts: Dict[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    chosen = []
    pool_idx = np.asarray(pool_idx)

    for label, count in target_counts.items():
        cands = pool_idx[y[pool_idx] == label]
        if len(cands) >= count:
            chosen.append(rng.choice(cands, size=count, replace=False))
        else:
            chosen.append(cands)

    if len(chosen) == 0:
        return np.array([], dtype=int)

    chosen = np.concatenate(chosen)
    total_needed = int(sum(target_counts.values()))
    if chosen.size < total_needed:
        remaining = np.setdiff1d(pool_idx, chosen)
        if remaining.size > 0:
            extra_n = min(total_needed - chosen.size, remaining.size)
            chosen = np.concatenate([chosen, rng.choice(remaining, size=extra_n, replace=False)])

    return chosen


def train_cavs_from_activations(
    embeddings_cav_train: np.ndarray,
    activations_cav_train: np.ndarray,
    y_cav_train: np.ndarray,
    scaler_emb,
    pure_quantile: float = 0.05,
    min_pure: int = 10,
    random_state: int = 42,
) -> Dict[Any, Any]:
    """
    Train one CAV per concept (column in activations).
    """
    rng = np.random.default_rng(random_state)
    cav_dict = {}
    cav_vectors = []

    n_samples, n_factors = activations_cav_train.shape

    for k in range(n_factors):
        scores = activations_cav_train[:, k]
        high_thr = np.quantile(scores, 1.0 - pure_quantile)
        low_thr = np.quantile(scores, pure_quantile)

        idx_pos = np.where(scores >= high_thr)[0]
        idx_neg_pool = np.where(scores <= low_thr)[0]

        if len(idx_pos) < min_pure or len(idx_neg_pool) < min_pure:
            continue

        labels, counts = np.unique(y_cav_train[idx_pos], return_counts=True)
        target_counts = {int(l): int(c) for l, c in zip(labels, counts)}
        idx_neg = sample_label_balanced(idx_neg_pool, y_cav_train, target_counts, rng)

        if len(idx_neg) == 0:
            continue

        Xc = np.vstack([embeddings_cav_train[idx_pos], embeddings_cav_train[idx_neg]])
        yc = np.hstack([np.ones(len(idx_pos), dtype=int), np.zeros(len(idx_neg), dtype=int)])

        clf = LogisticRegression(
            C=0.1,
            solver="liblinear",
            penalty="l2",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
        clf.fit(Xc, yc)

        w_std = clf.coef_.ravel()
        v_activ = _to_original_space_vector(w_std, scaler_emb)
        if np.linalg.norm(v_activ) == 0:
            continue

        cav_vectors.append(v_activ)

        cav_dict[int(k)] = {
            "factor": int(k),
            "size_pos": int(len(idx_pos)),
            "pos_idx": idx_pos,
            "neg_idx": idx_neg,
            "clf": clf,
            "v_activ": v_activ,
            "high_thr": float(high_thr),
            "low_thr": float(low_thr),
        }

    if len(cav_vectors) > 0:
        cav_dict["cav_vectors_matrix"] = np.column_stack(cav_vectors)

    return cav_dict


def train_sae_cavs_from_rules(
    rules_for_percentile: List[Dict[str, Any]],
    X_cav_train_df: pd.DataFrame,
    cav_train_embeddings_np: np.ndarray,
    codes_cav_train_sae: np.ndarray,
    y_cav_train: np.ndarray,
    feature_cols: List[str],
    scaler_emb,
    pure_quantile_sae: float,
    min_pure_sae: int = 50,
    random_state: int = 42,
) -> Dict[Any, Any]:
    """
    Notebook-parity SAE CAV training from decision-tree rules:
    - positives: samples matching rule path
    - negatives: non-matching low-activation subset
    """
    rng = np.random.default_rng(random_state)
    cav_dict_sae = {}
    cav_vectors_list_sae = []

    for rule in rules_for_percentile:
        factor = int(rule["Factor"])
        path = rule["Path"]

        scores = codes_cav_train_sae[:, factor]
        non_zero = scores[scores > 0]
        if len(non_zero) == 0:
            continue

        conditions = parse_decision_tree_rule(path)
        mask_cav_train = apply_rule_conditions(X_cav_train_df, conditions, feature_cols)
        matched_idx = np.where(mask_cav_train)[0]

        idx_neg_pool = np.where(~mask_cav_train)[0]
        low_thr = np.quantile(scores, max(pure_quantile_sae - 0.1, 0.0))
        idx_neg_low_activ = idx_neg_pool[scores[idx_neg_pool] <= low_thr]

        idx_pos = np.where(mask_cav_train)[0]
        n_neg_target = len(idx_pos)

        if len(idx_pos) < min_pure_sae:
            continue

        if len(idx_neg_low_activ) >= n_neg_target and n_neg_target > 0:
            idx_neg = rng.choice(idx_neg_low_activ, size=n_neg_target, replace=False)
        elif len(idx_neg_low_activ) >= min_pure_sae:
            idx_neg = idx_neg_low_activ
        else:
            continue

        Xc = np.vstack([cav_train_embeddings_np[idx_pos], cav_train_embeddings_np[idx_neg]])
        yc = np.hstack([np.ones(len(idx_pos), dtype=int), np.zeros(len(idx_neg), dtype=int)])

        clf = LogisticRegression(
            C=0.1,
            solver="liblinear",
            penalty="l2",
            class_weight="balanced",
            max_iter=5000,
            random_state=random_state,
        )
        clf.fit(Xc, yc)

        w_std = clf.coef_.ravel()
        v_original = w_std / scaler_emb.scale_ if hasattr(scaler_emb, "scale_") else w_std
        nrm = np.linalg.norm(v_original)
        if nrm == 0:
            continue
        v_activ = (v_original / nrm).astype(np.float32)

        cav_vectors_list_sae.append(v_activ)

        cav_dict_sae[int(factor)] = {
            "factor": int(factor),
            "size_pos": int(len(idx_pos)),
            "pos_idx": idx_pos,
            "neg_idx": idx_neg,
            "clf": clf,
            "v_activ": v_activ,
            "high_thr": 0.0,
            "low_thr": float(low_thr),
            "method": "sae_rule_conditioned",
        }

    if len(cav_vectors_list_sae) > 0:
        cav_dict_sae["cav_vectors_matrix"] = np.column_stack(cav_vectors_list_sae)

    return cav_dict_sae


# -----------------------------
# Gradients + TCAV
# -----------------------------
def compute_decoder_gradients_vmap(
    decoder: torch.nn.Module,
    embeddings: torch.Tensor,
    target_class: int = 1,
) -> torch.Tensor:
    """
    Compute per-sample gradients wrt decoder input embeddings.
    embeddings: [N, D]
    returns: [N, D]
    """
    from torch.func import vmap, grad

    def single_forward(single_input):
        logits = decoder(single_input.unsqueeze(0))
        if logits.ndim == 3:
            logits = logits[0]
        return logits[0, target_class]

    per_sample_grad = vmap(grad(single_forward))
    return per_sample_grad(embeddings)


def compute_tcav_from_gradients(
    all_grads: np.ndarray,
    cav_dict: Dict[Any, Any],
) -> Dict[int, TCAVGradientResult]:
    """
    Directional derivative based TCAV from precomputed gradients and CAV vectors.
    """
    results = {}
    for cid, entry in cav_dict.items():
        if not isinstance(cid, int):
            continue
        if "v_activ" not in entry:
            continue

        v = np.asarray(entry["v_activ"], dtype=np.float32)
        directional = all_grads @ v
        tcav_pos = float((directional > 0).mean())
        mean_derivative = float(np.mean(directional))

        results[int(cid)] = TCAVGradientResult(
            cluster_id=int(cid),
            tcav_positive_fraction=tcav_pos,
            mean_derivative=mean_derivative,
            derivatives=directional.astype(np.float32),
            all_grads=all_grads.astype(np.float32),
        )

    return results


# -----------------------------
# Robust significance testing
# -----------------------------
def train_cav_from_subset(
    embeddings: np.ndarray,
    idx_concept_full: np.ndarray,
    idx_non_concept_full: np.ndarray,
    scaler_emb,
    sample_fraction: float = 1.0,
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[LogisticRegression]]:
    """
    Train CAV on random concept/non-concept subsets.
    """
    rng = np.random.default_rng(random_state)

    n_concept = len(idx_concept_full)
    n_sample_concept = max(2, int(n_concept * sample_fraction))
    idx_concept = rng.choice(idx_concept_full, size=min(n_sample_concept, n_concept), replace=False)

    n_non = len(idx_non_concept_full)
    n_target_non = len(idx_concept)
    if n_non == 0:
        return None, None
    idx_non = rng.choice(idx_non_concept_full, size=min(n_target_non, n_non), replace=False)

    X_concept = embeddings[idx_concept]
    X_non = embeddings[idx_non]
    X_train = np.vstack([X_concept, X_non])
    y_train = np.hstack([np.ones(len(X_concept)), np.zeros(len(X_non))])

    clf = LogisticRegression(
        C=0.1,
        solver="liblinear",
        penalty="l2",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    v = _to_original_space_vector(clf.coef_.ravel(), scaler_emb)
    if np.linalg.norm(v) == 0:
        return None, None
    return v, clf


def generate_random_cav(
    embeddings: np.ndarray,
    scaler_emb,
    n_samples: int = 50,
    random_state: int = 42,
) -> Tuple[Optional[np.ndarray], Optional[LogisticRegression]]:
    """
    Random CAV by classifying random partition.
    """
    rng = np.random.default_rng(random_state)
    n_total = embeddings.shape[0]
    all_idx = np.arange(n_total)
    rng.shuffle(all_idx)

    n = min(n_samples, n_total // 2)
    if n < 2:
        return None, None

    idx_pos = all_idx[:n]
    idx_neg = all_idx[n: 2 * n]

    X_pos = embeddings[idx_pos]
    X_neg = embeddings[idx_neg]
    X_train = np.vstack([X_pos, X_neg])
    y_train = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    clf = LogisticRegression(
        C=0.1,
        solver="liblinear",
        penalty="l2",
        class_weight="balanced",
        max_iter=5000,
        random_state=random_state,
    )
    clf.fit(X_train, y_train)

    v = _to_original_space_vector(clf.coef_.ravel(), scaler_emb)
    if np.linalg.norm(v) == 0:
        return None, None
    return v, clf


def compute_tcav_score_for_cav(cav_vector: np.ndarray, gradients: np.ndarray) -> float:
    directional = gradients @ cav_vector
    return float((directional > 0).mean())


def robust_tcav_significance_test(
    concept_id: int,
    embeddings: np.ndarray,
    idx_concept: np.ndarray,
    idx_non_concept: np.ndarray,
    gradients: np.ndarray,
    scaler_emb,
    n_runs: int = 15,
    sample_fraction: float = 1.0,
    seed_base: int = 42,
) -> Dict[str, Any]:
    """
    Robust test:
      - N concept-CAV runs
      - N random-CAV runs
      - t-test + Cohen's d
    """
    concept_scores = []
    random_scores = []
    concept_cavs = []
    random_cavs = []

    for i in range(n_runs):
        seed = seed_base + i * 100
        cav_vec, _ = train_cav_from_subset(
            embeddings=embeddings,
            idx_concept_full=idx_concept,
            idx_non_concept_full=idx_non_concept,
            scaler_emb=scaler_emb,
            sample_fraction=sample_fraction,
            random_state=seed,
        )
        if cav_vec is None:
            continue
        concept_cavs.append(cav_vec)
        concept_scores.append(compute_tcav_score_for_cav(cav_vec, gradients))

    n_random_samples = max(20, min(len(idx_concept), len(idx_non_concept)) // 2)
    for i in range(n_runs):
        seed = seed_base + 10000 + i * 100
        cav_vec, _ = generate_random_cav(
            embeddings=embeddings,
            scaler_emb=scaler_emb,
            n_samples=n_random_samples,
            random_state=seed,
        )
        if cav_vec is None:
            continue
        random_cavs.append(cav_vec)
        random_scores.append(compute_tcav_score_for_cav(cav_vec, gradients))

    concept_scores = np.asarray(concept_scores, dtype=float)
    random_scores = np.asarray(random_scores, dtype=float)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_stat, p_value = stats.ttest_ind(concept_scores, random_scores)

    pooled_std = np.sqrt(
        (
            (len(concept_scores) - 1) * concept_scores.std() ** 2
            + (len(random_scores) - 1) * random_scores.std() ** 2
        )
        / max((len(concept_scores) + len(random_scores) - 2), 1)
    )
    cohens_d = (concept_scores.mean() - random_scores.mean()) / (pooled_std + 1e-10)

    return {
        "concept_id": int(concept_id),
        "concept_tcav_scores": concept_scores.tolist(),
        "random_tcav_scores": random_scores.tolist(),
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "cohens_d": float(cohens_d),
        "significant": bool(p_value < 0.05),
        "mean_concept_tcav": float(concept_scores.mean()) if len(concept_scores) else np.nan,
        "mean_random_tcav": float(random_scores.mean()) if len(random_scores) else np.nan,
        "std_concept_tcav": float(concept_scores.std()) if len(concept_scores) else np.nan,
        "std_random_tcav": float(random_scores.std()) if len(random_scores) else np.nan,
        "n_runs_effective": int(len(concept_scores)),
        "concept_cavs": concept_cavs,
        "random_cavs": random_cavs,
    }