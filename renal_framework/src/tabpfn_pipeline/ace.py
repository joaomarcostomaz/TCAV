"""
tabpfn_pipeline.ace

ACE-style concept sanity checks:
- embedding-level destruction/sufficiency
- optional input-level feature masking tests
- per-domain drift analysis
- validation summary tables
- rule validation on held-out activations
- temporal feature-trajectory utilities
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple, List, Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats


@dataclass
class ACESanityResult:
    concept_id: int

    destruction_delta_mean: float = 0.0
    destruction_delta_std: float = 0.0
    destruction_delta_high_concept: float = 0.0
    destruction_delta_low_concept: float = 0.0
    destruction_effect_ratio: float = 0.0

    sufficiency_ratio_mean: float = 0.0
    sufficiency_ratio_std: float = 0.0
    sufficiency_ratio_high_concept: float = 0.0
    sufficiency_ratio_low_concept: float = 0.0

    # input-level (feature masking) optional metrics
    feature_destruction_delta: float = 0.0
    feature_sufficiency_ratio: float = 0.0

    destruction_by_domain: Dict[int, float] = field(default_factory=dict)
    sufficiency_by_domain: Dict[int, float] = field(default_factory=dict)

    destruction_drift_index: float = 0.0
    sufficiency_drift_index: float = 0.0
    is_drift_sensitive: bool = False


def project_out_direction(embeddings: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    h_ablated = h - (h·v)v
    """
    v = direction / (np.linalg.norm(direction) + 1e-12)
    proj = np.dot(embeddings, v)[:, np.newaxis] * v[np.newaxis, :]
    return embeddings - proj


def project_onto_direction(embeddings: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    h_sufficient = (h·v)v
    """
    v = direction / (np.linalg.norm(direction) + 1e-12)
    return np.dot(embeddings, v)[:, np.newaxis] * v[np.newaxis, :]


def mask_features_by_importance(
    X: np.ndarray,
    feature_mask: np.ndarray,
    fill_value: str = "mean",
    random_state: int = 42,
) -> np.ndarray:
    """
    Input-level masking utility used for feature-level ACE probes.
    """
    X_masked = X.copy()
    feature_mask = np.asarray(feature_mask, dtype=bool)
    mask_indices = ~feature_mask

    if fill_value == "mean":
        col_means = np.nanmean(X, axis=0)
        X_masked[:, mask_indices] = col_means[mask_indices]
    elif fill_value == "zero":
        X_masked[:, mask_indices] = 0.0
    elif fill_value == "noise":
        rng = np.random.default_rng(random_state)
        for i in np.where(mask_indices)[0]:
            m = np.nanmean(X[:, i])
            s = np.nanstd(X[:, i])
            X_masked[:, i] = rng.normal(m, s if s > 0 else 1e-6, size=X.shape[0])
    else:
        raise ValueError("fill_value must be one of {'mean','zero','noise'}")

    return X_masked


@torch.no_grad()
def _predict_target_prob(
    decoder: nn.Module,
    emb_np: np.ndarray,
    device: torch.device | str,
    target_class: int = 1,
) -> np.ndarray:
    x = torch.tensor(emb_np, dtype=torch.float32, device=device)
    logits = decoder(x.unsqueeze(0))
    if logits.ndim == 3:
        logits = logits[0]
    probs = torch.softmax(logits, dim=-1)[:, target_class].detach().cpu().numpy()
    return probs


def run_embedding_destruction_test(
    embeddings: np.ndarray,
    cav_vectors: List[np.ndarray],
    concept_activations: np.ndarray,
    decoder: nn.Module,
    device: torch.device | str,
    quantile: float = 0.1,
    target_class: int = 1,
) -> Dict[str, float]:
    """
    Remove concept direction(s), compare prediction change.
    """
    high_thr = np.quantile(concept_activations, 1.0 - quantile)
    low_thr = np.quantile(concept_activations, quantile)
    idx_high = concept_activations >= high_thr
    idx_low = concept_activations <= low_thr

    emb_abl = embeddings.copy()
    for v in cav_vectors:
        emb_abl = project_out_direction(emb_abl, v)

    p_orig = _predict_target_prob(decoder, embeddings, device, target_class=target_class)
    p_abl = _predict_target_prob(decoder, emb_abl, device, target_class=target_class)

    delta = p_abl - p_orig
    return {
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "delta_high_concept": float(np.mean(delta[idx_high])) if idx_high.any() else 0.0,
        "delta_low_concept": float(np.mean(delta[idx_low])) if idx_low.any() else 0.0,
        "probs_orig_mean": float(np.mean(p_orig)),
        "probs_abl_mean": float(np.mean(p_abl)),
    }


def run_embedding_sufficiency_test(
    embeddings: np.ndarray,
    cav_vector: np.ndarray,
    concept_activations: np.ndarray,
    decoder: nn.Module,
    device: torch.device | str,
    quantile: float = 0.1,
    target_class: int = 1,
) -> Dict[str, float]:
    """
    Keep only concept direction, compute retained prediction ratio.
    """
    high_thr = np.quantile(concept_activations, 1.0 - quantile)
    low_thr = np.quantile(concept_activations, quantile)
    idx_high = concept_activations >= high_thr
    idx_low = concept_activations <= low_thr

    emb_suf = project_onto_direction(embeddings, cav_vector)

    p_orig = _predict_target_prob(decoder, embeddings, device, target_class=target_class)
    p_suf = _predict_target_prob(decoder, emb_suf, device, target_class=target_class)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(p_orig > 0.01, p_suf / p_orig, 0.0)

    return {
        "ratio_mean": float(np.mean(ratio)),
        "ratio_std": float(np.std(ratio)),
        "ratio_high_concept": float(np.mean(ratio[idx_high])) if idx_high.any() else 0.0,
        "ratio_low_concept": float(np.mean(ratio[idx_low])) if idx_low.any() else 0.0,
        "probs_orig_mean": float(np.mean(p_orig)),
        "probs_suf_mean": float(np.mean(p_suf)),
    }


def compute_drift_metrics(values_by_domain: Dict[int, float]) -> Tuple[float, float, bool]:
    """
    Drift sensitivity from per-domain metric values.
    """
    if len(values_by_domain) < 2:
        return 0.0, 0.0, False

    vals = np.asarray(list(values_by_domain.values()), dtype=float)
    variance = float(np.var(vals))
    v_range = float(np.max(vals) - np.min(vals))
    is_drift_sensitive = (variance > 0.01) or (v_range > 0.1)
    return variance, v_range, is_drift_sensitive


def run_tests_by_domain(
    embeddings: np.ndarray,
    cav_vector: np.ndarray,
    concept_activations: np.ndarray,
    domain_ids: np.ndarray,
    decoder: nn.Module,
    device: torch.device | str,
    target_class: int = 1,
) -> Tuple[Dict[int, Dict[str, float]], Dict[int, Dict[str, float]]]:
    """
    Run destruction/sufficiency per domain.
    """
    destruction_by_domain = {}
    sufficiency_by_domain = {}

    for d in sorted(np.unique(domain_ids)):
        mask = domain_ids == d
        if mask.sum() < 20:
            continue

        emb_d = embeddings[mask]
        act_d = concept_activations[mask]

        try:
            dr = run_embedding_destruction_test(
                embeddings=emb_d,
                cav_vectors=[cav_vector],
                concept_activations=act_d,
                decoder=decoder,
                device=device,
                quantile=0.1,
                target_class=target_class,
            )
            destruction_by_domain[int(d)] = dr
        except Exception:
            pass

        try:
            sr = run_embedding_sufficiency_test(
                embeddings=emb_d,
                cav_vector=cav_vector,
                concept_activations=act_d,
                decoder=decoder,
                device=device,
                quantile=0.1,
                target_class=target_class,
            )
            sufficiency_by_domain[int(d)] = sr
        except Exception:
            pass

    return destruction_by_domain, sufficiency_by_domain


def run_ace_for_concepts(
    concept_ids: List[int],
    cav_dict: Dict[Any, Any],
    embeddings_eval: np.ndarray,
    concept_activations_eval: np.ndarray,
    domain_ids_eval: np.ndarray,
    decoder: nn.Module,
    device: torch.device | str,
) -> Dict[int, ACESanityResult]:
    """
    Run full ACE workflow for a list of concepts.
    """
    results: Dict[int, ACESanityResult] = {}

    for cid in concept_ids:
        if cid not in cav_dict or "v_activ" not in cav_dict[cid]:
            continue

        v = np.asarray(cav_dict[cid]["v_activ"], dtype=np.float32)
        acts = concept_activations_eval[:, cid]

        out = ACESanityResult(concept_id=int(cid))

        dres = run_embedding_destruction_test(
            embeddings=embeddings_eval,
            cav_vectors=[v],
            concept_activations=acts,
            decoder=decoder,
            device=device,
            quantile=0.05,
            target_class=1,
        )
        out.destruction_delta_mean = dres["delta_mean"]
        out.destruction_delta_std = dres["delta_std"]
        out.destruction_delta_high_concept = dres["delta_high_concept"]
        out.destruction_delta_low_concept = dres["delta_low_concept"]

        if abs(out.destruction_delta_low_concept) > 1e-3:
            out.destruction_effect_ratio = abs(out.destruction_delta_high_concept) / abs(out.destruction_delta_low_concept)
        else:
            out.destruction_effect_ratio = abs(out.destruction_delta_high_concept) / 1e-3

        sres = run_embedding_sufficiency_test(
            embeddings=embeddings_eval,
            cav_vector=v,
            concept_activations=acts,
            decoder=decoder,
            device=device,
            quantile=0.05,
            target_class=1,
        )
        out.sufficiency_ratio_mean = sres["ratio_mean"]
        out.sufficiency_ratio_std = sres["ratio_std"]
        out.sufficiency_ratio_high_concept = sres["ratio_high_concept"]
        out.sufficiency_ratio_low_concept = sres["ratio_low_concept"]

        dbd, sbd = run_tests_by_domain(
            embeddings=embeddings_eval,
            cav_vector=v,
            concept_activations=acts,
            domain_ids=domain_ids_eval,
            decoder=decoder,
            device=device,
            target_class=1,
        )

        out.destruction_by_domain = {k: v["delta_high_concept"] for k, v in dbd.items()}
        out.sufficiency_by_domain = {k: v["ratio_high_concept"] for k, v in sbd.items()}

        if len(out.destruction_by_domain) > 1:
            var_d, _, drift_d = compute_drift_metrics(out.destruction_by_domain)
            out.destruction_drift_index = var_d
        else:
            drift_d = False

        if len(out.sufficiency_by_domain) > 1:
            var_s, _, drift_s = compute_drift_metrics(out.sufficiency_by_domain)
            out.sufficiency_drift_index = var_s
        else:
            drift_s = False

        out.is_drift_sensitive = drift_d or drift_s
        results[int(cid)] = out

    return results


def ace_validation_summary(
    ace_results: Dict[int, ACESanityResult],
    destruction_threshold: float = -0.01,
    effect_ratio_threshold: float = 1.2,
    sufficiency_threshold: float = 0.3,
):
    """
    Classify concepts by ACE validation criteria.
    """
    validated, partial, failed = [], [], []

    for cid, r in ace_results.items():
        passes_destruction = abs(r.destruction_delta_high_concept) > abs(destruction_threshold)
        passes_differential = r.destruction_effect_ratio > effect_ratio_threshold
        passes_sufficiency = r.sufficiency_ratio_high_concept > sufficiency_threshold
        passes_stability = not r.is_drift_sensitive

        n = sum([passes_destruction, passes_differential, passes_sufficiency, passes_stability])

        status = {
            "concept_id": cid,
            "passes_destruction": passes_destruction,
            "passes_differential": passes_differential,
            "passes_sufficiency": passes_sufficiency,
            "passes_stability": passes_stability,
            "n_passed": n,
            "result": r,
        }

        if n >= 3:
            validated.append(status)
        elif n >= 2:
            partial.append(status)
        else:
            failed.append(status)

    return validated, partial, failed


def build_ace_summary_df(
    ace_results: Dict[int, ACESanityResult],
    phenotype_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build notebook-style ACE summary DataFrame.
    """
    rows = []
    for concept_id, result in ace_results.items():
        phenotype_sig = "N/A"
        risk_category = "N/A"

        if phenotype_df is not None and not phenotype_df.empty and "Concept" in phenotype_df.columns:
            row = phenotype_df[phenotype_df["Concept"] == concept_id]
            if len(row) > 0:
                phenotype_sig = str(row.iloc[0].get("Phenotype Signature", "N/A"))
                risk_category = str(row.iloc[0].get("Risk Category", "N/A"))

        rows.append(
            {
                "Concept": concept_id,
                "Phenotype": (phenotype_sig[:30] + "...") if len(phenotype_sig) > 30 else phenotype_sig,
                "Risk": risk_category,
                "Δ_destroy (all)": result.destruction_delta_mean,
                "Δ_destroy (high)": result.destruction_delta_high_concept,
                "Δ_destroy (low)": result.destruction_delta_low_concept,
                "Effect Ratio": result.destruction_effect_ratio,
                "Suff Ratio (all)": result.sufficiency_ratio_mean,
                "Suff Ratio (high)": result.sufficiency_ratio_high_concept,
                "Destroy Drift": result.destruction_drift_index,
                "Suff Drift": result.sufficiency_drift_index,
                "Drift Sensitive": "⚠️" if result.is_drift_sensitive else "✓",
            }
        )

    return pd.DataFrame(rows).sort_values("Concept").reset_index(drop=True)


def validate_rules_on_heldout_activations(
    matched_df: pd.DataFrame,
    held_out_concept_activations: np.ndarray,
    X_held_out_eval_df: pd.DataFrame,
    feature_cols: List[str],
    parse_decision_tree_rule_fn,
    apply_rule_conditions_fn,
    rule_column: Optional[str] = None,
) -> Dict[int, Dict[str, Any]]:
    """
    Validate whether decision-tree rules capture higher concept activations on held-out data.
    """
    if rule_column is None:
        rule_column = "Path" if "Path" in matched_df.columns else "Rule"

    results = {}

    for _, row in matched_df.iterrows():
        factor_id = int(row["Factor"])
        rule_string = row[rule_column] if rule_column in row.index else ""

        conditions = parse_decision_tree_rule_fn(rule_string)
        if len(conditions) == 0:
            continue

        idx_matched = apply_rule_conditions_fn(X_held_out_eval_df, conditions, feature_cols)
        idx_not_matched = ~idx_matched

        n_matched = int(np.sum(idx_matched))
        n_not = int(np.sum(idx_not_matched))
        if n_matched < 5 or n_not < 5:
            continue

        if factor_id >= held_out_concept_activations.shape[1]:
            continue

        factor_acts = held_out_concept_activations[:, factor_id]
        matched_acts = factor_acts[idx_matched]
        not_matched_acts = factor_acts[idx_not_matched]

        pval = float(stats.mannwhitneyu(matched_acts, not_matched_acts, alternative="greater").pvalue)

        pooled_std = np.sqrt((np.std(matched_acts) ** 2 + np.std(not_matched_acts) ** 2) / 2.0)
        cohens_d = float((np.mean(matched_acts) - np.mean(not_matched_acts)) / pooled_std) if pooled_std > 0 else 0.0

        results[factor_id] = {
            "rule_string": rule_string,
            "conditions": conditions,
            "n_matched": n_matched,
            "n_not_matched": n_not,
            "idx_matched": idx_matched,
            "idx_not_matched": idx_not_matched,
            "matched_activation_mean": float(np.mean(matched_acts)),
            "matched_activation_std": float(np.std(matched_acts)),
            "not_matched_activation_mean": float(np.mean(not_matched_acts)),
            "not_matched_activation_std": float(np.std(not_matched_acts)),
            "mannwhitney_pval": pval,
            "cohens_d": cohens_d,
        }

    return results


def build_rule_validation_summary_df(
    tree_features_validation_results: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert rule validation dict to summary DataFrame.
    """
    rows = []
    for factor_id, res in tree_features_validation_results.items():
        rows.append(
            {
                "SAE Factor": factor_id,
                "N Matched": res["n_matched"],
                "N Not Matched": res["n_not_matched"],
                "Matched Act Mean": res["matched_activation_mean"],
                "Not Matched Act Mean": res["not_matched_activation_mean"],
                "Cohen's d": res["cohens_d"],
                "p-value": res["mannwhitney_pval"],
                "Significant": bool(res["mannwhitney_pval"] < 0.05),
                "N Conditions": int(len(res.get("conditions", []))),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("SAE Factor").reset_index(drop=True)


def compute_factor_activation_by_year(
    concept_activations: np.ndarray,
    years: np.ndarray,
    factors: List[int],
) -> pd.DataFrame:
    """
    Build per-year mean activation trajectories for selected factors.
    """
    years = np.asarray(years).astype(int)
    rows = []

    for k in factors:
        if k >= concept_activations.shape[1]:
            continue
        for y in sorted(np.unique(years)):
            m = years == y
            if m.sum() == 0:
                continue
            rows.append(
                {
                    "factor": int(k),
                    "year": int(y),
                    "mean_activation": float(np.mean(concept_activations[m, k])),
                    "std_activation": float(np.std(concept_activations[m, k])),
                    "n_samples": int(m.sum()),
                }
            )

    return pd.DataFrame(rows).sort_values(["factor", "year"]).reset_index(drop=True)


def compute_feature_means_by_year_for_matched_rules(
    matched_df: pd.DataFrame,
    test_rows_tcav_eval: pd.DataFrame,
    feature_cols: List[str],
    tree_features_validation_results: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Notebook-style utility:
    for each matched factor and year, compute mean feature values on rule-matched samples.
    """
    concepts_stats_per_year = []

    for _, row in matched_df.iterrows():
        factor_id = int(row["Factor"])
        if factor_id not in tree_features_validation_results:
            continue

        idx_matched = tree_features_validation_results[factor_id]["idx_matched"]
        rows_on_df = test_rows_tcav_eval.iloc[idx_matched]

        for year in sorted(rows_on_df["year"].unique()):
            means = rows_on_df[rows_on_df["year"] == year][feature_cols].mean()

            out_row = {"factor": factor_id, "year": int(year), "n_samples": int(idx_matched.sum())}
            for k, v in means.items():
                out_row[f"mean_{k}"] = float(v)
            concepts_stats_per_year.append(out_row)

    if not concepts_stats_per_year:
        return pd.DataFrame()

    return pd.DataFrame(concepts_stats_per_year).sort_values(["factor", "year"]).reset_index(drop=True)