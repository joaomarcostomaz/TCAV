"""
tabpfn_pipeline.phenotype

Phenotype characterization utilities:
- feature association analysis
- sparse readout (L1 regression)
- outcome association (OR / RR / risk diff)
- report table assembly
"""

from __future__ import annotations

from typing import Dict, List, Tuple, Any, Optional, Iterable

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import fisher_exact
from sklearn.decomposition import DictionaryLearning
from sklearn.linear_model import LassoCV, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler


def compute_feature_associations(
    concept_scores: np.ndarray,
    X_features: np.ndarray,
    feature_names: List[str],
    quantile: float = 0.1,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Compare feature distributions between high and low concept activation groups.
    """
    high_thr = np.quantile(concept_scores, 1.0 - quantile)
    low_thr = np.quantile(concept_scores, quantile)

    idx_high = np.where(concept_scores >= high_thr)[0]
    idx_low = np.where(concept_scores <= low_thr)[0]

    rows = []
    for i, fname in enumerate(feature_names):
        feat_high = X_features[idx_high, i]
        feat_low = X_features[idx_low, i]

        mh = np.nanmean(feat_high)
        ml = np.nanmean(feat_low)
        sh = np.nanstd(feat_high)
        sl = np.nanstd(feat_low)

        pooled = np.sqrt((sh**2 + sl**2) / 2)
        d = (mh - ml) / pooled if pooled > 0 else 0.0

        try:
            _, pval = stats.mannwhitneyu(feat_high, feat_low, alternative="two-sided")
        except Exception:
            pval = 1.0

        rows.append(
            {
                "feature": fname,
                "mean_high": mh,
                "mean_low": ml,
                "diff": mh - ml,
                "std_high": sh,
                "std_low": sl,
                "cohens_d": d,
                "pvalue": pval,
                "n_high": len(idx_high),
                "n_low": len(idx_low),
            }
        )

    df = pd.DataFrame(rows)
    df["significant"] = df["pvalue"] < 0.05
    df = df.sort_values("cohens_d", key=lambda s: np.abs(s), ascending=False).reset_index(drop=True)
    return df, idx_high, idx_low


def sparse_readout(
    concept_scores: np.ndarray,
    X_features: np.ndarray,
    feature_names: List[str],
    cv: int = 5,
) -> Dict[str, Any]:
    """
    Fit sparse linear readout:
      concept_score ~ X * beta
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_features)
    Xs = np.nan_to_num(Xs, nan=0.0)

    model = LassoCV(cv=cv, random_state=42, max_iter=10000)
    model.fit(Xs, concept_scores)

    coefs = model.coef_
    r2_train = model.score(Xs, concept_scores)

    cv_scores = cross_val_score(
        Lasso(alpha=model.alpha_, max_iter=10000),
        Xs,
        concept_scores,
        cv=cv,
        scoring="r2",
    )
    r2_cv = float(cv_scores.mean())
    r2_cv_std = float(cv_scores.std())

    idx = np.where(np.abs(coefs) > 1e-10)[0]
    selected_features = [feature_names[i] for i in idx]
    selected_coefs = coefs[idx]

    ord_idx = np.argsort(np.abs(selected_coefs))[::-1]
    selected_features = [selected_features[i] for i in ord_idx]
    selected_coefs = selected_coefs[ord_idx]

    return {
        "coefs": coefs,
        "alpha": float(model.alpha_),
        "r2_train": float(r2_train),
        "r2_cv": r2_cv,
        "r2_cv_std": r2_cv_std,
        "selected_features": selected_features,
        "selected_coefs": selected_coefs,
        "scaler": scaler,
        "model": model,
    }


def evaluate_sparse_readout(
    sparse_result: Dict[str, Any],
    X_features: np.ndarray,
    concept_scores: np.ndarray,
) -> Dict[str, float]:
    """
    Evaluate sparse readout on held-out split.
    """
    scaler = sparse_result["scaler"]
    model = sparse_result["model"]

    Xs = scaler.transform(X_features)
    Xs = np.nan_to_num(Xs, nan=0.0)

    r2 = float(model.score(Xs, concept_scores))
    pred = model.predict(Xs)
    corr = float(np.corrcoef(pred, concept_scores)[0, 1]) if len(concept_scores) > 1 else np.nan

    return {"r2": r2, "correlation": corr}


def compute_outcome_association(
    concept_scores: np.ndarray,
    y_labels: np.ndarray,
    quantile: float = 0.1,
) -> Dict[str, Any]:
    """
    Compute high-concept vs rest outcome association:
    odds ratio, CI, p-value, RR, risk diff.
    """
    high_thr = np.quantile(concept_scores, 1.0 - quantile)
    idx_high = concept_scores >= high_thr

    a = int(np.sum((idx_high) & (y_labels == 1)))
    b = int(np.sum((idx_high) & (y_labels == 0)))
    c = int(np.sum((~idx_high) & (y_labels == 1)))
    d = int(np.sum((~idx_high) & (y_labels == 0)))

    odds_ratio = (a * d) / (b * c) if (b * c) > 0 else (np.inf if (a * d) > 0 else 1.0)

    try:
        _, pval = fisher_exact([[a, b], [c, d]])
    except Exception:
        pval = 1.0

    if a > 0 and b > 0 and c > 0 and d > 0:
        log_or = np.log(odds_ratio)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci_low = float(np.exp(log_or - 1.96 * se))
        ci_high = float(np.exp(log_or + 1.96 * se))
    else:
        ci_low, ci_high = np.nan, np.nan

    mort_high = a / (a + b) if (a + b) > 0 else 0.0
    mort_rest = c / (c + d) if (c + d) > 0 else 0.0
    rr = mort_high / mort_rest if mort_rest > 0 else np.inf
    ard = mort_high - mort_rest

    return {
        "odds_ratio": float(odds_ratio),
        "or_ci_low": ci_low,
        "or_ci_high": ci_high,
        "pvalue": float(pval),
        "relative_risk": float(rr),
        "abs_risk_diff": float(ard),
        "mortality_high": float(mort_high),
        "mortality_rest": float(mort_rest),
        "n_high": int(np.sum(idx_high)),
        "n_rest": int(np.sum(~idx_high)),
        "deaths_high": a,
        "deaths_rest": c,
    }


def build_phenotype_report(
    matched_factors: List[int],
    sparse_readout_results: Dict[int, Dict[str, Any]],
    sparse_validation_results: Dict[int, Dict[str, Any]],
    feature_assoc_results: Dict[int, pd.DataFrame],
    outcome_df_primary: pd.DataFrame,
    outcome_df_validation: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Assemble phenotype summary report for paper-ready tables.
    """
    rows = []

    for k in matched_factors:
        sr = sparse_readout_results[k]
        sig = (
            ", ".join(
                [
                    f"{'↑' if c > 0 else '↓'}{f}"
                    for f, c in zip(sr["selected_features"][:3], sr["selected_coefs"][:3])
                ]
            )
            if len(sr["selected_features"]) > 0
            else "Not well-defined"
        )

        fa = feature_assoc_results[k].iloc[0]
        op = outcome_df_primary[outcome_df_primary["concept"] == k].iloc[0]

        risk_cat = "NEUTRAL"
        if op["odds_ratio"] > 1 and op["pvalue"] < 0.05:
            risk_cat = "RISK"
        elif op["odds_ratio"] < 1 and op["pvalue"] < 0.05:
            risk_cat = "PROTECTIVE"

        val_or = np.nan
        val_p = np.nan
        validated = True
        if outcome_df_validation is not None and len(outcome_df_validation) > 0:
            ov = outcome_df_validation[outcome_df_validation["concept"] == k].iloc[0]
            val_or = float(ov["odds_ratio"])
            val_p = float(ov["pvalue"])
            same_dir = (op["odds_ratio"] > 1) == (ov["odds_ratio"] > 1)
            validated = same_dir and (val_p < 0.1 or risk_cat == "NEUTRAL")

        rows.append(
            {
                "Concept": int(k),
                "Phenotype Signature": sig,
                "Signature R2 (CV)": float(sr["r2_cv"]),
                "Signature R2 (Validation)": float(
                    sparse_validation_results.get(k, {}).get("r2", np.nan)
                ),
                "Top Feature (Cohen d)": f"{fa['feature']} ({fa['cohens_d']:.2f})",
                "OR (Primary)": float(op["odds_ratio"]),
                "OR 95% CI": f"({op['or_ci_low']:.2f}-{op['or_ci_high']:.2f})",
                "p (Primary)": float(op["pvalue"]),
                "OR (Validation)": val_or,
                "p (Validation)": val_p,
                "Mortality High (Primary)": float(op["mortality_high"]),
                "Mortality Rest (Primary)": float(op["mortality_rest"]),
                "Risk Category": risk_cat,
                "Validated": "✓" if validated else "⚠️",
            }
        )

    return pd.DataFrame(rows)

def run_feature_association_dual_split(
    matched_factors: Iterable[int],
    tcav_eval_concept_activations: np.ndarray,
    held_out_concept_activations: np.ndarray,
    X_tcav_eval_raw: np.ndarray,
    X_held_out_raw: np.ndarray,
    feature_cols: list[str],
    quantile: float = 0.1,
) -> tuple[dict[int, pd.DataFrame], dict[int, pd.DataFrame], pd.DataFrame]:
    """
    Notebook-parity orchestration:
    - feature associations on TCAV eval (primary)
    - feature associations on held-out (validation)
    - top-5 overlap consistency summary
    """
    feature_association_results = {}
    feature_association_results_test = {}
    consistency_rows = []

    for k in matched_factors:
        concept_scores_tcav = tcav_eval_concept_activations[:, k]
        df_assoc_tcav, _, _ = compute_feature_associations(
            concept_scores=concept_scores_tcav,
            X_features=X_tcav_eval_raw,
            feature_names=feature_cols,
            quantile=quantile,
        )
        feature_association_results[int(k)] = df_assoc_tcav

        concept_scores_held_out = held_out_concept_activations[:, k]
        df_assoc_held_out, _, _ = compute_feature_associations(
            concept_scores=concept_scores_held_out,
            X_features=X_held_out_raw,
            feature_names=feature_cols,
            quantile=quantile,
        )
        feature_association_results_test[int(k)] = df_assoc_held_out

        top_tcav = set(df_assoc_tcav.head(5)["feature"].values.tolist())
        top_test = set(df_assoc_held_out.head(5)["feature"].values.tolist())
        overlap = len(top_tcav & top_test)

        consistency_rows.append(
            {
                "concept": int(k),
                "top5_overlap": int(overlap),
                "top5_overlap_ratio": float(overlap / 5.0),
            }
        )

    consistency_df = pd.DataFrame(consistency_rows).sort_values("concept").reset_index(drop=True)
    return feature_association_results, feature_association_results_test, consistency_df


def run_sparse_readout_dual_split(
    matched_factors: Iterable[int],
    cav_train_concept_activations: np.ndarray,
    tcav_eval_concept_activations: np.ndarray,
    held_out_concept_activations: np.ndarray,
    X_cav_train_raw: np.ndarray,
    X_tcav_eval_raw: np.ndarray,
    X_held_out_raw: np.ndarray,
    feature_cols: list[str],
    cv: int = 5,
    overfit_drop_warn_threshold: float = 0.2,
) -> tuple[dict[int, dict], dict[int, dict], pd.DataFrame]:
    """
    Notebook-parity sparse readout workflow:
    - fit on CAV train
    - validate on TCAV eval and held-out
    - return summary diagnostics
    """
    sparse_readout_results = {}
    sparse_readout_validation = {}
    summary_rows = []

    for k in matched_factors:
        result = sparse_readout(
            concept_scores=cav_train_concept_activations[:, k],
            X_features=X_cav_train_raw,
            feature_names=feature_cols,
            cv=cv,
        )
        sparse_readout_results[int(k)] = result

        tcav_eval_result = evaluate_sparse_readout(
                result,                      # <- positional
                X_tcav_eval_raw,             # <- positional
                tcav_eval_concept_activations[:, k],  # <- positional
            )

        held_out_result = evaluate_sparse_readout(
                result,                      # <- positional
                X_held_out_raw,              # <- positional
                held_out_concept_activations[:, k],   # <- positional
        )

        sparse_readout_validation[int(k)] = {
            "tcav_eval": tcav_eval_result,
            "test": held_out_result,
        }

        r2_drop_tcav = float(result["r2_cv"] - tcav_eval_result["r2"])
        r2_drop_test = float(result["r2_cv"] - held_out_result["r2"])

        summary_rows.append(
            {
                "concept": int(k),
                "alpha": float(result["alpha"]),
                "r2_train": float(result["r2_train"]),
                "r2_cv": float(result["r2_cv"]),
                "r2_cv_std": float(result["r2_cv_std"]),
                "r2_tcav_eval": float(tcav_eval_result["r2"]),
                "corr_tcav_eval": float(tcav_eval_result["correlation"]),
                "r2_test": float(held_out_result["r2"]),
                "corr_test": float(held_out_result["correlation"]),
                "r2_drop_tcav_eval": r2_drop_tcav,
                "r2_drop_test": r2_drop_test,
                "warn_overfit": bool(r2_drop_test > overfit_drop_warn_threshold),
                "n_selected_features": int(len(result["selected_features"])),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("concept").reset_index(drop=True)
    return sparse_readout_results, sparse_readout_validation, summary_df


def run_outcome_association_dual_split(
    matched_factors: Iterable[int],
    tcav_eval_concept_activations: np.ndarray,
    held_out_concept_activations: np.ndarray,
    y_tcav_eval: np.ndarray,
    y_test_labels: np.ndarray,
    quantile: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, list[int], pd.DataFrame]:
    """
    Notebook-parity outcome workflow:
    - compute outcome association on TCAV eval and held-out test
    - derive robust concept list using direction consistency and p<0.1 in test
    - return cross-split consistency table
    """
    rows_tcav = []
    rows_test = []

    for k in matched_factors:
        r_tcav = compute_outcome_association(
            concept_scores=tcav_eval_concept_activations[:, k],
            y_labels=y_tcav_eval,
            quantile=quantile,
        )
        r_tcav["concept"] = int(k)
        r_tcav["split"] = "tcav_eval"
        rows_tcav.append(r_tcav)

        r_test = compute_outcome_association(
            concept_scores=held_out_concept_activations[:, k],
            y_labels=y_test_labels,
            quantile=quantile,
        )
        r_test["concept"] = int(k)
        r_test["split"] = "test"
        rows_test.append(r_test)

    outcome_df_tcav = pd.DataFrame(rows_tcav).sort_values("odds_ratio", ascending=False).reset_index(drop=True)
    outcome_df_test = pd.DataFrame(rows_test).sort_values("odds_ratio", ascending=False).reset_index(drop=True)

    robust_concepts = []
    consistency_rows = []

    for k in matched_factors:
        tcav_row = outcome_df_tcav[outcome_df_tcav["concept"] == int(k)].iloc[0]
        test_row = outcome_df_test[outcome_df_test["concept"] == int(k)].iloc[0]

        tcav_sig = bool(tcav_row["pvalue"] < 0.05)
        test_sig = bool(test_row["pvalue"] < 0.05)
        same_direction = ((tcav_row["odds_ratio"] > 1) == (test_row["odds_ratio"] > 1))
        status = "CONSISTENT" if (tcav_sig == test_sig and same_direction) else "INCONSISTENT"

        if tcav_row["pvalue"] < 0.05:
            if same_direction and test_row["pvalue"] < 0.1:
                robust_concepts.append(int(k))

        consistency_rows.append(
            {
                "concept": int(k),
                "tcav_or": float(tcav_row["odds_ratio"]),
                "tcav_p": float(tcav_row["pvalue"]),
                "test_or": float(test_row["odds_ratio"]),
                "test_p": float(test_row["pvalue"]),
                "same_direction": bool(same_direction),
                "status": status,
            }
        )

    consistency_df = pd.DataFrame(consistency_rows).sort_values("concept").reset_index(drop=True)
    return outcome_df_tcav, outcome_df_test, robust_concepts, consistency_df


def build_clinical_interpretation_lines(report_df: pd.DataFrame) -> list[str]:
    """
    Build notebook-style narrative lines from phenotype report DataFrame.
    """
    lines = []
    n_robust = 0

    for _, row in report_df.iterrows():
        concept = row["Concept"]
        lines.append(f"📌 CONCEPT {concept}")
        lines.append(
            f"  Signature: {row['Phenotype Signature']} | "
            f"R²(CV)={row['Signature R2 (CV)']:.3f} | "
            f"R²(Validation)={row['Signature R2 (Validation)']:.3f}"
        )
        lines.append(f"  Top feature: {row['Top Feature (Cohen d)']}")

        risk = row["Risk Category"]
        if risk == "RISK":
            lines.append(
                f"  ⚠️ Risk factor | OR(primary)={row['OR (Primary)']:.2f}, p(primary)={row['p (Primary)']:.4f} | "
                f"OR(val)={row['OR (Validation)']:.2f}, p(val)={row['p (Validation)']:.4f}"
            )
        elif risk == "PROTECTIVE":
            lines.append(
                f"  ✅ Protective factor | OR(primary)={row['OR (Primary)']:.2f}, p(primary)={row['p (Primary)']:.4f} | "
                f"OR(val)={row['OR (Validation)']:.2f}, p(val)={row['p (Validation)']:.4f}"
            )
        else:
            lines.append(
                f"  ➖ Neutral | OR(primary)={row['OR (Primary)']:.2f}, p(primary)={row['p (Primary)']:.4f} | "
                f"OR(val)={row['OR (Validation)']:.2f}, p(val)={row['p (Validation)']:.4f}"
            )

        if row.get("Validated", "⚠️") == "✓":
            n_robust += 1
            lines.append("  ✓ Validation status: confirmed on held-out")
        else:
            lines.append("  ⚠️ Validation status: not confirmed on held-out")

        lines.append("")

    lines.append(f"📋 Summary: {int((report_df['Risk Category'] != 'NEUTRAL').sum())} significant concepts, {n_robust} validated.")
    return lines


def build_tcav_integration_lines(
    matched_factors: Iterable[int],
    report_df: pd.DataFrame,
    robust_tcav_results: dict[int, dict],
) -> list[str]:
    """
    Build notebook-style 'integration with TCAV' lines using robust_tcav_results format.
    """
    lines = []
    for k in matched_factors:
        if int(k) not in robust_tcav_results:
            continue
        phenotype_row = report_df[report_df["Concept"] == int(k)]
        if phenotype_row.empty:
            continue
        phenotype = phenotype_row.iloc[0]

        tcav_mean = float(robust_tcav_results[int(k)]["mean_concept_tcav"])
        lines.append(f"Concept {int(k)} ({phenotype['Phenotype Signature']})")
        lines.append(f"  • TCAV Score: {tcav_mean:.3f}")
        lines.append(f"  • Risk: {phenotype['Risk Category']}")
        lines.append(f"  • Validated: {phenotype['Validated']}")

        if tcav_mean > 0.6 and phenotype["Risk Category"] == "RISK":
            lines.append("  • Interpretation: High concept activation → higher mortality prediction")
        elif tcav_mean < 0.4 and phenotype["Risk Category"] == "PROTECTIVE":
            lines.append("  • Interpretation: High concept activation → lower mortality prediction")
        else:
            lines.append("  • Interpretation: Complex or mixed relationship")
        lines.append("")

    return lines