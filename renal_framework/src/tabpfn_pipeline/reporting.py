"""
tabpfn_pipeline.reporting

Small reporting/dataframe helpers for notebook-parity outputs.
"""

from __future__ import annotations

from typing import Dict, Any, List
import numpy as np
import pandas as pd


def decomposition_quality_report(
    codes_sae: np.ndarray,
    decoder_atoms_sae: np.ndarray,
    dict_codes: np.ndarray,
    dict_atoms: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute compact decomposition quality metrics for SAE vs Dictionary Learning.
    """
    # SAE atom cosine stats
    D = np.asarray(decoder_atoms_sae, dtype=float)  # expected [emb_dim, n_factors] or transpose-compatible
    if D.shape[0] < D.shape[1]:
        D = D.T
    Dn = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    cos_D = np.abs(Dn.T @ Dn)
    off_D = cos_D - np.eye(cos_D.shape[0])

    # SAE codes cosine stats
    C = np.asarray(codes_sae, dtype=float)
    Cn = C / (np.linalg.norm(C, axis=0, keepdims=True) + 1e-12)
    cos_C = np.abs(Cn.T @ Cn)
    off_C = cos_C - np.eye(cos_C.shape[0])

    # DL atoms cosine stats
    W = np.asarray(dict_atoms, dtype=float)
    if W.shape[0] < W.shape[1]:
        W = W.T
    Wn = W / (np.linalg.norm(W, axis=0, keepdims=True) + 1e-12)
    cos_W = np.abs(Wn.T @ Wn)
    off_W = cos_W - np.eye(cos_W.shape[0])

    # DL codes cosine stats
    A = np.asarray(dict_codes, dtype=float)
    An = A / (np.linalg.norm(A, axis=0, keepdims=True) + 1e-12)
    cos_A = np.abs(An.T @ An)
    off_A = cos_A - np.eye(cos_A.shape[0])

    return {
        "sae_atom_max_offdiag_cos": float(off_D.max()),
        "sae_atom_mean_offdiag_cos": float(off_D.mean()),
        "sae_atom_pairs_gt_thr": int((off_D > threshold).sum() // 2),
        "sae_code_max_offdiag_cos": float(off_C.max()),
        "sae_code_mean_offdiag_cos": float(off_C.mean()),
        "sae_code_pairs_gt_thr": int((off_C > threshold).sum() // 2),
        "dl_atom_max_offdiag_cos": float(off_W.max()),
        "dl_atom_mean_offdiag_cos": float(off_W.mean()),
        "dl_atom_pairs_gt_thr": int((off_W > threshold).sum() // 2),
        "dl_code_max_offdiag_cos": float(off_A.max()),
        "dl_code_mean_offdiag_cos": float(off_A.mean()),
        "dl_code_pairs_gt_thr": int((off_A > threshold).sum() // 2),
        "sae_sparsity": float((np.abs(C) <= 1e-5).mean()),
        "dl_sparsity": float((np.abs(A) <= 1e-5).mean()),
    }


def tcav_results_to_df(
    results: Dict[int, Any],
    cav_dict: Dict[int, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Convert TCAVGradientResult dict to dataframe for reporting.
    """
    rows = []
    for cid, res in results.items():
        if not isinstance(cid, int):
            continue
        rows.append(
            {
                "cluster": int(cid),
                "tcav_prop_positive": float(res.tcav_positive_fraction),
                "mean_derivative": float(res.mean_derivative),
                "size_pure": int(cav_dict[cid]["size_pos"]) if cid in cav_dict else np.nan,
                "size_neg": int(len(cav_dict[cid]["neg_idx"])) if cid in cav_dict else np.nan,
                "deriv_std": float(np.std(res.derivatives)),
                "deriv_range": float(np.max(res.derivatives) - np.min(res.derivatives)),
            }
        )
    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def robust_tcav_results_to_df(robust_tcav_results: Dict[int, Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for cid, r in robust_tcav_results.items():
        rows.append(
            {
                "concept": int(cid),
                "mean_concept_tcav": float(r.get("mean_concept_tcav", np.nan)),
                "std_concept_tcav": float(r.get("std_concept_tcav", np.nan)),
                "mean_random_tcav": float(r.get("mean_random_tcav", np.nan)),
                "std_random_tcav": float(r.get("std_random_tcav", np.nan)),
                "p_value": float(r.get("p_value", np.nan)),
                "cohens_d": float(r.get("cohens_d", np.nan)),
                "significant": bool(r.get("significant", False)),
            }
        )
    return pd.DataFrame(rows).sort_values("concept").reset_index(drop=True)