"""Reporting helpers for TCAV analyses."""
from __future__ import annotations

from typing import Dict, Iterable

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src.concepts import CAV, TCAVResult
from src import config

sns.set(style=config.PLOT_STYLE)


def tcav_results_to_frame(results: Dict[int, TCAVResult], cavs: Dict[int, CAV] | None = None) -> pd.DataFrame:
    rows = []
    for cluster_id, res in results.items():
        row = {
            "cluster": cluster_id,
            "tcav_prop_positive": res.tcav_positive_fraction,
            "mean_derivative": res.mean_derivative,
            "alpha_used": res.alpha_used,
        }
        if cavs and cluster_id in cavs:
            row["size_pure"] = cavs[cluster_id].size_pure
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["cluster", "tcav_prop_positive", "mean_derivative", "alpha_used", "size_pure"])
    return pd.DataFrame(rows).sort_values("tcav_prop_positive", ascending=False).reset_index(drop=True)


def plot_tcav_histograms(results: Dict[int, TCAVResult], *, top_k: int = 6) -> None:
    if not results:
        return
    ordered = sorted(results.values(), key=lambda r: r.tcav_positive_fraction, reverse=True)[:top_k]
    plt.figure(figsize=(10, 3 * len(ordered)))
    for idx, res in enumerate(ordered, start=1):
        plt.subplot(len(ordered), 1, idx)
        plt.hist(res.derivatives, bins=60, color=f"C{idx % 10}", alpha=0.8)
        plt.title(
            f"Cluster {res.cluster_id} – TCAV={res.tcav_positive_fraction:.3f}, "
            f"α={res.alpha_used:.2e}"
        )
        plt.xlabel("Derivative")
    plt.tight_layout()
    plt.show()


def plot_tcav_bar(results: Dict[int, TCAVResult]) -> None:
    if not results:
        return
    df = tcav_results_to_frame(results)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=df, x="cluster", y="tcav_prop_positive", palette="viridis")
    plt.ylabel("TCAV proportion (deriv > 0)")
    plt.xlabel("Cluster")
    plt.title("TCAV proportion by cluster")
    plt.tight_layout()
    plt.show()
