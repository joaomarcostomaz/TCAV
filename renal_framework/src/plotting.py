"""
Shared plotting utilities for yearly metrics and ranking charts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def set_plot_style() -> None:
    """Set default seaborn style for project plots."""
    sns.set_theme(style="whitegrid")


def plot_yearly_lines(
    yearly_df: pd.DataFrame,
    metric: str = "f1_pos",
    run_col: str = "run_id",
    year_col: str = "year",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
):
    """
    Plot yearly metric lines by run.
    """
    set_plot_style()
    plt.figure(figsize=(12, 6))
    for run_id, g in yearly_df.groupby(run_col):
        g = g.sort_values(year_col)
        plt.plot(g[year_col], g[metric], marker="o", linewidth=2, label=run_id)

    plt.ylim(0, 1)
    plt.xlabel("Year")
    plt.ylabel(metric)
    plt.title(title or f"{metric} by Year")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.show()


def plot_model_ranking_bar(
    summary_df: pd.DataFrame,
    metric: str = "avg_f1_pos",
    run_col: str = "run_id",
    title: Optional[str] = None,
    save_path: Optional[str | Path] = None,
):
    """
    Plot horizontal ranking bar chart for model summary metric.
    """
    set_plot_style()
    d = summary_df.sort_values(metric, ascending=False).copy()

    plt.figure(figsize=(10, max(4, 0.5 * len(d))))
    sns.barplot(data=d, y=run_col, x=metric)
    plt.xlim(0, 1)
    plt.xlabel(metric)
    plt.ylabel(run_col)
    plt.title(title or f"Ranking by {metric}")
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, dpi=300)
    plt.show()