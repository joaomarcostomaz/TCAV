"""Dataset ingestion and exploratory helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config

sns.set(style=config.PLOT_STYLE)


@dataclass
class DatasetSummary:
    """Structured information returned by :func:`summarize_dataset`."""

    shape: tuple[int, int]
    columns: list[str]
    missing_per_column: pd.Series
    numeric_description: pd.DataFrame


def load_free_light_chain_dataset(path: Optional[str] = None) -> pd.DataFrame:
    """Load the Multiple Myeloma free light chain dataset."""

    csv_path = config.get_data_path(path or "free_light_chain_mortality.csv")
    df = pd.read_csv(csv_path)
    if "sample.yr" in df.columns and "year" not in df.columns:
        df = df.rename(columns={"sample.yr": "year"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    return df


def summarize_dataset(df: pd.DataFrame, *, show_plots: bool = False) -> DatasetSummary:
    """Compute descriptive statistics and optionally show plots."""

    numeric_description = df.describe(include=["number"]).T
    summary = DatasetSummary(
        shape=df.shape,
        columns=list(df.columns),
        missing_per_column=df.isnull().sum(),
        numeric_description=numeric_description,
    )

    if show_plots and "year" in df.columns:
        _plot_year_distribution(df)
        if "death" in df.columns:
            _plot_death_rate_by_year(df)
    return summary


def _plot_year_distribution(df: pd.DataFrame) -> None:
    years = sorted(df["year"].dropna().astype(int).unique())
    plt.figure(figsize=(10, 4))
    sns.countplot(
        x=df["year"].astype(str),
        order=[str(y) for y in years],
        color="C0",
    )
    plt.title("Contagem por year")
    plt.xlabel("Year")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def _plot_death_rate_by_year(df: pd.DataFrame) -> None:
    agg = df.groupby("year")["death"].agg(["count", "sum"]).rename(columns={"sum": "deaths"})
    agg["death_rate"] = agg["deaths"] / agg["count"]
    plt.figure(figsize=(10, 4))
    plt.bar(agg.index.astype(str), agg["death_rate"], color="C3")
    plt.title("Taxa de death por year")
    plt.xlabel("Year")
    plt.ylabel("Death rate")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
