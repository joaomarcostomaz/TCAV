"""Preprocessing utilities mirrored from the original notebook."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src import config


@dataclass
class PreprocessResult:
    feature_frame: pd.DataFrame
    target: np.ndarray
    years: np.ndarray
    feature_names: list[str]


@dataclass
class TemporalSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    years_train: np.ndarray
    years_test: np.ndarray
    train_years: list[int]
    test_years: list[int]
    year_to_domain: dict[int, int]


@dataclass
class NormalizedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    scaler: StandardScaler


NUMERIC_FEATURE_CANDIDATES = [
    "age",
    "kappa",
    "lambda",
    "flc.grp",
    "creatinine",
    "mgus",
    "futime",
]


def build_feature_matrix(df: pd.DataFrame, *, include_sex: bool = True) -> PreprocessResult:
    feature_cols: list[str] = []
    proc = df.copy()
    if include_sex and "sex" in proc.columns:
        proc["sex_bin"] = proc["sex"].astype(str).str.upper().map({"F": 0, "M": 1})
        feature_cols.append("sex_bin")
    for col in NUMERIC_FEATURE_CANDIDATES:
        if col in proc.columns:
            feature_cols.append(col)
    X = proc[feature_cols].copy()
    y = proc["death"].fillna(0).astype(int).to_numpy()
    years = proc["year"].astype("Int64").to_numpy()
    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
    return PreprocessResult(X_imputed, y, years, feature_cols)


def temporal_split(
    X: pd.DataFrame,
    y: np.ndarray,
    years: np.ndarray,
    *,
    train_frac: float = config.TRAIN_YEAR_FRAC,
) -> TemporalSplit:
    valid_years = sorted(pd.Series(years).dropna().astype(int).unique())
    n_train_years = max(1, int(np.floor(len(valid_years) * train_frac)))
    train_years = valid_years[:n_train_years]
    test_years = valid_years[n_train_years:]
    train_mask = np.isin(years, train_years)
    test_mask = np.isin(years, test_years)
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError("Temporal split resulted in empty partition; adjust TRAIN_YEAR_FRAC.")
    year_to_domain = {yr: idx for idx, yr in enumerate(valid_years)}
    return TemporalSplit(
        X_train=X.loc[train_mask].reset_index(drop=True),
        X_test=X.loc[test_mask].reset_index(drop=True),
        y_train=y[train_mask],
        y_test=y[test_mask],
        years_train=years[train_mask],
        years_test=years[test_mask],
        train_years=train_years,
        test_years=test_years,
        year_to_domain=year_to_domain,
    )


def normalize_features(split: TemporalSplit, *, scaler: StandardScaler | None = None) -> NormalizedData:
    scaler = scaler or StandardScaler()
    scaler.fit(split.X_train)
    X_train_norm = pd.DataFrame(
        scaler.transform(split.X_train),
        columns=split.X_train.columns,
    )
    X_test_norm = pd.DataFrame(
        scaler.transform(split.X_test),
        columns=split.X_test.columns,
    )
    return NormalizedData(X_train_norm, X_test_norm, scaler)
