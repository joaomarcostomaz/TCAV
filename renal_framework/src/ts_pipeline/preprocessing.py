"""
preprocessing.py

Data loading, canonical preprocessing, patient/year split helpers, pivot builders,
feature-selection preparation, and sequence construction utilities extracted from
the notebook workflow.

This module keeps logic explicit and reproducible, while avoiding hidden globals.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


# -----------------------------
# Data classes
# -----------------------------
@dataclass
class SelectionConfig:
    rng_seed: int = 42
    final_top_k: int = 500
    m_candidates: int = 3000
    train_years_forced_start: int = 1997
    train_years_forced_end: int = 2006


@dataclass
class LGBSelectionConfig:
    params: Dict
    num_boost_round: int = 200


# -----------------------------
# Core loading + canonical prep
# -----------------------------
def load_and_prepare_feather(data_feather: Path) -> pd.DataFrame:
    """
    Load feather dataset and apply canonical preprocessing used in notebooks:
    - date to datetime
    - year from date
    - patient_id as str
    """
    if not data_feather.exists():
        raise FileNotFoundError(f"Feather file not found: {data_feather}")

    df = pd.read_feather(data_feather).copy()
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year.astype(int)
    df["patient_id"] = df["patient_id"].astype(str)
    return df


# -----------------------------
# Temporal split helpers
# -----------------------------
def infer_train_test_years(
    years_all: Sequence[int],
    forced_start: int = 1997,
    forced_end: int = 2006,
) -> Tuple[List[int], List[int]]:
    """
    Match notebook behavior:
    - if 1997..2006 exists, use as TRAIN_YEARS
    - otherwise first half of sorted years
    """
    years_all = sorted(set(map(int, years_all)))
    forced_set = set(range(forced_start, forced_end + 1))

    if forced_set.issubset(set(years_all)):
        train_years = list(range(forced_start, forced_end + 1))
    else:
        half = len(years_all) // 2
        train_years = years_all[:half]

    test_years = sorted([y for y in years_all if y not in set(train_years)])
    return sorted(train_years), test_years


def split_patients(
    patient_ids: Sequence[str],
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Random patient split preserving notebook protocol.
    """
    arr = np.asarray(patient_ids, dtype=str)
    train_patients, test_patients = train_test_split(
        arr, test_size=test_size, random_state=random_state
    )
    return sorted(map(str, train_patients)), sorted(map(str, test_patients))


# -----------------------------
# Pivot builders
# -----------------------------
def build_pivot_preserve_presence(
    df_input: pd.DataFrame,
    patients: Sequence[str],
    years: Sequence[int],
    events: Sequence[str],
) -> pd.DataFrame:
    """
    Pivot counts by (patient_id, year, event) WITHOUT full product reindex.
    """
    df_sub = df_input[
        df_input["patient_id"].isin(patients)
        & df_input["year"].isin(years)
        & df_input["event"].isin(events)
    ].copy()

    pivot = pd.pivot_table(
        df_sub,
        index=["patient_id", "year"],
        columns="event",
        values="date",
        aggfunc="count",
        fill_value=0,
    )
    return pivot.astype(int).sort_index()


def trim_post_death_rows(pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove patient-year rows where year > first death year for that patient.
    """
    if "DEATH" not in pivot_df.columns:
        return pivot_df.copy()

    death_rows = (
        pivot_df[pivot_df["DEATH"] > 0]
        .reset_index()[["patient_id", "year"]]
        .drop_duplicates()
    )
    death_year_map = death_rows.groupby("patient_id")["year"].min().to_dict()

    idx_df = pivot_df.index.to_frame(index=False)
    keep_mask = []
    for _, row in idx_df.iterrows():
        pid = row["patient_id"]
        yr = int(row["year"])
        dy = death_year_map.get(pid, None)
        keep_mask.append(False if (dy is not None and yr > dy) else True)

    return pivot_df.iloc[np.asarray(keep_mask, dtype=bool)].copy()


# -----------------------------
# Patient balancing helpers
# -----------------------------
def build_patient_availability_table(
    df_train_long: pd.DataFrame,
    candidate_train_patients: Sequence[str],
) -> pd.DataFrame:
    """
    Build patient availability table with:
    patient_id, n_avail_rows, years_avail, is_pos, death_year, first_year, last_year
    """
    death_py = df_train_long[
        df_train_long["event"].str.contains("DEATH", case=False, na=False)
    ][["patient_id", "year"]].drop_duplicates()

    pos_patient_candidates = death_py["patient_id"].unique().tolist()
    death_year_map = death_py.groupby("patient_id")["year"].min().to_dict()

    presence = df_train_long[["patient_id", "year"]].drop_duplicates()
    presence_groups = (
        presence.groupby("patient_id")["year"]
        .apply(lambda s: sorted(s.tolist()))
        .to_dict()
    )

    def available_years_for_patient(pid: str) -> List[int]:
        yrs = presence_groups.get(pid, [])
        dy = death_year_map.get(pid, None)
        if dy is not None:
            yrs = [y for y in yrs if y <= dy]
        return yrs

    rows = []
    for pid in sorted(set(candidate_train_patients)):
        years_avail = available_years_for_patient(pid)
        rows.append(
            {
                "patient_id": pid,
                "n_avail_rows": len(years_avail),
                "years_avail": years_avail,
                "is_pos": pid in pos_patient_candidates,
                "death_year": death_year_map.get(pid, None),
                "first_year": min(years_avail) if years_avail else None,
                "last_year": max(years_avail) if years_avail else None,
            }
        )

    patients_df = pd.DataFrame(rows)
    return patients_df[patients_df["n_avail_rows"] > 0].copy()


def select_full_balanced_patients(
    patients_df: pd.DataFrame,
    rng_seed: int = 42,
) -> Tuple[List[str], List[str], int]:
    """
    Full balanced selection:
    choose same number of pos/neg patients = min(n_pos, n_neg).
    """
    pos_df = (
        patients_df[patients_df["is_pos"] == True]
        .sample(frac=1, random_state=rng_seed)
        .reset_index(drop=True)
    )
    neg_df = (
        patients_df[patients_df["is_pos"] == False]
        .sample(frac=1, random_state=rng_seed + 1)
        .reset_index(drop=True)
    )

    n_pat_each = min(len(pos_df), len(neg_df))
    if n_pat_each == 0:
        raise RuntimeError("No sufficient patients in one class for balanced selection.")

    selected_pos = pos_df["patient_id"].iloc[:n_pat_each].tolist()
    selected_neg = neg_df["patient_id"].iloc[:n_pat_each].tolist()
    return selected_pos, selected_neg, n_pat_each


# -----------------------------
# Feature selection via LGBM
# -----------------------------
def select_top_events_via_lgbm(
    df: pd.DataFrame,
    selected_train_patients: Sequence[str],
    train_years: Sequence[int],
    cfg: SelectionConfig,
    lgb_cfg: LGBSelectionConfig,
) -> List[str]:
    """
    Candidate events by frequency (excluding DEATH), then LGBM gain importance,
    return top_k events.
    """
    df_train_py_event = df[
        df["patient_id"].isin(selected_train_patients)
        & df["year"].isin(train_years)
    ][["patient_id", "year", "event"]].drop_duplicates()

    event_freq = df_train_py_event.groupby("event").size().sort_values(ascending=False)
    candidate_events = [
        e for e in event_freq.index if str(e).upper() != "DEATH"
    ][: cfg.m_candidates]

    events_for_pivot = set(candidate_events) | {"DEATH"}
    pivot_candidate = build_pivot_preserve_presence(
        df, selected_train_patients, train_years, list(events_for_pivot)
    )

    if "DEATH" not in pivot_candidate.columns:
        pivot_candidate["DEATH"] = 0

    pivot_candidate = pivot_candidate.sort_index().astype(int)
    available_candidate_events = [c for c in candidate_events if c in pivot_candidate.columns]

    X_lgb = pivot_candidate[available_candidate_events].astype(np.float32)
    y_lgb = (pivot_candidate["DEATH"] > 0).astype(int).values

    pos = int(y_lgb.sum())
    neg = int(len(y_lgb) - pos)
    if pos == 0:
        raise RuntimeError("No positive rows found for LGB feature selection.")

    params = dict(lgb_cfg.params)
    params["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0

    lgb_train = lgb.Dataset(X_lgb, label=y_lgb)
    gbm = lgb.train(params, lgb_train, num_boost_round=lgb_cfg.num_boost_round)

    importance_df = pd.DataFrame(
        {
            "feature": available_candidate_events,
            "importance": gbm.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    return importance_df["feature"].tolist()[: cfg.final_top_k]


# -----------------------------
# Final train/test rows builder
# -----------------------------
def build_train_test_rows(
    df: pd.DataFrame,
    selected_train_patients: Sequence[str],
    candidate_test_patients: Sequence[str],
    train_years: Sequence[int],
    test_years: Sequence[int],
    top_k_events: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build final train_rows/test_rows with columns:
    patient_id, year, top_k_events..., DEATH
    """
    events_to_keep = set(top_k_events) | {"DEATH"}

    # train
    pivot_final = build_pivot_preserve_presence(
        df, selected_train_patients, train_years, list(events_to_keep)
    )
    if "DEATH" not in pivot_final.columns:
        pivot_final["DEATH"] = 0
    pivot_final = pivot_final.sort_index().astype(int)
    pivot_final_trimmed = trim_post_death_rows(pivot_final)
    train_rows = pivot_final_trimmed.reset_index().copy()

    # test
    pivot_test = build_pivot_preserve_presence(
        df, candidate_test_patients, test_years, list(events_to_keep)
    )
    if "DEATH" not in pivot_test.columns:
        pivot_test["DEATH"] = 0
    pivot_test = pivot_test.astype(int)
    test_rows = pivot_test.reset_index().copy()

    missing_cols = [c for c in top_k_events if c not in test_rows.columns]
    if missing_cols:
        test_rows = pd.concat(
            [test_rows, pd.DataFrame(0, index=test_rows.index, columns=missing_cols)],
            axis=1,
        )

    train_rows = train_rows[["patient_id", "year"] + list(top_k_events) + ["DEATH"]]
    test_rows = test_rows[["patient_id", "year"] + list(top_k_events) + ["DEATH"]]

    return train_rows, test_rows


# -----------------------------
# Sequence builders
# -----------------------------
def _build_patient_year_matrix(
    rows_df: pd.DataFrame, feature_cols: List[str]
) -> Dict[str, pd.DataFrame]:
    out = {}
    for pid, g in rows_df.groupby("patient_id"):
        gg = g.sort_values("year").copy()
        out[pid] = gg[["year"] + feature_cols + ["DEATH"]].reset_index(drop=True)
    return out


def _make_sequence_for_row(
    patient_df: pd.DataFrame,
    row_idx: int,
    feature_cols: List[str],
    lookback: int,
) -> np.ndarray:
    start = max(0, row_idx - lookback + 1)
    hist = patient_df.iloc[start : row_idx + 1][feature_cols].to_numpy(dtype=np.float32)

    if hist.shape[0] < lookback:
        pad = np.zeros((lookback - hist.shape[0], hist.shape[1]), dtype=np.float32)
        hist = np.vstack([pad, hist])
    return hist


def build_temporal_sequences(
    rows_df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert patient-year rows to sequence tensor:
    X: [N, L, F], y: [N], years: [N], pids: [N]
    """
    by_patient = _build_patient_year_matrix(rows_df, feature_cols)

    X_list, y_list, year_list, pid_list = [], [], [], []

    for pid, pdf in by_patient.items():
        for i in range(len(pdf)):
            x_seq = _make_sequence_for_row(pdf, i, feature_cols, lookback)
            y = 1.0 if float(pdf.iloc[i]["DEATH"]) > 0 else 0.0
            yr = int(pdf.iloc[i]["year"])

            X_list.append(x_seq)
            y_list.append(y)
            year_list.append(yr)
            pid_list.append(pid)

    X = np.stack(X_list, axis=0).astype(np.float32)
    y = np.asarray(y_list, dtype=np.float32)
    years = np.asarray(year_list, dtype=np.int64)
    pids = np.asarray(pid_list, dtype=str)

    return X, y, years, pids