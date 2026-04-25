"""
tabpfn_pipeline.preprocessing

Preprocessing and dataset-building utilities for the TabPFN interpretability pipeline.

This module includes:
- canonical event preprocessing
- temporal split logic
- patient balancing with row caps
- event selection via LightGBM
- train/test row construction for TabPFN workflows
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split


@dataclass
class TabPFNPrepConfig:
    rng_seed: int = 42
    target_pos_lines: int = 5000
    target_neg_lines: int = 5000
    max_total_rows: int = 10000
    final_top_k: int = 500
    m_candidates: int = 3000
    forced_train_year_start: int = 1997
    forced_train_year_end: int = 2006


def canonicalize_event_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Canonical preprocessing used across notebooks.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    out["year"] = out["date"].dt.year.astype(int)
    out["patient_id"] = out["patient_id"].astype(str)
    return out


def infer_train_test_years(
    years_all: Sequence[int],
    forced_start: int = 1997,
    forced_end: int = 2006,
) -> Tuple[List[int], List[int]]:
    """
    Match notebook behavior:
    - if forced range exists in data, use it as train
    - else use first half as train
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
    all_patient_ids: Sequence[str],
    test_size: float = 0.1,
    random_state: int = 42,
) -> Tuple[List[str], List[str]]:
    """
    Patient-level split (no leakage across patient).
    """
    arr = np.asarray(all_patient_ids, dtype=str)
    train_pat, test_pat = train_test_split(arr, test_size=test_size, random_state=random_state)
    return sorted(map(str, train_pat)), sorted(map(str, test_pat))


def build_pivot_preserve_presence(
    df_input: pd.DataFrame,
    patients: Sequence[str],
    years: Sequence[int],
    events: Sequence[str],
) -> pd.DataFrame:
    """
    Pivot counts only for observed patient-year rows.
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
    Remove rows where year > first death year for each patient.
    """
    if "DEATH" not in pivot_df.columns:
        return pivot_df.copy()

    death_rows = (
        pivot_df[pivot_df["DEATH"] > 0]
        .reset_index()[["patient_id", "year"]]
        .drop_duplicates()
    )
    death_year_map = death_rows.groupby("patient_id")["year"].min().to_dict()

    idx = pivot_df.index.to_frame(index=False)
    keep_mask = []
    for _, r in idx.iterrows():
        pid = r["patient_id"]
        yr = int(r["year"])
        dy = death_year_map.get(pid, None)
        keep_mask.append(False if (dy is not None and yr > dy) else True)

    return pivot_df.iloc[np.asarray(keep_mask, dtype=bool)].copy()


def select_equal_patients_with_line_cap(
    patients_df: pd.DataFrame,
    target_pos_lines: int,
    target_neg_lines: int,
    rng_seed: int = 42,
):
    """
    Same selection strategy from notebooks:
    pick equal #patients per class while satisfying row caps when possible.
    """
    pos_df = patients_df[patients_df["is_pos"] == True].copy()
    neg_df = patients_df[patients_df["is_pos"] == False].copy()

    pos_df = pos_df.sample(frac=1, random_state=rng_seed).reset_index(drop=True)
    neg_df = neg_df.sample(frac=1, random_state=rng_seed + 1).reset_index(drop=True)

    pos_df["cum_rows"] = pos_df["n_avail_rows"].cumsum()
    neg_df["cum_rows"] = neg_df["n_avail_rows"].cumsum()

    max_n_possible = min(len(pos_df), len(neg_df))
    feasible_ns = []
    for n in range(1, max_n_possible + 1):
        pos_rows = int(pos_df.loc[n - 1, "cum_rows"])
        neg_rows = int(neg_df.loc[n - 1, "cum_rows"])
        if pos_rows <= target_pos_lines and neg_rows <= target_neg_lines:
            feasible_ns.append((n, pos_rows, neg_rows))

    if feasible_ns:
        n_selected, pos_rows_sel, neg_rows_sel = feasible_ns[-1]
        selected_pos = pos_df["patient_id"].iloc[:n_selected].tolist()
        selected_neg = neg_df["patient_id"].iloc[:n_selected].tolist()
        return selected_pos, selected_neg, n_selected, pos_rows_sel, neg_rows_sel

    n_pos_max = int((pos_df["cum_rows"] <= target_pos_lines).sum())
    n_neg_max = int((neg_df["cum_rows"] <= target_neg_lines).sum())
    n_selected = min(n_pos_max, n_neg_max)

    if n_selected > 0:
        pos_rows_sel = int(pos_df.loc[n_selected - 1, "cum_rows"])
        neg_rows_sel = int(neg_df.loc[n_selected - 1, "cum_rows"])
        selected_pos = pos_df["patient_id"].iloc[:n_selected].tolist()
        selected_neg = neg_df["patient_id"].iloc[:n_selected].tolist()
        return selected_pos, selected_neg, n_selected, pos_rows_sel, neg_rows_sel

    return [], [], 0, 0, 0


def _build_patient_availability_table(
    df_train_long: pd.DataFrame,
    candidate_train_patients: Sequence[str],
) -> pd.DataFrame:
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
            }
        )
    out = pd.DataFrame(rows)
    return out[out["n_avail_rows"] > 0].copy()


def select_top_events_lgbm(
    df: pd.DataFrame,
    selected_train_patients: Sequence[str],
    train_years: Sequence[int],
    m_candidates: int,
    final_top_k: int,
    lgb_params: Dict,
) -> List[str]:
    """
    Candidate event filtering + LGB gain importance ranking.
    """
    df_train_py_event = df[
        df["patient_id"].isin(selected_train_patients)
        & df["year"].isin(train_years)
    ][["patient_id", "year", "event"]].drop_duplicates()

    event_freq = df_train_py_event.groupby("event").size().sort_values(ascending=False)
    candidate_events = [e for e in event_freq.index if str(e).upper() != "DEATH"][:m_candidates]

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

    params = dict(lgb_params)
    params["scale_pos_weight"] = (neg / pos) if pos > 0 else 1.0

    lgb_train = lgb.Dataset(X_lgb, label=y_lgb)
    gbm = lgb.train(params, lgb_train, num_boost_round=200)

    importance_df = pd.DataFrame(
        {
            "feature": available_candidate_events,
            "importance": gbm.feature_importance(importance_type="gain"),
        }
    ).sort_values("importance", ascending=False)

    return importance_df["feature"].tolist()[:final_top_k]


def build_train_test_rows(
    df: pd.DataFrame,
    selected_train_patients: Sequence[str],
    candidate_test_patients: Sequence[str],
    train_years: Sequence[int],
    test_years: Sequence[int],
    top_k_events: Sequence[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build final train/test rows for TabPFN pipeline.
    """
    events_to_keep = set(top_k_events) | {"DEATH"}

    pivot_train = build_pivot_preserve_presence(df, selected_train_patients, train_years, list(events_to_keep))
    if "DEATH" not in pivot_train.columns:
        pivot_train["DEATH"] = 0
    pivot_train = pivot_train.sort_index().astype(int)
    train_rows = trim_post_death_rows(pivot_train).reset_index().copy()

    pivot_test = build_pivot_preserve_presence(df, candidate_test_patients, test_years, list(events_to_keep))
    if "DEATH" not in pivot_test.columns:
        pivot_test["DEATH"] = 0
    test_rows = pivot_test.astype(int).reset_index().copy()

    missing_cols = [c for c in top_k_events if c not in test_rows.columns]
    if missing_cols:
        test_rows = pd.concat(
            [test_rows, pd.DataFrame(0, index=test_rows.index, columns=missing_cols)],
            axis=1,
        )

    cols = ["patient_id", "year"] + list(top_k_events) + ["DEATH"]
    train_rows = train_rows[cols]
    test_rows = test_rows[cols]
    return train_rows, test_rows


def prepare_tabpfn_rows(
    df: pd.DataFrame,
    cfg: TabPFNPrepConfig,
    lgb_params: Dict,
) -> Dict[str, object]:
    """
    End-to-end preprocessing orchestration for TabPFN tabular training/eval rows.
    Returns a dict with artifacts used by downstream modules.
    """
    years_all = sorted(df["year"].unique())
    train_years, test_years = infer_train_test_years(
        years_all,
        forced_start=cfg.forced_train_year_start,
        forced_end=cfg.forced_train_year_end,
    )

    all_patients = np.asarray(df["patient_id"].unique(), dtype=str)
    train_patients, test_patients = split_patients(
        all_patients, test_size=0.1, random_state=cfg.rng_seed
    )

    df_train_long = df[
        df["patient_id"].isin(train_patients) & df["year"].isin(train_years)
    ][["patient_id", "year", "event", "date"]].drop_duplicates()

    patients_df = _build_patient_availability_table(df_train_long, train_patients)

    pos_sel, neg_sel, n_each, _, _ = select_equal_patients_with_line_cap(
        patients_df=patients_df,
        target_pos_lines=cfg.target_pos_lines,
        target_neg_lines=cfg.target_neg_lines,
        rng_seed=cfg.rng_seed,
    )

    if n_each == 0:
        # strict fallback (same notebook spirit)
        n_fallback = min(
            len(patients_df[patients_df["is_pos"] == True]),
            len(patients_df[patients_df["is_pos"] == False]),
        )
        if n_fallback == 0:
            raise RuntimeError("No sufficient patients in one class for fallback selection.")

        pos_df = (
            patients_df[patients_df["is_pos"] == True]
            .sample(frac=1, random_state=cfg.rng_seed)
            .reset_index(drop=True)
        )
        neg_df = (
            patients_df[patients_df["is_pos"] == False]
            .sample(frac=1, random_state=cfg.rng_seed + 1)
            .reset_index(drop=True)
        )
        pos_sel = pos_df["patient_id"].iloc[:n_fallback].tolist()
        neg_sel = neg_df["patient_id"].iloc[:n_fallback].tolist()

    selected_train_patients = pos_sel + neg_sel

    top_k_events = select_top_events_lgbm(
        df=df,
        selected_train_patients=selected_train_patients,
        train_years=train_years,
        m_candidates=cfg.m_candidates,
        final_top_k=cfg.final_top_k,
        lgb_params=lgb_params,
    )

    train_rows, test_rows = build_train_test_rows(
        df=df,
        selected_train_patients=selected_train_patients,
        candidate_test_patients=test_patients,
        train_years=train_years,
        test_years=test_years,
        top_k_events=top_k_events,
    )

    if len(train_rows) > cfg.max_total_rows:
        train_rows = train_rows.sample(n=cfg.max_total_rows, random_state=cfg.rng_seed).reset_index(drop=True)

    return {
        "train_rows": train_rows,
        "test_rows": test_rows,
        "top_k_events": top_k_events,
        "train_years": train_years,
        "test_years": test_years,
        "selected_train_patients": selected_train_patients,
    }

def build_pivot_for_events_fullreindex(df_input, patients, years, events):
    """
    pivot with full product reindex patient x years (In this case will be zeros in some rows).
    """
    df_sub = df_input[df_input['patient_id'].isin(patients) & df_input['year'].isin(
        years) & df_input['event'].isin(events)].copy()
    pivot = pd.pivot_table(
        df_sub,
        index=['patient_id', 'year'],
        columns='event',
        values='date',
        aggfunc='count',
        fill_value=0
    )
    full_index = pd.MultiIndex.from_product(
        [sorted(map(str, patients)), years], names=['patient_id', 'year'])
    pivot = pivot.reindex(full_index, fill_value=0).sort_index()
    return pivot