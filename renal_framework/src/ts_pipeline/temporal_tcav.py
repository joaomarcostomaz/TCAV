from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset


@dataclass
class TemporalTCAVConfig:
    C: float = 0.1
    max_iter: int = 5000
    class_weight: str = "balanced"
    solver: str = "liblinear"
    standardize_per_timestep: bool = True
    random_state: int = 42


@dataclass
class DTWClusterConfig:
    n_clusters: int = 3
    metric: str = "dtw"
    max_iter: int = 50
    n_init: int = 4
    random_state: int = 42


@torch.no_grad()
def extract_hidden_states_batched(
    model: torch.nn.Module,
    loader,
    device: str = "cpu",
) -> Dict[str, np.ndarray]:
    """
    Assumes model can return hidden states [B,T,H] when called with return_hidden=True.
    If not, add a forward hook in model wrapper.
    """
    model.eval()
    all_h, all_y, all_years, all_pids = [], [], [], []

    for xb, yb, yrs, pids in loader:
        xb = xb.to(device, non_blocking=True)
        out = model(xb, return_hidden=True)  # expected dict or tuple convention
        if isinstance(out, dict):
            h = out["hidden_states"]
            logits = out.get("logits", None)
        else:
            # tuple convention: (logits, hidden_states)
            logits, h = out

        all_h.append(h.detach().cpu())
        all_y.append(yb.numpy())
        all_years.append(yrs.numpy())
        all_pids.extend(list(pids))

    hidden = torch.cat(all_h, dim=0).numpy().astype(np.float32)  # [N,T,H]
    y = np.concatenate(all_y).astype(int)
    years = np.concatenate(all_years).astype(int)
    pids = np.asarray(all_pids, dtype=str)

    return {"hidden_states": hidden, "y": y, "years": years, "pids": pids}


def compute_sequential_cavs(
    hidden_states_nth: np.ndarray,   # [N,T,H]
    concept_labels_n: np.ndarray,    # [N]
    cfg: TemporalTCAVConfig = TemporalTCAVConfig(),
) -> Dict[str, np.ndarray]:
    """
    Train one probe per timestep t => CAV_t.
    Returns:
      cavs: [T,H]
      trajectories: [N,T] where traj[i,t] = <h_{i,t}, cav_t>
    """
    X = np.asarray(hidden_states_nth, dtype=np.float32)
    y = np.asarray(concept_labels_n).astype(int)

    if X.ndim != 3:
        raise ValueError(f"Expected [N,T,H], got {X.shape}")
    N, T, H = X.shape
    if y.shape[0] != N:
        raise ValueError("concept_labels length mismatch.")

    cavs = np.zeros((T, H), dtype=np.float32)
    intercepts = np.zeros((T,), dtype=np.float32)
    probe_scores = np.zeros((T,), dtype=np.float32)

    for t in range(T):
        Xt = X[:, t, :]  # [N,H]

        if cfg.standardize_per_timestep:
            scaler = StandardScaler()
            Xt_fit = scaler.fit_transform(Xt)
        else:
            Xt_fit = Xt

        clf = LogisticRegression(
            C=cfg.C,
            max_iter=cfg.max_iter,
            class_weight=cfg.class_weight,
            solver=cfg.solver,
            random_state=cfg.random_state,
        )
        clf.fit(Xt_fit, y)

        w = clf.coef_.reshape(-1).astype(np.float32)
        nrm = np.linalg.norm(w)
        if nrm > 0:
            w = w / nrm

        cavs[t] = w
        intercepts[t] = float(clf.intercept_[0])
        probe_scores[t] = float(clf.score(Xt_fit, y))

    trajectories = np.einsum("nth,th->nt", X, cavs).astype(np.float32)  # [N,T]

    return {
        "cavs": cavs,
        "intercepts": intercepts,
        "probe_scores": probe_scores,
        "trajectories": trajectories,
    }


def cluster_trajectories_dtw(
    trajectories_nt: np.ndarray,   # [N,T]
    cfg: DTWClusterConfig = DTWClusterConfig(),
) -> Dict[str, np.ndarray]:
    traj = np.asarray(trajectories_nt, dtype=np.float32)
    if traj.ndim != 2:
        raise ValueError(f"Expected [N,T], got {traj.shape}")

    X_ts = to_time_series_dataset(traj[..., None])  # [N,T,1]

    km = TimeSeriesKMeans(
        n_clusters=cfg.n_clusters,
        metric=cfg.metric,
        max_iter=cfg.max_iter,
        n_init=cfg.n_init,
        random_state=cfg.random_state,
        verbose=False,
    )
    labels = km.fit_predict(X_ts)
    centroids = km.cluster_centers_.squeeze(-1)  # [K,T]

    K, T = centroids.shape
    mean_traj = np.zeros((K, T), dtype=np.float32)
    std_traj = np.zeros((K, T), dtype=np.float32)
    sizes = np.zeros((K,), dtype=int)

    for k in range(K):
        idx = np.where(labels == k)[0]
        sizes[k] = len(idx)
        if len(idx) > 0:
            mean_traj[k] = traj[idx].mean(axis=0)
            std_traj[k] = traj[idx].std(axis=0)

    return {
        "labels": labels.astype(int),
        "centroids": centroids.astype(np.float32),
        "cluster_mean_trajectories": mean_traj,
        "cluster_std_trajectories": std_traj,
        "cluster_sizes": sizes,
    }


def summarize_temporal_phenotypes(
    trajectories_nt: np.ndarray,
    labels_n: np.ndarray,
    y_outcome_n: Optional[np.ndarray] = None,
) -> Dict[int, Dict[str, Any]]:
    traj = np.asarray(trajectories_nt, dtype=np.float32)
    labels = np.asarray(labels_n).astype(int)
    K = int(labels.max()) + 1 if labels.size > 0 else 0
    T = traj.shape[1]

    summary: Dict[int, Dict[str, Any]] = {}
    for k in range(K):
        idx = np.where(labels == k)[0]
        if len(idx) == 0:
            summary[k] = {"size": 0, "mean_trajectory": [0.0] * T}
            continue

        tk = traj[idx]
        mean_t = tk.mean(axis=0)
        peak_t = int(np.argmax(mean_t))
        onset_w = max(2, T // 5)
        onset_slope = float(np.polyfit(np.arange(onset_w), mean_t[:onset_w], 1)[0])

        outcome_prev = None
        if y_outcome_n is not None:
            y = np.asarray(y_outcome_n).astype(int)
            outcome_prev = float(y[idx].mean())

        summary[k] = {
            "size": int(len(idx)),
            "mean_trajectory": mean_t.tolist(),
            "std_trajectory": tk.std(axis=0).tolist(),
            "peak_time": peak_t,
            "onset_slope": onset_slope,
            "outcome_prevalence": outcome_prev,
        }

    return summary