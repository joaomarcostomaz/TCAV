from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
import numpy as np
import pandas as pd
import json


def save_temporal_tcav_outputs(out_dir: str | Path, payload: Dict[str, Any]) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "temporal_cavs.npy", payload["cavs"])
    np.save(out_dir / "temporal_cav_intercepts.npy", payload["intercepts"])
    np.save(out_dir / "probe_scores_by_t.npy", payload["probe_scores"])
    np.save(out_dir / "concept_trajectories.npy", payload["trajectories"])

    cl = payload["cluster"]
    np.save(out_dir / "cluster_labels.npy", cl["labels"])
    np.save(out_dir / "cluster_centroids.npy", cl["centroids"])
    np.save(out_dir / "cluster_mean_trajectories.npy", cl["cluster_mean_trajectories"])
    np.save(out_dir / "cluster_std_trajectories.npy", cl["cluster_std_trajectories"])
    np.save(out_dir / "cluster_sizes.npy", cl["cluster_sizes"])

    with open(out_dir / "cluster_phenotypes.json", "w", encoding="utf-8") as f:
        json.dump(payload["phenotypes"], f, indent=2)

    rows = []
    for k, v in payload["phenotypes"].items():
        rows.append({
            "cluster": int(k),
            "size": v.get("size"),
            "peak_time": v.get("peak_time"),
            "onset_slope": v.get("onset_slope"),
            "outcome_prevalence": v.get("outcome_prevalence"),
        })
    pd.DataFrame(rows).to_csv(out_dir / "cluster_summary.csv", index=False)