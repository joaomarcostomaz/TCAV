"""
datasets.py

Torch Dataset and DataLoader builders for temporal sequence classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class LoaderConfig:
    batch_size: int = 256
    num_workers: int = 0
    pin_memory: bool = False


class TemporalSequenceDataset(Dataset):
    """
    Each sample:
      x_seq: [L, F] float32
      y: scalar float32 (0/1)
      year: int64
      patient_id: str
    """

    def __init__(
        self,
        X_seq: np.ndarray,
        y: np.ndarray,
        years: np.ndarray,
        patient_ids: np.ndarray,
    ) -> None:
        self.X_seq = X_seq.astype(np.float32)
        self.y = y.astype(np.float32)
        self.years = years.astype(np.int64)
        self.patient_ids = patient_ids.astype(str)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X_seq[idx]),
            torch.tensor(self.y[idx], dtype=torch.float32),
            torch.tensor(self.years[idx], dtype=torch.long),
            self.patient_ids[idx],
        )


def build_loader(
    dataset: Dataset,
    cfg: LoaderConfig,
    shuffle: bool,
) -> DataLoader:
    """
    Standard DataLoader builder.
    """
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def build_model_aware_loader(
    dataset: Dataset,
    model_name: str,
    base_cfg: LoaderConfig,
    train: bool,
) -> DataLoader:
    """
    Model-specific loader policy from notebooks:
    - PatchTST may require smaller batch due to attention memory.
    """
    bs = base_cfg.batch_size
    if model_name.lower() == "patchtst":
        bs = min(64, bs)

    cfg = LoaderConfig(
        batch_size=bs,
        num_workers=base_cfg.num_workers,
        pin_memory=base_cfg.pin_memory,
    )
    return build_loader(dataset, cfg=cfg, shuffle=train)