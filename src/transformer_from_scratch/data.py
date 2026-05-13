"""Dataset and dataloader helpers for numpy token shards."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class NumpyDataset(Dataset):
    def __init__(self, source: str | Path | np.ndarray):
        if isinstance(source, np.ndarray):
            data = source
        else:
            data = np.load(Path(source), allow_pickle=False)
        if data.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {data.shape}")
        self.data = data.astype(np.int64, copy=False)

    def __len__(self) -> int:
        return int(self.data.shape[0])

    def __getitem__(self, idx: int) -> torch.Tensor:
        return torch.as_tensor(self.data[idx], dtype=torch.long)


def lm_collate(batch: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    sequences = torch.stack(batch, dim=0)
    return sequences[:, :-1], sequences[:, 1:]


def build_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lm_collate,
        drop_last=False,
    )


class StreamingNumpyShards:
    """Keep a list of shards that can grow over time and be reloaded."""

    def __init__(self, shards: Sequence[str | Path] | None = None):
        self.shards = [str(Path(p)) for p in (shards or [])]

    def add_shard(self, shard: str | Path) -> None:
        self.shards.append(str(Path(shard)))

    def load_all(self) -> np.ndarray:
        if not self.shards:
            raise ValueError("No shards registered")
        arrays = [np.load(path, allow_pickle=False) for path in self.shards]
        return np.concatenate(arrays, axis=0)

    def dataset(self) -> NumpyDataset:
        return NumpyDataset(self.load_all())
