from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class PositionBatch:
    patches: torch.Tensor
    realized_returns: torch.Tensor
    forecast_risk: torch.Tensor | None = None


class CachedDistributionDataset(Dataset):
    """Dataset for cached FinCast distribution patches.

    Patches are [T, H, C] and realized_returns [T]. When `asset_names` is
    provided, sequences are sliced within each asset's contiguous block so a
    returned window never spans across assets.
    """

    def __init__(
        self,
        patches: torch.Tensor,
        realized_returns: torch.Tensor,
        *,
        seq_len: int = 32,
        forecast_risk: torch.Tensor | None = None,
        stride: int | None = None,
        asset_names: np.ndarray | None = None,
        episode_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        patches = torch.as_tensor(patches, dtype=torch.float32)
        realized_returns = torch.as_tensor(realized_returns, dtype=torch.float32).flatten()
        if patches.ndim != 3:
            raise ValueError(f"patches must be [T, H, C], got {tuple(patches.shape)}")
        if patches.shape[0] != realized_returns.shape[0]:
            raise ValueError(
                "patches and realized_returns must agree on T, "
                f"got {patches.shape[0]} and {realized_returns.shape[0]}"
            )
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")

        self.patches = patches
        self.realized_returns = realized_returns
        self.seq_len = int(seq_len)
        self.stride = int(stride or seq_len)
        self.asset_names: np.ndarray | None = None
        self.episode_ids: np.ndarray | None = None
        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        if forecast_risk is None:
            self.forecast_risk = None
        else:
            risk = torch.as_tensor(forecast_risk, dtype=torch.float32).flatten()
            if risk.shape[0] != patches.shape[0]:
                raise ValueError("forecast_risk must have one value per patch.")
            self.forecast_risk = risk

        self.starts: list[int] = []
        if asset_names is not None:
            asset_names = np.asarray(asset_names)
            if asset_names.shape[0] != patches.shape[0]:
                raise ValueError("asset_names must have one entry per patch.")
            self.asset_names = asset_names
        if episode_ids is not None:
            episode_ids = np.asarray(episode_ids)
            if episode_ids.shape[0] != patches.shape[0]:
                raise ValueError("episode_ids must have one entry per patch.")
            self.episode_ids = episode_ids
            for run_start, run_end in _contiguous_runs(episode_ids):
                run_length = run_end - run_start
                if run_length < self.seq_len:
                    continue
                last_start = run_end - self.seq_len
                for s in range(run_start, last_start + 1, self.stride):
                    self.starts.append(s)
            if not self.starts:
                raise ValueError("No episode has enough samples for the requested seq_len.")
        elif asset_names is None:
            if patches.shape[0] < self.seq_len:
                raise ValueError("Not enough samples for the requested seq_len.")
            self.starts = list(range(0, patches.shape[0] - self.seq_len + 1, self.stride))
        else:
            for run_start, run_end in _contiguous_runs(asset_names):
                run_length = run_end - run_start
                if run_length < self.seq_len:
                    continue
                last_start = run_end - self.seq_len
                for s in range(run_start, last_start + 1, self.stride):
                    self.starts.append(s)
            if not self.starts:
                raise ValueError("No asset has enough samples for the requested seq_len.")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.starts[index]
        end = start + self.seq_len
        item = {
            "patches": self.patches[start:end],
            "realized_returns": self.realized_returns[start:end],
        }
        if self.forecast_risk is not None:
            item["forecast_risk"] = self.forecast_risk[start:end]
        return item


class CachedFeatureDataset(Dataset):
    """Dataset for cached per-date feature vectors.

    Features are `[T, D]` and realized_returns `[T]`. With `asset_names`,
    returned sequences stay within each asset's contiguous block, matching
    `CachedDistributionDataset` split semantics.
    """

    def __init__(
        self,
        features: torch.Tensor,
        realized_returns: torch.Tensor,
        *,
        seq_len: int = 32,
        stride: int | None = None,
        asset_names: np.ndarray | None = None,
        episode_ids: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        features = torch.as_tensor(features, dtype=torch.float32)
        realized_returns = torch.as_tensor(realized_returns, dtype=torch.float32).flatten()
        if features.ndim != 2:
            raise ValueError(f"features must be [T, D], got {tuple(features.shape)}")
        if features.shape[0] != realized_returns.shape[0]:
            raise ValueError(
                "features and realized_returns must agree on T, "
                f"got {features.shape[0]} and {realized_returns.shape[0]}"
            )
        if seq_len <= 0:
            raise ValueError("seq_len must be positive.")

        self.features = features
        self.realized_returns = realized_returns
        self.seq_len = int(seq_len)
        self.stride = int(stride or seq_len)
        self.asset_names: np.ndarray | None = None
        self.episode_ids: np.ndarray | None = None
        if self.stride <= 0:
            raise ValueError("stride must be positive.")

        self.starts: list[int] = []
        if asset_names is not None:
            asset_names = np.asarray(asset_names)
            if asset_names.shape[0] != features.shape[0]:
                raise ValueError("asset_names must have one entry per feature row.")
            self.asset_names = asset_names
        if episode_ids is not None:
            episode_ids = np.asarray(episode_ids)
            if episode_ids.shape[0] != features.shape[0]:
                raise ValueError("episode_ids must have one entry per feature row.")
            self.episode_ids = episode_ids
            for run_start, run_end in _contiguous_runs(episode_ids):
                run_length = run_end - run_start
                if run_length < self.seq_len:
                    continue
                last_start = run_end - self.seq_len
                for s in range(run_start, last_start + 1, self.stride):
                    self.starts.append(s)
            if not self.starts:
                raise ValueError("No episode has enough samples for the requested seq_len.")
        elif asset_names is None:
            if features.shape[0] < self.seq_len:
                raise ValueError("Not enough samples for the requested seq_len.")
            self.starts = list(range(0, features.shape[0] - self.seq_len + 1, self.stride))
        else:
            for run_start, run_end in _contiguous_runs(asset_names):
                run_length = run_end - run_start
                if run_length < self.seq_len:
                    continue
                last_start = run_end - self.seq_len
                for s in range(run_start, last_start + 1, self.stride):
                    self.starts.append(s)
            if not self.starts:
                raise ValueError("No asset has enough samples for the requested seq_len.")

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        start = self.starts[index]
        end = start + self.seq_len
        return {
            "features": self.features[start:end],
            "realized_returns": self.realized_returns[start:end],
        }


def time_ordered_train_test_indices(
    dataset: CachedDistributionDataset | CachedFeatureDataset,
    *,
    test_fraction: float = 0.2,
) -> tuple[list[int], list[int]]:
    """Split sequence indices by time order, independently within each asset.

    For single-asset data this is simply the first 80% of sequences for train
    and the last 20% for test. For multi-asset caches, each asset receives the
    same within-asset split so the test set is not just the final assets in the
    concatenated cache.
    """

    train_indices, validation_indices, test_indices = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=0.0,
        test_fraction=test_fraction,
    )
    return train_indices + validation_indices, test_indices


def time_ordered_train_validation_test_indices(
    dataset: CachedDistributionDataset | CachedFeatureDataset,
    *,
    validation_fraction: float = 0.1,
    test_fraction: float = 0.2,
) -> tuple[list[int], list[int], list[int]]:
    """Split sequence indices by time order into train/validation/test.

    The split is applied independently within each asset, preserving chronology
    and keeping the final time block for test. Defaults are 70/10/20.
    """

    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between 0 and 1.")
    if validation_fraction + test_fraction >= 1.0:
        raise ValueError("validation_fraction + test_fraction must be less than 1.")

    groups: dict[str, list[int]] = {}
    for dataset_index, start in enumerate(dataset.starts):
        if dataset.asset_names is None:
            asset = "__all__"
        else:
            asset = str(dataset.asset_names[start])
        groups.setdefault(asset, []).append(dataset_index)

    train_indices: list[int] = []
    validation_indices: list[int] = []
    test_indices: list[int] = []
    for asset, indices in groups.items():
        n_total = len(indices)
        n_test = max(1, int(n_total * test_fraction))
        n_validation = (
            max(1, int(n_total * validation_fraction))
            if validation_fraction > 0.0
            else 0
        )
        n_train = n_total - n_validation - n_test
        if n_train <= 0:
            raise ValueError(
                "Not enough sequences for the requested validation/test fractions "
                f"in asset {asset!r}."
            )
        train_indices.extend(indices[:n_train])
        validation_indices.extend(indices[n_train : n_train + n_validation])
        test_indices.extend(indices[n_train + n_validation :])
    return train_indices, validation_indices, test_indices


def _contiguous_runs(asset_names: np.ndarray) -> list[tuple[int, int]]:
    """Return [start, end) index ranges for each maximal run of equal asset names."""
    if len(asset_names) == 0:
        return []
    runs: list[tuple[int, int]] = []
    run_start = 0
    current = asset_names[0]
    for i in range(1, len(asset_names)):
        if asset_names[i] != current:
            runs.append((run_start, i))
            run_start = i
            current = asset_names[i]
    runs.append((run_start, len(asset_names)))
    return runs
