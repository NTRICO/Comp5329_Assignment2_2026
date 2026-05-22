from __future__ import annotations

import numpy as np
import torch

from src.datasets.trader_dataset import (
    CachedFeatureDataset,
    CachedDistributionDataset,
    time_ordered_train_test_indices,
    time_ordered_train_validation_test_indices,
)


def test_train_validation_test_split_preserves_asset_order() -> None:
    patches = torch.zeros(40, 2, 3)
    returns = torch.zeros(40)
    asset_names = np.asarray(["AAA"] * 20 + ["BBB"] * 20, dtype=object)
    dataset = CachedDistributionDataset(
        patches=patches,
        realized_returns=returns,
        seq_len=2,
        stride=2,
        asset_names=asset_names,
    )

    train_idx, val_idx, test_idx = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=0.2,
        test_fraction=0.2,
    )

    assert len(train_idx) == 12
    assert len(val_idx) == 4
    assert len(test_idx) == 4
    for indices in (train_idx, val_idx, test_idx):
        assert len({asset_names[dataset.starts[i]] for i in indices}) == 2

    for asset in ("AAA", "BBB"):
        train_starts = [dataset.starts[i] for i in train_idx if asset_names[dataset.starts[i]] == asset]
        val_starts = [dataset.starts[i] for i in val_idx if asset_names[dataset.starts[i]] == asset]
        test_starts = [dataset.starts[i] for i in test_idx if asset_names[dataset.starts[i]] == asset]
        assert max(train_starts) < min(val_starts)
        assert max(val_starts) < min(test_starts)


def test_train_test_wrapper_matches_no_validation_split() -> None:
    patches = torch.zeros(20, 2, 3)
    returns = torch.zeros(20)
    dataset = CachedDistributionDataset(
        patches=patches,
        realized_returns=returns,
        seq_len=2,
        stride=2,
    )

    train_idx, test_idx = time_ordered_train_test_indices(dataset, test_fraction=0.2)
    train_idx_3way, val_idx, test_idx_3way = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=0.0,
        test_fraction=0.2,
    )

    assert val_idx == []
    assert train_idx == train_idx_3way
    assert test_idx == test_idx_3way


def test_cached_feature_dataset_uses_episode_ids_for_sequence_boundaries() -> None:
    features = torch.zeros(8, 3)
    returns = torch.zeros(8)
    asset_names = np.asarray(["stock_0"] * 8, dtype=object)
    episode_ids = np.asarray([10] * 4 + [11] * 4)

    dataset = CachedFeatureDataset(
        features=features,
        realized_returns=returns,
        seq_len=3,
        stride=1,
        asset_names=asset_names,
        episode_ids=episode_ids,
    )

    assert dataset.starts == [0, 1, 4, 5]
    for start in dataset.starts:
        assert len(set(episode_ids[start : start + dataset.seq_len].tolist())) == 1


def test_cached_distribution_dataset_uses_episode_ids_for_sequence_boundaries() -> None:
    patches = torch.zeros(8, 2, 1)
    returns = torch.zeros(8)
    asset_names = np.asarray(["stock_0"] * 8, dtype=object)
    episode_ids = np.asarray([10] * 4 + [11] * 4)

    dataset = CachedDistributionDataset(
        patches=patches,
        realized_returns=returns,
        seq_len=3,
        stride=1,
        asset_names=asset_names,
        episode_ids=episode_ids,
    )

    assert dataset.starts == [0, 1, 4, 5]
    for start in dataset.starts:
        assert len(set(episode_ids[start : start + dataset.seq_len].tolist())) == 1
