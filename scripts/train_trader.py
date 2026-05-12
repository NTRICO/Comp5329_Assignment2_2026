"""Train the position-aware GRU controller on cached FinCast distribution patches.

Headless mirror of notebooks/02_trader_training.ipynb. Edit the constants at the
top of main() to tweak hyperparameters — there are no longer any YAML configs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader, Subset

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.trader_dataset import CachedDistributionDataset, time_ordered_train_test_indices
from src.fincast_io.cache_builder import load_distribution_cache
from src.fincast_io.forecast_features import forecast_to_return_patch
from src.trader.cnn_gru import PositionAwareGRUPolicy
from src.trader.encoder_transformer import (
    EncoderTransformerPolicy,
    EncoderTransformerPolicyConfig,
)
from src.training.losses import MeanVarianceTurnoverLoss
from src.training.trainer import evaluate_policy, train_one_epoch
from src.utils.config import PositionControllerConfig, PositionLossConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the position-aware GRU trader.")
    parser.add_argument("--cache", default=str(WORKSPACE_ROOT / "data" / "cache" / "position_fincast_daily_cache.npz"))
    parser.add_argument("--ckpt-dir", default=str(WORKSPACE_ROOT / "outputs" / "checkpoints"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = 42
    input_horizon = 5
    model_kind = "encoder_transformer"
    controller_cfg = (
        EncoderTransformerPolicyConfig(horizon_len=input_horizon)
        if model_kind == "encoder_transformer"
        else PositionControllerConfig(horizon_len=input_horizon)
    )
    loss_cfg = PositionLossConfig()
    seq_len = 32
    dataset_stride = 32
    test_fraction = 0.2
    batch_size = 32
    epochs = 50
    learning_rate = 1e-3
    weight_decay = 0.0
    grad_clip = 1.0
    initial_position = 0.0

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = load_distribution_cache(args.cache)
    level_patches = torch.as_tensor(cache["full_outputs"], dtype=torch.float32)
    last_values = torch.as_tensor(cache["last_values"], dtype=torch.float32)
    realized_returns = torch.as_tensor(cache["realized_returns"], dtype=torch.float32)
    asset_names = cache["asset_names"]

    if not 1 <= input_horizon <= level_patches.shape[1]:
        raise ValueError(f"input_horizon must be in [1, {level_patches.shape[1]}].")
    return_patches = forecast_to_return_patch(level_patches, last_values)
    return_patches = return_patches[:, :input_horizon, :]

    dataset = CachedDistributionDataset(
        patches=return_patches,
        realized_returns=realized_returns,
        seq_len=seq_len,
        stride=dataset_stride,
        asset_names=asset_names if asset_names.size else None,
    )

    train_indices, test_indices = time_ordered_train_test_indices(
        dataset,
        test_fraction=test_fraction,
    )
    n_train = len(train_indices)
    n_test = len(test_indices)
    n_total = len(dataset)
    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if model_kind == "encoder_transformer":
        model = EncoderTransformerPolicy(controller_cfg).to(device)
    elif model_kind == "cnn_gru":
        model = PositionAwareGRUPolicy(controller_cfg).to(device)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")
    loss_fn = MeanVarianceTurnoverLoss(loss_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_test = float("inf")
    best_path = ckpt_dir / f"trader_daily_{model_kind}_h{input_horizon}.pt"

    print(f"model_kind={model_kind}")
    print(f"input_horizon={input_horizon}")
    print(f"sequences: train={n_train} test={n_test} (total {n_total}, time-based per asset)")
    for epoch in range(1, epochs + 1):
        train_stats = train_one_epoch(
            model,
            loss_fn,
            train_loader,
            optimizer,
            device=device,
            initial_position=initial_position,
            grad_clip=grad_clip,
        )
        test_stats = evaluate_policy(
            model,
            loss_fn,
            test_loader,
            device=device,
            initial_position=initial_position,
        )
        print(
            f"epoch {epoch:03d}  "
            f"train_loss={train_stats['loss']:.5f}  "
            f"test_loss={test_stats['loss']:.5f}  "
            f"test_meanret={test_stats['mean_return']:.5f}  "
            f"test_turnover={test_stats['turnover']:.5f}"
        )
        if test_stats["loss"] < best_test:
            best_test = test_stats["loss"]
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "controller_config": controller_cfg.__dict__,
                    "loss_config": loss_cfg.__dict__,
                    "model_kind": model_kind,
                    "input_horizon": input_horizon,
                    "seq_len": seq_len,
                    "dataset_stride": dataset_stride,
                    "test_stats": test_stats,
                },
                best_path,
            )
            print(f"  -> saved best to {best_path}")

    print(f"best test_loss: {best_test:.5f}")


if __name__ == "__main__":
    main()
