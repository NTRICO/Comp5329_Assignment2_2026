"""Train the encoder-only position policy on cached feature vectors.

Headless mirror of notebooks/02_trader_training.ipynb. Edit the constants at the
top of main() to tweak hyperparameters — there are no longer any YAML configs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.trader_dataset import (
    CachedFeatureDataset,
    time_ordered_train_validation_test_indices,
)
from src.eval.backtest import run_policy_backtest
from src.eval.metrics import BacktestMetrics, BacktestMetricsConfig
from src.trader.encoder_policy import (
    EncoderFeatureControllerConfig,
    EncoderFeatureOnlyPolicy,
)
from src.training.losses import MeanVarianceTurnoverLoss
from src.training.trainer import evaluate_policy, train_one_epoch
from src.utils.config import PositionLossConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the encoder-only feature trader.")
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_fincast_encoder_daily_cache.npz"),
    )
    parser.add_argument("--ckpt-dir", default=str(WORKSPACE_ROOT / "outputs" / "checkpoints"))
    parser.add_argument(
        "--model-kind",
        choices=["encoder_only"],
        default="encoder_only",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["loss", "sharpe_like", "risk_adjusted"],
        default="risk_adjusted",
    )
    parser.add_argument("--selection-drawdown-weight", type=float, default=0.5)
    parser.add_argument("--selection-turnover-weight", type=float, default=0.0)
    parser.add_argument("--selection-max-drawdown", type=float, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--encoder-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--max-trade", type=float, default=0.25)
    parser.add_argument("--lambda-variance", type=float, default=1.0)
    parser.add_argument("--lambda-turnover", type=float, default=0.001)
    parser.add_argument("--lambda-forecast-risk", type=float, default=0.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    seed = 42
    input_horizon = 5
    model_kind = args.model_kind
    loss_cfg = PositionLossConfig(
        lambda_variance=args.lambda_variance,
        lambda_turnover=args.lambda_turnover,
        lambda_forecast_risk=args.lambda_forecast_risk,
    )
    seq_len = 32
    dataset_stride = 32
    validation_fraction = args.validation_fraction
    test_fraction = args.test_fraction
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    grad_clip = 1.0
    initial_position = 0.0

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cache = np.load(args.cache, allow_pickle=True)
    features = torch.as_tensor(cache["encoder_features"], dtype=torch.float32)
    realized_returns = torch.as_tensor(cache["realized_returns"], dtype=torch.float32)
    asset_names = cache["asset_names"] if "asset_names" in cache else np.asarray([])
    episode_ids = cache["episode_ids"] if "episode_ids" in cache else np.asarray([])
    dataset = CachedFeatureDataset(
        features=features,
        realized_returns=realized_returns,
        seq_len=seq_len,
        stride=dataset_stride,
        asset_names=asset_names if asset_names.size else None,
        episode_ids=episode_ids if episode_ids.size else None,
    )
    controller_cfg = EncoderFeatureControllerConfig(
        feature_dim=int(features.shape[-1]),
        encoder_dim=args.encoder_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        max_trade=args.max_trade,
    )

    train_indices, validation_indices, test_indices = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    n_train = len(train_indices)
    n_validation = len(validation_indices)
    n_test = len(test_indices)
    n_total = len(dataset)
    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, validation_indices)
    test_set = Subset(dataset, test_indices)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    if model_kind == "encoder_only":
        model = EncoderFeatureOnlyPolicy(controller_cfg).to(device)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")
    loss_fn = MeanVarianceTurnoverLoss(loss_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    metrics_config = BacktestMetricsConfig(
        transaction_cost_bps=args.transaction_cost_bps,
        initial_position=initial_position,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_validation_loss = float("inf")
    best_selection_score = float("-inf")
    best_path = ckpt_dir / f"trader_daily_{model_kind}_h{input_horizon}.pt"

    print(f"model_kind={model_kind}")
    print(f"feature_dim={features.shape[-1]}")
    print(
        f"epochs={epochs} batch_size={batch_size} lr={learning_rate:g} "
        f"dropout={args.dropout:g} encoder_dim={args.encoder_dim} hidden_dim={args.hidden_dim}"
    )
    print(
        f"selection={args.selection_metric} drawdown_weight={args.selection_drawdown_weight:g} "
        f"turnover_weight={args.selection_turnover_weight:g} transaction_cost_bps={args.transaction_cost_bps:g}"
    )
    print(
        f"loss_lambda_variance={args.lambda_variance:g} "
        f"loss_lambda_turnover={args.lambda_turnover:g} "
        f"loss_lambda_forecast_risk={args.lambda_forecast_risk:g}"
    )
    print(
        f"sequences: train={n_train} validation={n_validation} test={n_test} "
        f"(total {n_total}, time-based per asset)"
    )
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
        validation_stats = evaluate_policy(
            model,
            loss_fn,
            validation_loader,
            device=device,
            initial_position=initial_position,
        )
        validation_result = run_policy_backtest(
            name="validation",
            model=model,
            loader=validation_loader,
            device=device,
            metrics_config=metrics_config,
        )
        validation_metrics = validation_result.metrics
        selection_score = _selection_score(
            metric=args.selection_metric,
            validation_stats=validation_stats,
            validation_metrics=validation_metrics,
            drawdown_weight=args.selection_drawdown_weight,
            turnover_weight=args.selection_turnover_weight,
            max_drawdown=args.selection_max_drawdown,
        )
        print(
            f"epoch {epoch:03d}  "
            f"train_loss={train_stats['loss']:.5f}  "
            f"val_loss={validation_stats['loss']:.5f}  "
            f"val_meanret={validation_stats['mean_return']:.5f}  "
            f"val_total={validation_metrics.cumulative_return:.5f}  "
            f"val_sharpe={validation_metrics.sharpe_like:.5f}  "
            f"val_mdd={validation_metrics.max_drawdown:.5f}  "
            f"select={selection_score:.5f}"
        )
        if validation_stats["loss"] < best_validation_loss:
            best_validation_loss = validation_stats["loss"]
        if selection_score > best_selection_score:
            best_selection_score = selection_score
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
                    "selection_metric": args.selection_metric,
                    "selection_score": selection_score,
                    "selection_drawdown_weight": args.selection_drawdown_weight,
                    "selection_turnover_weight": args.selection_turnover_weight,
                    "selection_max_drawdown": args.selection_max_drawdown,
                    "transaction_cost_bps": args.transaction_cost_bps,
                    "loss_config": {
                        "lambda_variance": args.lambda_variance,
                        "lambda_turnover": args.lambda_turnover,
                        "lambda_forecast_risk": args.lambda_forecast_risk,
                    },
                    "validation_fraction": validation_fraction,
                    "test_fraction": test_fraction,
                    "validation_stats": validation_stats,
                    "validation_metrics": validation_metrics.to_dict(),
                },
                best_path,
            )
            print(f"  -> saved best to {best_path}")

    best_checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_checkpoint["model"])
    test_stats = evaluate_policy(
        model,
        loss_fn,
        test_loader,
        device=device,
        initial_position=initial_position,
    )
    test_result = run_policy_backtest(
        name="test",
        model=model,
        loader=test_loader,
        device=device,
        metrics_config=metrics_config,
    )
    best_checkpoint["test_stats"] = test_stats
    best_checkpoint["test_metrics"] = test_result.metrics.to_dict()
    torch.save(best_checkpoint, best_path)
    print(f"best selection_score: {best_selection_score:.5f}")
    print(f"best validation_loss seen: {best_validation_loss:.5f}")
    print(
        f"final test_loss={test_stats['loss']:.5f}  "
        f"test_total={test_result.metrics.cumulative_return:.5f}  "
        f"test_sharpe={test_result.metrics.sharpe_like:.5f}  "
        f"test_mdd={test_result.metrics.max_drawdown:.5f}  "
        f"test_turnover={test_result.metrics.average_turnover:.5f}"
    )


def _selection_score(
    *,
    metric: str,
    validation_stats: dict[str, float],
    validation_metrics: BacktestMetrics,
    drawdown_weight: float,
    turnover_weight: float,
    max_drawdown: float | None,
) -> float:
    if metric == "loss":
        score = -float(validation_stats["loss"])
    elif metric == "sharpe_like":
        score = float(validation_metrics.sharpe_like)
    elif metric == "risk_adjusted":
        score = (
            float(validation_metrics.sharpe_like)
            - float(drawdown_weight) * float(validation_metrics.max_drawdown)
            - float(turnover_weight) * float(validation_metrics.average_turnover)
        )
    else:
        raise ValueError(f"Unsupported selection metric: {metric}")

    if max_drawdown is not None and validation_metrics.max_drawdown > max_drawdown:
        excess = float(validation_metrics.max_drawdown) - float(max_drawdown)
        score -= 1_000.0 + excess * 100.0
    return score


if __name__ == "__main__":
    main()
