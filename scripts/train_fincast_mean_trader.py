"""Train an encoder trader using only FinCast mean forecasts.

This is the high-frequency FinCast path: Optiver WAP1 contexts are first passed
through frozen FinCast, then this script trains the decision model on the
mean-only forecast path `[H, 1]`. Quantile channels are intentionally absent.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.baselines import RandomPositionBaselineConfig
from src.datasets.trader_dataset import (
    CachedDistributionDataset,
    time_ordered_train_validation_test_indices,
)
from src.eval.backtest import (
    BacktestResult,
    run_constant_position_backtest,
    run_policy_backtest,
    run_random_position_backtest,
)
from src.eval.metrics import BacktestMetrics, BacktestMetricsConfig
from src.fincast_io.forecast_features import forecast_to_return_patch
from src.trader.encoder_transformer import EncoderTransformerPolicy, EncoderTransformerPolicyConfig
from src.training.losses import MeanVarianceTurnoverLoss
from src.training.trainer import evaluate_policy, train_one_epoch
from src.utils.config import PositionLossConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a mean-only FinCast encoder trader.")
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "optiver_8stocks_fincast_mean_smoke_8192.npz"),
    )
    parser.add_argument(
        "--ckpt-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "checkpoints" / "optiver_fincast_mean"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(WORKSPACE_ROOT / "outputs" / "backtests" / "optiver_fincast_mean_backtest.csv"),
    )
    parser.add_argument("--input-horizon", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=8)
    parser.add_argument("--dataset-stride", type=int, default=4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--model-dim", type=int, default=64)
    parser.add_argument("--state-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--ff-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-trade", type=float, default=0.05)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--random-samples", type=int, default=20)
    parser.add_argument("--random-seed", type=int, default=8100)
    parser.add_argument("--random-max-trade", type=float, default=0.05)
    parser.add_argument("--lambda-variance", type=float, default=1.0)
    parser.add_argument("--lambda-turnover", type=float, default=0.0)
    parser.add_argument("--lambda-forecast-risk", type=float, default=0.0)
    parser.add_argument("--selection-drawdown-weight", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = build_data(
        cache_path=Path(args.cache),
        input_horizon=args.input_horizon,
        seq_len=args.seq_len,
        dataset_stride=args.dataset_stride,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        batch_size=args.batch_size,
    )
    controller_cfg = EncoderTransformerPolicyConfig(
        horizon_len=args.input_horizon,
        forecast_channels=int(data["forecast_channels"]),
        model_dim=args.model_dim,
        state_dim=args.state_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
        max_trade=args.max_trade,
    )
    loss_cfg = PositionLossConfig(
        lambda_variance=args.lambda_variance,
        lambda_turnover=args.lambda_turnover,
        lambda_forecast_risk=args.lambda_forecast_risk,
    )
    model = EncoderTransformerPolicy(controller_cfg).to(device)
    loss_fn = MeanVarianceTurnoverLoss(loss_cfg).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    metrics_config = BacktestMetricsConfig(
        transaction_cost_bps=args.transaction_cost_bps,
        initial_position=0.0,
    )

    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "trader_fincast_mean_encoder_transformer.pt"
    best_score = float("-inf")

    print("FinCast mean-only encoder training")
    print("----------------------------------")
    print(f"device:                  {device}")
    print(f"cache:                   {Path(args.cache)}")
    print(f"patches:                 {tuple(data['patches'].shape)}")
    print(f"forecast_channels:       {data['forecast_channels']}")
    print(f"seq_len / stride:        {args.seq_len} / {args.dataset_stride}")
    print(
        "train/val/test sequences: "
        f"{len(data['train_indices'])} / {len(data['validation_indices'])} / {len(data['test_indices'])}"
    )
    print(
        f"loss lambdas: variance={args.lambda_variance:g}, "
        f"turnover={args.lambda_turnover:g}, forecast_risk={args.lambda_forecast_risk:g}"
    )

    for epoch in range(1, args.epochs + 1):
        train_stats = train_one_epoch(
            model,
            loss_fn,
            data["train_loader"],
            optimizer,
            device=device,
            initial_position=0.0,
        )
        validation_stats = evaluate_policy(
            model,
            loss_fn,
            data["validation_loader"],
            device=device,
            initial_position=0.0,
        )
        validation_result = run_policy_backtest(
            name="validation",
            model=model,
            loader=data["validation_loader"],
            device=device,
            metrics_config=metrics_config,
        )
        validation_metrics = validation_result.metrics
        score = _selection_score(
            validation_stats=validation_stats,
            validation_metrics=validation_metrics,
            drawdown_weight=args.selection_drawdown_weight,
        )
        print(
            f"epoch {epoch:03d}  "
            f"train_loss={train_stats['loss']:.6g}  "
            f"val_loss={validation_stats['loss']:.6g}  "
            f"val_total={validation_metrics.cumulative_return:.6g}  "
            f"val_sharpe={validation_metrics.sharpe_like:.6g}  "
            f"val_mdd={validation_metrics.max_drawdown:.6g}  "
            f"select={score:.6g}"
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "epoch": epoch,
                    "model_kind": "fincast_mean_encoder_transformer",
                    "model": model.state_dict(),
                    "best_state": model.state_dict(),
                    "controller_config": controller_cfg.__dict__,
                    "loss_config": loss_cfg.__dict__,
                    "input_horizon": args.input_horizon,
                    "seq_len": args.seq_len,
                    "dataset_stride": args.dataset_stride,
                    "validation_fraction": args.validation_fraction,
                    "test_fraction": args.test_fraction,
                    "selection_score": score,
                    "validation_stats": validation_stats,
                    "validation_metrics": validation_metrics.to_dict(),
                    "cache": str(Path(args.cache)),
                    "forecast_channels": ["mean"],
                },
                best_path,
            )
            print(f"  -> saved best to {best_path}")

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["best_state"])
    results = run_backtests(
        model=model,
        loader=data["test_loader"],
        device=device,
        metrics_config=metrics_config,
        random_samples=args.random_samples,
        random_seed=args.random_seed,
        random_max_trade=args.random_max_trade,
    )
    rows = result_rows(results)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print()
    print_result_table(rows)
    summarize_policy_vs_random(rows)
    print(f"\ncheckpoint saved -> {best_path}")
    print(f"backtest saved   -> {output_csv}")


def build_data(
    *,
    cache_path: Path,
    input_horizon: int,
    seq_len: int,
    dataset_stride: int,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
) -> dict[str, Any]:
    cache = np.load(cache_path, allow_pickle=True)
    if "mean_outputs" not in cache and "full_outputs" not in cache:
        raise KeyError("Mean-only FinCast cache must contain mean_outputs or full_outputs.")
    if "mean_outputs" in cache:
        full_outputs = cache["mean_outputs"][:, :, None].astype(np.float32)
    else:
        full_outputs = cache["full_outputs"].astype(np.float32)
        if full_outputs.shape[-1] != 1:
            raise ValueError(f"Expected mean-only forecast channel, got {full_outputs.shape[-1]} channels.")
    if not 1 <= input_horizon <= full_outputs.shape[1]:
        raise ValueError(f"input_horizon must be in [1, {full_outputs.shape[1]}].")

    last_values = torch.as_tensor(cache["last_values"], dtype=torch.float32)
    realized_returns = torch.as_tensor(cache["realized_returns"], dtype=torch.float32)
    return_patches = forecast_to_return_patch(full_outputs[:, :input_horizon, :], last_values)
    asset_names = cache["asset_names"] if "asset_names" in cache else np.asarray([])
    episode_ids = cache["episode_ids"] if "episode_ids" in cache else np.asarray([])
    dataset = CachedDistributionDataset(
        patches=return_patches,
        realized_returns=realized_returns,
        seq_len=seq_len,
        stride=dataset_stride,
        asset_names=asset_names if asset_names.size else None,
        episode_ids=episode_ids if episode_ids.size else None,
    )
    train_indices, validation_indices, test_indices = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(Subset(dataset, validation_indices), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)
    return {
        "patches": return_patches,
        "forecast_channels": int(return_patches.shape[-1]),
        "dataset": dataset,
        "train_indices": train_indices,
        "validation_indices": validation_indices,
        "test_indices": test_indices,
        "train_loader": train_loader,
        "validation_loader": validation_loader,
        "test_loader": test_loader,
    }


def run_backtests(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    metrics_config: BacktestMetricsConfig,
    random_samples: int,
    random_seed: int,
    random_max_trade: float | None,
) -> list[BacktestResult]:
    results = [
        run_constant_position_backtest(
            name="cash",
            loader=loader,
            position=0.0,
            metrics_config=metrics_config,
        ),
        run_constant_position_backtest(
            name="buy_hold",
            loader=loader,
            position=1.0,
            metrics_config=metrics_config,
        ),
    ]
    for i in range(max(0, random_samples)):
        seed = int(random_seed) + i
        results.append(
            run_random_position_backtest(
                name=f"random_uniform_s{seed}",
                loader=loader,
                baseline_config=RandomPositionBaselineConfig(
                    seed=seed,
                    max_trade=random_max_trade,
                    round_step=0.01,
                ),
                metrics_config=metrics_config,
            )
        )
    results.append(
        run_policy_backtest(
            name="policy:fincast_mean_encoder",
            model=model,
            loader=loader,
            device=device,
            metrics_config=metrics_config,
        )
    )
    return results


def _selection_score(
    *,
    validation_stats: dict[str, float],
    validation_metrics: BacktestMetrics,
    drawdown_weight: float,
) -> float:
    del validation_stats
    return float(validation_metrics.sharpe_like) - float(drawdown_weight) * float(validation_metrics.max_drawdown)


def result_rows(results: list[BacktestResult]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        row: dict[str, float | int | str] = {"strategy": result.name}
        row.update(result.metrics.to_dict())
        row["position_min"] = float(result.positions.min().item())
        row["position_max"] = float(result.positions.max().item())
        row["return_min"] = float(result.returns.net_returns.min().item())
        row["return_max"] = float(result.returns.net_returns.max().item())
        rows.append(row)
    return rows


def print_result_table(rows: list[dict[str, float | int | str]]) -> None:
    columns = [
        "strategy",
        "n_steps",
        "mean_return",
        "volatility",
        "sharpe_like",
        "cumulative_return",
        "max_drawdown",
        "average_position",
        "average_turnover",
    ]
    widths = {col: max(len(col), *(len(format_value(row[col])) for row in rows)) for col in columns}
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(format_value(row[col]).ljust(widths[col]) for col in columns))


def summarize_policy_vs_random(rows: list[dict[str, float | int | str]]) -> None:
    df = pd.DataFrame(rows)
    policy = df[df["strategy"] == "policy:fincast_mean_encoder"]
    random_rows = df[df["strategy"].str.startswith("random_uniform")]
    if policy.empty or random_rows.empty:
        return
    p = policy.iloc[0]
    print()
    print(
        "policy beats random: "
        f"cumulative={int((p['cumulative_return'] > random_rows['cumulative_return']).sum())}/{len(random_rows)}, "
        f"sharpe={int((p['sharpe_like'] > random_rows['sharpe_like']).sum())}/{len(random_rows)}"
    )
    print(
        "random mean: "
        f"cum={random_rows['cumulative_return'].mean():.6g}, "
        f"sharpe={random_rows['sharpe_like'].mean():.6g}, "
        f"mdd={random_rows['max_drawdown'].mean():.6g}"
    )


def format_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def dataclass_from_dict(cls: type, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


if __name__ == "__main__":
    main()
