"""Run trading sanity checks on the current FinCast trader checkpoint.

These checks are diagnostic only. They use the existing daily cache and current
checkpoint to separate objective mismatch, evaluation reset effects, execution
constraints, and raw FinCast signal strength.
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

from src.baselines import MeanVarianceBaselineConfig
from src.datasets.trader_dataset import CachedDistributionDataset, time_ordered_train_validation_test_indices
from src.eval.backtest import (
    BacktestResult,
    run_constant_position_backtest,
    run_markowitz_backtest,
    run_oracle_backtest,
    run_policy_backtest,
)
from src.eval.metrics import BacktestMetricsConfig, compute_strategy_returns, summarize_backtest
from src.fincast_io.cache_builder import load_distribution_cache
from src.fincast_io.forecast_features import forecast_to_return_patch
from src.trader.cnn_gru import PositionAwareGRUPolicy
from src.trader.encoder_transformer import EncoderTransformerPolicy, EncoderTransformerPolicyConfig
from src.training.losses import MeanVarianceTurnoverLoss
from src.training.trainer import evaluate_policy
from src.utils.config import PositionControllerConfig, PositionLossConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trader sanity checks.")
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_fincast_daily_cache.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(WORKSPACE_ROOT / "outputs" / "checkpoints" / "trader_daily_encoder_transformer_h5.pt"),
    )
    parser.add_argument("--model-kind", default=None, choices=["encoder_transformer", "cnn_gru", "vanilla_encoder"])
    parser.add_argument("--input-horizon", type=int, default=None)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dataset-stride", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "sanity_checks"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = _load_checkpoint(Path(args.checkpoint))
    input_horizon = int(_coalesce(args.input_horizon, checkpoint, "input_horizon", 5))
    seq_len = int(_coalesce(args.seq_len, checkpoint, "seq_len", 32))
    dataset_stride = int(_coalesce(args.dataset_stride, checkpoint, "dataset_stride", seq_len))
    validation_fraction = float(_coalesce(args.validation_fraction, checkpoint, "validation_fraction", 0.1))
    test_fraction = float(_coalesce(args.test_fraction, checkpoint, "test_fraction", 0.2))
    model_kind = str(args.model_kind or checkpoint.get("model_kind", "encoder_transformer"))

    data = _build_data(
        cache_path=Path(args.cache),
        input_horizon=input_horizon,
        seq_len=seq_len,
        dataset_stride=dataset_stride,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        batch_size=args.batch_size,
    )
    loss_cfg = _dataclass_from_dict(PositionLossConfig, dict(checkpoint.get("loss_config", {})))
    loss_fn = MeanVarianceTurnoverLoss(loss_cfg).to(device)
    model = _build_model_from_checkpoint(
        checkpoint=checkpoint,
        model_kind=model_kind,
        device=device,
    )
    model_trade_1 = _build_model_from_checkpoint(
        checkpoint=checkpoint,
        model_kind=model_kind,
        device=device,
        max_trade_override=1.0,
    )

    loss_rows = _constant_loss_rows(
        loss_fn=loss_fn,
        loader=data["test_loader"],
        device=device,
        model=model,
    )
    _save_and_print("1_constant_position_loss", loss_rows, output_dir)

    eval_rows = _evaluation_mode_rows(
        model=model,
        return_patches=data["return_patches"],
        realized_returns=data["realized_returns"],
        spans=data["test_spans"],
        loader=data["test_loader"],
        device=device,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    _save_and_print("2_window_reset_vs_continuous", eval_rows, output_dir)

    fairness_rows = _fairness_rows(
        model=model,
        model_trade_1=model_trade_1,
        loader=data["test_loader"],
        device=device,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    _save_and_print("3_initial_position_and_max_trade", fairness_rows, output_dir)

    signal_rows = _signal_ic_rows(
        return_patches_full=data["return_patches_full"],
        realized_returns=data["realized_returns"],
        test_sample_indices=data["test_sample_indices"],
        input_horizon=input_horizon,
    )
    _save_and_print("4_signal_ic", signal_rows, output_dir)

    financial_rows = _financial_metric_rows(
        model=model,
        loader=data["test_loader"],
        device=device,
        input_horizon=input_horizon,
        transaction_cost_bps=args.transaction_cost_bps,
    )
    _save_and_print("5_financial_metrics", financial_rows, output_dir)

    print("\nSanity check context")
    print("--------------------")
    print(f"device:               {device}")
    print(f"cache:                {Path(args.cache)}")
    print(f"checkpoint:           {Path(args.checkpoint)}")
    print(f"model_kind:           {model_kind}")
    print(f"input_horizon:        {input_horizon}")
    print(f"seq_len / stride:     {seq_len} / {dataset_stride}")
    print(
        "train/val/test sequences: "
        f"{len(data['train_indices'])} / {len(data['validation_indices'])} / {len(data['test_indices'])}"
    )
    print(f"test spans:           {len(data['test_spans'])} assets")
    print(f"saved directory:      {output_dir}")


def _build_data(
    *,
    cache_path: Path,
    input_horizon: int,
    seq_len: int,
    dataset_stride: int,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
) -> dict[str, Any]:
    cache = load_distribution_cache(cache_path)
    level_patches = torch.as_tensor(cache["full_outputs"], dtype=torch.float32)
    last_values = torch.as_tensor(cache["last_values"], dtype=torch.float32)
    realized_returns = torch.as_tensor(cache["realized_returns"], dtype=torch.float32)
    asset_names = cache["asset_names"]
    if not 1 <= input_horizon <= level_patches.shape[1]:
        raise ValueError(f"input_horizon must be in [1, {level_patches.shape[1]}].")

    return_patches_full = forecast_to_return_patch(level_patches, last_values)
    return_patches = return_patches_full[:, :input_horizon, :]
    dataset = CachedDistributionDataset(
        patches=return_patches,
        realized_returns=realized_returns,
        seq_len=seq_len,
        stride=dataset_stride,
        asset_names=asset_names if asset_names.size else None,
    )
    train_indices, validation_indices, test_indices = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size, shuffle=False)
    return {
        "asset_names": asset_names,
        "dataset": dataset,
        "realized_returns": realized_returns,
        "return_patches": return_patches,
        "return_patches_full": return_patches_full,
        "test_indices": test_indices,
        "test_loader": test_loader,
        "test_sample_indices": _test_sample_indices(dataset, test_indices),
        "test_spans": _test_asset_spans(dataset, test_indices),
        "train_indices": train_indices,
        "validation_indices": validation_indices,
    }


@torch.no_grad()
def _constant_loss_rows(
    *,
    loss_fn: MeanVarianceTurnoverLoss,
    loader: DataLoader,
    device: torch.device,
    model: torch.nn.Module,
) -> list[dict[str, float | str]]:
    rows = []
    for p in [0.0, 0.25, 0.5, 0.75, 1.0]:
        stats = _evaluate_constant_position_loss(loss_fn, loader, device, p=p, initial_position=0.0)
        rows.append({"strategy": f"constant_p={p:g}_init0", "p": p, **stats})
    policy_stats = evaluate_policy(
        model,
        loss_fn,
        loader,
        device=device,
        initial_position=0.0,
    )
    rows.append({"strategy": "policy_init0", "p": np.nan, **policy_stats})
    return rows


@torch.no_grad()
def _evaluate_constant_position_loss(
    loss_fn: MeanVarianceTurnoverLoss,
    loader: DataLoader,
    device: torch.device,
    *,
    p: float,
    initial_position: float,
) -> dict[str, float]:
    totals = {"loss": 0.0, "mean_return": 0.0, "variance": 0.0, "turnover": 0.0}
    count = 0
    for batch in loader:
        rets = batch["realized_returns"].to(device)
        positions = torch.full_like(rets, float(p))
        deltas = torch.zeros_like(rets)
        deltas[:, 0] = float(p) - float(initial_position)
        breakdown = loss_fn(positions, rets, deltas=deltas)
        totals["loss"] += float(breakdown.loss.cpu())
        totals["mean_return"] += float(breakdown.mean_return.cpu())
        totals["variance"] += float(breakdown.variance.cpu())
        totals["turnover"] += float(breakdown.turnover.cpu())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def _evaluation_mode_rows(
    *,
    model: torch.nn.Module,
    return_patches: torch.Tensor,
    realized_returns: torch.Tensor,
    spans: list[tuple[str, int, int]],
    loader: DataLoader,
    device: torch.device,
    transaction_cost_bps: float,
) -> list[dict[str, float | int | str]]:
    reset_config = BacktestMetricsConfig(
        transaction_cost_bps=transaction_cost_bps,
        initial_position=0.0,
    )
    reset = run_policy_backtest(
        name="policy_window_reset_init0",
        model=model,
        loader=loader,
        device=device,
        metrics_config=reset_config,
    )
    continuous = _run_continuous_policy(
        name="policy_continuous_init0",
        model=model,
        return_patches=return_patches,
        realized_returns=realized_returns,
        spans=spans,
        device=device,
        metrics_config=reset_config,
    )
    return [_result_row(reset), _result_row(continuous)]


@torch.no_grad()
def _fairness_rows(
    *,
    model: torch.nn.Module,
    model_trade_1: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    transaction_cost_bps: float,
) -> list[dict[str, float | int | str]]:
    cfg_init0 = BacktestMetricsConfig(transaction_cost_bps=transaction_cost_bps, initial_position=0.0)
    cfg_init1 = BacktestMetricsConfig(transaction_cost_bps=transaction_cost_bps, initial_position=1.0)
    rows = [
        _result_row(
            run_constant_position_backtest(
                name="buy_hold_init0",
                loader=loader,
                position=1.0,
                metrics_config=cfg_init0,
            )
        ),
        _result_row(
            run_constant_position_backtest(
                name="buy_hold_init1",
                loader=loader,
                position=1.0,
                metrics_config=cfg_init1,
            )
        ),
        _result_row(
            run_policy_backtest(
                name="policy_init0_max_trade0.25",
                model=model,
                loader=loader,
                device=device,
                metrics_config=cfg_init0,
            )
        ),
        _result_row(
            run_policy_backtest(
                name="policy_init1_max_trade0.25",
                model=model,
                loader=loader,
                device=device,
                metrics_config=cfg_init1,
            )
        ),
        _result_row(
            run_policy_backtest(
                name="policy_init0_max_trade1.0",
                model=model_trade_1,
                loader=loader,
                device=device,
                metrics_config=cfg_init0,
            )
        ),
    ]
    return rows


def _signal_ic_rows(
    *,
    return_patches_full: torch.Tensor,
    realized_returns: torch.Tensor,
    test_sample_indices: np.ndarray,
    input_horizon: int,
) -> list[dict[str, float | str]]:
    patches = return_patches_full.detach().cpu().numpy()
    target = realized_returns.detach().cpu().numpy()
    test_idx = np.unique(test_sample_indices)

    signals = {
        "mean_h1": patches[:, 0, 0],
        "median_h1": patches[:, 0, 5],
        f"mean_avg_h{input_horizon}": patches[:, :input_horizon, 0].mean(axis=1),
        f"median_avg_h{input_horizon}": patches[:, :input_horizon, 5].mean(axis=1),
    }
    rows: list[dict[str, float | str]] = []
    for split_name, idx in [("all_cache", np.arange(len(target))), ("test_steps", test_idx)]:
        for signal_name, signal in signals.items():
            rows.append(
                {
                    "split": split_name,
                    "signal": signal_name,
                    **_signal_stats(signal[idx], target[idx]),
                }
            )
    return rows


def _signal_stats(signal: np.ndarray, target: np.ndarray) -> dict[str, float]:
    signal = np.asarray(signal, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    mask = np.isfinite(signal) & np.isfinite(target)
    signal = signal[mask]
    target = target[mask]
    if len(signal) == 0 or np.std(signal) == 0 or np.std(target) == 0:
        ic = np.nan
    else:
        ic = float(np.corrcoef(signal, target)[0, 1])
    return {
        "n": int(len(signal)),
        "ic": ic,
        "direction_acc": float(((signal > 0) == (target > 0)).mean()),
        "signal_positive_rate": float((signal > 0).mean()),
        "target_positive_rate": float((target > 0).mean()),
        "signal_mean": float(signal.mean()),
        "target_mean": float(target.mean()),
    }


@torch.no_grad()
def _financial_metric_rows(
    *,
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    input_horizon: int,
    transaction_cost_bps: float,
) -> list[dict[str, float | int | str]]:
    cfg = BacktestMetricsConfig(transaction_cost_bps=transaction_cost_bps, initial_position=0.0)
    markowitz_config = MeanVarianceBaselineConfig(
        horizon=input_horizon,
        risk_aversion=25.0,
        max_trade=0.25,
        round_step=0.01,
    )
    results = [
        run_constant_position_backtest(name="cash", loader=loader, position=0.0, metrics_config=cfg),
        run_constant_position_backtest(name="buy_hold", loader=loader, position=1.0, metrics_config=cfg),
        run_markowitz_backtest(
            name="markowitz",
            loader=loader,
            baseline_config=markowitz_config,
            metrics_config=cfg,
        ),
        run_oracle_backtest(name="oracle_binary", loader=loader, metrics_config=cfg, max_trade=None),
        run_oracle_backtest(name="oracle_trade_cap", loader=loader, metrics_config=cfg, max_trade=0.25),
        run_policy_backtest(
            name="policy",
            model=model,
            loader=loader,
            device=device,
            metrics_config=cfg,
        ),
    ]
    return [_result_row(result) for result in results]


@torch.no_grad()
def _run_continuous_policy(
    *,
    name: str,
    model: torch.nn.Module,
    return_patches: torch.Tensor,
    realized_returns: torch.Tensor,
    spans: list[tuple[str, int, int]],
    device: torch.device,
    metrics_config: BacktestMetricsConfig,
) -> BacktestResult:
    model.eval()
    positions: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    realized: list[torch.Tensor] = []
    for _, start, end in spans:
        patches = return_patches[start:end].unsqueeze(0).to(device)
        rollout = model(patches, initial_position=metrics_config.initial_position)
        positions.append(rollout.positions.squeeze(0).detach().cpu())
        deltas.append(rollout.deltas.squeeze(0).detach().cpu())
        realized.append(realized_returns[start:end].detach().cpu())
    all_positions = torch.cat(positions, dim=0)
    all_deltas = torch.cat(deltas, dim=0)
    all_realized = torch.cat(realized, dim=0)
    returns = compute_strategy_returns(
        all_positions,
        all_realized,
        deltas=all_deltas,
        config=metrics_config,
    )
    metrics = summarize_backtest(
        all_positions,
        all_realized,
        deltas=all_deltas,
        config=metrics_config,
    )
    return BacktestResult(
        name=name,
        positions=all_positions,
        realized_returns=all_realized,
        deltas=all_deltas,
        metrics=metrics,
        returns=returns,
        metadata={"source": "policy_continuous", "asset_count": len(spans)},
    )


def _test_sample_indices(dataset: CachedDistributionDataset, test_indices: list[int]) -> np.ndarray:
    sample_indices: list[int] = []
    for dataset_index in test_indices:
        start = dataset.starts[dataset_index]
        sample_indices.extend(range(start, start + dataset.seq_len))
    return np.asarray(sample_indices, dtype=np.int64)


def _test_asset_spans(
    dataset: CachedDistributionDataset,
    test_indices: list[int],
) -> list[tuple[str, int, int]]:
    by_asset: dict[str, list[int]] = {}
    for dataset_index in test_indices:
        start = dataset.starts[dataset_index]
        asset = "__all__" if dataset.asset_names is None else str(dataset.asset_names[start])
        by_asset.setdefault(asset, []).append(start)
    spans = []
    for asset, starts in sorted(by_asset.items()):
        spans.append((asset, min(starts), max(starts) + dataset.seq_len))
    return spans


def _result_row(result: BacktestResult) -> dict[str, float | int | str]:
    row: dict[str, float | int | str] = {"strategy": result.name}
    row.update(result.metrics.to_dict())
    row["position_min"] = float(result.positions.min().item())
    row["position_max"] = float(result.positions.max().item())
    row["return_min"] = float(result.returns.net_returns.min().item())
    row["return_max"] = float(result.returns.net_returns.max().item())
    return row


def _build_model_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model_kind: str,
    device: torch.device,
    max_trade_override: float | None = None,
) -> torch.nn.Module:
    state = checkpoint.get("best_state") or checkpoint.get("model")
    if state is None:
        raise KeyError("Checkpoint must contain 'best_state' or 'model'.")

    config_dict = dict(checkpoint.get("controller_config", {}))
    if max_trade_override is not None:
        config_dict["max_trade"] = float(max_trade_override)
    if model_kind in {"encoder_transformer", "vanilla_encoder"}:
        config = _dataclass_from_dict(EncoderTransformerPolicyConfig, config_dict)
        model = EncoderTransformerPolicy(config)
    elif model_kind == "cnn_gru":
        config = _dataclass_from_dict(PositionControllerConfig, config_dict)
        model = PositionAwareGRUPolicy(config)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")
    model.load_state_dict(state)
    return model.to(device)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _dataclass_from_dict(cls: type, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _coalesce(value: Any, checkpoint: dict[str, Any], key: str, default: Any) -> Any:
    if value is not None:
        return value
    if key in checkpoint:
        return checkpoint[key]
    return default


def _save_and_print(name: str, rows: list[dict[str, Any]], output_dir: Path) -> None:
    df = pd.DataFrame(rows)
    path = output_dir / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"\n{name}")
    print("-" * len(name))
    print(df.to_string(index=False, max_cols=12))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
