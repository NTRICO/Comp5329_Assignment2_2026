"""Smoke backtest scaffold for cached FinCast trader outputs.

This script is intentionally a wiring check, not a final research conclusion.
It evaluates independent non-overlapping test sequences with the same
time-based split used by the training notebook.
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

from src.baselines import (
    RandomPositionBaselineConfig,
    RollingArGarchBaselineConfig,
)
from src.datasets.trader_dataset import (
    CachedFeatureDataset,
    time_ordered_train_validation_test_indices,
)
from src.eval.backtest import (
    BacktestResult,
    run_constant_position_backtest,
    run_oracle_backtest,
    run_policy_backtest,
    run_random_position_backtest,
    run_rolling_ar_garch_backtest,
)
from src.eval.metrics import BacktestMetricsConfig
from src.trader.encoder_policy import (
    EncoderFeatureControllerConfig,
    EncoderFeatureOnlyPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a smoke backtest on cached FinCast trader data.")
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_fincast_encoder_daily_cache.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            WORKSPACE_ROOT
            / "outputs"
            / "checkpoints"
            / "risk_select_mt05"
            / "trader_daily_encoder_only_h5.pt"
        ),
    )
    parser.add_argument(
        "--model-kind",
        default=None,
        choices=["encoder_only"],
    )
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dataset-stride", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--initial-position", type=float, default=0.0)
    parser.add_argument("--random-samples", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--random-max-trade", type=float, default=None)
    parser.add_argument("--random-uncapped", action="store_true")
    parser.add_argument("--skip-random", action="store_true")
    parser.add_argument("--ar-garch-lookback", type=int, default=16)
    parser.add_argument("--ar-garch-risk-aversion", type=float, default=25.0)
    parser.add_argument("--ar-garch-max-trade", type=float, default=0.05)
    parser.add_argument("--skip-ar-garch", action="store_true")
    parser.add_argument("--skip-oracle", action="store_true")
    parser.add_argument("--skip-policy", action="store_true")
    parser.add_argument(
        "--output-csv",
        default=str(WORKSPACE_ROOT / "outputs" / "backtests" / "smoke_backtest_summary.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(args.checkpoint)
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_path.exists() else None

    seq_len = _coalesce(args.seq_len, checkpoint, "seq_len", 32)
    dataset_stride = _coalesce(args.dataset_stride, checkpoint, "dataset_stride", seq_len)
    validation_fraction = _coalesce(args.validation_fraction, checkpoint, "validation_fraction", 0.1)
    test_fraction = _coalesce(args.test_fraction, checkpoint, "test_fraction", 0.2)
    model_kind = args.model_kind or (checkpoint or {}).get("model_kind", "encoder_only")
    if model_kind != "encoder_only":
        raise ValueError(f"Only encoder_only is supported by the active backtest path, got {model_kind!r}.")
    random_max_trade = _resolve_random_max_trade(
        explicit_max_trade=args.random_max_trade,
        random_uncapped=args.random_uncapped,
        checkpoint=checkpoint,
    )

    loader, info = _build_test_loader(
        cache_path=Path(args.cache),
        seq_len=int(seq_len),
        dataset_stride=int(dataset_stride),
        validation_fraction=float(validation_fraction),
        test_fraction=float(test_fraction),
        batch_size=args.batch_size,
    )

    metrics_config = BacktestMetricsConfig(
        transaction_cost_bps=args.transaction_cost_bps,
        initial_position=args.initial_position,
    )
    ar_garch_config = RollingArGarchBaselineConfig(
        lookback=args.ar_garch_lookback,
        risk_aversion=args.ar_garch_risk_aversion,
        max_trade=args.ar_garch_max_trade,
        round_step=0.01,
    )

    results: list[BacktestResult] = [
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
    if not args.skip_random:
        for i in range(max(0, args.random_samples)):
            seed = int(args.random_seed) + i
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
    if not args.skip_ar_garch:
        results.append(
            run_rolling_ar_garch_backtest(
                name="rolling_ar1_garch",
                loader=loader,
                baseline_config=ar_garch_config,
                metrics_config=metrics_config,
            )
        )
    if not args.skip_oracle:
        results.extend(
            [
                run_oracle_backtest(
                    name="oracle_binary",
                    loader=loader,
                    metrics_config=metrics_config,
                    max_trade=None,
                    round_step=0.01,
                ),
                run_oracle_backtest(
                    name="oracle_trade_cap",
                    loader=loader,
                    metrics_config=metrics_config,
                    max_trade=0.25,
                    round_step=0.01,
                ),
            ]
        )

    policy_loaded = False
    if not args.skip_policy:
        if checkpoint is None:
            print(f"policy checkpoint not found, skipping policy smoke: {checkpoint_path}")
        else:
            model = _build_model_from_checkpoint(
                checkpoint=checkpoint,
                model_kind=str(model_kind),
                device=device,
            )
            results.append(
                run_policy_backtest(
                    name=f"policy:{model_kind}",
                    model=model,
                    loader=loader,
                    device=device,
                    metrics_config=metrics_config,
                )
            )
            policy_loaded = True

    rows = _result_rows(results)
    print("\nSmoke backtest scaffold")
    print("-----------------------")
    print("Interpretation: pooled non-overlapping test sequences; each sequence resets initial position.")
    print("Use this for wiring checks now, not for final live-trading conclusions.")
    print(f"device:                  {device}")
    print(f"cache:                   {Path(args.cache)}")
    print(f"checkpoint:              {checkpoint_path if policy_loaded else 'skipped'}")
    print(f"model_kind:              {model_kind}")
    print(f"seq_len / stride:        {seq_len} / {dataset_stride}")
    print(
        "train/val/test sequences: "
        f"{info['train_sequences']} / {info['validation_sequences']} / {info['test_sequences']}"
    )
    print(f"test realized shape:     {results[0].realized_returns.shape}")
    print(f"transaction_cost_bps:    {args.transaction_cost_bps}")
    print(f"random_max_trade:        {_format_optional_float(random_max_trade)}")
    print()
    _print_table(rows)

    if args.output_csv:
        output_path = Path(args.output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output_path, index=False)
        print(f"\nsummary saved -> {output_path}")


def _build_test_loader(
    *,
    cache_path: Path,
    seq_len: int,
    dataset_stride: int,
    validation_fraction: float,
    test_fraction: float,
    batch_size: int,
) -> tuple[DataLoader, dict[str, int]]:
    cache = np.load(cache_path, allow_pickle=True)
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
    train_indices, validation_indices, test_indices = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    test_set = Subset(dataset, test_indices)
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    return loader, {
        "train_sequences": len(train_indices),
        "validation_sequences": len(validation_indices),
        "test_sequences": len(test_indices),
        "total_sequences": len(dataset),
    }


def _build_model_from_checkpoint(
    *,
    checkpoint: dict[str, Any],
    model_kind: str,
    device: torch.device,
) -> torch.nn.Module:
    state = checkpoint.get("best_state") or checkpoint.get("model")
    if state is None:
        raise KeyError("Checkpoint must contain 'best_state' or 'model'.")

    config_dict = dict(checkpoint.get("controller_config", {}))
    if model_kind == "encoder_only":
        config = _dataclass_from_dict(EncoderFeatureControllerConfig, config_dict)
        model = EncoderFeatureOnlyPolicy(config)
    else:
        raise ValueError(f"Unsupported model_kind: {model_kind}")

    model.load_state_dict(state)
    return model.to(device)


def _load_checkpoint(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _dataclass_from_dict(cls: type, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


def _coalesce(value: Any, checkpoint: dict[str, Any] | None, key: str, default: Any) -> Any:
    if value is not None:
        return value
    if checkpoint is not None and key in checkpoint:
        return checkpoint[key]
    return default


def _resolve_random_max_trade(
    *,
    explicit_max_trade: float | None,
    random_uncapped: bool,
    checkpoint: dict[str, Any] | None,
) -> float | None:
    if random_uncapped:
        return None
    if explicit_max_trade is not None:
        return float(explicit_max_trade)
    if checkpoint is None:
        return None
    controller_config = checkpoint.get("controller_config", {})
    if not isinstance(controller_config, dict):
        return None
    max_trade = controller_config.get("max_trade")
    if max_trade is None:
        return None
    return float(max_trade)


def _result_rows(results: list[BacktestResult]) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        row = {"strategy": result.name}
        row.update(result.metrics.to_dict())
        row["position_min"] = float(result.positions.min().item())
        row["position_max"] = float(result.positions.max().item())
        row["return_min"] = float(result.returns.net_returns.min().item())
        row["return_max"] = float(result.returns.net_returns.max().item())
        rows.append(row)
    return rows


def _print_table(rows: list[dict[str, float | int | str]]) -> None:
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
    widths = {
        col: max(len(col), *(len(_format_value(row[col])) for row in rows))
        for col in columns
    }
    print("  ".join(col.ljust(widths[col]) for col in columns))
    print("  ".join("-" * widths[col] for col in columns))
    for row in rows:
        print("  ".join(_format_value(row[col]).ljust(widths[col]) for col in columns))


def _format_value(value: float | int | str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "uncapped"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
