"""Plot position paths for several strategies on the feature-cache test set."""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.baselines import RandomPositionBaselineConfig, RollingArGarchBaselineConfig
from src.datasets.trader_dataset import (
    CachedFeatureDataset,
    time_ordered_train_validation_test_indices,
)
from src.eval.backtest import (
    BacktestResult,
    run_constant_position_backtest,
    run_policy_backtest,
    run_random_position_backtest,
    run_rolling_ar_garch_backtest,
)
from src.eval.metrics import BacktestMetricsConfig
from src.trader.encoder_policy import EncoderFeatureControllerConfig, EncoderFeatureOnlyPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot strategy position paths over test time.")
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
    parser.add_argument("--asset", default=None, help="Asset to plot; defaults to SPY when available.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--dataset-stride", type=int, default=None)
    parser.add_argument("--validation-fraction", type=float, default=None)
    parser.add_argument("--test-fraction", type=float, default=None)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--initial-position", type=float, default=0.0)
    parser.add_argument("--random-seed", type=int, default=1234)
    parser.add_argument("--random-max-trade", type=float, default=None)
    parser.add_argument("--random-uncapped", action="store_true")
    parser.add_argument("--ar-garch-lookback", type=int, default=16)
    parser.add_argument("--ar-garch-risk-aversion", type=float, default=25.0)
    parser.add_argument("--ar-garch-max-trade", type=float, default=0.05)
    parser.add_argument("--max-points", type=int, default=0, help="Optional cap on plotted time points.")
    parser.add_argument(
        "--output-png",
        default=str(WORKSPACE_ROOT / "outputs" / "visualizations" / "strategy_positions" / "strategy_positions.png"),
    )
    parser.add_argument(
        "--output-csv",
        default=str(WORKSPACE_ROOT / "outputs" / "visualizations" / "strategy_positions" / "strategy_positions.csv"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_kind = checkpoint.get("model_kind", "encoder_only")
    if model_kind != "encoder_only":
        raise ValueError(f"Only encoder_only checkpoints are supported, got {model_kind!r}.")

    seq_len = int(_coalesce(args.seq_len, checkpoint, "seq_len", 32))
    dataset_stride = int(_coalesce(args.dataset_stride, checkpoint, "dataset_stride", seq_len))
    validation_fraction = float(_coalesce(args.validation_fraction, checkpoint, "validation_fraction", 0.1))
    test_fraction = float(_coalesce(args.test_fraction, checkpoint, "test_fraction", 0.2))

    dataset, metadata = _build_dataset(Path(args.cache), seq_len=seq_len, dataset_stride=dataset_stride)
    _, _, test_indices = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    loader = DataLoader(Subset(dataset, test_indices), batch_size=args.batch_size, shuffle=False)
    metrics_config = BacktestMetricsConfig(
        transaction_cost_bps=args.transaction_cost_bps,
        initial_position=args.initial_position,
    )

    random_max_trade = _resolve_random_max_trade(
        explicit_max_trade=args.random_max_trade,
        random_uncapped=args.random_uncapped,
        checkpoint=checkpoint,
    )
    model = _build_model_from_checkpoint(checkpoint=checkpoint, device=device)
    results = _run_position_backtests(
        loader=loader,
        model=model,
        device=device,
        metrics_config=metrics_config,
        random_seed=args.random_seed,
        random_max_trade=random_max_trade,
        ar_garch_lookback=args.ar_garch_lookback,
        ar_garch_risk_aversion=args.ar_garch_risk_aversion,
        ar_garch_max_trade=args.ar_garch_max_trade,
    )

    asset = _resolve_asset(args.asset, metadata["asset_names"], dataset, test_indices)
    trace = _build_position_trace(
        results=results,
        dataset=dataset,
        test_indices=test_indices,
        asset=asset,
        dates=metadata["dates"],
        max_points=max(0, int(args.max_points)),
    )
    if trace.empty:
        raise ValueError(f"No test-set rows found for asset {asset!r}.")

    output_png = Path(args.output_png)
    output_csv = Path(args.output_csv)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    trace.to_csv(output_csv, index=False)
    _plot_positions(trace, output_png, asset=asset)

    print("Strategy position visualization")
    print("--------------------------------")
    print(f"asset:           {asset}")
    print(f"rows plotted:    {len(trace)}")
    print(f"checkpoint:      {checkpoint_path}")
    print(f"cache:           {Path(args.cache)}")
    print(f"random_max_trade:{_format_optional_float(random_max_trade)}")
    print(f"figure saved:    {output_png}")
    print(f"trace saved:     {output_csv}")


def _build_dataset(
    cache_path: Path,
    *,
    seq_len: int,
    dataset_stride: int,
) -> tuple[CachedFeatureDataset, dict[str, np.ndarray]]:
    cache = np.load(cache_path, allow_pickle=True)
    features = torch.as_tensor(cache["encoder_features"], dtype=torch.float32)
    realized_returns = torch.as_tensor(cache["realized_returns"], dtype=torch.float32)
    asset_names = cache["asset_names"] if "asset_names" in cache else np.asarray([])
    dates = cache["dates"] if "dates" in cache else np.asarray([])
    episode_ids = cache["episode_ids"] if "episode_ids" in cache else np.asarray([])
    dataset = CachedFeatureDataset(
        features=features,
        realized_returns=realized_returns,
        seq_len=seq_len,
        stride=dataset_stride,
        asset_names=asset_names if asset_names.size else None,
        episode_ids=episode_ids if episode_ids.size else None,
    )
    return dataset, {"asset_names": asset_names, "dates": dates}


def _run_position_backtests(
    *,
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    metrics_config: BacktestMetricsConfig,
    random_seed: int,
    random_max_trade: float | None,
    ar_garch_lookback: int,
    ar_garch_risk_aversion: float,
    ar_garch_max_trade: float,
) -> list[BacktestResult]:
    return [
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
        run_random_position_backtest(
            name=f"random_s{random_seed}",
            loader=loader,
            baseline_config=RandomPositionBaselineConfig(
                seed=random_seed,
                max_trade=random_max_trade,
                round_step=0.01,
            ),
            metrics_config=metrics_config,
        ),
        run_rolling_ar_garch_backtest(
            name="rolling_ar1_garch",
            loader=loader,
            baseline_config=RollingArGarchBaselineConfig(
                lookback=ar_garch_lookback,
                risk_aversion=ar_garch_risk_aversion,
                max_trade=ar_garch_max_trade,
                round_step=0.01,
            ),
            metrics_config=metrics_config,
        ),
        run_policy_backtest(
            name="policy_encoder_only",
            model=model,
            loader=loader,
            device=device,
            metrics_config=metrics_config,
        ),
    ]


def _resolve_asset(
    requested_asset: str | None,
    asset_names: np.ndarray,
    dataset: CachedFeatureDataset,
    test_indices: list[int],
) -> str:
    if requested_asset:
        return requested_asset
    if asset_names.size:
        test_assets = [str(asset_names[dataset.starts[index]]) for index in test_indices]
        if "SPY" in test_assets:
            return "SPY"
        return test_assets[0]
    return "__all__"


def _build_position_trace(
    *,
    results: list[BacktestResult],
    dataset: CachedFeatureDataset,
    test_indices: list[int],
    asset: str,
    dates: np.ndarray,
    max_points: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    seq_rank_for_asset = 0
    for result_row, dataset_index in enumerate(test_indices):
        start = dataset.starts[dataset_index]
        row_asset = "__all__" if dataset.asset_names is None else str(dataset.asset_names[start])
        if row_asset != asset:
            continue
        for t in range(dataset.seq_len):
            global_index = start + t
            row: dict[str, Any] = {
                "time_index": len(rows),
                "date": _metadata_at(dates, global_index),
                "asset": row_asset,
                "sequence_rank": seq_rank_for_asset,
                "sequence_t": t,
                "global_index": global_index,
                "realized_return": float(dataset.realized_returns[global_index].item()),
            }
            for result in results:
                row[result.name] = float(result.positions[result_row, t].item())
            rows.append(row)
            if max_points and len(rows) >= max_points:
                return pd.DataFrame(rows)
        seq_rank_for_asset += 1
    return pd.DataFrame(rows)


def _plot_positions(trace: pd.DataFrame, output_png: Path, *, asset: str) -> None:
    strategy_columns = [
        col
        for col in trace.columns
        if col not in {"time_index", "date", "asset", "sequence_rank", "sequence_t", "global_index", "realized_return"}
    ]
    x = pd.to_datetime(trace["date"], errors="coerce")
    use_dates = not x.isna().all()
    if not use_dates:
        x = trace["time_index"]

    fig, ax = plt.subplots(figsize=(14, 6.5), constrained_layout=True)
    styles = {
        "cash": {"color": "#555555", "linestyle": "--", "linewidth": 1.4},
        "buy_hold": {"color": "#111111", "linestyle": "--", "linewidth": 1.4},
        "random_s1234": {"color": "#9c6ade", "linestyle": "-", "linewidth": 1.4, "alpha": 0.85},
        "rolling_ar1_garch": {"color": "#2f80ed", "linestyle": "-", "linewidth": 1.8},
        "policy_encoder_only": {"color": "#d04f3a", "linestyle": "-", "linewidth": 2.2},
    }
    for col in strategy_columns:
        kwargs = styles.get(col, {"linewidth": 1.6})
        ax.step(x, trace[col], where="post", label=col, **kwargs)

    ax.set_title(f"Strategy Position Over Test Time ({asset})")
    ax.set_xlabel("time")
    ax.set_ylabel("position")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper left", ncols=2)
    if use_dates:
        locator = mdates.AutoDateLocator(minticks=4, maxticks=9)
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
    fig.savefig(output_png, dpi=180)
    plt.close(fig)


def _build_model_from_checkpoint(*, checkpoint: dict[str, Any], device: torch.device) -> torch.nn.Module:
    state = checkpoint.get("best_state") or checkpoint.get("model")
    if state is None:
        raise KeyError("Checkpoint must contain 'best_state' or 'model'.")
    config = _dataclass_from_dict(
        EncoderFeatureControllerConfig,
        dict(checkpoint.get("controller_config", {})),
    )
    model = EncoderFeatureOnlyPolicy(config)
    model.load_state_dict(state)
    return model.to(device)


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


def _metadata_at(values: np.ndarray, index: int) -> str:
    if values.size == 0:
        return str(index)
    return str(values[index])


def _format_optional_float(value: float | None) -> str:
    if value is None:
        return "uncapped"
    return f"{value:.6g}"


if __name__ == "__main__":
    main()
