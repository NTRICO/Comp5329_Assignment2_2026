"""Visualize what a feature-cache trader policy receives and emits.

The feature policies consume one pooled FinCast encoder vector per decision
step plus the previous position. This script plots a single test sequence:

- compressed FinCast encoder feature heatmap, shaped [seq_len, feature_bins]
- feature summary statistics over time
- previous position, output position, and position delta
- realized returns, strategy returns, and equity curve
"""
from __future__ import annotations

import argparse
from dataclasses import fields
from pathlib import Path
import sys
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.trader_dataset import (
    CachedFeatureDataset,
    time_ordered_train_validation_test_indices,
)
from src.eval.metrics import BacktestMetricsConfig, compute_strategy_returns
from src.trader.encoder_policy import (
    EncoderFeatureControllerConfig,
    EncoderFeatureOnlyPolicy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize model inputs and outputs for one test sequence.")
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_fincast_encoder_daily_cache.npz"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(WORKSPACE_ROOT / "outputs" / "checkpoints" / "risk_select_mt05" / "trader_daily_encoder_only_h5.pt"),
    )
    parser.add_argument("--sequence-rank", type=int, default=0)
    parser.add_argument("--feature-bins", type=int, default=128)
    parser.add_argument("--transaction-cost-bps", type=float, default=1.0)
    parser.add_argument("--initial-position", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "visualizations" / "model_io"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_kind = checkpoint.get("model_kind", "encoder_only")
    if model_kind != "encoder_only":
        raise ValueError(f"Only encoder_only is supported by the active visualization path, got {model_kind!r}.")

    cache = np.load(args.cache, allow_pickle=True)
    features = torch.as_tensor(cache["encoder_features"], dtype=torch.float32)
    realized_returns = torch.as_tensor(cache["realized_returns"], dtype=torch.float32)
    asset_names = cache["asset_names"] if "asset_names" in cache else np.asarray([])
    dates = cache["dates"] if "dates" in cache else np.asarray([])
    episode_ids = cache["episode_ids"] if "episode_ids" in cache else np.asarray([])

    seq_len = int(checkpoint.get("seq_len", 32))
    dataset_stride = int(checkpoint.get("dataset_stride", seq_len))
    validation_fraction = float(checkpoint.get("validation_fraction", 0.1))
    test_fraction = float(checkpoint.get("test_fraction", 0.2))
    dataset = CachedFeatureDataset(
        features=features,
        realized_returns=realized_returns,
        seq_len=seq_len,
        stride=dataset_stride,
        asset_names=asset_names if asset_names.size else None,
        episode_ids=episode_ids if episode_ids.size else None,
    )
    _, _, test_indices = time_ordered_train_validation_test_indices(
        dataset,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
    )
    if not 0 <= args.sequence_rank < len(test_indices):
        raise ValueError(f"sequence-rank must be in [0, {len(test_indices) - 1}].")

    dataset_index = test_indices[args.sequence_rank]
    sequence_start = dataset.starts[dataset_index]
    item = dataset[dataset_index]
    feature_sequence = item["features"]
    returns = item["realized_returns"]
    sequence_dates = _slice_metadata(dates, sequence_start, seq_len)
    sequence_assets = _slice_metadata(asset_names, sequence_start, seq_len)
    asset_name = str(sequence_assets[0]) if len(sequence_assets) else "unknown"

    model = _build_model_from_checkpoint(checkpoint, model_kind)
    model.eval()
    trace = _trace_policy(
        model,
        feature_sequence,
        initial_position=float(args.initial_position),
    )

    metrics_config = BacktestMetricsConfig(
        transaction_cost_bps=float(args.transaction_cost_bps),
        initial_position=float(args.initial_position),
    )
    returns_breakdown = compute_strategy_returns(
        trace["positions"].unsqueeze(0),
        returns.unsqueeze(0),
        deltas=trace["deltas"].unsqueeze(0),
        config=metrics_config,
    )
    net_returns = returns_breakdown.net_returns.squeeze(0)
    equity = torch.cumprod(1.0 + net_returns, dim=0)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{model_kind}_seq{args.sequence_rank:03d}_{asset_name}"
    table_path = output_dir / f"{stem}_trace.csv"
    fig_path = output_dir / f"{stem}_io.png"
    feature_path = output_dir / f"{stem}_feature_snapshot.csv"

    _write_trace_table(
        path=table_path,
        sequence_start=sequence_start,
        dates=sequence_dates,
        assets=sequence_assets,
        features=feature_sequence,
        trace=trace,
        realized_returns=returns,
        net_returns=net_returns,
        equity=equity,
    )
    _write_feature_snapshot(feature_path, feature_sequence)
    _plot_model_io(
        path=fig_path,
        feature_sequence=feature_sequence,
        trace=trace,
        realized_returns=returns,
        net_returns=net_returns,
        equity=equity,
        model_kind=model_kind,
        asset_name=asset_name,
        sequence_rank=args.sequence_rank,
        feature_bins=args.feature_bins,
    )

    print("Model I/O visualization")
    print("-----------------------")
    print(f"model_kind:       {model_kind}")
    print(f"checkpoint:       {checkpoint_path}")
    print(f"cache:            {Path(args.cache)}")
    print(f"asset:            {asset_name}")
    print(f"dataset_index:    {dataset_index}")
    print(f"sequence_start:   {sequence_start}")
    print(f"feature shape:    {tuple(feature_sequence.shape)}")
    print(f"positions shape:  {tuple(trace['positions'].shape)}")
    print(f"encoded shape:    {tuple(trace['encoded'].shape)}")
    print(f"figure saved:     {fig_path}")
    print(f"trace saved:      {table_path}")
    print(f"snapshot saved:   {feature_path}")


def _build_model_from_checkpoint(checkpoint: dict[str, Any], model_kind: str) -> torch.nn.Module:
    config = _dataclass_from_dict(
        EncoderFeatureControllerConfig,
        dict(checkpoint.get("controller_config", {})),
    )
    if model_kind == "encoder_only":
        model = EncoderFeatureOnlyPolicy(config)
    else:
        raise ValueError(f"visualize_model_io supports encoder_only only, got {model_kind!r}.")

    state = checkpoint.get("best_state") or checkpoint.get("model")
    if state is None:
        raise KeyError("Checkpoint must contain 'best_state' or 'model'.")
    model.load_state_dict(state)
    return model


def _dataclass_from_dict(cls: type, values: dict[str, Any]):
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in allowed})


@torch.no_grad()
def _trace_policy(
    model: torch.nn.Module,
    features: torch.Tensor,
    *,
    initial_position: float,
) -> dict[str, torch.Tensor]:
    prev_position = torch.full((1,), float(initial_position), dtype=features.dtype)
    prev_positions = []
    positions = []
    deltas = []
    encoded = []

    for t in range(features.shape[0]):
        prev_positions.append(prev_position.squeeze(0))
        next_position, delta, z = model.step(features[t].unsqueeze(0), prev_position)
        positions.append(next_position.squeeze(0))
        deltas.append(delta.squeeze(0))
        encoded.append(z.squeeze(0))
        prev_position = next_position

    return {
        "prev_positions": torch.stack(prev_positions),
        "positions": torch.stack(positions),
        "deltas": torch.stack(deltas),
        "encoded": torch.stack(encoded),
    }


def _write_trace_table(
    *,
    path: Path,
    sequence_start: int,
    dates: np.ndarray,
    assets: np.ndarray,
    features: torch.Tensor,
    trace: dict[str, torch.Tensor],
    realized_returns: torch.Tensor,
    net_returns: torch.Tensor,
    equity: torch.Tensor,
) -> None:
    rows = []
    for t in range(features.shape[0]):
        feature_t = features[t]
        rows.append(
            {
                "t": t,
                "global_index": sequence_start + t,
                "date": _metadata_at(dates, t),
                "asset": _metadata_at(assets, t),
                "feature_dim": int(feature_t.numel()),
                "feature_mean": float(feature_t.mean().item()),
                "feature_std": float(feature_t.std(unbiased=False).item()),
                "feature_min": float(feature_t.min().item()),
                "feature_max": float(feature_t.max().item()),
                "feature_l2_norm": float(feature_t.norm().item()),
                "prev_position_input": float(trace["prev_positions"][t].item()),
                "output_position": float(trace["positions"][t].item()),
                "output_delta": float(trace["deltas"][t].item()),
                "encoded_l2_norm": float(trace["encoded"][t].norm().item()),
                "realized_return": float(realized_returns[t].item()),
                "strategy_net_return": float(net_returns[t].item()),
                "equity": float(equity[t].item()),
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_feature_snapshot(path: Path, features: torch.Tensor, *, top_k: int = 64) -> None:
    rows = []
    for t in range(features.shape[0]):
        feature_t = features[t]
        top_idx = torch.topk(feature_t.abs(), k=min(top_k, feature_t.numel())).indices
        for rank, dim in enumerate(top_idx.tolist(), start=1):
            rows.append(
                {
                    "t": t,
                    "rank_abs_value": rank,
                    "feature_dim_index": dim,
                    "feature_value": float(feature_t[dim].item()),
                    "abs_feature_value": float(feature_t[dim].abs().item()),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def _plot_model_io(
    *,
    path: Path,
    feature_sequence: torch.Tensor,
    trace: dict[str, torch.Tensor],
    realized_returns: torch.Tensor,
    net_returns: torch.Tensor,
    equity: torch.Tensor,
    model_kind: str,
    asset_name: str,
    sequence_rank: int,
    feature_bins: int,
) -> None:
    t = np.arange(feature_sequence.shape[0])
    heatmap = _compress_features(feature_sequence, feature_bins)
    feature_mean = feature_sequence.mean(dim=1).numpy()
    feature_std = feature_sequence.std(dim=1, unbiased=False).numpy()
    feature_norm = feature_sequence.norm(dim=1).numpy()

    fig, axes = plt.subplots(
        nrows=5,
        ncols=1,
        figsize=(14, 13),
        gridspec_kw={"height_ratios": [2.8, 1.3, 1.4, 1.5, 1.2]},
        constrained_layout=True,
    )

    im = axes[0].imshow(heatmap, aspect="auto", cmap="coolwarm", interpolation="nearest")
    axes[0].set_title(f"Input FinCast encoder features, compressed heatmap ({asset_name}, seq {sequence_rank})")
    axes[0].set_ylabel("time step")
    axes[0].set_xlabel("compressed feature bin")
    fig.colorbar(im, ax=axes[0], label="z-scored feature-bin mean")

    axes[1].plot(t, feature_mean, label="feature mean", linewidth=1.8)
    axes[1].plot(t, feature_std, label="feature std", linewidth=1.8)
    axes[1].plot(t, feature_norm / max(feature_norm.max(), 1e-8), label="feature L2 norm / max", linewidth=1.8)
    axes[1].set_title("Input vector summary per step")
    axes[1].set_ylabel("value")
    axes[1].legend(loc="upper left", ncols=3)
    axes[1].grid(alpha=0.25)

    axes[2].step(t, trace["prev_positions"].numpy(), where="post", label="previous position input", linewidth=1.8)
    axes[2].step(t, trace["positions"].numpy(), where="post", label="model output position", linewidth=2.2)
    axes[2].bar(t, trace["deltas"].numpy(), label="output delta", alpha=0.35)
    axes[2].set_title("Position input and model output")
    axes[2].set_ylabel("position")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].legend(loc="upper left", ncols=3)
    axes[2].grid(alpha=0.25)

    axes[3].bar(t, realized_returns.numpy(), label="realized return", alpha=0.45)
    axes[3].plot(t, net_returns.numpy(), label="strategy net return", linewidth=2.0)
    axes[3].axhline(0.0, color="black", linewidth=0.8)
    axes[3].set_title("Return seen after taking the position")
    axes[3].set_ylabel("return")
    axes[3].legend(loc="upper left", ncols=2)
    axes[3].grid(alpha=0.25)

    axes[4].plot(t, equity.numpy(), label="strategy equity", linewidth=2.0)
    axes[4].plot(t, trace["encoded"].norm(dim=1).numpy(), label="encoded vector norm", linewidth=1.6)
    axes[4].set_title(f"Output path and internal activation scale ({model_kind})")
    axes[4].set_xlabel("time step")
    axes[4].legend(loc="upper left", ncols=2)
    axes[4].grid(alpha=0.25)

    fig.savefig(path, dpi=180)
    plt.close(fig)


def _compress_features(features: torch.Tensor, bins: int) -> np.ndarray:
    x = features.detach().cpu()
    if bins <= 0:
        raise ValueError("feature-bins must be positive.")
    if bins > x.shape[1]:
        bins = x.shape[1]
    trim = (x.shape[1] // bins) * bins
    x = x[:, :trim].reshape(x.shape[0], bins, trim // bins).mean(dim=2)
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-8)
    return ((x - mean) / std).numpy()


def _slice_metadata(values: np.ndarray, start: int, length: int) -> np.ndarray:
    if values.size == 0:
        return np.asarray([])
    return values[start : start + length]


def _metadata_at(values: np.ndarray, index: int) -> str:
    if values.size == 0:
        return ""
    return str(values[index])


if __name__ == "__main__":
    main()
