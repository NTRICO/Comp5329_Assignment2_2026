from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_optiver_spectral_denoise_patchtst import (  # noqa: E402
    parse_asset_stock_id,
    select_device,
    set_seed,
)
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
    TARGET_HORIZON_STEPS,
    ScaleData,
    append_baseline_rows,
    append_model_rows,
    apply_training_regime,
    frame_to_markdown,
    load_raw_backbone_checkpoint,
    make_all_loaders,
    make_scale_specs,
    parse_stock_list,
    save_summary,
    train_model,
)
from src.baselines.scale_aware_asd_patchtst import (  # noqa: E402
    RawMultiScalePatchTST,
    ScaleAwareASDMultiScalePatchTST,
    ScaleSpec,
    build_multiscale_patchtst,
)


PRICE_LEVEL_CHANNELS = ("wap1", "wap2", "mid1", "mid2")
STATE_CHANNELS = (
    "rel_spread1",
    "rel_spread2",
    "imbalance1",
    "imbalance2",
    "log_total_size1",
    "log_total_size2",
    "total_imbalance",
    "updates_in_second",
    "is_observed_update",
    "seconds_since_update",
    "second_frac",
)
MULTICHANNEL_NAMES = tuple(f"{name}_log_return" for name in PRICE_LEVEL_CHANNELS) + STATE_CHANNELS


class SplitAccumulator:
    def __init__(self, cap: int, *, rng: np.random.Generator) -> None:
        self.cap = int(cap)
        self.rng = rng
        self.x: list[np.ndarray] = []
        self.y: list[float] = []
        self.last: list[float] = []
        self.seen = 0

    def add(self, x: np.ndarray, y: np.ndarray, last: np.ndarray) -> None:
        if len(y) == 0:
            return
        if self.cap <= 0:
            self.x.extend(np.asarray(x, dtype=np.float32))
            self.y.extend(float(v) for v in np.asarray(y, dtype=np.float32))
            self.last.extend(float(v) for v in np.asarray(last, dtype=np.float32))
            self.seen += int(len(y))
            return
        for row_x, row_y, row_last in zip(x, y, last):
            self.seen += 1
            if len(self.y) < self.cap:
                self.x.append(np.asarray(row_x, dtype=np.float32))
                self.y.append(float(row_y))
                self.last.append(float(row_last))
            else:
                index = int(self.rng.integers(0, self.seen))
                if index < self.cap:
                    self.x[index] = np.asarray(row_x, dtype=np.float32)
                    self.y[index] = float(row_y)
                    self.last[index] = float(row_last)

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        if not self.y:
            raise ValueError("No windows accumulated.")
        return (
            np.stack(self.x, axis=0).astype(np.float32),
            np.asarray(self.y, dtype=np.float32),
            np.asarray(self.last, dtype=np.float32),
            int(self.seen),
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run multi-channel PatchTST/LoRA-MoE intraday experiments.")
    parser.add_argument(
        "--cache",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "cache"
            / "position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "multichannel_patchtst_true_hour_60_30_10"),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "multichannel_patchtst_true_hour_60_30_10.md"),
    )
    parser.add_argument("--train-stocks", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--zero-shot-stock", type=int, default=9)
    parser.add_argument("--scales", nargs="+", choices=SCALE_ORDER, default=list(SCALE_ORDER))
    parser.add_argument("--patch-preset", choices=sorted(PATCH_PRESETS), default="balanced_60_30_10")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int, default=12)
    parser.add_argument("--small-train-cap", type=int, default=4096)
    parser.add_argument("--small-validation-cap", type=int, default=1024)
    parser.add_argument("--small-test-cap", type=int, default=1024)
    parser.add_argument("--small-zero-shot-cap", type=int, default=1024)
    parser.add_argument("--full-train-cap", type=int, default=0)
    parser.add_argument("--full-validation-cap", type=int, default=0)
    parser.add_argument("--full-test-cap", type=int, default=0)
    parser.add_argument("--full-zero-shot-cap", type=int, default=0)
    parser.add_argument("--lora-moe-rank", type=int, default=8)
    parser.add_argument("--lora-moe-alpha", type=float, default=16.0)
    parser.add_argument("--lora-moe-n-experts", type=int, default=4)
    parser.add_argument("--lora-moe-top-k", type=int, default=2)
    parser.add_argument("--lora-moe-dropout", type=float, default=0.1)
    parser.add_argument("--router-balance-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
    return parser.parse_args()


def small_caps(args: argparse.Namespace) -> dict[str, int]:
    return {
        "train": int(args.small_train_cap),
        "validation": int(args.small_validation_cap),
        "test": int(args.small_test_cap),
        "zero_shot": int(args.small_zero_shot_cap),
    }


def build_multichannel_scale_data(
    args: argparse.Namespace,
    *,
    cache_path: Path,
    caps: dict[str, int],
    scale_specs: dict[str, ScaleSpec],
) -> dict[str, ScaleData]:
    raw = build_multichannel_arrays(
        cache_path=cache_path,
        scales=list(args.scales),
        scale_specs=scale_specs,
        train_stocks=parse_stock_list(args.train_stocks),
        zero_shot_stock=int(args.zero_shot_stock),
        caps=caps,
        seed=int(args.seed),
    )
    out: dict[str, ScaleData] = {}
    for scale in args.scales:
        arrays = raw[scale]
        normalizer = fit_multichannel_normalizer(arrays["train_x"], arrays["train_y"])
        out[scale] = ScaleData(name=scale, spec=scale_specs[scale], arrays=arrays, normalizer=normalizer)
        print(
            f"{scale}: train={len(arrays['train_y'])} validation={len(arrays['validation_y'])} "
            f"test={len(arrays['test_y'])} zero_shot={len(arrays['zero_shot_y'])}",
            flush=True,
        )
    return out


def build_multichannel_arrays(
    *,
    cache_path: Path,
    scales: list[str],
    scale_specs: dict[str, ScaleSpec],
    train_stocks: list[int],
    zero_shot_stock: int,
    caps: dict[str, int],
    seed: int,
) -> dict[str, dict[str, Any]]:
    arrays = np.load(cache_path, allow_pickle=True)
    features = np.asarray(arrays["encoder_features"], dtype=np.float32)
    targets = np.asarray(arrays["realized_returns"], dtype=np.float32)
    asset_names = np.asarray(arrays["asset_names"]).astype(str)
    episode_ids = np.asarray(arrays["episode_ids"], dtype=np.int64)
    time_ids = np.asarray(arrays["time_ids"], dtype=np.int64)
    seconds = np.asarray(arrays["seconds_in_bucket"], dtype=np.int64)
    feature_names = [str(name) for name in arrays["feature_names"]]
    if "seconds_per_bucket" in arrays:
        seconds_per_bucket = int(np.asarray(arrays["seconds_per_bucket"]).reshape(-1)[0])
    else:
        seconds_per_bucket = int(seconds.max()) + 1 if len(seconds) else 600
    required = set(PRICE_LEVEL_CHANNELS + STATE_CHANNELS)
    missing_features = sorted(required - set(feature_names))
    if missing_features:
        raise ValueError(f"Cache is missing features: {missing_features}")

    stock_ids = np.asarray([parse_asset_stock_id(name) for name in asset_names], dtype=np.int64)
    needed = set(train_stocks + [int(zero_shot_stock)])
    missing_stocks = sorted(needed - set(stock_ids.tolist()))
    if missing_stocks:
        raise ValueError(f"Cache is missing stocks {missing_stocks}.")

    rng = np.random.default_rng(seed)
    accumulators: dict[str, dict[str, SplitAccumulator]] = {
        scale: {
            split: SplitAccumulator(int(caps.get(split, 0)), rng=rng)
            for split in ["train", "validation", "test", "zero_shot"]
        }
        for scale in scales
    }

    for stock_id in train_stocks + [int(zero_shot_stock)]:
        stock_mask = stock_ids == stock_id
        stock_episode_ids = np.unique(episode_ids[stock_mask])
        stock_episode_ids = stock_episode_ids[
            np.argsort([time_ids[stock_mask & (episode_ids == episode)].min() for episode in stock_episode_ids])
        ]
        if stock_id == int(zero_shot_stock):
            split_plan = {"zero_shot": stock_episode_ids}
        else:
            n = len(stock_episode_ids)
            n_train = max(1, int(math.floor(n * 0.8)))
            n_validation = max(1, int(math.floor(n * 0.1)))
            if n_train + n_validation >= n:
                n_validation = max(1, n - n_train - 1)
            split_plan = {
                "train": stock_episode_ids[:n_train],
                "validation": stock_episode_ids[n_train : n_train + n_validation],
                "test": stock_episode_ids[n_train + n_validation :],
            }

        for split_name, split_episodes in split_plan.items():
            if "hour" in scales:
                hour_rows = []
                for episode in split_episodes:
                    idx = np.flatnonzero(stock_mask & (episode_ids == episode))
                    idx = idx[np.argsort(seconds[idx])]
                    if len(idx):
                        hour_rows.append(features[idx[-1]])
                built = build_windows_from_feature_rows(
                    np.asarray(hour_rows, dtype=np.float32),
                    feature_names=feature_names,
                    context_length=scale_specs["hour"].context_length,
                    target_horizon_steps=TARGET_HORIZON_STEPS["hour"],
                )
                add_to_accumulator(accumulators["hour"][split_name], built)

            for episode in split_episodes:
                idx = np.flatnonzero(stock_mask & (episode_ids == episode))
                idx = idx[np.argsort(seconds[idx])]
                if not len(idx):
                    continue
                if "second" in scales:
                    matrix = make_feature_matrix(features[idx], feature_names)
                    future_returns = np.log1p(
                        np.clip(targets[idx].astype(np.float64), -0.999999, None)
                    ).astype(np.float32)
                    built = build_windows_from_matrix(
                        matrix,
                        future_returns,
                        context_length=scale_specs["second"].context_length,
                        target_horizon_steps=TARGET_HORIZON_STEPS["second"],
                    )
                    add_to_accumulator(accumulators["second"][split_name], built)
                if "minute" in scales:
                    minute_rows = aggregate_minute_feature_rows(
                        features[idx],
                        seconds[idx],
                        seconds_per_bucket=seconds_per_bucket,
                    )
                    built = build_windows_from_feature_rows(
                        minute_rows,
                        feature_names=feature_names,
                        context_length=scale_specs["minute"].context_length,
                        target_horizon_steps=TARGET_HORIZON_STEPS["minute"],
                    )
                    add_to_accumulator(accumulators["minute"][split_name], built)

    out: dict[str, dict[str, Any]] = {}
    for scale, split_accumulators in accumulators.items():
        scale_out: dict[str, Any] = {
            "meta": {
                "scale": scale,
                "context_length": int(scale_specs[scale].context_length),
                "target_horizon_steps": int(TARGET_HORIZON_STEPS[scale]),
                "seconds_per_bucket": int(seconds_per_bucket),
                "channel_names": list(MULTICHANNEL_NAMES),
            }
        }
        for split_name, accumulator in split_accumulators.items():
            x, y, last, total = accumulator.finalize()
            scale_out[f"{split_name}_x"] = x
            scale_out[f"{split_name}_y"] = y
            scale_out[f"{split_name}_last_return"] = last
            scale_out["meta"][f"{split_name}_total_windows"] = int(total)
            scale_out["meta"][f"{split_name}_evaluated_windows"] = int(len(y))
        out[scale] = scale_out
    return out


def add_to_accumulator(
    accumulator: SplitAccumulator,
    built: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> None:
    if built is None:
        return
    accumulator.add(*built)


def aggregate_minute_feature_rows(
    rows: np.ndarray,
    seconds: np.ndarray,
    *,
    seconds_per_bucket: int = 600,
) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    seconds = np.asarray(seconds, dtype=np.int64)
    minute_rows = []
    n_minutes = int(math.ceil(float(seconds_per_bucket) / 60.0))
    for minute in range(n_minutes):
        start = minute * 60
        stop = min((minute + 1) * 60, int(seconds_per_bucket))
        positions = np.flatnonzero((seconds >= start) & (seconds < stop))
        if len(positions):
            minute_rows.append(rows[positions[-1]])
    return np.asarray(minute_rows, dtype=np.float32)


def build_windows_from_feature_rows(
    rows: np.ndarray,
    *,
    feature_names: list[str],
    context_length: int,
    target_horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or len(rows) < context_length + target_horizon_steps:
        return None
    matrix = make_feature_matrix(rows, feature_names)
    wap1 = rows[:, feature_names.index("wap1")]
    log_wap1 = np.log(np.clip(wap1.astype(np.float64), 1e-12, None)).astype(np.float32)
    future_returns = np.diff(log_wap1)
    return build_windows_from_matrix(
        matrix,
        future_returns,
        context_length=context_length,
        target_horizon_steps=target_horizon_steps,
    )


def make_feature_matrix(rows: np.ndarray, feature_names: list[str]) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float32)
    level_values = np.stack([rows[:, feature_names.index(name)] for name in PRICE_LEVEL_CHANNELS], axis=1)
    safe_levels = np.clip(level_values.astype(np.float64), 1e-12, None)
    price_returns = np.zeros_like(level_values, dtype=np.float32)
    if len(rows) > 1:
        price_returns[1:] = np.diff(np.log(safe_levels), axis=0).astype(np.float32)
    state_values = np.stack([rows[:, feature_names.index(name)] for name in STATE_CHANNELS], axis=1)
    matrix = np.concatenate([price_returns, state_values.astype(np.float32)], axis=1)
    finite = np.isfinite(matrix).all(axis=1)
    matrix = np.where(np.isfinite(matrix), matrix, 0.0).astype(np.float32)
    if not finite.all():
        matrix[~finite] = 0.0
    return matrix


def build_windows_from_matrix(
    matrix: np.ndarray,
    future_returns: np.ndarray,
    *,
    context_length: int,
    target_horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    matrix = np.asarray(matrix, dtype=np.float32)
    future_returns = np.asarray(future_returns, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < context_length or len(future_returns) < target_horizon_steps:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(matrix, context_length, axis=0)
    windows = np.moveaxis(windows, -1, 1)
    horizon_targets = np.lib.stride_tricks.sliding_window_view(future_returns, target_horizon_steps).sum(axis=1)
    end_positions = np.arange(context_length - 1, context_length - 1 + len(windows))
    valid = end_positions < len(horizon_targets)
    if not np.any(valid):
        return None
    valid_end_positions = end_positions[valid]
    last_return = matrix[valid_end_positions, 0] * float(target_horizon_steps)
    return (
        windows[valid].astype(np.float32),
        horizon_targets[valid_end_positions].astype(np.float32),
        last_return.astype(np.float32),
    )


def fit_multichannel_normalizer(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    x = np.asarray(x, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x_mean = x.mean(axis=(0, 1), keepdims=True)
    x_std = x.std(axis=(0, 1), keepdims=True)
    x_std = np.maximum(x_std, 1e-6).astype(np.float32)
    return {
        "x_mean": x_mean.astype(np.float32),
        "x_std": x_std,
        "y_mean": float(y.mean()),
        "y_std": float(max(y.std(), 1e-6)),
    }


def build_model_for_channels(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
    *,
    input_channels: int,
    lora_moe_mode: str,
) -> RawMultiScalePatchTST:
    backbone = build_multiscale_patchtst(
        scale_specs,
        input_channels=input_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lora_moe_mode=lora_moe_mode,
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
        target_mode="all_channels",
    )
    return RawMultiScalePatchTST(backbone)


def build_asd_model_for_channels(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
    *,
    input_channels: int,
    lora_moe_mode: str,
) -> ScaleAwareASDMultiScalePatchTST:
    backbone = build_multiscale_patchtst(
        scale_specs,
        input_channels=input_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lora_moe_mode=lora_moe_mode,
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
        target_mode="all_channels",
    )
    return ScaleAwareASDMultiScalePatchTST(backbone, init_gate=-4.0)


def diagnostic_rows(result: dict[str, Any], *, model_name: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale, diagnostics in result.get("diagnostics", {}).items():
        row: dict[str, Any] = {"model": model_name, "scale": scale}
        row.update({key: float(value) for key, value in diagnostics.items()})
        rows.append(row)
    return rows


def write_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Multi-Channel PatchTST Experiment")
    lines.append("")
    lines.append(
        "This run uses a 15-channel intraday input: WAP/MID returns plus spread, imbalance, size, update, and time features. "
        "PatchTST still shares the encoder across channels, but the target head flattens all channel tokens to predict WAP1 future return."
    )
    lines.append("")
    lines.append(
        f"patch preset: `{args.patch_preset}`; epochs={args.epochs}; "
        f"balanced steps/epoch={args.steps_per_epoch}; channels={len(MULTICHANNEL_NAMES)}."
    )
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("")
    test = summary[(summary["split"] == "test") & (summary["model"] != "last_return")].copy()
    keep = ["model", "scale", "n", "nmse", "mse", "mae", "direction_accuracy_nonzero", "corr"]
    lines.extend(frame_to_markdown(test[[column for column in keep if column in test.columns]]))
    lines.append("")
    lines.append("## Test NMSE Relative To Multi-Channel Raw")
    lines.append("")
    rel_rows: list[dict[str, Any]] = []
    raw = test[test["model"] == "multichannel_raw_joint"].set_index("scale")
    for _, row in test.iterrows():
        if row["model"] in {"zero", "multichannel_raw_joint"}:
            continue
        scale = row["scale"]
        if scale in raw.index:
            rel_rows.append(
                {
                    "model": row["model"],
                    "scale": scale,
                    "nmse": float(row["nmse"]),
                    "nmse_vs_multichannel_raw_pct": (float(row["nmse"]) / float(raw.loc[scale, "nmse"]) - 1.0)
                    * 100.0,
                }
            )
    lines.extend(frame_to_markdown(pd.DataFrame(rel_rows)))
    lines.append("")
    lines.append("## Router Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("No diagnostics.")
    else:
        keep_diag = [
            "model",
            "scale",
            "router_entropy",
            "router_balance_loss",
            "expert_prob_0",
            "expert_prob_1",
            "expert_prob_2",
            "expert_prob_3",
            "mean_abs_delta",
        ]
        lines.extend(frame_to_markdown(diagnostics[[column for column in keep_diag if column in diagnostics.columns]]))
    lines.append("")
    lines.append("## Channels")
    lines.append("")
    lines.append(", ".join(f"`{name}`" for name in MULTICHANNEL_NAMES))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_specs = make_scale_specs(args)
    scale_data = build_multichannel_scale_data(
        args,
        cache_path=Path(args.cache),
        caps=small_caps(args),
        scale_specs=scale_specs,
    )
    loaders = make_all_loaders(scale_data, batch_size=args.batch_size, device=device)
    input_channels = len(MULTICHANNEL_NAMES)
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    extra = {
        "patch_preset": args.patch_preset,
        "input_channels": input_channels,
        "target_mode": "all_channels",
    }
    append_baseline_rows(rows, "small", scale_data, extra=extra)

    raw_model = build_model_for_channels(
        args,
        scale_specs,
        input_channels=input_channels,
        lora_moe_mode="none",
    ).to(device)
    apply_training_regime(raw_model, "raw_joint")
    raw_result = train_model(
        model=raw_model,
        model_name="multichannel_raw_joint",
        scale_data=scale_data,
        loaders=loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name="multichannel_raw_joint",
    )
    append_model_rows(rows, "small", "multichannel_raw_joint", raw_result, extra=extra)
    diag_rows.extend(diagnostic_rows(raw_result, model_name="multichannel_raw_joint"))
    raw_checkpoint = Path(raw_result["checkpoint"])

    lora_model = build_model_for_channels(
        args,
        scale_specs,
        input_channels=input_channels,
        lora_moe_mode="last1",
    ).to(device)
    load_raw_backbone_checkpoint(lora_model, raw_checkpoint)
    apply_training_regime(lora_model, "lora_moe_frozen_base_train_moe_head")
    lora_result = train_model(
        model=lora_model,
        model_name="multichannel_lora_moe_frozen_base_train_moe_head",
        scale_data=scale_data,
        loaders=loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name="multichannel_lora_moe_frozen_base_train_moe_head",
        router_balance_weight=args.router_balance_weight,
    )
    append_model_rows(
        rows,
        "small",
        "multichannel_lora_moe_frozen_base_train_moe_head",
        lora_result,
        extra={**extra, "adapter_rank": args.lora_moe_rank},
    )
    diag_rows.extend(diagnostic_rows(lora_result, model_name="multichannel_lora_moe_frozen_base_train_moe_head"))

    asd_model = build_asd_model_for_channels(
        args,
        scale_specs,
        input_channels=input_channels,
        lora_moe_mode="none",
    ).to(device)
    load_raw_backbone_checkpoint(asd_model, raw_checkpoint)
    apply_training_regime(asd_model, "asd_frozen_encoder_train_head")
    asd_result = train_model(
        model=asd_model,
        model_name="multichannel_asd_frozen_encoder_train_head",
        scale_data=scale_data,
        loaders=loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name="multichannel_asd_frozen_encoder_train_head",
    )
    append_model_rows(
        rows,
        "small",
        "multichannel_asd_frozen_encoder_train_head",
        asd_result,
        extra=extra,
    )
    diag_rows.extend(diagnostic_rows(asd_result, model_name="multichannel_asd_frozen_encoder_train_head"))

    asd_lora_model = build_asd_model_for_channels(
        args,
        scale_specs,
        input_channels=input_channels,
        lora_moe_mode="last1",
    ).to(device)
    load_raw_backbone_checkpoint(asd_lora_model, raw_checkpoint)
    apply_training_regime(asd_lora_model, "asd_lora_moe_frozen_base_train_adapters_head")
    asd_lora_result = train_model(
        model=asd_lora_model,
        model_name="multichannel_asd_lora_moe_frozen_adapters_head",
        scale_data=scale_data,
        loaders=loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name="multichannel_asd_lora_moe_frozen_adapters_head",
        router_balance_weight=args.router_balance_weight,
    )
    append_model_rows(
        rows,
        "small",
        "multichannel_asd_lora_moe_frozen_adapters_head",
        asd_lora_result,
        extra={**extra, "adapter_rank": args.lora_moe_rank},
    )
    diag_rows.extend(diagnostic_rows(asd_lora_result, model_name="multichannel_asd_lora_moe_frozen_adapters_head"))

    summary = save_summary(rows, output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "patch_preset": args.patch_preset,
                "channel_names": list(MULTICHANNEL_NAMES),
                "scale_specs": {name: spec.__dict__ for name, spec in scale_specs.items()},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(path=Path(args.report_path), summary=summary, diagnostics=diagnostics, args=args)
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
