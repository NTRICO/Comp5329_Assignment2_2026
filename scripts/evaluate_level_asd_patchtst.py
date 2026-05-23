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
    aggregate_minute_levels,
    cap_indices,
    fit_normalizer,
    parse_asset_stock_id,
    select_device,
    set_seed,
)
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    DEFAULT_SCALE_SPECS,
    PATCH_PRESETS,
    SCALE_ORDER,
    TARGET_HORIZON_STEPS,
    ScaleData,
    append_baseline_rows,
    append_model_rows,
    apply_training_regime,
    build_model,
    build_multiscale_patchtst,
    caps_for_preset,
    collect_diagnostics,
    frame_to_markdown,
    load_raw_backbone_checkpoint,
    load_scale_data,
    make_all_loaders,
    make_scale_specs,
    parse_stock_list,
    save_summary,
    train_model,
)
from src.baselines.scale_aware_asd_patchtst import LevelASDMultiScalePatchTST, ScaleSpec  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare return-ASD with price-domain ASD before converting to returns."
    )
    parser.add_argument(
        "--cache",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "cache"
            / "position_optiver_hf_second_feature_cache_11stocks_512t.npz"
        ),
    )
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "level_asd_patchtst"))
    parser.add_argument("--report-path", default=str(WORKSPACE_ROOT / "report" / "level_asd_patchtst_experiment.md"))
    parser.add_argument("--train-stocks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--zero-shot-stock", type=int, default=10)
    parser.add_argument("--scales", nargs="+", choices=SCALE_ORDER, default=list(SCALE_ORDER))
    parser.add_argument("--patch-preset", choices=sorted(PATCH_PRESETS), default="short_second")
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
    parser.add_argument("--scale-aware-init-gate", type=float, default=-4.0)
    parser.add_argument("--encoder-spectral-mode", choices=["none", "last1"], default="none")
    parser.add_argument("--encoder-spectral-init-gate", type=float, default=-4.0)
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


def build_level_scale_data(
    args: argparse.Namespace,
    *,
    cache_path: Path,
    caps: dict[str, int],
    scale_specs: dict[str, ScaleSpec],
    price_mode: str,
    return_scale_data: dict[str, ScaleData],
) -> dict[str, ScaleData]:
    train_stocks = parse_stock_list(args.train_stocks)
    raw_arrays = build_level_arrays(
        cache_path=cache_path,
        scales=list(args.scales),
        scale_specs=scale_specs,
        train_stocks=train_stocks,
        zero_shot_stock=int(args.zero_shot_stock),
        price_mode=price_mode,
        caps=caps,
        seed=int(args.seed),
    )
    out: dict[str, ScaleData] = {}
    for scale in args.scales:
        arrays = raw_arrays[scale]
        return_normalizer = return_scale_data[scale].normalizer
        normalizer = fit_normalizer(arrays["train_x"], arrays["train_y"])
        normalizer["x_mean"] = 0.0
        normalizer["x_std"] = 1.0
        normalizer["y_mean"] = float(return_normalizer["y_mean"])
        normalizer["y_std"] = float(return_normalizer["y_std"])
        out[scale] = ScaleData(name=scale, spec=scale_specs[scale], arrays=arrays, normalizer=normalizer)
        print(
            f"{price_mode}/{scale}: train={len(arrays['train_y'])} "
            f"validation={len(arrays['validation_y'])} test={len(arrays['test_y'])} "
            f"zero_shot={len(arrays['zero_shot_y'])}",
            flush=True,
        )
    return out


def build_level_arrays(
    *,
    cache_path: Path,
    scales: list[str],
    scale_specs: dict[str, ScaleSpec],
    train_stocks: list[int],
    zero_shot_stock: int,
    price_mode: str,
    caps: dict[str, int],
    seed: int,
) -> dict[str, dict[str, Any]]:
    if price_mode not in {"log_price", "raw_price"}:
        raise ValueError("price_mode must be 'log_price' or 'raw_price'.")
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
    if "wap1" not in feature_names:
        raise ValueError("wap1 is required for price-domain ASD.")
    wap1_index = feature_names.index("wap1")
    stock_ids = np.asarray([parse_asset_stock_id(name) for name in asset_names], dtype=np.int64)

    needed = set(train_stocks + [int(zero_shot_stock)])
    missing = sorted(needed - set(stock_ids.tolist()))
    if missing:
        raise ValueError(f"Cache is missing stocks {missing}.")

    by_scale: dict[str, dict[str, list[np.ndarray]]] = {}
    for scale in scales:
        by_scale[scale] = {
            "train_x": [],
            "train_y": [],
            "train_last_return": [],
            "validation_x": [],
            "validation_y": [],
            "validation_last_return": [],
            "test_x": [],
            "test_y": [],
            "test_last_return": [],
            "zero_shot_x": [],
            "zero_shot_y": [],
            "zero_shot_last_return": [],
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
                hour_levels = []
                for episode in split_episodes:
                    idx = np.flatnonzero(stock_mask & (episode_ids == episode))
                    idx = idx[np.argsort(seconds[idx])]
                    if len(idx):
                        hour_levels.append(float(features[idx[-1], wap1_index]))
                add_built_window(
                    by_scale["hour"],
                    split_name,
                    build_level_windows_from_levels(
                        np.asarray(hour_levels, dtype=np.float32),
                        context_length=scale_specs["hour"].context_length,
                        target_horizon_steps=TARGET_HORIZON_STEPS["hour"],
                        price_mode=price_mode,
                    ),
                )

            for episode in split_episodes:
                idx = np.flatnonzero(stock_mask & (episode_ids == episode))
                idx = idx[np.argsort(seconds[idx])]
                if not len(idx):
                    continue
                if "second" in scales:
                    levels = features[idx, wap1_index].astype(np.float32)
                    future_returns = np.log1p(
                        np.clip(targets[idx].astype(np.float64), -0.999999, None)
                    ).astype(np.float32)
                    add_built_window(
                        by_scale["second"],
                        split_name,
                        build_level_windows_from_aligned_returns(
                            levels,
                            future_returns,
                            context_length=scale_specs["second"].context_length,
                            target_horizon_steps=TARGET_HORIZON_STEPS["second"],
                            price_mode=price_mode,
                        ),
                    )
                if "minute" in scales:
                    minute_levels = aggregate_minute_levels(
                        features[idx, wap1_index],
                        seconds[idx],
                        seconds_per_bucket=seconds_per_bucket,
                    )
                    add_built_window(
                        by_scale["minute"],
                        split_name,
                        build_level_windows_from_levels(
                            minute_levels,
                            context_length=scale_specs["minute"].context_length,
                            target_horizon_steps=TARGET_HORIZON_STEPS["minute"],
                            price_mode=price_mode,
                        ),
                    )

    rng = np.random.default_rng(seed)
    out: dict[str, dict[str, Any]] = {}
    for scale, split_lists in by_scale.items():
        scale_out: dict[str, Any] = {
            "meta": {
                "scale": scale,
                "price_mode": price_mode,
                "context_length": int(scale_specs[scale].context_length),
                "target_horizon_steps": int(TARGET_HORIZON_STEPS[scale]),
                "seconds_per_bucket": int(seconds_per_bucket),
            }
        }
        for split_name in ["train", "validation", "test", "zero_shot"]:
            x_key = f"{split_name}_x"
            y_key = f"{split_name}_y"
            last_key = f"{split_name}_last_return"
            if not split_lists[x_key]:
                raise ValueError(f"No {price_mode}/{scale} windows built for split {split_name}.")
            x = np.concatenate(split_lists[x_key], axis=0)
            y = np.concatenate(split_lists[y_key], axis=0)
            last = np.concatenate(split_lists[last_key], axis=0)
            selected = cap_indices(len(y), int(caps.get(split_name, 0)), rng=rng, random=(split_name == "train"))
            scale_out[x_key] = x[selected]
            scale_out[y_key] = y[selected]
            scale_out[last_key] = last[selected]
            scale_out["meta"][f"{split_name}_total_windows"] = int(len(y))
            scale_out["meta"][f"{split_name}_evaluated_windows"] = int(len(selected))
        out[scale] = scale_out
    return out


def add_built_window(
    split_lists: dict[str, list[np.ndarray]],
    split_name: str,
    built: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> None:
    if built is None:
        return
    x, y, last = built
    split_lists[f"{split_name}_x"].append(x)
    split_lists[f"{split_name}_y"].append(y)
    split_lists[f"{split_name}_last_return"].append(last)


def build_level_windows_from_levels(
    levels: np.ndarray,
    *,
    context_length: int,
    target_horizon_steps: int,
    price_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    levels = np.asarray(levels, dtype=np.float32)
    levels = levels[np.isfinite(levels) & (levels > 0.0)]
    if len(levels) < context_length + target_horizon_steps:
        return None
    log_levels = np.log(np.clip(levels.astype(np.float64), 1e-12, None)).astype(np.float32)
    future_returns = np.diff(log_levels)
    return build_level_windows_from_aligned_returns(
        levels,
        future_returns,
        context_length=context_length,
        target_horizon_steps=target_horizon_steps,
        price_mode=price_mode,
    )


def build_level_windows_from_aligned_returns(
    levels: np.ndarray,
    future_returns: np.ndarray,
    *,
    context_length: int,
    target_horizon_steps: int,
    price_mode: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    levels = np.asarray(levels, dtype=np.float32)
    valid = np.isfinite(levels) & (levels > 0.0)
    levels = levels[valid]
    future_returns = np.asarray(future_returns, dtype=np.float32)
    future_returns = future_returns[: len(levels)]
    if len(levels) < context_length or len(future_returns) < target_horizon_steps:
        return None
    level_values = transform_levels(levels, price_mode)
    windows = np.lib.stride_tricks.sliding_window_view(level_values, context_length)
    horizon_targets = np.lib.stride_tricks.sliding_window_view(future_returns, target_horizon_steps).sum(axis=1)
    end_positions = np.arange(context_length - 1, context_length - 1 + len(windows))
    valid_positions = end_positions < len(horizon_targets)
    if not np.any(valid_positions):
        return None
    selected_end_positions = end_positions[valid_positions]
    log_levels = np.log(np.clip(levels.astype(np.float64), 1e-12, None)).astype(np.float32)
    observed_returns = np.zeros_like(log_levels, dtype=np.float32)
    observed_returns[1:] = np.diff(log_levels)
    return (
        windows[valid_positions].astype(np.float32)[:, :, None],
        horizon_targets[selected_end_positions].astype(np.float32),
        (observed_returns[selected_end_positions] * float(target_horizon_steps)).astype(np.float32),
    )


def transform_levels(levels: np.ndarray, price_mode: str) -> np.ndarray:
    if price_mode == "log_price":
        return np.log(np.clip(levels.astype(np.float64), 1e-12, None)).astype(np.float32)
    if price_mode == "raw_price":
        return levels.astype(np.float32)
    raise ValueError(f"Unsupported price_mode={price_mode!r}")


def build_level_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
    price_mode: str,
    *,
    lora_moe_mode: str = "none",
) -> LevelASDMultiScalePatchTST:
    backbone = build_multiscale_patchtst(
        scale_specs,
        input_channels=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        encoder_spectral_mode="none",
        encoder_spectral_init_gate=args.encoder_spectral_init_gate,
        lora_moe_mode=lora_moe_mode,
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
    )
    return LevelASDMultiScalePatchTST(
        backbone,
        price_mode=price_mode,
        init_gate=args.scale_aware_init_gate,
    )


def set_return_stats(model: LevelASDMultiScalePatchTST, return_scale_data: dict[str, ScaleData]) -> None:
    for scale, data in return_scale_data.items():
        model.set_return_input_stats(
            scale,
            mean=float(data.normalizer["x_mean"]),
            std=float(data.normalizer["x_std"]),
        )


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
    lines.append("# Level-ASD PatchTST Experiment")
    lines.append("")
    lines.append(
        "This small experiment compares return-domain ASD with price-domain ASD. "
        "For price-domain ASD, the module first cleans WAP1 price/log-price, then the model converts the cleaned path to returns."
    )
    lines.append("")
    lines.append(
        f"cache: `{args.cache}`; patch preset: `{args.patch_preset}`; "
        f"epochs={args.epochs}; balanced steps/epoch={args.steps_per_epoch}; init_gate={args.scale_aware_init_gate}."
    )
    lines.append("")
    lines.append("## Test NMSE")
    lines.append("")
    test = summary[summary["split"] == "test"].copy()
    keep = [
        "model",
        "scale",
        "n",
        "nmse",
        "mse",
        "mae",
        "direction_accuracy_nonzero",
        "corr",
    ]
    lines.extend(frame_to_markdown(test[[column for column in keep if column in test.columns]]))
    lines.append("")
    lines.append("## Test NMSE Relative To Raw")
    lines.append("")
    rel_rows: list[dict[str, Any]] = []
    raw_rows = test[test["model"] == "raw_joint"].set_index("scale")
    for _, row in test.iterrows():
        scale = row["scale"]
        if row["model"] in {"zero", "last_return", "raw_joint"} or scale not in raw_rows.index:
            continue
        rel_rows.append(
            {
                "model": row["model"],
                "scale": scale,
                "nmse": row["nmse"],
                "nmse_vs_raw_pct": (float(row["nmse"]) / float(raw_rows.loc[scale, "nmse"]) - 1.0) * 100.0,
            }
        )
    lines.extend(frame_to_markdown(pd.DataFrame(rel_rows)))
    lines.append("")
    lines.append("## ASD Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("No diagnostics.")
    else:
        keep_diag = [
            "model",
            "scale",
            "gate_mean",
            "tau_mean",
            "mean_abs_delta",
            "level_asd_gate_mean",
            "level_asd_tau_mean",
            "level_asd_mean_abs_delta",
            "clean_return_abs_mean",
        ]
        lines.extend(frame_to_markdown(diagnostics[[column for column in keep_diag if column in diagnostics.columns]]))
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- summary: `{Path(args.output_dir) / 'summary.csv'}`")
    lines.append(f"- diagnostics: `{Path(args.output_dir) / 'diagnostics.csv'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_specs = make_scale_specs(args)
    caps = caps_for_preset(args, "small")

    return_scale_data = load_scale_data(
        args,
        cache_path=Path(args.cache),
        caps=caps,
        scale_specs=scale_specs,
    )
    return_loaders = make_all_loaders(return_scale_data, batch_size=args.batch_size, device=device)
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    append_baseline_rows(rows, "small", return_scale_data, extra={"patch_preset": args.patch_preset})

    raw_model = build_model("raw_patchtst", args, scale_specs).to(device)
    apply_training_regime(raw_model, "raw_joint")
    raw_result = train_model(
        model=raw_model,
        model_name="raw_joint",
        scale_data=return_scale_data,
        loaders=return_loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name="raw_joint",
    )
    append_model_rows(rows, "small", "raw_joint", raw_result, extra={"patch_preset": args.patch_preset})
    diag_rows.extend(diagnostic_rows(raw_result, model_name="raw_joint"))
    raw_checkpoint = Path(raw_result["checkpoint"])

    return_asd_model = build_model("scale_aware_asd_patchtst", args, scale_specs).to(device)
    load_raw_backbone_checkpoint(return_asd_model, raw_checkpoint)
    apply_training_regime(return_asd_model, "asd_frozen_encoder_train_head")
    return_asd_result = train_model(
        model=return_asd_model,
        model_name="return_asd_frozen_encoder_train_head",
        scale_data=return_scale_data,
        loaders=return_loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name="return_asd_frozen_encoder_train_head",
    )
    append_model_rows(
        rows,
        "small",
        "return_asd_frozen_encoder_train_head",
        return_asd_result,
        extra={"patch_preset": args.patch_preset},
    )
    diag_rows.extend(diagnostic_rows(return_asd_result, model_name="return_asd_frozen_encoder_train_head"))

    lora_moe_model = build_model("lora_moe_patchtst", args, scale_specs).to(device)
    load_raw_backbone_checkpoint(lora_moe_model, raw_checkpoint)
    apply_training_regime(lora_moe_model, "lora_moe_frozen_base_train_moe_head")
    lora_moe_result = train_model(
        model=lora_moe_model,
        model_name="lora_moe_frozen_base_train_moe_head",
        scale_data=return_scale_data,
        loaders=return_loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name="lora_moe_frozen_base_train_moe_head",
        router_balance_weight=args.router_balance_weight,
    )
    append_model_rows(
        rows,
        "small",
        "lora_moe_frozen_base_train_moe_head",
        lora_moe_result,
        extra={"patch_preset": args.patch_preset, "adapter_rank": args.lora_moe_rank},
    )
    diag_rows.extend(diagnostic_rows(lora_moe_result, model_name="lora_moe_frozen_base_train_moe_head"))

    for price_mode in ["log_price", "raw_price"]:
        level_scale_data = build_level_scale_data(
            args,
            cache_path=Path(args.cache),
            caps=caps,
            scale_specs=scale_specs,
            price_mode=price_mode,
            return_scale_data=return_scale_data,
        )
        level_loaders = make_all_loaders(level_scale_data, batch_size=args.batch_size, device=device)
        level_model = build_level_model(args, scale_specs, price_mode).to(device)
        set_return_stats(level_model, return_scale_data)
        load_raw_backbone_checkpoint(level_model, raw_checkpoint)
        apply_training_regime(level_model, "asd_frozen_encoder_train_head")
        model_name = f"level_asd_{price_mode}_frozen_encoder_train_head"
        result = train_model(
            model=level_model,
            model_name=model_name,
            scale_data=level_scale_data,
            loaders=level_loaders,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=output_dir,
            checkpoint_name=model_name,
        )
        append_model_rows(rows, "small", model_name, result, extra={"patch_preset": args.patch_preset})
        diag_rows.extend(diagnostic_rows(result, model_name=model_name))

    raw_price_data = build_level_scale_data(
        args,
        cache_path=Path(args.cache),
        caps=caps,
        scale_specs=scale_specs,
        price_mode="raw_price",
        return_scale_data=return_scale_data,
    )
    raw_price_loaders = make_all_loaders(raw_price_data, batch_size=args.batch_size, device=device)
    raw_price_lora_model = build_level_model(args, scale_specs, "raw_price", lora_moe_mode="last1").to(device)
    set_return_stats(raw_price_lora_model, return_scale_data)
    load_raw_backbone_checkpoint(raw_price_lora_model, raw_checkpoint)
    apply_training_regime(raw_price_lora_model, "asd_lora_moe_frozen_base_train_adapters_head")
    raw_price_lora_name = "level_asd_raw_price_lora_moe_frozen_adapters_head"
    raw_price_lora_result = train_model(
        model=raw_price_lora_model,
        model_name=raw_price_lora_name,
        scale_data=raw_price_data,
        loaders=raw_price_loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name=raw_price_lora_name,
        router_balance_weight=args.router_balance_weight,
    )
    append_model_rows(
        rows,
        "small",
        raw_price_lora_name,
        raw_price_lora_result,
        extra={"patch_preset": args.patch_preset, "adapter_rank": args.lora_moe_rank},
    )
    diag_rows.extend(diagnostic_rows(raw_price_lora_result, model_name=raw_price_lora_name))

    summary = save_summary(rows, output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "patch_preset": args.patch_preset,
                "scale_specs": {name: spec.__dict__ for name, spec in scale_specs.items()},
                "price_modes": ["log_price", "raw_price"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(path=Path(args.report_path), summary=summary, diagnostics=diagnostics, args=args)
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
