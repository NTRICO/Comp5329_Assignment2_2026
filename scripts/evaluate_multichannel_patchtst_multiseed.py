from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import torch


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_multichannel_patchtst import (  # noqa: E402
    MULTICHANNEL_NAMES,
    build_asd_model_for_channels,
    build_model_for_channels,
    build_multichannel_scale_data,
    diagnostic_rows,
    small_caps,
)
from evaluate_optiver_spectral_denoise_patchtst import select_device, set_seed  # noqa: E402
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
    append_baseline_rows,
    append_model_rows,
    apply_training_regime,
    frame_to_markdown,
    load_raw_backbone_checkpoint,
    make_all_loaders,
    make_scale_specs,
    save_summary,
    train_model,
)


DEFAULT_TRAIN_STOCKS = ",".join(
    str(item)
    for item in [
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        26,
        27,
        28,
        29,
        30,
        31,
        32,
        33,
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run multi-seed multi-channel ASD + LoRA-MoE PatchTST confirmation."
    )
    parser.add_argument(
        "--cache",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "cache"
            / "position_optiver_hf_second_feature_cache_32stocks_512t.npz"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "multichannel_asd_lora_moe_32stock_multiseed"),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "multichannel_asd_lora_moe_32stock_multiseed.md"),
    )
    parser.add_argument("--train-stocks", default=DEFAULT_TRAIN_STOCKS)
    parser.add_argument("--zero-shot-stock", type=int, default=34)
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
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=50)
    parser.add_argument("--small-train-cap", type=int, default=20000)
    parser.add_argument("--small-validation-cap", type=int, default=4096)
    parser.add_argument("--small-test-cap", type=int, default=4096)
    parser.add_argument("--small-zero-shot-cap", type=int, default=4096)
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
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "multichannel_raw_joint",
            "multichannel_lora_moe_frozen_base_train_moe_head",
            "multichannel_asd_frozen_encoder_train_head",
            "multichannel_asd_lora_moe_frozen_adapters_head",
        ],
        default=[
            "multichannel_raw_joint",
            "multichannel_asd_lora_moe_frozen_adapters_head",
        ],
    )
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
    return parser.parse_args()


def aggregate_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["n", "nmse", "mse", "mae", "direction_accuracy_nonzero", "corr"]
    frame = summary.copy()
    for column in metric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.groupby(["model", "split", "scale"], dropna=False)[metric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    return grouped


def aggregate_diagnostics(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return diagnostics
    numeric_columns = [column for column in diagnostics.columns if column not in {"model", "scale", "seed"}]
    frame = diagnostics.copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.groupby(["model", "scale"], dropna=False)[numeric_columns]
        .agg(["mean", "std"])
        .reset_index()
    )
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    return grouped


def comparison_rows(aggregate: pd.DataFrame, *, split: str = "test") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    raw = aggregate[
        (aggregate["split"] == split)
        & (aggregate["model"] == "multichannel_raw_joint")
    ].set_index("scale")
    if raw.empty:
        return pd.DataFrame()
    candidates = aggregate[
        (aggregate["split"] == split)
        & (~aggregate["model"].isin(["multichannel_raw_joint", "zero", "last_return"]))
    ]
    for _, row in candidates.iterrows():
        scale = str(row["scale"])
        if scale not in raw.index:
            continue
        raw_row = raw.loc[scale]
        rows.append(
            {
                "model": row["model"],
                "scale": scale,
                "n": row["n_mean"],
                "improvement_pct_mean": (
                    (float(raw_row["nmse_mean"]) - float(row["nmse_mean"]))
                    / float(raw_row["nmse_mean"])
                    * 100.0
                ),
                "direction_pp_mean": (
                    float(row["direction_accuracy_nonzero_mean"])
                    - float(raw_row["direction_accuracy_nonzero_mean"])
                )
                * 100.0,
                "corr_delta_mean": float(row["corr_mean"]) - float(raw_row["corr_mean"]),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    *,
    path: Path,
    aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Multi-Channel ASD + LoRA-MoE Multi-Seed Confirmation")
    lines.append("")
    lines.append(
        "本报告只展示相对 `multichannel_raw_joint` 的百分比变化；不把原始归一化误差写入主表。"
    )
    lines.append("")
    lines.append(
        f"cache: `{args.cache}`; seeds: `{', '.join(map(str, args.seeds))}`; "
        f"patch preset: `{args.patch_preset}`; epochs={args.epochs}; "
        f"balanced steps/epoch={args.steps_per_epoch}; channels={len(MULTICHANNEL_NAMES)}."
    )
    lines.append("")
    lines.append("## Test Improvement")
    lines.append("")
    test_comparison = comparison_rows(aggregate, split="test")
    lines.extend(frame_to_markdown(test_comparison))
    lines.append("")
    lines.append("## Zero-Shot Improvement")
    lines.append("")
    zero_comparison = comparison_rows(aggregate, split="zero_shot")
    lines.extend(frame_to_markdown(zero_comparison))
    lines.append("")
    lines.append("## Router Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("No diagnostics.")
    else:
        keep = [
            "model",
            "scale",
            "router_entropy_mean",
            "router_balance_loss_mean",
            "expert_prob_0_mean",
            "expert_prob_1_mean",
            "expert_prob_2_mean",
            "expert_prob_3_mean",
            "mean_abs_delta_mean",
        ]
        lines.extend(frame_to_markdown(diagnostics[[column for column in keep if column in diagnostics.columns]]))
    lines.append("")
    lines.append("## Channels")
    lines.append("")
    lines.append(", ".join(f"`{name}`" for name in MULTICHANNEL_NAMES))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed(args: argparse.Namespace, *, seed: int, device: torch.device, output_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = int(seed)
    set_seed(int(seed))
    seed_dir = output_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    scale_specs = make_scale_specs(seed_args)
    scale_data = build_multichannel_scale_data(
        seed_args,
        cache_path=Path(seed_args.cache),
        caps=small_caps(seed_args),
        scale_specs=scale_specs,
    )
    loaders = make_all_loaders(scale_data, batch_size=seed_args.batch_size, device=device)
    input_channels = len(MULTICHANNEL_NAMES)
    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    extra = {
        "patch_preset": seed_args.patch_preset,
        "input_channels": input_channels,
        "target_mode": "all_channels",
        "adapter_rank": seed_args.lora_moe_rank,
        "seed": int(seed),
    }
    append_baseline_rows(rows, "small", scale_data, extra=extra)

    raw_model = build_model_for_channels(
        seed_args,
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
        epochs=seed_args.epochs,
        steps_per_epoch=seed_args.steps_per_epoch,
        learning_rate=seed_args.learning_rate,
        weight_decay=seed_args.weight_decay,
        device=device,
        output_dir=seed_dir,
        checkpoint_name="multichannel_raw_joint",
    )
    append_model_rows(rows, "small", "multichannel_raw_joint", raw_result, extra=extra)
    diag_rows.extend(diagnostic_rows(raw_result, model_name="multichannel_raw_joint"))
    raw_checkpoint = Path(raw_result["checkpoint"])

    requested_models = set(seed_args.models)
    if "multichannel_lora_moe_frozen_base_train_moe_head" in requested_models:
        lora_model = build_model_for_channels(
            seed_args,
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
            epochs=seed_args.epochs,
            steps_per_epoch=seed_args.steps_per_epoch,
            learning_rate=seed_args.learning_rate,
            weight_decay=seed_args.weight_decay,
            device=device,
            output_dir=seed_dir,
            checkpoint_name="multichannel_lora_moe_frozen_base_train_moe_head",
            router_balance_weight=seed_args.router_balance_weight,
        )
        append_model_rows(
            rows,
            "small",
            "multichannel_lora_moe_frozen_base_train_moe_head",
            lora_result,
            extra=extra,
        )
        diag_rows.extend(diagnostic_rows(lora_result, model_name="multichannel_lora_moe_frozen_base_train_moe_head"))

    if "multichannel_asd_frozen_encoder_train_head" in requested_models:
        asd_model = build_asd_model_for_channels(
            seed_args,
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
            epochs=seed_args.epochs,
            steps_per_epoch=seed_args.steps_per_epoch,
            learning_rate=seed_args.learning_rate,
            weight_decay=seed_args.weight_decay,
            device=device,
            output_dir=seed_dir,
            checkpoint_name="multichannel_asd_frozen_encoder_train_head",
        )
        append_model_rows(rows, "small", "multichannel_asd_frozen_encoder_train_head", asd_result, extra=extra)
        diag_rows.extend(diagnostic_rows(asd_result, model_name="multichannel_asd_frozen_encoder_train_head"))

    if "multichannel_asd_lora_moe_frozen_adapters_head" in requested_models:
        asd_lora_model = build_asd_model_for_channels(
            seed_args,
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
            epochs=seed_args.epochs,
            steps_per_epoch=seed_args.steps_per_epoch,
            learning_rate=seed_args.learning_rate,
            weight_decay=seed_args.weight_decay,
            device=device,
            output_dir=seed_dir,
            checkpoint_name="multichannel_asd_lora_moe_frozen_adapters_head",
            router_balance_weight=seed_args.router_balance_weight,
        )
        append_model_rows(
            rows,
            "small",
            "multichannel_asd_lora_moe_frozen_adapters_head",
            asd_lora_result,
            extra=extra,
        )
        diag_rows.extend(
            diagnostic_rows(asd_lora_result, model_name="multichannel_asd_lora_moe_frozen_adapters_head")
        )

    summary = save_summary(rows, seed_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics["seed"] = int(seed)
    diagnostics.to_csv(seed_dir / "diagnostics.csv", index=False)
    return summary, diagnostics


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    all_summary: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    for seed in args.seeds:
        print(f"running_seed={seed}", flush=True)
        summary, diagnostics = run_seed(args, seed=int(seed), device=device, output_root=output_root)
        all_summary.append(summary)
        all_diagnostics.append(diagnostics)

    combined_summary = pd.concat(all_summary, ignore_index=True)
    combined_summary.to_csv(output_root / "summary.csv", index=False)
    diagnostics = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    aggregate = aggregate_metrics(combined_summary)
    aggregate.to_csv(output_root / "aggregate.csv", index=False)
    diagnostic_aggregate = aggregate_diagnostics(diagnostics)
    diagnostic_aggregate.to_csv(output_root / "diagnostic_aggregate.csv", index=False)
    (output_root / "metadata.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "patch_preset": args.patch_preset,
                "seeds": [int(seed) for seed in args.seeds],
                "models": list(args.models),
                "channel_names": list(MULTICHANNEL_NAMES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        path=Path(args.report_path),
        aggregate=aggregate,
        diagnostics=diagnostic_aggregate,
        args=args,
    )
    print(f"saved_summary={output_root / 'summary.csv'}", flush=True)
    print(f"saved_aggregate={output_root / 'aggregate.csv'}", flush=True)
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
