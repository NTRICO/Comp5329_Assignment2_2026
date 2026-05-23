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

from evaluate_optiver_spectral_denoise_patchtst import select_device, set_seed  # noqa: E402
from evaluate_prepatch_asd_adapter_patchtst import (  # noqa: E402
    apply_preprocessed_regime,
    build_preprocessed_model,
    build_scale_specific_preprocessed_model,
    train_and_record,
)
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
    append_baseline_rows,
    apply_training_regime,
    build_model,
    caps_for_preset,
    frame_to_markdown,
    load_raw_backbone_checkpoint,
    load_scale_data,
    make_all_loaders,
    make_scale_specs,
    save_summary,
)


DEFAULT_TRAIN_STOCKS = ",".join(
    str(stock) for stock in range(10)
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-seed confirmation for gated pre-ASD + LoRA-MoE."
    )
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
        default=str(
            WORKSPACE_ROOT
            / "outputs"
            / "prepatch_asd_adapter_patchtst"
            / "gated_pre_asd_true_hour_60_30_10_h10"
        ),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "gated_pre_asd_true_hour_60_30_10_h10.md"),
    )
    parser.add_argument("--train-stocks", default=DEFAULT_TRAIN_STOCKS)
    parser.add_argument("--zero-shot-stock", type=int, default=9)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
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
    parser.add_argument("--scale-aware-init-gate", type=float, default=-4.0)
    parser.add_argument("--encoder-spectral-mode", choices=["none", "last1"], default="none")
    parser.add_argument("--encoder-spectral-init-gate", type=float, default=-4.0)
    parser.add_argument("--lora-moe-rank", type=int, default=8)
    parser.add_argument("--lora-moe-alpha", type=float, default=16.0)
    parser.add_argument("--lora-moe-n-experts", type=int, default=4)
    parser.add_argument("--lora-moe-top-k", type=int, default=2)
    parser.add_argument("--lora-moe-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-moe-bottleneck", type=int, default=8)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--router-balance-weight", type=float, default=1e-3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "raw_joint",
            "post_return_lora_moe_head",
            "post_return_lora_moe_mlp_head",
            "gated_pre_return_asd_lora_moe_patchtst",
            "gated_pre_return_asd_lora_moe_mlp_head",
            "scale_specific_gated_pre_asd_moe_patchtst",
        ],
        default=[
            "raw_joint",
            "post_return_lora_moe_head",
            "post_return_lora_moe_mlp_head",
            "gated_pre_return_asd_lora_moe_patchtst",
            "gated_pre_return_asd_lora_moe_mlp_head",
            "scale_specific_gated_pre_asd_moe_patchtst",
        ],
    )
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
        target_default = 10 if scale == "second" else 1
        parser.add_argument(f"--{scale}-target-horizon-steps", type=int, default=target_default)
    return parser.parse_args()


def aggregate_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "n",
        "nmse",
        "mse",
        "mae",
        "direction_accuracy_nonzero",
        "corr",
    ]
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
    numeric_columns = [
        column
        for column in diagnostics.columns
        if column not in {"model", "scale", "seed"}
    ]
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


def write_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    aggregate: pd.DataFrame,
    diagnostics_aggregate: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Gated Pre-ASD True-Hour 60/30/10 Multi-Seed Confirmation")
    lines.append("")
    lines.append(
        "This run checks whether `gated_pre_return_asd_lora_moe_patchtst` remains useful "
        "on the additional-data true-hour cache across seeds. Training is balanced across "
        "`second/minute/hour`; no day data is included."
    )
    lines.append("")
    lines.append(
        f"cache: `{args.cache}`; seeds: `{', '.join(map(str, args.seeds))}`; "
        f"patch preset: `{args.patch_preset}`; epochs={args.epochs}; "
        f"steps/epoch={args.steps_per_epoch}; train cap={args.small_train_cap}; "
        f"eval cap={args.small_test_cap}."
    )
    lines.append("")
    lines.append("## Test Mean / Std")
    lines.append("")
    test = aggregate[aggregate["split"] == "test"].copy()
    keep = [
        "model",
        "scale",
        "n_mean",
        "nmse_mean",
        "nmse_std",
        "mae_mean",
        "mae_std",
        "direction_accuracy_nonzero_mean",
        "direction_accuracy_nonzero_std",
        "corr_mean",
        "corr_std",
    ]
    lines.extend(frame_to_markdown(test[[column for column in keep if column in test.columns]]))
    lines.append("")
    lines.append("## Seed-Level Test Rows")
    lines.append("")
    seed_test = summary[
        (summary["split"] == "test")
        & (~summary["model"].isin(["zero", "last_return"]))
    ].copy()
    seed_keep = ["seed", "model", "scale", "n", "nmse", "mae", "direction_accuracy_nonzero", "corr"]
    lines.extend(frame_to_markdown(seed_test[[column for column in seed_keep if column in seed_test.columns]]))
    lines.append("")
    lines.append("## Diagnostics Mean / Std")
    lines.append("")
    if diagnostics_aggregate.empty:
        lines.append("No diagnostics.")
    else:
        diag_keep = [
            "model",
            "scale",
            "asd_gate_mean_mean",
            "asd_gate_mean_std",
            "asd_tau_mean_mean",
            "asd_tau_mean_std",
            "final_gate_mean_mean",
            "final_gate_mean_std",
            "final_mean_abs_delta_mean",
            "final_mean_abs_delta_std",
            "router_entropy_mean",
            "router_entropy_std",
            "expert_prob_0_mean",
            "expert_prob_1_mean",
            "expert_prob_2_mean",
            "expert_prob_3_mean",
        ]
        lines.extend(
            frame_to_markdown(
                diagnostics_aggregate[
                    [column for column in diag_keep if column in diagnostics_aggregate.columns]
                ]
            )
        )
    lines.append("")
    out = Path(args.output_dir)
    lines.append("## Files")
    lines.append("")
    lines.append(f"- seed summaries: `{out / 'seed_*' / 'summary.csv'}`")
    lines.append(f"- all summary: `{out / 'summary_all.csv'}`")
    lines.append(f"- aggregate: `{out / 'aggregate.csv'}`")
    lines.append(f"- diagnostics: `{out / 'diagnostics_all.csv'}`")
    lines.append(f"- diagnostics aggregate: `{out / 'diagnostics_aggregate.csv'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed(args: argparse.Namespace, seed: int, device: torch.device) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = seed
    linear_args = argparse.Namespace(**vars(seed_args))
    linear_args.head_type = "linear"
    mlp_args = argparse.Namespace(**vars(seed_args))
    mlp_args.head_type = "mlp"
    seed_output_dir = Path(args.output_dir) / f"seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    scale_specs = make_scale_specs(linear_args)
    caps = caps_for_preset(linear_args, "small")
    scale_data = load_scale_data(linear_args, cache_path=Path(args.cache), caps=caps, scale_specs=scale_specs)
    loaders = make_all_loaders(scale_data, batch_size=args.batch_size, device=device)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    extra = {
        "patch_preset": args.patch_preset,
        "seed": seed,
        "adapter_rank": args.lora_moe_rank,
        "init_gate": args.scale_aware_init_gate,
        "train_cap": args.small_train_cap,
        "eval_cap": args.small_test_cap,
        "steps_per_epoch": args.steps_per_epoch,
    }
    append_baseline_rows(rows, "multi_seed", scale_data, extra=extra)

    raw_model = build_model("raw_patchtst", linear_args, scale_specs).to(device)
    apply_training_regime(raw_model, "raw_joint")
    raw_result = train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=raw_model,
        model_name="raw_joint",
        scale_data=scale_data,
        loaders=loaders,
        args=linear_args,
        device=device,
        output_dir=seed_output_dir,
        extra=extra,
    )
    raw_checkpoint = Path(raw_result["checkpoint"])

    if "post_return_lora_moe_head" in args.models:
        post_lora = build_model("lora_moe_patchtst", linear_args, scale_specs).to(device)
        load_raw_backbone_checkpoint(post_lora, raw_checkpoint)
        apply_training_regime(post_lora, "lora_moe_frozen_base_train_moe_head")
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=post_lora,
            model_name="post_return_lora_moe_head",
            scale_data=scale_data,
            loaders=loaders,
            args=linear_args,
            device=device,
            output_dir=seed_output_dir,
            extra={**extra, "architecture": "post_return_lora_moe_head"},
            router_balance_weight=args.router_balance_weight,
        )

    if "post_return_lora_moe_mlp_head" in args.models:
        post_lora_mlp = build_model("lora_moe_patchtst", mlp_args, scale_specs).to(device)
        load_raw_backbone_checkpoint(post_lora_mlp, raw_checkpoint)
        apply_training_regime(post_lora_mlp, "lora_moe_frozen_base_train_moe_head")
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=post_lora_mlp,
            model_name="post_return_lora_moe_mlp_head",
            scale_data=scale_data,
            loaders=loaders,
            args=mlp_args,
            device=device,
            output_dir=seed_output_dir,
            extra={**extra, "architecture": "post_return_lora_moe_mlp_head", "head_type": "mlp"},
            router_balance_weight=args.router_balance_weight,
        )

    if "gated_pre_return_asd_lora_moe_patchtst" in args.models:
        gated_pre_lora = build_preprocessed_model(
            linear_args,
            scale_specs,
            input_mode="return",
            adapter_kind="lora_moe",
            residual_to_raw=True,
            final_gate_init=-2.0,
        ).to(device)
        load_raw_backbone_checkpoint(gated_pre_lora, raw_checkpoint)
        apply_preprocessed_regime(gated_pre_lora)
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=gated_pre_lora,
            model_name="gated_pre_return_asd_lora_moe_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            args=linear_args,
            device=device,
            output_dir=seed_output_dir,
            extra={**extra, "architecture": "gated_pre_return_asd_lora_moe"},
            router_balance_weight=args.router_balance_weight,
        )

    if "gated_pre_return_asd_lora_moe_mlp_head" in args.models:
        gated_pre_lora_mlp = build_preprocessed_model(
            mlp_args,
            scale_specs,
            input_mode="return",
            adapter_kind="lora_moe",
            residual_to_raw=True,
            final_gate_init=-2.0,
        ).to(device)
        load_raw_backbone_checkpoint(gated_pre_lora_mlp, raw_checkpoint)
        apply_preprocessed_regime(gated_pre_lora_mlp)
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=gated_pre_lora_mlp,
            model_name="gated_pre_return_asd_lora_moe_mlp_head",
            scale_data=scale_data,
            loaders=loaders,
            args=mlp_args,
            device=device,
            output_dir=seed_output_dir,
            extra={**extra, "architecture": "gated_pre_return_asd_lora_moe_mlp_head", "head_type": "mlp"},
            router_balance_weight=args.router_balance_weight,
        )

    if "scale_specific_gated_pre_asd_moe_patchtst" in args.models:
        scale_specific = build_scale_specific_preprocessed_model(linear_args, scale_specs).to(device)
        load_raw_backbone_checkpoint(scale_specific, raw_checkpoint)
        apply_preprocessed_regime(scale_specific)
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=scale_specific,
            model_name="scale_specific_gated_pre_asd_moe_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            args=linear_args,
            device=device,
            output_dir=seed_output_dir,
            extra={**extra, "architecture": "scale_specific_gated_pre_asd_moe"},
            router_balance_weight=args.router_balance_weight,
        )

    summary = save_summary(rows, seed_output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    if not diagnostics.empty:
        diagnostics["seed"] = seed
    diagnostics.to_csv(seed_output_dir / "diagnostics.csv", index=False)
    return summary, diagnostics


def main() -> None:
    args = parse_args()
    if "day" in args.scales:
        raise ValueError("This runner intentionally supports only second/minute/hour.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    all_summary: list[pd.DataFrame] = []
    all_diagnostics: list[pd.DataFrame] = []
    for seed in args.seeds:
        print(f"=== running seed={seed} ===", flush=True)
        summary, diagnostics = run_seed(args, int(seed), device)
        all_summary.append(summary)
        all_diagnostics.append(diagnostics)

    summary_all = pd.concat(all_summary, ignore_index=True)
    diagnostics_all = pd.concat(all_diagnostics, ignore_index=True) if all_diagnostics else pd.DataFrame()
    aggregate = aggregate_metrics(summary_all)
    diagnostics_aggregate = aggregate_diagnostics(diagnostics_all)

    summary_all.to_csv(output_dir / "summary_all.csv", index=False)
    diagnostics_all.to_csv(output_dir / "diagnostics_all.csv", index=False)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    diagnostics_aggregate.to_csv(output_dir / "diagnostics_aggregate.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "seeds": [int(seed) for seed in args.seeds],
                "models": list(args.models),
                "train_stocks": args.train_stocks,
                "zero_shot_stock": int(args.zero_shot_stock),
                "patch_preset": args.patch_preset,
                "epochs": int(args.epochs),
                "steps_per_epoch": int(args.steps_per_epoch),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        path=Path(args.report_path),
        summary=summary_all,
        aggregate=aggregate,
        diagnostics_aggregate=diagnostics_aggregate,
        args=args,
    )
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
