from __future__ import annotations

import argparse
import json
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

from evaluate_level_asd_patchtst import build_level_arrays  # noqa: E402
from evaluate_optiver_spectral_denoise_patchtst import fit_normalizer, select_device, set_seed  # noqa: E402
from evaluate_prepatch_asd_adapter_patchtst import (  # noqa: E402
    apply_preprocessed_regime,
    build_preprocessed_model,
    diagnostic_rows,
    train_and_record,
)
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
    ScaleData,
    append_baseline_rows,
    apply_training_regime,
    build_model,
    caps_for_preset,
    frame_to_markdown,
    load_raw_backbone_checkpoint,
    make_all_loaders,
    make_scale_specs,
    save_summary,
)


DEFAULT_TRAIN_STOCKS = ",".join(
    str(stock)
    for stock in [
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
        description="Test direct raw/log price inputs for return forecasting with PatchTST variants."
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
        default=str(WORKSPACE_ROOT / "outputs" / "direct_price_input_patchtst_32stock_multiseed"),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "direct_price_input_patchtst_32stock_multiseed.md"),
    )
    parser.add_argument("--train-stocks", default=DEFAULT_TRAIN_STOCKS)
    parser.add_argument("--zero-shot-stock", type=int, default=34)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--price-modes", nargs="+", choices=["log_price", "raw_price"], default=["log_price", "raw_price"])
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
    parser.add_argument("--scale-aware-init-gate", type=float, default=-4.0)
    parser.add_argument("--encoder-spectral-mode", choices=["none", "last1"], default="none")
    parser.add_argument("--encoder-spectral-init-gate", type=float, default=-4.0)
    parser.add_argument("--lora-moe-rank", type=int, default=8)
    parser.add_argument("--lora-moe-alpha", type=float, default=16.0)
    parser.add_argument("--lora-moe-n-experts", type=int, default=4)
    parser.add_argument("--lora-moe-top-k", type=int, default=2)
    parser.add_argument("--lora-moe-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-moe-bottleneck", type=int, default=8)
    parser.add_argument("--router-balance-weight", type=float, default=1e-3)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "direct_price_raw_patchtst",
            "direct_price_lora_moe_head",
            "direct_price_gated_pre_asd_lora_moe",
        ],
        default=[
            "direct_price_raw_patchtst",
            "direct_price_lora_moe_head",
            "direct_price_gated_pre_asd_lora_moe",
        ],
    )
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
    return parser.parse_args()


def build_direct_price_scale_data(
    args: argparse.Namespace,
    *,
    price_mode: str,
    seed: int,
) -> dict[str, ScaleData]:
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = seed
    scale_specs = make_scale_specs(seed_args)
    caps = caps_for_preset(seed_args, "small")
    raw_arrays = build_level_arrays(
        cache_path=Path(args.cache),
        scales=list(args.scales),
        scale_specs=scale_specs,
        train_stocks=[int(item) for item in str(args.train_stocks).split(",") if item.strip()],
        zero_shot_stock=int(args.zero_shot_stock),
        price_mode=price_mode,
        caps=caps,
        seed=seed,
    )
    out: dict[str, ScaleData] = {}
    for scale in args.scales:
        arrays = raw_arrays[scale]
        normalizer = fit_normalizer(arrays["train_x"], arrays["train_y"])
        out[scale] = ScaleData(name=scale, spec=scale_specs[scale], arrays=arrays, normalizer=normalizer)
        print(
            f"{price_mode}/{scale}: train={len(arrays['train_y'])} "
            f"validation={len(arrays['validation_y'])} test={len(arrays['test_y'])} "
            f"zero_shot={len(arrays['zero_shot_y'])}",
            flush=True,
        )
    return out


def aggregate_metrics(summary: pd.DataFrame) -> pd.DataFrame:
    metric_columns = ["n", "nmse", "mse", "mae", "direction_accuracy_nonzero", "corr"]
    frame = summary.copy()
    for column in metric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.groupby(["price_mode", "model", "split", "scale"], dropna=False)[metric_columns]
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
        if column not in {"price_mode", "model", "scale", "seed"}
    ]
    frame = diagnostics.copy()
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    grouped = (
        frame.groupby(["price_mode", "model", "scale"], dropna=False)[numeric_columns]
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
    lines.append("# Direct Price Input PatchTST Experiment")
    lines.append("")
    lines.append(
        "This experiment feeds `raw_price` or `log_price` windows directly into PatchTST variants "
        "and keeps the target as future return. Inputs are normalized using train-split level statistics; "
        "targets use return statistics. No day data is included."
    )
    lines.append("")
    lines.append(
        f"cache: `{args.cache}`; price modes: `{', '.join(args.price_modes)}`; "
        f"seeds: `{', '.join(map(str, args.seeds))}`; patch preset: `{args.patch_preset}`; "
        f"epochs={args.epochs}; steps/epoch={args.steps_per_epoch}; train cap={args.small_train_cap}."
    )
    lines.append("")
    lines.append("## Test Mean / Std")
    lines.append("")
    test = aggregate[aggregate["split"] == "test"].copy()
    keep = [
        "price_mode",
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
    seed_keep = ["seed", "price_mode", "model", "scale", "n", "nmse", "mae", "direction_accuracy_nonzero", "corr"]
    lines.extend(frame_to_markdown(seed_test[[column for column in seed_keep if column in seed_test.columns]]))
    lines.append("")
    lines.append("## Diagnostics Mean / Std")
    lines.append("")
    if diagnostics_aggregate.empty:
        lines.append("No diagnostics.")
    else:
        diag_keep = [
            "price_mode",
            "model",
            "scale",
            "asd_gate_mean_mean",
            "final_gate_mean_mean",
            "final_mean_abs_delta_mean",
            "router_entropy_mean",
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
    lines.append(f"- all summary: `{out / 'summary_all.csv'}`")
    lines.append(f"- aggregate: `{out / 'aggregate.csv'}`")
    lines.append(f"- diagnostics: `{out / 'diagnostics_all.csv'}`")
    lines.append(f"- diagnostics aggregate: `{out / 'diagnostics_aggregate.csv'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_seed_price_mode(
    args: argparse.Namespace,
    *,
    seed: int,
    price_mode: str,
    device: torch.device,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    set_seed(seed)
    seed_args = argparse.Namespace(**vars(args))
    seed_args.seed = seed
    scale_specs = make_scale_specs(seed_args)
    seed_output_dir = Path(args.output_dir) / f"{price_mode}_seed_{seed}"
    seed_output_dir.mkdir(parents=True, exist_ok=True)
    scale_data = build_direct_price_scale_data(args, price_mode=price_mode, seed=seed)
    loaders = make_all_loaders(scale_data, batch_size=args.batch_size, device=device)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    extra = {
        "price_mode": price_mode,
        "patch_preset": args.patch_preset,
        "seed": seed,
        "adapter_rank": args.lora_moe_rank,
        "init_gate": args.scale_aware_init_gate,
        "train_cap": args.small_train_cap,
        "eval_cap": args.small_test_cap,
        "steps_per_epoch": args.steps_per_epoch,
        "input_kind": "direct_price",
    }
    append_baseline_rows(rows, "direct_price", scale_data, extra=extra)

    raw_checkpoint: Path | None = None
    if "direct_price_raw_patchtst" in args.models:
        raw_model = build_model("raw_patchtst", seed_args, scale_specs).to(device)
        apply_training_regime(raw_model, "raw_joint")
        raw_result = train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=raw_model,
            model_name="direct_price_raw_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            args=seed_args,
            device=device,
            output_dir=seed_output_dir,
            extra=extra,
        )
        raw_checkpoint = Path(raw_result["checkpoint"])

    if raw_checkpoint is None:
        raw_model = build_model("raw_patchtst", seed_args, scale_specs).to(device)
        apply_training_regime(raw_model, "raw_joint")
        raw_result = train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=raw_model,
            model_name="direct_price_raw_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            args=seed_args,
            device=device,
            output_dir=seed_output_dir,
            extra=extra,
        )
        raw_checkpoint = Path(raw_result["checkpoint"])

    if "direct_price_lora_moe_head" in args.models:
        lora_model = build_model("lora_moe_patchtst", seed_args, scale_specs).to(device)
        load_raw_backbone_checkpoint(lora_model, raw_checkpoint)
        apply_training_regime(lora_model, "lora_moe_frozen_base_train_moe_head")
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=lora_model,
            model_name="direct_price_lora_moe_head",
            scale_data=scale_data,
            loaders=loaders,
            args=seed_args,
            device=device,
            output_dir=seed_output_dir,
            extra={**extra, "architecture": "direct_price_lora_moe_head"},
            router_balance_weight=args.router_balance_weight,
        )

    if "direct_price_gated_pre_asd_lora_moe" in args.models:
        gated_model = build_preprocessed_model(
            seed_args,
            scale_specs,
            input_mode="return",
            adapter_kind="lora_moe",
            residual_to_raw=True,
            final_gate_init=-2.0,
        ).to(device)
        load_raw_backbone_checkpoint(gated_model, raw_checkpoint)
        apply_preprocessed_regime(gated_model)
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=gated_model,
            model_name="direct_price_gated_pre_asd_lora_moe",
            scale_data=scale_data,
            loaders=loaders,
            args=seed_args,
            device=device,
            output_dir=seed_output_dir,
            extra={**extra, "architecture": "direct_price_gated_pre_asd_lora_moe"},
            router_balance_weight=args.router_balance_weight,
        )

    summary = save_summary(rows, seed_output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    if not diagnostics.empty:
        diagnostics["seed"] = seed
        diagnostics["price_mode"] = price_mode
    diagnostics.to_csv(seed_output_dir / "diagnostics.csv", index=False)
    return summary, diagnostics


def main() -> None:
    args = parse_args()
    if "day" in args.scales:
        raise ValueError("This runner intentionally supports only second/minute/hour.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)

    summaries: list[pd.DataFrame] = []
    diagnostics_frames: list[pd.DataFrame] = []
    for seed in args.seeds:
        for price_mode in args.price_modes:
            print(f"=== running price_mode={price_mode} seed={seed} ===", flush=True)
            summary, diagnostics = run_seed_price_mode(args, seed=int(seed), price_mode=price_mode, device=device)
            summaries.append(summary)
            diagnostics_frames.append(diagnostics)

    summary_all = pd.concat(summaries, ignore_index=True)
    diagnostics_all = (
        pd.concat(diagnostics_frames, ignore_index=True) if diagnostics_frames else pd.DataFrame()
    )
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
                "price_modes": list(args.price_modes),
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
