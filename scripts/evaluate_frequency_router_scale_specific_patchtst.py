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

from evaluate_optiver_spectral_denoise_patchtst import select_device, set_seed  # noqa: E402
from evaluate_prepatch_asd_adapter_patchtst import diagnostic_rows  # noqa: E402
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
    append_baseline_rows,
    append_model_rows,
    apply_training_regime,
    build_model,
    caps_for_preset,
    frame_to_markdown,
    load_scale_data,
    make_all_loaders,
    make_scale_specs,
    save_summary,
    train_model,
)
from src.baselines.patchtst_lora import count_parameters  # noqa: E402
from src.baselines.scale_aware_asd_patchtst import (  # noqa: E402
    PaddedFrequencyRouterSharedASDPatchTST,
    PaddedFrequencyRouterScaleSpecificASDPatchTST,
    ScaleSpecificPatchTST,
    ScaleSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Small-data test for padded frequency-router scale-specific ASD PatchTST."
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
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "frequency_router_scale_specific_patchtst_small"),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "frequency_router_scale_specific_patchtst_small.md"),
    )
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
    parser.add_argument("--mlp-moe-bottleneck", type=int, default=8)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--router-top-k", type=int, default=2)
    parser.add_argument("--router-balance-weight", type=float, default=1e-3)
    parser.add_argument("--scale-router-prior", type=float, default=1.0)
    parser.add_argument("--no-identity-expert", action="store_true")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=[
            "shared_raw_patchtst",
            "scale_specific_raw_patchtst",
            "padded_freq_router_scale_specific_asd_patchtst",
            "padded_freq_router_shared_asd_patchtst",
            "padded_freq_router_shared_asd_mid_moe_patchtst",
        ],
        default=[
            "shared_raw_patchtst",
            "scale_specific_raw_patchtst",
            "padded_freq_router_scale_specific_asd_patchtst",
            "padded_freq_router_shared_asd_patchtst",
            "padded_freq_router_shared_asd_mid_moe_patchtst",
        ],
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
    return parser.parse_args()


def build_scale_specific_patchtst(args: argparse.Namespace, scale_specs: dict[str, ScaleSpec]) -> ScaleSpecificPatchTST:
    return ScaleSpecificPatchTST(
        scale_specs,
        input_channels=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
    )


def build_frequency_router_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
) -> PaddedFrequencyRouterScaleSpecificASDPatchTST:
    return PaddedFrequencyRouterScaleSpecificASDPatchTST(
        scale_specs,
        input_channels=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        init_gate=args.scale_aware_init_gate,
        top_k=args.router_top_k,
        scale_prior_strength=args.scale_router_prior,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
    )


def build_frequency_router_shared_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
    *,
    backbone_lora_moe_mode: str = "none",
) -> PaddedFrequencyRouterSharedASDPatchTST:
    return PaddedFrequencyRouterSharedASDPatchTST(
        scale_specs,
        input_channels=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        init_gate=args.scale_aware_init_gate,
        top_k=args.router_top_k,
        scale_prior_strength=args.scale_router_prior,
        include_identity_expert=not args.no_identity_expert,
        backbone_lora_moe_mode=backbone_lora_moe_mode,
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
    )


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


def write_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Padded Frequency-Router PatchTST Small Run")
    lines.append("")
    lines.append("本实验测试一个新的小数据架构：")
    lines.append("")
    lines.append(
        "`normalized input -> padding + mask -> FFT frequency embedding -> MoE router -> "
        "identity/scale-specific ASD experts -> crop -> PatchTST -> scale head`"
    )
    lines.append("")
    lines.append(
        "训练仍采用 mixed-frequency balanced step：每个 optimizer step 同时取 "
        "`second/minute/hour` 各一个 batch，并平均 loss。"
    )
    lines.append("")
    identity_status = "off" if args.no_identity_expert else "on"
    lines.append(
        f"cache: `{args.cache}`; patch preset: `{args.patch_preset}`; seed={args.seed}; "
        f"epochs={args.epochs}; steps/epoch={args.steps_per_epoch}; train cap={args.small_train_cap}; "
        f"eval cap={args.small_test_cap}; identity expert={identity_status}."
    )
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("")
    test = summary[(summary["split"] == "test") & (summary["model"] != "last_return")].copy()
    keep = ["model", "scale", "n", "nmse", "mse", "mae", "direction_accuracy_nonzero", "corr"]
    lines.extend(frame_to_markdown(test[[column for column in keep if column in test.columns]]))
    lines.append("")
    lines.append("## Test Ranking By Scale")
    lines.append("")
    ranking = (
        test[~test["model"].isin(["zero"])]
        .sort_values(["scale", "nmse"])
        .groupby("scale")
        .head(5)
    )
    lines.extend(frame_to_markdown(ranking[[column for column in keep if column in ranking.columns]]))
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("No diagnostics.")
    else:
        keep_diag = [
            "model",
            "scale",
            "router_entropy",
            "router_balance_loss",
            "asd_router_entropy",
            "asd_router_balance_loss",
            "mid_moe_router_entropy",
            "mid_moe_router_balance_loss",
            "backbone_mid_moe_router_entropy",
            "backbone_mid_moe_router_balance_loss",
            "frequency_embedding_norm",
            "mean_abs_delta",
            "asd_gate_mean",
            "asd_tau_mean",
            "expert_prob_0",
            "expert_prob_1",
            "expert_prob_2",
            "expert_prob_3",
            "expert_identity_prob",
            "expert_second_prob",
            "expert_minute_prob",
            "expert_hour_prob",
        ]
        lines.extend(frame_to_markdown(diagnostics[[column for column in keep_diag if column in diagnostics.columns]]))
    lines.append("")
    lines.append("## Files")
    lines.append("")
    out = Path(args.output_dir)
    lines.append(f"- summary: `{out / 'summary.csv'}`")
    lines.append(f"- diagnostics: `{out / 'diagnostics.csv'}`")
    lines.append(f"- aggregate: `{out / 'aggregate.csv'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if "day" in args.scales:
        raise ValueError("This runner intentionally supports only second/minute/hour.")
    set_seed(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    scale_specs = make_scale_specs(args)
    caps = caps_for_preset(args, "small")
    scale_data = load_scale_data(args, cache_path=Path(args.cache), caps=caps, scale_specs=scale_specs)
    loaders = make_all_loaders(scale_data, batch_size=args.batch_size, device=device)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    extra = {
        "patch_preset": args.patch_preset,
        "seed": args.seed,
        "train_cap": args.small_train_cap,
        "eval_cap": args.small_test_cap,
        "steps_per_epoch": args.steps_per_epoch,
        "router_top_k": args.router_top_k,
        "scale_router_prior": args.scale_router_prior,
    }
    append_baseline_rows(rows, "small", scale_data, extra=extra)
    requested_models = set(args.models)
    parameter_counts: dict[str, dict[str, int]] = {}

    if "shared_raw_patchtst" in requested_models:
        raw_shared = build_model("raw_patchtst", args, scale_specs).to(device)
        apply_training_regime(raw_shared, "raw_joint")
        shared_result = train_model(
            model=raw_shared,
            model_name="shared_raw_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=output_dir,
            checkpoint_name="shared_raw_patchtst",
        )
        append_model_rows(rows, "small", "shared_raw_patchtst", shared_result, extra=extra)
        diag_rows.extend(diagnostic_rows(shared_result, model_name="shared_raw_patchtst"))
        parameter_counts["shared_raw_patchtst"] = count_parameters(raw_shared)

    if "scale_specific_raw_patchtst" in requested_models:
        scale_specific = build_scale_specific_patchtst(args, scale_specs).to(device)
        scale_specific_result = train_model(
            model=scale_specific,
            model_name="scale_specific_raw_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=output_dir,
            checkpoint_name="scale_specific_raw_patchtst",
        )
        append_model_rows(rows, "small", "scale_specific_raw_patchtst", scale_specific_result, extra=extra)
        diag_rows.extend(diagnostic_rows(scale_specific_result, model_name="scale_specific_raw_patchtst"))
        parameter_counts["scale_specific_raw_patchtst"] = count_parameters(scale_specific)

    if "padded_freq_router_scale_specific_asd_patchtst" in requested_models:
        routed_model = build_frequency_router_model(args, scale_specs).to(device)
        routed_result = train_model(
            model=routed_model,
            model_name="padded_freq_router_scale_specific_asd_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=output_dir,
            checkpoint_name="padded_freq_router_scale_specific_asd_patchtst",
            router_balance_weight=args.router_balance_weight,
        )
        append_model_rows(
            rows,
            "small",
            "padded_freq_router_scale_specific_asd_patchtst",
            routed_result,
            extra=extra,
        )
        diag_rows.extend(
            diagnostic_rows(routed_result, model_name="padded_freq_router_scale_specific_asd_patchtst")
        )
        parameter_counts["padded_freq_router_scale_specific_asd_patchtst"] = count_parameters(routed_model)

    if "padded_freq_router_shared_asd_patchtst" in requested_models:
        routed_shared_model = build_frequency_router_shared_model(args, scale_specs).to(device)
        routed_shared_result = train_model(
            model=routed_shared_model,
            model_name="padded_freq_router_shared_asd_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=output_dir,
            checkpoint_name="padded_freq_router_shared_asd_patchtst",
            router_balance_weight=args.router_balance_weight,
        )
        append_model_rows(
            rows,
            "small",
            "padded_freq_router_shared_asd_patchtst",
            routed_shared_result,
            extra=extra,
        )
        diag_rows.extend(
            diagnostic_rows(routed_shared_result, model_name="padded_freq_router_shared_asd_patchtst")
        )
        parameter_counts["padded_freq_router_shared_asd_patchtst"] = count_parameters(routed_shared_model)

    if "padded_freq_router_shared_asd_mid_moe_patchtst" in requested_models:
        routed_shared_mid_moe_model = build_frequency_router_shared_model(
            args,
            scale_specs,
            backbone_lora_moe_mode="after1",
        ).to(device)
        routed_shared_mid_moe_result = train_model(
            model=routed_shared_mid_moe_model,
            model_name="padded_freq_router_shared_asd_mid_moe_patchtst",
            scale_data=scale_data,
            loaders=loaders,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=output_dir,
            checkpoint_name="padded_freq_router_shared_asd_mid_moe_patchtst",
            router_balance_weight=args.router_balance_weight,
        )
        append_model_rows(
            rows,
            "small",
            "padded_freq_router_shared_asd_mid_moe_patchtst",
            routed_shared_mid_moe_result,
            extra=extra,
        )
        diag_rows.extend(
            diagnostic_rows(routed_shared_mid_moe_result, model_name="padded_freq_router_shared_asd_mid_moe_patchtst")
        )
        parameter_counts["padded_freq_router_shared_asd_mid_moe_patchtst"] = count_parameters(
            routed_shared_mid_moe_model
        )

    summary = save_summary(rows, output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    aggregate = aggregate_metrics(summary)
    aggregate.to_csv(output_dir / "aggregate.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "patch_preset": args.patch_preset,
                "scale_specs": {name: spec.__dict__ for name, spec in scale_specs.items()},
                "models": sorted(summary["model"].dropna().unique().tolist()),
                "parameters": parameter_counts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(path=Path(args.report_path), summary=summary, diagnostics=diagnostics, args=args)
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
