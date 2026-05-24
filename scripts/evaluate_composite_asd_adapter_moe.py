from __future__ import annotations

import argparse
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
from evaluate_prepatch_asd_adapter_patchtst import train_and_record  # noqa: E402
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
from src.baselines.patchtst_lora import count_parameters  # noqa: E402
from src.baselines.scale_aware_asd_patchtst import (  # noqa: E402
    RoutedCompositeASDAdapterPatchTST,
    ScaleSpec,
    build_multiscale_patchtst,
)


DEFAULT_TRAIN_STOCKS = ",".join(str(stock) for stock in range(9))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Quick ablation for routed composite ASD+LoRA/MLP experts before PatchTST."
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
        default=str(WORKSPACE_ROOT / "outputs" / "composite_asd_adapter_moe_quick"),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "composite_asd_adapter_moe_quick.md"),
    )
    parser.add_argument("--train-stocks", default=DEFAULT_TRAIN_STOCKS)
    parser.add_argument("--zero-shot-stock", type=int, default=9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--scales", nargs="+", choices=SCALE_ORDER, default=list(SCALE_ORDER))
    parser.add_argument("--patch-preset", choices=sorted(PATCH_PRESETS), default="balanced_60_45_24")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps-per-epoch", type=int, default=12)
    parser.add_argument("--data-preset", choices=["small", "full"], default="small")
    parser.add_argument("--small-train-cap", type=int, default=4096)
    parser.add_argument("--small-validation-cap", type=int, default=1024)
    parser.add_argument("--small-test-cap", type=int, default=1024)
    parser.add_argument("--small-zero-shot-cap", type=int, default=1024)
    parser.add_argument("--full-train-cap", type=int, default=0)
    parser.add_argument("--full-validation-cap", type=int, default=0)
    parser.add_argument("--full-test-cap", type=int, default=0)
    parser.add_argument("--full-zero-shot-cap", type=int, default=0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=0.0)
    parser.add_argument("--scale-aware-init-gate", type=float, default=-4.0)
    parser.add_argument("--encoder-spectral-mode", choices=["none", "last1"], default="none")
    parser.add_argument("--encoder-spectral-init-gate", type=float, default=-4.0)
    parser.add_argument("--lora-moe-rank", type=int, default=8)
    parser.add_argument("--lora-moe-alpha", type=float, default=16.0)
    parser.add_argument("--lora-moe-n-experts", type=int, default=4)
    parser.add_argument("--lora-moe-top-k", type=int, default=2)
    parser.add_argument("--lora-moe-dropout", type=float, default=0.1)
    parser.add_argument("--mlp-moe-bottleneck", type=int, default=8)
    parser.add_argument(
        "--expert-pattern",
        choices=["alternating", "lora_first", "all_lora", "all_mlp", "one_mlp", "three_mlp"],
        default="one_mlp",
    )
    parser.add_argument("--final-gate-init", type=float, default=-2.0)
    parser.add_argument("--router-balance-weight", type=float, default=1e-3)
    parser.add_argument("--head-type", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--head-hidden-dim", type=int, default=128)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
        target_default = 10 if scale == "second" else 1
        parser.add_argument(f"--{scale}-target-horizon-steps", type=int, default=target_default)
    return parser.parse_args()


def build_composite_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
) -> RoutedCompositeASDAdapterPatchTST:
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
        lora_moe_mode="none",
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
        head_type=args.head_type,
        head_hidden_dim=args.head_hidden_dim,
    )
    return RoutedCompositeASDAdapterPatchTST(
        backbone,
        init_gate=args.scale_aware_init_gate,
        n_experts=args.lora_moe_n_experts,
        rank=args.lora_moe_rank,
        alpha=args.lora_moe_alpha,
        bottleneck=args.mlp_moe_bottleneck,
        top_k=args.lora_moe_top_k,
        dropout=args.lora_moe_dropout,
        expert_pattern=args.expert_pattern,
        final_gate_init=args.final_gate_init,
    )


def apply_composite_regime(model: RoutedCompositeASDAdapterPatchTST) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.composite_moe.parameters():
        parameter.requires_grad = True
    for parameter in model.final_gate_projection.parameters():
        parameter.requires_grad = True
    for parameter in model.backbone.heads.parameters():
        parameter.requires_grad = True
    counts = count_parameters(model)
    return {key: int(value) for key, value in counts.items()}


def write_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
    parameter_counts: dict[str, dict[str, int]],
) -> None:
    lines: list[str] = []
    lines.append("# Routed Composite ASD-Adapter MoE Quick Ablation")
    lines.append("")
    lines.append(
        "This quick run tests a routed composite expert design: each expert owns its own ASD module "
        "and either a LoRA or MLP value enhancer. The router operates per return position with scale "
        "and position information, then a final residual gate mixes the adapted window back with raw returns."
    )
    lines.append("")
    lines.append(
        f"cache: `{args.cache}`; patch preset: `{args.patch_preset}`; seed={args.seed}; "
        f"epochs={args.epochs}; steps/epoch={args.steps_per_epoch}; "
        f"train cap={args.small_train_cap}; eval cap={args.small_test_cap}."
    )
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("")
    test_rows = summary[summary["split"] == "test"].copy()
    keep = [
        "model",
        "scale",
        "n",
        "nmse",
        "mse",
        "mae",
        "direction_accuracy_nonzero",
        "corr",
        "architecture",
    ]
    lines.extend(frame_to_markdown(test_rows[[column for column in keep if column in test_rows.columns]]))
    lines.append("")
    if not diagnostics.empty:
        lines.append("## Diagnostics")
        lines.append("")
        diag_keep = [
            "model",
            "scale",
            "router_entropy",
            "router_balance_loss",
            "final_gate_mean",
            "final_mean_abs_delta",
            "composite_mean_abs_delta",
            "expert_prob_0",
            "expert_prob_1",
            "expert_prob_2",
            "expert_prob_3",
        ]
        lines.extend(frame_to_markdown(diagnostics[[column for column in diag_keep if column in diagnostics.columns]]))
        lines.append("")
    lines.append("## Parameter Counts")
    lines.append("")
    param_frame = pd.DataFrame(
        [
            {
                "model": model_name,
                "total": int(counts.get("total", 0)),
                "trainable": int(counts.get("trainable", 0)),
            }
            for model_name, counts in parameter_counts.items()
        ]
    )
    lines.extend(frame_to_markdown(param_frame))
    lines.append("")
    out = Path(args.output_dir)
    lines.append("## Files")
    lines.append("")
    lines.append(f"- summary: `{out / 'summary.csv'}`")
    lines.append(f"- diagnostics: `{out / 'diagnostics.csv'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    scale_specs = make_scale_specs(args)
    caps = caps_for_preset(args, args.data_preset)
    scale_data = load_scale_data(args, cache_path=Path(args.cache), caps=caps, scale_specs=scale_specs)
    loaders = make_all_loaders(scale_data, batch_size=args.batch_size, device=device)

    rows: list[dict[str, Any]] = []
    diag_rows: list[dict[str, Any]] = []
    extra = {
        "patch_preset": args.patch_preset,
        "seed": args.seed,
        "adapter_rank": args.lora_moe_rank,
        "n_experts": args.lora_moe_n_experts,
        "top_k": args.lora_moe_top_k,
        "expert_pattern": args.expert_pattern,
        "architecture": "baseline",
        "steps_per_epoch": args.steps_per_epoch,
    }
    append_baseline_rows(rows, "quick", scale_data, extra=extra)

    raw_model = build_model("raw_patchtst", args, scale_specs).to(device)
    apply_training_regime(raw_model, "raw_joint")
    parameter_counts: dict[str, dict[str, int]] = {
        "raw_joint": {key: int(value) for key, value in count_parameters(raw_model).items()}
    }
    raw_result = train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=raw_model,
        model_name="raw_joint",
        scale_data=scale_data,
        loaders=loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "raw_patchtst"},
    )
    raw_checkpoint = Path(raw_result["checkpoint"])

    composite = build_composite_model(args, scale_specs).to(device)
    load_raw_backbone_checkpoint(composite, raw_checkpoint)
    freeze_info = apply_composite_regime(composite)
    print(f"routed_composite_asd_adapter_moe_patchtst freeze_info={freeze_info}", flush=True)
    parameter_counts["routed_composite_asd_adapter_moe_patchtst"] = freeze_info
    train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=composite,
        model_name="routed_composite_asd_adapter_moe_patchtst",
        scale_data=scale_data,
        loaders=loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "routed_composite_asd_adapter_moe"},
        router_balance_weight=args.router_balance_weight,
    )

    summary = save_summary(rows, output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    write_report(
        path=Path(args.report_path),
        summary=summary,
        diagnostics=diagnostics,
        args=args,
        parameter_counts=parameter_counts,
    )
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
