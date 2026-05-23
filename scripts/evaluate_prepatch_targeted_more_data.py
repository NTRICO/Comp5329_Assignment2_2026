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
    diagnostic_rows,
    train_and_record,
)
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
    append_baseline_rows,
    apply_training_regime,
    build_multiscale_patchtst,
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
from src.baselines.scale_aware_asd_patchtst import RawMultiScalePatchTST, ScaleSpec  # noqa: E402


def build_mid_token_asd_lora_moe_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
) -> RawMultiScalePatchTST:
    backbone = build_multiscale_patchtst(
        scale_specs,
        input_channels=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        encoder_spectral_mode="last1",
        encoder_spectral_init_gate=args.encoder_spectral_init_gate,
        lora_moe_mode="last1",
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
    )
    return RawMultiScalePatchTST(backbone)


def build_internal_after1_asd_lora_moe_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
) -> RawMultiScalePatchTST:
    backbone = build_multiscale_patchtst(
        scale_specs,
        input_channels=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        encoder_spectral_mode="after1",
        encoder_spectral_init_gate=args.encoder_spectral_init_gate,
        lora_moe_mode="last1",
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
    )
    return RawMultiScalePatchTST(backbone)


def apply_mid_token_asd_lora_moe_regime(model: RawMultiScalePatchTST) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    if getattr(model.backbone, "encoder_spectral", None) is None:
        raise RuntimeError("mid-token ASD model requires backbone.encoder_spectral.")
    if getattr(model.backbone, "lora_moe", None) is None:
        raise RuntimeError("mid-token ASD model requires backbone.lora_moe.")
    for parameter in model.backbone.encoder_spectral.parameters():
        parameter.requires_grad = True
    for parameter in model.backbone.lora_moe.parameters():
        parameter.requires_grad = True
    for parameter in model.backbone.heads.parameters():
        parameter.requires_grad = True
    counts = count_parameters(model)
    return {"total_parameters": int(counts["total"]), "trainable_parameters": int(counts["trainable"])}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Targeted more-data run for pre-PatchTST adapter candidates.")
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
        default=str(WORKSPACE_ROOT / "outputs" / "prepatch_asd_adapter_patchtst" / "targeted_more_data"),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "prepatch_asd_adapter_targeted_more_data.md"),
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
    return parser.parse_args()


def write_targeted_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Pre-PatchTST Targeted More-Data Run")
    lines.append("")
    lines.append(
        "This run compares targeted architectures with larger sample caps and more balanced training steps. "
        "`mid_token_asd_lora_moe_patchtst` places token-level spectral denoising between PatchTST and LoRA-MoE. "
        "`internal_after1_asd_lora_moe_patchtst` places it after the first encoder layer, before later encoder layers."
    )
    lines.append("")
    lines.append(
        f"patch preset: `{args.patch_preset}`; layers={args.n_layers}; epochs={args.epochs}; "
        f"steps/epoch={args.steps_per_epoch}; "
        f"train cap={args.small_train_cap}; eval cap={args.small_test_cap}; seed={args.seed}."
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
    ranking = test[~test["model"].isin(["zero"])].sort_values(["scale", "nmse"]).groupby("scale").head(5)
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
            "asd_gate_mean",
            "asd_tau_mean",
            "asd_mean_abs_delta",
            "gate_mean",
            "tau_mean",
            "local_mask_mean",
            "pre_adapter_gate_mean",
            "pre_adapter_mean_abs_delta",
            "final_gate_mean",
            "final_mean_abs_delta",
            "router_entropy",
            "expert_prob_0",
            "expert_prob_1",
            "expert_prob_2",
            "expert_prob_3",
            "adapter_kind_id",
        ]
        lines.extend(frame_to_markdown(diagnostics[[column for column in keep_diag if column in diagnostics.columns]]))
    lines.append("")
    lines.append("## Files")
    lines.append("")
    out = Path(args.output_dir)
    lines.append(f"- summary: `{out / 'summary.csv'}`")
    lines.append(f"- diagnostics: `{out / 'diagnostics.csv'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if "day" in args.scales:
        raise ValueError("This runner intentionally supports only second/minute/hour.")
    set_seed(int(args.seed))
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
        "adapter_rank": args.lora_moe_rank,
        "init_gate": args.scale_aware_init_gate,
        "train_cap": args.small_train_cap,
        "eval_cap": args.small_test_cap,
        "steps_per_epoch": args.steps_per_epoch,
    }
    append_baseline_rows(rows, "more_data", scale_data, extra=extra)

    raw_model = build_model("raw_patchtst", args, scale_specs).to(device)
    apply_training_regime(raw_model, "raw_joint")
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
        extra=extra,
    )
    raw_checkpoint = Path(raw_result["checkpoint"])

    post_lora = build_model("lora_moe_patchtst", args, scale_specs).to(device)
    load_raw_backbone_checkpoint(post_lora, raw_checkpoint)
    apply_training_regime(post_lora, "lora_moe_frozen_base_train_moe_head")
    train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=post_lora,
        model_name="post_return_lora_moe_head",
        scale_data=scale_data,
        loaders=loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "post_return_lora_moe_head"},
        router_balance_weight=args.router_balance_weight,
    )

    mid_token = build_mid_token_asd_lora_moe_model(args, scale_specs).to(device)
    load_raw_backbone_checkpoint(mid_token, raw_checkpoint)
    freeze_info = apply_mid_token_asd_lora_moe_regime(mid_token)
    print(f"mid_token_asd_lora_moe_patchtst freeze_info={freeze_info}", flush=True)
    train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=mid_token,
        model_name="mid_token_asd_lora_moe_patchtst",
        scale_data=scale_data,
        loaders=loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "mid_token_asd_lora_moe"},
        router_balance_weight=args.router_balance_weight,
    )

    internal_after1 = build_internal_after1_asd_lora_moe_model(args, scale_specs).to(device)
    load_raw_backbone_checkpoint(internal_after1, raw_checkpoint)
    freeze_info = apply_mid_token_asd_lora_moe_regime(internal_after1)
    print(f"internal_after1_asd_lora_moe_patchtst freeze_info={freeze_info}", flush=True)
    train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=internal_after1,
        model_name="internal_after1_asd_lora_moe_patchtst",
        scale_data=scale_data,
        loaders=loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "internal_after1_asd_lora_moe"},
        router_balance_weight=args.router_balance_weight,
    )

    pre_lora = build_preprocessed_model(
        args,
        scale_specs,
        input_mode="return",
        adapter_kind="lora_moe",
        residual_to_raw=False,
    ).to(device)
    load_raw_backbone_checkpoint(pre_lora, raw_checkpoint)
    apply_preprocessed_regime(pre_lora)
    train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=pre_lora,
        model_name="pre_return_asd_lora_moe_patchtst",
        scale_data=scale_data,
        loaders=loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "pre_return_asd_lora_moe"},
        router_balance_weight=args.router_balance_weight,
    )

    gated_pre_lora = build_preprocessed_model(
        args,
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
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "gated_pre_return_asd_lora_moe"},
        router_balance_weight=args.router_balance_weight,
    )

    scale_specific = build_scale_specific_preprocessed_model(args, scale_specs).to(device)
    load_raw_backbone_checkpoint(scale_specific, raw_checkpoint)
    apply_preprocessed_regime(scale_specific)
    train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=scale_specific,
        model_name="scale_specific_gated_pre_asd_moe_patchtst",
        scale_data=scale_data,
        loaders=loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**extra, "architecture": "scale_specific_gated_pre_asd_moe"},
        router_balance_weight=args.router_balance_weight,
    )

    summary = save_summary(rows, output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diag_rows)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "cache": str(args.cache),
                "patch_preset": args.patch_preset,
                "scale_specs": {name: spec.__dict__ for name, spec in scale_specs.items()},
                "models": sorted(summary["model"].dropna().unique().tolist()),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_targeted_report(path=Path(args.report_path), summary=summary, diagnostics=diagnostics, args=args)
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
