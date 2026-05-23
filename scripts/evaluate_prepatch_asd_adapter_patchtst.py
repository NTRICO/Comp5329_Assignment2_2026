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

from evaluate_level_asd_patchtst import build_level_scale_data  # noqa: E402
from evaluate_optiver_spectral_denoise_patchtst import select_device, set_seed  # noqa: E402
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
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
from src.baselines.patchtst_lora import count_parameters  # noqa: E402
from src.baselines.scale_aware_asd_patchtst import (  # noqa: E402
    PreprocessedASDAdapterPatchTST,
    ScaleSpecificGatedPreprocessedASDAdapterPatchTST,
    ScaleSpec,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate pre-PatchTST ASD+LoRA-MoE/MLP-MoE adapters on return and price inputs."
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
        default=str(WORKSPACE_ROOT / "outputs" / "prepatch_asd_adapter_patchtst_true_hour_60_45_24"),
    )
    parser.add_argument(
        "--report-path",
        default=str(WORKSPACE_ROOT / "report" / "prepatch_asd_adapter_patchtst_true_hour_60_45_24.md"),
    )
    parser.add_argument("--train-stocks", default="0,1,2,3,4,5,6,7,8")
    parser.add_argument("--zero-shot-stock", type=int, default=9)
    parser.add_argument("--scales", nargs="+", choices=SCALE_ORDER, default=list(SCALE_ORDER))
    parser.add_argument("--patch-preset", choices=sorted(PATCH_PRESETS), default="balanced_60_45_24")
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
    parser.add_argument("--router-balance-weight", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
        parser.add_argument(f"--{scale}-target-horizon-steps", type=int, default=None)
    return parser.parse_args()


def build_preprocessed_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
    *,
    input_mode: str,
    price_mode: str = "log_price",
    adapter_kind: str,
    residual_to_raw: bool = False,
    final_gate_init: float = -2.0,
) -> PreprocessedASDAdapterPatchTST:
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
        head_type=getattr(args, "head_type", "linear"),
        head_hidden_dim=getattr(args, "head_hidden_dim", 128),
    )
    return PreprocessedASDAdapterPatchTST(
        backbone,
        input_mode=input_mode,
        price_mode=price_mode,
        adapter_kind=adapter_kind,
        init_gate=args.scale_aware_init_gate,
        n_experts=args.lora_moe_n_experts,
        rank=args.lora_moe_rank,
        alpha=args.lora_moe_alpha,
        bottleneck=args.mlp_moe_bottleneck,
        top_k=args.lora_moe_top_k,
        dropout=args.lora_moe_dropout,
        residual_to_raw=residual_to_raw,
        final_gate_init=final_gate_init,
    )


def build_scale_specific_preprocessed_model(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
) -> ScaleSpecificGatedPreprocessedASDAdapterPatchTST:
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
        head_type=getattr(args, "head_type", "linear"),
        head_hidden_dim=getattr(args, "head_hidden_dim", 128),
    )
    return ScaleSpecificGatedPreprocessedASDAdapterPatchTST(
        backbone,
        init_gate=args.scale_aware_init_gate,
        n_experts=args.lora_moe_n_experts,
        rank=args.lora_moe_rank,
        alpha=args.lora_moe_alpha,
        bottleneck=args.mlp_moe_bottleneck,
        top_k=args.lora_moe_top_k,
        dropout=args.lora_moe_dropout,
        scale_adapter_kind={"second": "lora_moe", "minute": "mlp_moe", "hour": "lora_moe"},
        scale_gate_init={"second": -6.0, "minute": -2.5, "hour": -1.5},
    )


def apply_preprocessed_regime(
    model: PreprocessedASDAdapterPatchTST | ScaleSpecificGatedPreprocessedASDAdapterPatchTST,
) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.denoiser.parameters():
        parameter.requires_grad = True
    if hasattr(model, "pre_adapter"):
        for parameter in model.pre_adapter.parameters():
            parameter.requires_grad = True
    if hasattr(model, "adapters"):
        for parameter in model.adapters.parameters():
            parameter.requires_grad = True
    if hasattr(model, "scale_gate_bias"):
        for parameter in model.scale_gate_bias.parameters():
            parameter.requires_grad = True
    if hasattr(model, "final_gate_projection"):
        for parameter in model.final_gate_projection.parameters():
            parameter.requires_grad = True
    for parameter in model.backbone.heads.parameters():
        parameter.requires_grad = True
    counts = count_parameters(model)
    return {"total_parameters": int(counts["total"]), "trainable_parameters": int(counts["trainable"])}


def set_return_stats(model: PreprocessedASDAdapterPatchTST, return_scale_data: dict[str, ScaleData]) -> None:
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


def train_and_record(
    *,
    rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
    model: torch.nn.Module,
    model_name: str,
    scale_data: dict[str, ScaleData],
    loaders: dict[str, dict[str, torch.utils.data.DataLoader]],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
    extra: dict[str, Any],
    router_balance_weight: float = 0.0,
) -> dict[str, Any]:
    result = train_model(
        model=model,
        model_name=model_name,
        scale_data=scale_data,
        loaders=loaders,
        epochs=args.epochs,
        steps_per_epoch=args.steps_per_epoch,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=output_dir,
        checkpoint_name=model_name,
        router_balance_weight=router_balance_weight,
    )
    append_model_rows(rows, "small", model_name, result, extra=extra)
    diag_rows.extend(diagnostic_rows(result, model_name=model_name))
    return result


def write_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# Pre-PatchTST ASD Adapter Experiment")
    lines.append("")
    lines.append(
        "This small experiment tests whether a learnable front-end can clean/adapt price or return sequences before "
        "the frozen PatchTST backbone sees returns. The front-end is ASD plus either LoRA-MoE or MLP-MoE."
    )
    lines.append("")
    lines.append(
        f"cache: `{args.cache}`; patch preset: `{args.patch_preset}`; epochs={args.epochs}; "
        f"balanced steps/epoch={args.steps_per_epoch}; rank={args.lora_moe_rank}; init_gate={args.scale_aware_init_gate}."
    )
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("")
    test = summary[(summary["split"] == "test") & (summary["model"] != "last_return")].copy()
    keep = ["model", "scale", "n", "nmse", "mse", "mae", "direction_accuracy_nonzero", "corr"]
    lines.extend(frame_to_markdown(test[[column for column in keep if column in test.columns]]))
    lines.append("")
    lines.append("## Test NMSE Relative To Raw PatchTST")
    lines.append("")
    rel_rows: list[dict[str, Any]] = []
    raw_rows = test[test["model"] == "raw_joint"].set_index("scale")
    for _, row in test.iterrows():
        scale = row["scale"]
        if row["model"] in {"zero", "raw_joint"} or scale not in raw_rows.index:
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
    lines.append("## Front-End Diagnostics")
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
            "pre_adapter_gate_mean",
            "pre_adapter_mean_abs_delta",
            "router_entropy",
            "router_balance_loss",
            "expert_prob_0",
            "expert_prob_1",
            "expert_prob_2",
            "expert_prob_3",
            "patch_input_abs_mean",
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
    if "day" in args.scales:
        raise ValueError("This experiment intentionally supports only second/minute/hour.")
    parse_stock_list(args.train_stocks)
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
    base_extra = {
        "patch_preset": args.patch_preset,
        "seed": args.seed,
        "adapter_rank": args.lora_moe_rank,
        "init_gate": args.scale_aware_init_gate,
    }
    append_baseline_rows(rows, "small", return_scale_data, extra=base_extra)

    raw_model = build_model("raw_patchtst", args, scale_specs).to(device)
    apply_training_regime(raw_model, "raw_joint")
    raw_result = train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=raw_model,
        model_name="raw_joint",
        scale_data=return_scale_data,
        loaders=return_loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra=base_extra,
    )
    raw_checkpoint = Path(raw_result["checkpoint"])

    post_lora_model = build_model("lora_moe_patchtst", args, scale_specs).to(device)
    load_raw_backbone_checkpoint(post_lora_model, raw_checkpoint)
    apply_training_regime(post_lora_model, "lora_moe_frozen_base_train_moe_head")
    train_and_record(
        rows=rows,
        diag_rows=diag_rows,
        model=post_lora_model,
        model_name="post_return_lora_moe_head",
        scale_data=return_scale_data,
        loaders=return_loaders,
        args=args,
        device=device,
        output_dir=output_dir,
        extra={**base_extra, "adapter_position": "post_patch", "adapter_kind": "lora_moe"},
        router_balance_weight=args.router_balance_weight,
    )

    for adapter_kind in ["lora_moe", "mlp_moe"]:
        model = build_preprocessed_model(
            args,
            scale_specs,
            input_mode="return",
            adapter_kind=adapter_kind,
        ).to(device)
        load_raw_backbone_checkpoint(model, raw_checkpoint)
        freeze_info = apply_preprocessed_regime(model)
        model_name = f"pre_return_asd_{adapter_kind}_patchtst"
        print(f"{model_name} freeze_info={freeze_info}", flush=True)
        train_and_record(
            rows=rows,
            diag_rows=diag_rows,
            model=model,
            model_name=model_name,
            scale_data=return_scale_data,
            loaders=return_loaders,
            args=args,
            device=device,
            output_dir=output_dir,
            extra={**base_extra, "adapter_position": "pre_patch", "adapter_kind": adapter_kind, "input_mode": "return"},
            router_balance_weight=args.router_balance_weight,
        )

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
        for adapter_kind in ["lora_moe", "mlp_moe"]:
            model = build_preprocessed_model(
                args,
                scale_specs,
                input_mode="level",
                price_mode=price_mode,
                adapter_kind=adapter_kind,
            ).to(device)
            set_return_stats(model, return_scale_data)
            load_raw_backbone_checkpoint(model, raw_checkpoint)
            freeze_info = apply_preprocessed_regime(model)
            model_name = f"pre_{price_mode}_asd_{adapter_kind}_to_return_patchtst"
            print(f"{model_name} freeze_info={freeze_info}", flush=True)
            train_and_record(
                rows=rows,
                diag_rows=diag_rows,
                model=model,
                model_name=model_name,
                scale_data=level_scale_data,
                loaders=level_loaders,
                args=args,
                device=device,
                output_dir=output_dir,
                extra={
                    **base_extra,
                    "adapter_position": "pre_patch",
                    "adapter_kind": adapter_kind,
                    "input_mode": "level",
                    "price_mode": price_mode,
                },
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
    write_report(path=Path(args.report_path), summary=summary, diagnostics=diagnostics, args=args)
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
