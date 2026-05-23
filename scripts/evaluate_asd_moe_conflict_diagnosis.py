from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_optiver_spectral_denoise_patchtst import select_device, set_seed  # noqa: E402
from evaluate_scale_aware_asd_patchtst import (  # noqa: E402
    PATCH_PRESETS,
    SCALE_ORDER,
    ScaleData,
    append_baseline_rows,
    append_model_rows,
    caps_for_preset,
    collect_diagnostics,
    evaluate_all_scales,
    frame_to_markdown,
    load_raw_backbone_checkpoint,
    load_scale_data,
    make_all_loaders,
    make_scale_specs,
    next_batch,
    resolve_steps_per_epoch,
    save_summary,
)
from src.baselines.patchtst_lora import count_parameters  # noqa: E402
from src.baselines.scale_aware_asd_patchtst import (  # noqa: E402
    RawMultiScalePatchTST,
    ScaleAwareASDMultiScalePatchTST,
    ScaleSpec,
    SideASDFeatureMultiScalePatchTST,
    build_multiscale_patchtst,
)


DIAGNOSIS_MODELS = (
    "lora_moe_head",
    "asd_lora_moe",
    "asd_lora_moe_router_frozen",
    "asd_lora_moe_router_kl",
    "asd_side_feature_lora_moe",
    "patch_token_asd_lora_moe",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose ASD and LoRA-MoE interaction conflicts.")
    parser.add_argument("--mode", default="asd_moe_conflict_diagnosis")
    parser.add_argument(
        "--cache",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "cache"
            / "position_optiver_hf_second_feature_cache_11stocks_512t.npz"
        ),
    )
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "asd_moe_conflict_diagnosis"))
    parser.add_argument("--report-path", default=str(WORKSPACE_ROOT / "report" / "asd_moe_conflict_diagnosis.md"))
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
    parser.add_argument("--small-epochs", type=int, default=3)
    parser.add_argument("--small-steps-per-epoch", type=int, default=12)
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
    parser.add_argument("--router-kl-weight", type=float, default=1e-2)
    parser.add_argument("--scale-aware-init-gate", type=float, default=-4.0)
    parser.add_argument("--encoder-spectral-init-gate", type=float, default=-4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
    return parser.parse_args()


def build_backbone(
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
    *,
    input_channels: int = 1,
    target_mode: str = "per_channel",
    lora_moe_mode: str = "last1",
    encoder_spectral_mode: str = "none",
) -> torch.nn.Module:
    return build_multiscale_patchtst(
        scale_specs,
        input_channels=input_channels,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        encoder_spectral_mode=encoder_spectral_mode,
        encoder_spectral_init_gate=args.encoder_spectral_init_gate,
        lora_moe_mode=lora_moe_mode,
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
        target_mode=target_mode,
    )


def build_model(model_key: str, args: argparse.Namespace, scale_specs: dict[str, ScaleSpec]) -> torch.nn.Module:
    if model_key == "raw_joint":
        return RawMultiScalePatchTST(build_backbone(args, scale_specs, lora_moe_mode="none"))
    if model_key == "lora_moe_head":
        return RawMultiScalePatchTST(build_backbone(args, scale_specs, lora_moe_mode="last1"))
    if model_key in {"asd_lora_moe", "asd_lora_moe_router_frozen", "asd_lora_moe_router_kl"}:
        return ScaleAwareASDMultiScalePatchTST(
            build_backbone(args, scale_specs, lora_moe_mode="last1"),
            init_gate=args.scale_aware_init_gate,
        )
    if model_key == "asd_side_feature_lora_moe":
        return SideASDFeatureMultiScalePatchTST(
            build_backbone(
                args,
                scale_specs,
                input_channels=2,
                target_mode="all_channels",
                lora_moe_mode="last1",
            ),
            init_gate=args.scale_aware_init_gate,
        )
    if model_key == "patch_token_asd_lora_moe":
        return RawMultiScalePatchTST(
            build_backbone(
                args,
                scale_specs,
                lora_moe_mode="last1",
                encoder_spectral_mode="last1",
            )
        )
    raise ValueError(f"Unknown model_key={model_key!r}")


def freeze_for_regime(model: torch.nn.Module, model_key: str) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = False

    if model_key == "raw_joint":
        for parameter in model.parameters():
            parameter.requires_grad = True
    elif model_key == "lora_moe_head":
        enable_lora_and_heads(model)
    elif model_key in {"asd_lora_moe", "asd_lora_moe_router_kl"}:
        enable_asd_lora_and_heads(model)
    elif model_key == "asd_lora_moe_router_frozen":
        enable_asd_lora_and_heads(model)
        lora_moe = model.backbone.lora_moe
        for parameter in lora_moe.router.parameters():
            parameter.requires_grad = False
        for parameter in lora_moe.scale_router.parameters():
            parameter.requires_grad = False
    elif model_key == "asd_side_feature_lora_moe":
        enable_asd_lora_and_heads(model)
    elif model_key == "patch_token_asd_lora_moe":
        if getattr(model.backbone, "encoder_spectral", None) is None:
            raise RuntimeError("patch_token_asd_lora_moe requires encoder_spectral.")
        for parameter in model.backbone.encoder_spectral.parameters():
            parameter.requires_grad = True
        enable_lora_and_heads(model)
    else:
        raise ValueError(f"Unknown model_key={model_key!r}")

    counts = count_parameters(model)
    return {"total_parameters": int(counts["total"]), "trainable_parameters": int(counts["trainable"])}


def enable_lora_and_heads(model: torch.nn.Module) -> None:
    if getattr(model.backbone, "lora_moe", None) is None:
        raise RuntimeError("Expected backbone.lora_moe.")
    for parameter in model.backbone.lora_moe.parameters():
        parameter.requires_grad = True
    for parameter in model.backbone.heads.parameters():
        parameter.requires_grad = True


def enable_asd_lora_and_heads(model: torch.nn.Module) -> None:
    if not hasattr(model, "denoiser"):
        raise RuntimeError("Expected model.denoiser.")
    for parameter in model.denoiser.parameters():
        parameter.requires_grad = True
    enable_lora_and_heads(model)


def assert_router_frozen(model: torch.nn.Module) -> None:
    lora_moe = model.backbone.lora_moe
    router_flags = [parameter.requires_grad for parameter in lora_moe.router.parameters()]
    scale_flags = [parameter.requires_grad for parameter in lora_moe.scale_router.parameters()]
    if any(router_flags) or any(scale_flags):
        raise RuntimeError("router_frozen regime left router parameters trainable.")
    expert_flags = [parameter.requires_grad for expert in lora_moe.experts for parameter in expert.parameters()]
    if not any(expert_flags):
        raise RuntimeError("router_frozen regime did not enable expert parameters.")


def train_conflict_model(
    *,
    model: torch.nn.Module,
    model_key: str,
    scale_data: dict[str, ScaleData],
    loaders: dict[str, dict[str, torch.utils.data.DataLoader]],
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    output_dir: Path,
    checkpoint_name: str,
    router_balance_weight: float = 0.0,
    teacher: torch.nn.Module | None = None,
    router_kl_weight: float = 0.0,
) -> dict[str, Any]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"{model_key}: no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_nmse = float("inf")
    history: list[dict[str, float]] = []
    start = perf_counter()
    if teacher is not None:
        teacher.eval()

    for epoch in range(1, epochs + 1):
        model.train()
        train_iters = {scale: iter(loaders[scale]["train"]) for scale in scale_data}
        running_loss = 0.0
        running_router_kl = 0.0
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            router_losses: list[torch.Tensor] = []
            router_kls: list[torch.Tensor] = []
            for scale in scale_data:
                xb, yb, train_iters[scale] = next_batch(loaders[scale]["train"], train_iters[scale])
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb, scale, return_diagnostics=True)
                pred, diagnostics = out if isinstance(out, tuple) else (out, {})
                pred = pred.squeeze(1)
                losses.append(F.huber_loss(pred, yb, delta=1.0, reduction="mean"))
                if router_balance_weight > 0.0 and "router_balance_loss" in diagnostics:
                    router_losses.append(diagnostics["router_balance_loss"])
                if teacher is not None and router_kl_weight > 0.0:
                    with torch.no_grad():
                        teacher_probs = router_probabilities(teacher, xb, scale)
                    student_probs = router_probabilities(model, xb, scale)
                    router_kls.append(router_kl(student_probs, teacher_probs))
            loss = torch.stack(losses).mean()
            if router_losses:
                loss = loss + float(router_balance_weight) * torch.stack(router_losses).mean()
            if router_kls:
                kl_loss = torch.stack(router_kls).mean()
                loss = loss + float(router_kl_weight) * kl_loss
                running_router_kl += float(kl_loss.detach().cpu())
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_key}: non-finite loss encountered.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        validation_metrics = evaluate_all_scales(model, scale_data, loaders, split="validation", device=device)
        validation_nmse = float(np.mean([metrics["nmse"] for metrics in validation_metrics.values()]))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_scaled": running_loss / max(steps_per_epoch, 1),
                "router_kl_scaled": running_router_kl / max(steps_per_epoch, 1),
                "validation_mean_nmse": validation_nmse,
            }
        )
        if validation_nmse < best_validation_nmse:
            best_validation_nmse = validation_nmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"{model_key} epoch={epoch}/{epochs} "
            f"train_loss={history[-1]['train_loss_scaled']:.6f} val_nmse={validation_nmse:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = perf_counter() - start
    split_metrics = {
        split: evaluate_all_scales(model, scale_data, loaders, split=split, device=device)
        for split in ["validation", "test", "zero_shot"]
    }
    diagnostics = collect_diagnostics(model, loaders, device)
    checkpoint_path = output_dir / f"{checkpoint_name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_key": model_key,
            "history": history,
            "split_metrics": split_metrics,
            "diagnostics": diagnostics,
        },
        checkpoint_path,
    )
    return {
        "model": model_key,
        "parameters": count_parameters(model),
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": float(elapsed),
        "history": history,
        "best_validation_mean_nmse": best_validation_nmse,
        "split_metrics": split_metrics,
        "diagnostics": diagnostics,
    }


def router_kl(student_probs: torch.Tensor, teacher_probs: torch.Tensor) -> torch.Tensor:
    return (student_probs * (torch.log(student_probs + 1e-8) - torch.log(teacher_probs + 1e-8))).sum(dim=-1).mean()


def router_probabilities(model: torch.nn.Module, xb: torch.Tensor, scale: str) -> torch.Tensor:
    backbone, adapter_input = adapter_backbone_and_input(model, xb, scale)
    h, scale_emb, _, _, _ = backbone.adapter_input_tokens(adapter_input, scale)
    if backbone.lora_moe is None:
        raise RuntimeError("router_probabilities requires a LoRA-MoE backbone.")
    return backbone.lora_moe.routing_weights(h, scale_emb, sparse=False)


def adapter_tokens_for_model(model: torch.nn.Module, xb: torch.Tensor, scale: str) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    backbone, adapter_input = adapter_backbone_and_input(model, xb, scale)
    h, scale_emb, batch_size, channels, _ = backbone.adapter_input_tokens(adapter_input, scale)
    return h, scale_emb, batch_size, channels


def adapter_backbone_and_input(model: torch.nn.Module, xb: torch.Tensor, scale: str) -> tuple[Any, torch.Tensor]:
    if isinstance(model, ScaleAwareASDMultiScalePatchTST):
        scale_emb = model.backbone.scale_embedding_for(scale, xb.shape[0], xb.device)
        clean = model.denoiser(xb, scale_emb)
        if isinstance(clean, tuple):
            clean = clean[0]
        return model.backbone, clean
    if isinstance(model, SideASDFeatureMultiScalePatchTST):
        scale_emb = model.backbone.scale_embedding_for(scale, xb.shape[0], xb.device)
        clean = model.denoiser(xb, scale_emb)
        if isinstance(clean, tuple):
            clean = clean[0]
        return model.backbone, torch.cat([xb, clean - xb], dim=-1)
    if isinstance(model, RawMultiScalePatchTST):
        return model.backbone, xb
    raise TypeError(f"Unsupported model type for adapter diagnostics: {type(model).__name__}")


@torch.no_grad()
def collect_conflict_diagnostics(
    *,
    teacher: torch.nn.Module,
    model: torch.nn.Module,
    model_key: str,
    loaders: dict[str, dict[str, torch.utils.data.DataLoader]],
    normalizers: dict[str, dict[str, float]],
    device: torch.device,
) -> list[dict[str, Any]]:
    teacher.eval()
    model.eval()
    rows: list[dict[str, Any]] = []
    for scale, split_loaders in loaders.items():
        xb, _ = next(iter(split_loaders["validation"]))
        xb = xb.to(device, non_blocking=True)
        teacher_pred = forward_prediction(teacher, xb, scale)
        model_pred = forward_prediction(model, xb, scale)
        y_std = float(normalizers[scale]["y_std"])
        pred_delta = torch.mean(torch.abs(model_pred - teacher_pred)) * y_std

        row: dict[str, Any] = {
            "model": model_key,
            "scale": scale,
            "prediction_delta_abs_mean": float(pred_delta.detach().cpu()),
        }
        if has_lora_moe(model):
            teacher_probs = router_probabilities(teacher, xb, scale)
            student_probs = router_probabilities(model, xb, scale)
            student_probs, teacher_probs = align_probabilities(student_probs, teacher_probs, xb.shape[0])
            row["router_kl_to_teacher"] = float(router_kl(student_probs, teacher_probs).detach().cpu())
            row["router_l1_to_teacher"] = float(torch.mean(torch.abs(student_probs - teacher_probs)).detach().cpu())
        teacher_h, _, teacher_batch, teacher_channels = adapter_tokens_for_model(teacher, xb, scale)
        model_h, _, model_batch, model_channels = adapter_tokens_for_model(model, xb, scale)
        if teacher_h.shape == model_h.shape:
            row["token_cosine_to_teacher"] = float(
                F.cosine_similarity(model_h.flatten(), teacher_h.flatten(), dim=0).detach().cpu()
            )
            row["token_l2_to_teacher"] = float(torch.mean((model_h - teacher_h).pow(2)).sqrt().detach().cpu())
        elif model_batch == teacher_batch and model_channels > teacher_channels:
            model_h_view = model_h.reshape(model_batch, model_channels, model_h.shape[1], model_h.shape[2])[:, 0]
            teacher_h_view = teacher_h.reshape(teacher_batch, teacher_channels, teacher_h.shape[1], teacher_h.shape[2])[:, 0]
            row["token_cosine_to_teacher"] = float(
                F.cosine_similarity(model_h_view.flatten(), teacher_h_view.flatten(), dim=0).detach().cpu()
            )
            row["token_l2_to_teacher"] = float(torch.mean((model_h_view - teacher_h_view).pow(2)).sqrt().detach().cpu())
        rows.append(row)
    return rows


def align_probabilities(
    student_probs: torch.Tensor,
    teacher_probs: torch.Tensor,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if student_probs.shape == teacher_probs.shape:
        return student_probs, teacher_probs
    if student_probs.shape[0] % batch_size == 0 and teacher_probs.shape[0] == batch_size:
        channels = student_probs.shape[0] // batch_size
        student_probs = student_probs.reshape(batch_size, channels, student_probs.shape[1], student_probs.shape[2])[:, 0]
        return student_probs, teacher_probs
    if teacher_probs.shape[0] % batch_size == 0 and student_probs.shape[0] == batch_size:
        channels = teacher_probs.shape[0] // batch_size
        teacher_probs = teacher_probs.reshape(batch_size, channels, teacher_probs.shape[1], teacher_probs.shape[2])[:, 0]
        return student_probs, teacher_probs
    raise ValueError(f"Cannot align router probabilities {tuple(student_probs.shape)} vs {tuple(teacher_probs.shape)}")


def forward_prediction(model: torch.nn.Module, xb: torch.Tensor, scale: str) -> torch.Tensor:
    out = model(xb, scale)
    return out[0] if isinstance(out, tuple) else out


def has_lora_moe(model: torch.nn.Module) -> bool:
    return hasattr(model, "backbone") and getattr(model.backbone, "lora_moe", None) is not None


def collect_gradient_diagnostics(
    *,
    model: torch.nn.Module,
    model_key: str,
    loaders: dict[str, dict[str, torch.utils.data.DataLoader]],
    scale_data: dict[str, ScaleData],
    device: torch.device,
) -> list[dict[str, Any]]:
    model.train()
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad = None
    losses: list[torch.Tensor] = []
    for scale in scale_data:
        xb, yb = next(iter(loaders[scale]["validation"]))
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        out = model(xb, scale)
        pred = (out[0] if isinstance(out, tuple) else out).squeeze(1)
        losses.append(F.huber_loss(pred, yb, delta=1.0, reduction="mean"))
    loss = torch.stack(losses).mean()
    loss.backward()
    row = {
        "model": model_key,
        "asd_grad_norm": grad_norm_for_keywords(model, ["denoiser"]),
        "moe_grad_norm": grad_norm_for_keywords(model, ["lora_moe"]),
        "head_grad_norm": grad_norm_for_keywords(model, ["heads"]),
        "encoder_spectral_grad_norm": grad_norm_for_keywords(model, ["encoder_spectral"]),
    }
    row["asd_moe_grad_ratio"] = safe_ratio(row["asd_grad_norm"], row["moe_grad_norm"])
    row["asd_moe_grad_cosine_truncated"] = truncated_grad_cosine(model, ["denoiser"], ["lora_moe"])
    return [row]


def grad_norm_for_keywords(model: torch.nn.Module, keywords: list[str]) -> float:
    total = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not any(keyword in name for keyword in keywords):
            continue
        total += float(parameter.grad.detach().pow(2).sum().cpu())
    return math.sqrt(total)


def truncated_grad_cosine(model: torch.nn.Module, left_keywords: list[str], right_keywords: list[str]) -> float:
    left = flatten_grads(model, left_keywords)
    right = flatten_grads(model, right_keywords)
    if left.numel() == 0 or right.numel() == 0:
        return float("nan")
    n = min(left.numel(), right.numel())
    return float(F.cosine_similarity(left[:n], right[:n], dim=0).detach().cpu())


def flatten_grads(model: torch.nn.Module, keywords: list[str]) -> torch.Tensor:
    chunks = []
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and any(keyword in name for keyword in keywords):
            chunks.append(parameter.grad.detach().flatten())
    if not chunks:
        return torch.empty(0)
    return torch.cat(chunks)


def safe_ratio(num: float, den: float) -> float:
    return float(num / den) if abs(float(den)) > 1e-12 else float("nan")


def result_diagnostic_rows(result: dict[str, Any], *, model_key: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for scale, diagnostics in result.get("diagnostics", {}).items():
        row: dict[str, Any] = {"model": model_key, "scale": scale}
        row.update({key: float(value) for key, value in diagnostics.items()})
        rows.append(row)
    return rows


def write_report(
    *,
    path: Path,
    summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
    conflict: pd.DataFrame,
    gradients: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines: list[str] = []
    lines.append("# ASD 与 LoRA-MoE 冲突诊断")
    lines.append("")
    lines.append(
        "本轮目标是验证 ASD 是否通过改变 patch/token 表示导致 LoRA-MoE router 分工漂移，并测试 router 稳定化、side denoising、post-patch spectral filtering 三类修复。"
    )
    lines.append("")
    lines.append(
        f"配置：patch preset `{args.patch_preset}`，rank={args.lora_moe_rank}，ASD init gate={args.scale_aware_init_gate}，"
        f"epochs={args.small_epochs}，balanced steps/epoch={args.small_steps_per_epoch}。"
    )
    lines.append("")
    lines.append("## Test Metrics")
    lines.append("")
    test = summary[(summary["split"] == "test") & (summary["model"] != "last_return")].copy()
    keep = ["model", "scale", "n", "nmse", "mse", "mae", "direction_accuracy_nonzero", "corr"]
    lines.extend(frame_to_markdown(test[[column for column in keep if column in test.columns]]))
    lines.append("")
    lines.append("## Router / ASD Diagnostics")
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
            "router_entropy",
            "router_balance_loss",
            "expert_prob_0",
            "expert_prob_1",
            "expert_prob_2",
            "expert_prob_3",
            "side_residual_abs_mean",
            "local_mask_mean",
        ]
        lines.extend(frame_to_markdown(diagnostics[[column for column in keep_diag if column in diagnostics.columns]]))
    lines.append("")
    lines.append("## Conflict Metrics vs LoRA-MoE Teacher")
    lines.append("")
    lines.extend(frame_to_markdown(conflict))
    lines.append("")
    lines.append("## Gradient Diagnostics")
    lines.append("")
    lines.extend(frame_to_markdown(gradients))
    lines.append("")
    lines.append("## Decision Notes")
    lines.append("")
    lines.append("- `router_frozen` / `router_kl` 若恢复 hour，说明主要问题是 router distribution shift。")
    lines.append("- `asd_side_feature_lora_moe` 若更稳，说明 ASD 有信息价值但不应替换 raw path。")
    lines.append("- `patch_token_asd_lora_moe` 若更稳，说明 ASD 应后移到 patch/token 表示。")
    lines.append("- 若所有 ASD 变体都弱于 `lora_moe_head`，主模型应保持 no-ASD LoRA-MoE。")
    lines.append("")
    lines.append("## Files")
    lines.append("")
    out = Path(args.output_dir)
    lines.append(f"- summary: `{out / 'summary.csv'}`")
    lines.append(f"- diagnostics: `{out / 'diagnostics.csv'}`")
    lines.append(f"- conflict diagnostics: `{out / 'conflict_diagnostics.csv'}`")
    lines.append(f"- gradient diagnostics: `{out / 'gradient_diagnostics.csv'}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.mode != "asd_moe_conflict_diagnosis":
        raise ValueError("This script only supports --mode asd_moe_conflict_diagnosis.")
    if "day" in args.scales:
        raise ValueError("This runner intentionally supports only second/minute/hour.")
    set_seed(int(args.seed))
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    scale_specs = make_scale_specs(args)
    caps = caps_for_preset(args, "small")
    scale_data = load_scale_data(args, cache_path=Path(args.cache), caps=caps, scale_specs=scale_specs)
    loaders = make_all_loaders(scale_data, batch_size=args.batch_size, device=device)
    steps_per_epoch = resolve_steps_per_epoch(scale_data, args.batch_size, int(args.small_steps_per_epoch))
    normalizers = {scale: data.normalizer for scale, data in scale_data.items()}
    rows: list[dict[str, Any]] = []
    diagnostics_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    gradient_rows: list[dict[str, Any]] = []
    row_extra = {
        "patch_preset": args.patch_preset,
        "seed": args.seed,
        "adapter_rank": args.lora_moe_rank,
        "init_gate": args.scale_aware_init_gate,
    }
    append_baseline_rows(rows, "small", scale_data, extra=row_extra)

    raw_model = build_model("raw_joint", args, scale_specs).to(device)
    freeze_for_regime(raw_model, "raw_joint")
    raw_result = train_conflict_model(
        model=raw_model,
        model_key="raw_joint",
        scale_data=scale_data,
        loaders=loaders,
        epochs=int(args.small_epochs),
        steps_per_epoch=steps_per_epoch,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        device=device,
        output_dir=output_dir,
        checkpoint_name="raw_joint",
    )
    append_model_rows(rows, "small", "raw_joint", raw_result, extra=row_extra)
    raw_checkpoint = Path(raw_result["checkpoint"])

    teacher_model = build_model("lora_moe_head", args, scale_specs).to(device)
    load_raw_backbone_checkpoint(teacher_model, raw_checkpoint)
    freeze_for_regime(teacher_model, "lora_moe_head")
    teacher_result = train_conflict_model(
        model=teacher_model,
        model_key="lora_moe_head",
        scale_data=scale_data,
        loaders=loaders,
        epochs=int(args.small_epochs),
        steps_per_epoch=steps_per_epoch,
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        device=device,
        output_dir=output_dir,
        checkpoint_name="lora_moe_head",
        router_balance_weight=float(args.router_balance_weight),
    )
    append_model_rows(rows, "small", "lora_moe_head", teacher_result, extra=row_extra)
    diagnostics_rows.extend(result_diagnostic_rows(teacher_result, model_key="lora_moe_head"))
    teacher_checkpoint = Path(teacher_result["checkpoint"])

    trained_models: dict[str, torch.nn.Module] = {"lora_moe_head": teacher_model}
    for model_key in DIAGNOSIS_MODELS[1:]:
        model = build_model(model_key, args, scale_specs).to(device)
        if model_key in {"asd_lora_moe_router_frozen", "asd_lora_moe_router_kl", "patch_token_asd_lora_moe"}:
            load_raw_backbone_checkpoint(model, teacher_checkpoint)
        else:
            load_raw_backbone_checkpoint(model, raw_checkpoint)
        freeze_info = freeze_for_regime(model, model_key)
        if model_key == "asd_lora_moe_router_frozen":
            assert_router_frozen(model)
        print(f"{model_key} freeze_info={freeze_info}", flush=True)
        result = train_conflict_model(
            model=model,
            model_key=model_key,
            scale_data=scale_data,
            loaders=loaders,
            epochs=int(args.small_epochs),
            steps_per_epoch=steps_per_epoch,
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            device=device,
            output_dir=output_dir,
            checkpoint_name=model_key,
            router_balance_weight=float(args.router_balance_weight),
            teacher=teacher_model if model_key == "asd_lora_moe_router_kl" else None,
            router_kl_weight=float(args.router_kl_weight) if model_key == "asd_lora_moe_router_kl" else 0.0,
        )
        append_model_rows(rows, "small", model_key, result, extra=row_extra)
        diagnostics_rows.extend(result_diagnostic_rows(result, model_key=model_key))
        conflict_rows.extend(
            collect_conflict_diagnostics(
                teacher=teacher_model,
                model=model,
                model_key=model_key,
                loaders=loaders,
                normalizers=normalizers,
                device=device,
            )
        )
        gradient_rows.extend(
            collect_gradient_diagnostics(
                model=model,
                model_key=model_key,
                loaders=loaders,
                scale_data=scale_data,
                device=device,
            )
        )
        trained_models[model_key] = model

    conflict_rows.extend(
        collect_conflict_diagnostics(
            teacher=teacher_model,
            model=teacher_model,
            model_key="lora_moe_head",
            loaders=loaders,
            normalizers=normalizers,
            device=device,
        )
    )
    gradient_rows.extend(
        collect_gradient_diagnostics(
            model=teacher_model,
            model_key="lora_moe_head",
            loaders=loaders,
            scale_data=scale_data,
            device=device,
        )
    )

    summary = save_summary(rows, output_dir / "summary.csv")
    diagnostics = pd.DataFrame(diagnostics_rows)
    conflict = pd.DataFrame(conflict_rows)
    gradients = pd.DataFrame(gradient_rows)
    diagnostics.to_csv(output_dir / "diagnostics.csv", index=False)
    conflict.to_csv(output_dir / "conflict_diagnostics.csv", index=False)
    gradients.to_csv(output_dir / "gradient_diagnostics.csv", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(
            {
                "summary_rows": len(summary),
                "diagnostic_rows": len(diagnostics),
                "conflict_rows": len(conflict),
                "gradient_rows": len(gradients),
                "models": list(trained_models),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_report(
        path=Path(args.report_path),
        summary=summary,
        diagnostics=diagnostics,
        conflict=conflict,
        gradients=gradients,
        args=args,
    )
    print(f"saved_report={args.report_path}", flush=True)


if __name__ == "__main__":
    main()
