from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.baselines.patchtst_lora import (  # noqa: E402
    LoRAConfig,
    PatchTSTForecastConfig,
    PatchTSTLoRA,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare vanilla PatchTST against a spectral-denoised PatchTST on "
            "Optiver per-second stock data, with stock10 reserved for zero-shot."
        )
    )
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_optiver_hf_second_feature_cache_11stocks_256t.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "optiver_spectral_denoise_patchtst"),
    )
    parser.add_argument("--train-stocks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--zero-shot-stock", type=int, default=10)
    parser.add_argument("--scale", choices=["second", "minute", "hour"], default="second")
    parser.add_argument("--feature-name", default="wap1_log_return_1s")
    parser.add_argument(
        "--target-horizon-seconds",
        type=int,
        default=1,
        help="Predict the cumulative future log return over this many seconds inside each time_id.",
    )
    parser.add_argument(
        "--target-horizon-steps",
        type=int,
        default=None,
        help="For minute/hour scales, predict this many future scale steps. Defaults to 1.",
    )
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--train-cap", type=int, default=0, help="0 means use all available windows.")
    parser.add_argument("--validation-cap", type=int, default=0, help="0 means use all available windows.")
    parser.add_argument("--test-cap", type=int, default=0, help="0 means use all available windows.")
    parser.add_argument("--zero-shot-cap", type=int, default=0, help="0 means use all available windows.")
    parser.add_argument("--spectral-keep-ratio", type=float, default=0.35)
    parser.add_argument("--spectral-blend-init", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return parser.parse_args()


class AdaptiveSpectralDenoiser(nn.Module):
    """Low-pass spectral front-end with a learned residual blend.

    The block is intentionally model-agnostic: it takes a normalized time
    window [batch, time, channel], filters high-frequency components, and
    returns another window with the same shape before the forecasting encoder.
    """

    def __init__(
        self,
        *,
        context_length: int,
        channels: int,
        keep_ratio: float = 0.35,
        blend_init: float = 0.65,
    ) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if not 0.0 < blend_init < 1.0:
            raise ValueError("blend_init must be in (0, 1).")
        self.context_length = int(context_length)
        self.keep_ratio = float(keep_ratio)
        self.register_buffer("_mask", self._build_mask(context_length, keep_ratio), persistent=False)
        logit = math.log(blend_init / (1.0 - blend_init))
        self.logit_blend = nn.Parameter(torch.full((1, 1, channels), logit, dtype=torch.float32))

    @staticmethod
    def _build_mask(context_length: int, keep_ratio: float) -> torch.Tensor:
        freq_count = context_length // 2 + 1
        keep = max(2, min(freq_count, int(math.ceil(freq_count * keep_ratio))))
        mask = torch.zeros(1, freq_count, 1, dtype=torch.float32)
        mask[:, :keep, :] = 1.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected [B,L,C], got {tuple(x.shape)}")
        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        spectrum = torch.fft.rfft(centered, dim=1)
        filtered = torch.fft.irfft(spectrum * self._mask.to(x.device), n=x.shape[1], dim=1) + mean
        blend = torch.sigmoid(self.logit_blend)
        return x + blend * (filtered - x)

    def extra_repr(self) -> str:
        return f"context_length={self.context_length}, keep_ratio={self.keep_ratio}"


class SpectralDenoisedPatchTST(nn.Module):
    def __init__(
        self,
        config: PatchTSTForecastConfig,
        *,
        keep_ratio: float,
        blend_init: float,
    ) -> None:
        super().__init__()
        self.denoiser = AdaptiveSpectralDenoiser(
            context_length=config.context_length,
            channels=config.input_channels,
            keep_ratio=keep_ratio,
            blend_init=blend_init,
        )
        self.forecaster = PatchTSTLoRA(config)
        self.forecaster.set_lora_enabled(False)

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        return self.forecaster(self.denoiser(past_values))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = select_device(args.device)

    train_stocks = parse_stock_list(args.train_stocks)
    target_horizon_steps = (
        int(args.target_horizon_seconds)
        if args.scale == "second"
        else int(args.target_horizon_steps or 1)
    )
    data = build_datasets(
        cache_path=Path(args.cache),
        scale=args.scale,
        train_stocks=train_stocks,
        zero_shot_stock=args.zero_shot_stock,
        feature_name=args.feature_name,
        target_horizon_steps=target_horizon_steps,
        context_length=args.context_length,
        train_fraction=args.train_fraction,
        validation_fraction=args.validation_fraction,
        caps={
            "train": args.train_cap,
            "validation": args.validation_cap,
            "test": args.test_cap,
            "zero_shot": args.zero_shot_cap,
        },
        seed=args.seed,
    )
    normalizer = fit_normalizer(data["train_x"], data["train_y"])
    config = PatchTSTForecastConfig(
        context_length=args.context_length,
        prediction_length=1,
        input_channels=1,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lora=LoRAConfig(rank=0, alpha=1.0, dropout=0.0, enabled=False),
    )

    print(f"device={device}", flush=True)
    print(f"cache={Path(args.cache)}", flush=True)
    print(
        f"scale={args.scale} horizon_steps={target_horizon_steps} "
        f"train_stocks={train_stocks} zero_shot_stock={args.zero_shot_stock}",
        flush=True,
    )
    print(
        "windows "
        f"train={len(data['train_y'])} validation={len(data['validation_y'])} "
        f"test={len(data['test_y'])} zero_shot={len(data['zero_shot_y'])}",
        flush=True,
    )

    raw_model = PatchTSTLoRA(config).to(device)
    raw_model.set_lora_enabled(False)
    raw_result, raw_predictions = train_and_evaluate(
        model=raw_model,
        model_name="raw_patchtst",
        config=config,
        data=data,
        normalizer=normalizer,
        args=args,
        output_dir=output_dir,
        device=device,
        seed=args.seed,
    )

    set_seed(args.seed)
    denoised_model = SpectralDenoisedPatchTST(
        config,
        keep_ratio=args.spectral_keep_ratio,
        blend_init=args.spectral_blend_init,
    ).to(device)
    denoised_result, denoised_predictions = train_and_evaluate(
        model=denoised_model,
        model_name="spectral_denoised_patchtst",
        config=config,
        data=data,
        normalizer=normalizer,
        args=args,
        output_dir=output_dir,
        device=device,
        seed=args.seed,
    )

    summary = build_summary_frame(
        data=data,
        raw_result=raw_result,
        denoised_result=denoised_result,
    )
    comparisons = build_comparisons(summary)
    results = {
        "experiment": "optiver_scaled_spectral_denoise_patchtst_no_lora",
        "cache": str(Path(args.cache)),
        "scale": args.scale,
        "target": target_description(args.scale, target_horizon_steps),
        "input_feature": args.feature_name if args.scale == "second" else "derived WAP1 log returns",
        "train_stocks": train_stocks,
        "zero_shot_stock": int(args.zero_shot_stock),
        "protocol": {
            "train": "stock0-9, first train_fraction of time_id episodes per stock",
            "validation": "stock0-9, next validation_fraction of time_id episodes per stock",
            "test": "stock0-9, remaining held-out time_id episodes per stock",
            "zero_shot": "stock10, all available time_id episodes, never used for fitting or normalization",
            "lora": "disabled; LoRA rank is 0 for both models",
        },
        "config": asdict(config),
        "spectral_denoiser": {
            "keep_ratio": float(args.spectral_keep_ratio),
            "blend_init": float(args.spectral_blend_init),
            "learned_blend": denoised_result["learned_blend"],
        },
        "normalizer": normalizer,
        "data_meta": data["meta"],
        "raw_patchtst": raw_result,
        "spectral_denoised_patchtst": denoised_result,
        "comparisons": comparisons,
    }

    summary_path = output_dir / "summary.csv"
    metrics_path = output_dir / "metrics.json"
    pred_path = output_dir / "prediction_head.csv"
    summary.to_csv(summary_path, index=False)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    write_prediction_head(pred_path, data, raw_predictions, denoised_predictions)

    print(f"saved_summary={summary_path}", flush=True)
    print(f"saved_metrics={metrics_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("comparisons", json.dumps(comparisons, indent=2), flush=True)


def build_datasets(
    *,
    cache_path: Path,
    scale: str,
    train_stocks: list[int],
    zero_shot_stock: int,
    feature_name: str,
    target_horizon_steps: int,
    context_length: int,
    train_fraction: float,
    validation_fraction: float,
    caps: dict[str, int],
    seed: int,
) -> dict[str, object]:
    if train_fraction <= 0.0 or validation_fraction <= 0.0 or train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction and validation_fraction must be positive and leave a test split.")
    if target_horizon_steps <= 0:
        raise ValueError("target_horizon_steps must be positive.")
    arrays = np.load(cache_path, allow_pickle=True)
    features = np.asarray(arrays["encoder_features"], dtype=np.float32)
    targets = np.asarray(arrays["realized_returns"], dtype=np.float32)
    asset_names = np.asarray(arrays["asset_names"]).astype(str)
    episode_ids = np.asarray(arrays["episode_ids"], dtype=np.int64)
    time_ids = np.asarray(arrays["time_ids"], dtype=np.int64)
    seconds = np.asarray(arrays["seconds_in_bucket"], dtype=np.int64)
    feature_names = [str(name) for name in arrays["feature_names"]]
    if feature_name not in feature_names:
        raise ValueError(f"feature {feature_name!r} not found. Available: {feature_names}")
    feature_index = feature_names.index(feature_name)
    wap1_index = feature_names.index("wap1")
    x_values = features[:, feature_index].astype(np.float32)
    stock_ids = np.asarray([parse_asset_stock_id(name) for name in asset_names], dtype=np.int64)

    needed = set(train_stocks + [int(zero_shot_stock)])
    available = set(stock_ids.tolist())
    missing = sorted(needed - available)
    if missing:
        raise ValueError(f"Cache is missing stocks {missing}; available stocks include {sorted(available)[:20]}.")

    split_x: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": [], "zero_shot": []}
    split_y: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": [], "zero_shot": []}
    split_last: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": [], "zero_shot": []}
    split_meta: dict[str, object] = {
        "cache_rows": int(len(targets)),
        "scale": scale,
        "context_length": int(context_length),
        "target_horizon_steps": int(target_horizon_steps),
        "feature_names": feature_names,
        "source_files": [str(path) for path in arrays["source_files"]],
        "splits": {},
    }

    for stock_id in train_stocks + [int(zero_shot_stock)]:
        stock_episode_ids = np.unique(episode_ids[stock_ids == stock_id])
        stock_episode_ids = stock_episode_ids[np.argsort([time_ids[episode_ids == episode].min() for episode in stock_episode_ids])]
        if stock_id == int(zero_shot_stock):
            split_plan = {"zero_shot": stock_episode_ids}
        else:
            n = len(stock_episode_ids)
            n_train = max(1, int(math.floor(n * train_fraction)))
            n_validation = max(1, int(math.floor(n * validation_fraction)))
            if n_train + n_validation >= n:
                n_validation = max(1, n - n_train - 1)
            split_plan = {
                "train": stock_episode_ids[:n_train],
                "validation": stock_episode_ids[n_train : n_train + n_validation],
                "test": stock_episode_ids[n_train + n_validation :],
            }
        split_meta["splits"][f"stock_{stock_id}"] = {
            name: int(len(episodes)) for name, episodes in split_plan.items()
        }
        for split_name, episodes in split_plan.items():
            if scale == "hour":
                episode_levels = []
                for episode in episodes:
                    idx = np.flatnonzero(episode_ids == episode)
                    idx = idx[np.argsort(seconds[idx])]
                    if len(idx):
                        episode_levels.append(float(features[idx[-1], wap1_index]))
                built = build_windows_from_levels(
                    levels=np.asarray(episode_levels, dtype=np.float32),
                    context_length=context_length,
                    target_horizon_steps=target_horizon_steps,
                )
                if built is not None:
                    x_windows, y_values, last_values = built
                    split_x[split_name].append(x_windows)
                    split_y[split_name].append(y_values)
                    split_last[split_name].append(last_values)
                continue

            for episode in episodes:
                idx = np.flatnonzero(episode_ids == episode)
                idx = idx[np.argsort(seconds[idx])]
                if scale == "second":
                    if len(idx) < context_length:
                        continue
                    values = x_values[idx]
                    one_step_simple_returns = targets[idx]
                    one_step_log_returns = np.log1p(
                        np.clip(one_step_simple_returns.astype(np.float64), -0.999999, None)
                    ).astype(np.float32)
                    built = build_windows_from_aligned_returns(
                        values=values,
                        future_returns=one_step_log_returns,
                        context_length=context_length,
                        target_horizon_steps=target_horizon_steps,
                    )
                elif scale == "minute":
                    minute_levels = aggregate_minute_levels(features[idx, wap1_index], seconds[idx])
                    built = build_windows_from_levels(
                        levels=minute_levels,
                        context_length=context_length,
                        target_horizon_steps=target_horizon_steps,
                    )
                else:
                    raise ValueError(f"Unsupported scale: {scale}")
                if built is None:
                    continue
                x_windows, y_values, last_values = built
                split_x[split_name].append(x_windows)
                split_y[split_name].append(y_values)
                split_last[split_name].append(last_values)

    rng = np.random.default_rng(seed)
    output: dict[str, object] = {"meta": split_meta}
    for split_name in ["train", "validation", "test", "zero_shot"]:
        if not split_x[split_name]:
            raise ValueError(f"No windows built for split {split_name}.")
        x = np.concatenate(split_x[split_name], axis=0)
        y = np.concatenate(split_y[split_name], axis=0)
        last = np.concatenate(split_last[split_name], axis=0)
        selected = cap_indices(len(y), int(caps.get(split_name, 0)), rng=rng, random=(split_name == "train"))
        output[f"{split_name}_x"] = x[selected]
        output[f"{split_name}_y"] = y[selected]
        output[f"{split_name}_last_return"] = last[selected]
        split_meta[f"{split_name}_total_windows"] = int(len(y))
        split_meta[f"{split_name}_evaluated_windows"] = int(len(selected))
    return output


def build_windows_from_aligned_returns(
    *,
    values: np.ndarray,
    future_returns: np.ndarray,
    context_length: int,
    target_horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if len(values) < context_length or len(future_returns) < target_horizon_steps:
        return None
    windows = np.lib.stride_tricks.sliding_window_view(values, context_length)
    horizon_targets = np.lib.stride_tricks.sliding_window_view(
        future_returns,
        target_horizon_steps,
    ).sum(axis=1)
    end_positions = np.arange(context_length - 1, context_length - 1 + len(windows))
    valid = end_positions < len(horizon_targets)
    if not np.any(valid):
        return None
    valid_end_positions = end_positions[valid]
    return (
        windows[valid].astype(np.float32)[:, :, None],
        horizon_targets[valid_end_positions].astype(np.float32),
        (values[valid_end_positions] * float(target_horizon_steps)).astype(np.float32),
    )


def build_windows_from_levels(
    *,
    levels: np.ndarray,
    context_length: int,
    target_horizon_steps: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    levels = np.asarray(levels, dtype=np.float32)
    levels = levels[np.isfinite(levels) & (levels > 0.0)]
    if len(levels) < context_length + target_horizon_steps + 1:
        return None
    log_levels = np.log(np.clip(levels.astype(np.float64), 1e-12, None)).astype(np.float32)
    observed_returns = np.zeros_like(log_levels, dtype=np.float32)
    observed_returns[1:] = np.diff(log_levels)
    future_returns = np.diff(log_levels)
    return build_windows_from_aligned_returns(
        values=observed_returns,
        future_returns=future_returns,
        context_length=context_length,
        target_horizon_steps=target_horizon_steps,
    )


def aggregate_minute_levels(levels: np.ndarray, seconds: np.ndarray) -> np.ndarray:
    levels = np.asarray(levels, dtype=np.float32)
    seconds = np.asarray(seconds, dtype=np.int64)
    minute_levels = []
    for minute in range(10):
        start = minute * 60
        stop = min((minute + 1) * 60, 600)
        positions = np.flatnonzero((seconds >= start) & (seconds < stop))
        if len(positions):
            minute_levels.append(float(levels[positions[-1]]))
    return np.asarray(minute_levels, dtype=np.float32)


def cap_indices(n: int, cap: int, *, rng: np.random.Generator, random: bool) -> np.ndarray:
    indices = np.arange(n)
    if cap <= 0 or cap >= n:
        return indices
    if random:
        return np.sort(rng.choice(indices, size=cap, replace=False))
    positions = np.linspace(0, n - 1, cap).round().astype(np.int64)
    return indices[positions]


def train_and_evaluate(
    *,
    model: nn.Module,
    model_name: str,
    config: PatchTSTForecastConfig,
    data: dict[str, object],
    normalizer: dict[str, float],
    args: argparse.Namespace,
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    set_seed(seed)
    train_loader = make_loader(
        data["train_x"],
        data["train_y"],
        normalizer,
        batch_size=args.batch_size,
        shuffle=True,
        device=device,
    )
    validation_loader = make_loader(
        data["validation_x"],
        data["validation_y"],
        normalizer,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )
    test_loader = make_loader(
        data["test_x"],
        data["test_y"],
        normalizer,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )
    zero_loader = make_loader(
        data["zero_shot_x"],
        data["zero_shot_y"],
        normalizer,
        batch_size=args.batch_size,
        shuffle=False,
        device=device,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    history: list[dict[str, float]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_val = float("inf")
    start_time = perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(1)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        train_loss = total / max(count, 1)
        validation_metrics, _ = evaluate_model(model, validation_loader, normalizer, device)
        history.append(
            {
                "epoch": int(epoch),
                "train_loss_scaled": float(train_loss),
                **{f"validation_{key}": float(value) for key, value in validation_metrics.items()},
            }
        )
        if validation_metrics["mse"] < best_val:
            best_val = validation_metrics["mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"{model_name} epoch={epoch}/{args.epochs} "
            f"train_loss={train_loss:.6f} val_mse={validation_metrics['mse']:.8g} "
            f"val_dir={validation_metrics['direction_accuracy_nonzero']:.4f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = perf_counter() - start_time
    validation_metrics, validation_pred = evaluate_model(model, validation_loader, normalizer, device)
    test_metrics, test_pred = evaluate_model(model, test_loader, normalizer, device)
    zero_metrics, zero_pred = evaluate_model(model, zero_loader, normalizer, device)
    learned_blend = None
    denoiser = getattr(model, "denoiser", None)
    if denoiser is not None and hasattr(denoiser, "logit_blend"):
        learned_blend = torch.sigmoid(denoiser.logit_blend.detach().cpu()).numpy().reshape(-1).astype(float).tolist()

    checkpoint_path = output_dir / f"{model_name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "normalizer": normalizer,
            "history": history,
            "validation": validation_metrics,
            "test": test_metrics,
            "zero_shot": zero_metrics,
        },
        checkpoint_path,
    )
    return (
        {
            "model": model_name,
            "parameters": count_parameters(model),
            "checkpoint": str(checkpoint_path),
            "elapsed_seconds": float(elapsed),
            "history": history,
            "validation": validation_metrics,
            "test": test_metrics,
            "zero_shot": zero_metrics,
            "learned_blend": learned_blend,
        },
        {
            "validation": validation_pred,
            "test": test_pred,
            "zero_shot": zero_pred,
        },
    )


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: dict[str, float],
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    x_norm = (np.asarray(x, dtype=np.float32) - normalizer["x_mean"]) / normalizer["x_std"]
    y_norm = (np.asarray(y, dtype=np.float32).reshape(-1, 1) - normalizer["y_mean"]) / normalizer["y_std"]
    dataset = TensorDataset(torch.as_tensor(x_norm, dtype=torch.float32), torch.as_tensor(y_norm, dtype=torch.float32))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    normalizer: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    for xb, yb in loader:
        pred_scaled = model(xb.to(device, non_blocking=True)).squeeze(1).cpu().numpy().reshape(-1)
        pred = pred_scaled * normalizer["y_std"] + normalizer["y_mean"]
        actual = yb.cpu().numpy().reshape(-1) * normalizer["y_std"] + normalizer["y_mean"]
        preds.append(pred.astype(np.float32))
        actuals.append(actual.astype(np.float32))
    pred_arr = np.concatenate(preds)
    actual_arr = np.concatenate(actuals)
    return metric_dict(pred_arr, actual_arr), pred_arr


def fit_normalizer(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    return {
        "x_mean": float(np.mean(x)),
        "x_std": float(max(np.std(x), 1e-8)),
        "y_mean": float(np.mean(y)),
        "y_std": float(max(np.std(y), 1e-8)),
    }


def metric_dict(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    pred = np.asarray(prediction, dtype=np.float64).reshape(-1)
    y = np.asarray(actual, dtype=np.float64).reshape(-1)
    mse = float(np.mean((pred - y) ** 2))
    mae = float(np.mean(np.abs(pred - y)))
    nonzero = y != 0
    direction = float(np.mean((pred[nonzero] > 0) == (y[nonzero] > 0))) if nonzero.any() else float("nan")
    corr = float(np.corrcoef(pred, y)[0, 1]) if np.std(pred) > 0 and np.std(y) > 0 else float("nan")
    return {
        "mse": mse,
        "rmse": float(math.sqrt(mse)),
        "mae": mae,
        "direction_accuracy_nonzero": direction,
        "corr": corr,
    }


def build_summary_frame(
    *,
    data: dict[str, object],
    raw_result: dict[str, object],
    denoised_result: dict[str, object],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for split in ["validation", "test", "zero_shot"]:
        actual = data[f"{split}_y"]
        last_return = data[f"{split}_last_return"]
        baselines = {
            "zero": metric_dict(np.zeros_like(actual), actual),
            "last_return": metric_dict(last_return, actual),
        }
        for model_name, result in [
            ("raw_patchtst", raw_result),
            ("spectral_denoised_patchtst", denoised_result),
        ]:
            append_metric_row(rows, split, model_name, result[split])
        for model_name, metrics in baselines.items():
            append_metric_row(rows, split, model_name, metrics)
    return pd.DataFrame(rows)


def append_metric_row(
    rows: list[dict[str, object]],
    split: str,
    model_name: str,
    metrics: dict[str, float],
) -> None:
    rows.append(
        {
            "split": split,
            "model": model_name,
            "mse": metrics["mse"],
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "direction_accuracy_nonzero": metrics["direction_accuracy_nonzero"],
            "corr": metrics["corr"],
        }
    )


def build_comparisons(summary: pd.DataFrame) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for split in ["validation", "test", "zero_shot"]:
        raw = summary[(summary["split"] == split) & (summary["model"] == "raw_patchtst")].iloc[0]
        denoised = summary[(summary["split"] == split) & (summary["model"] == "spectral_denoised_patchtst")].iloc[0]
        out[split] = {
            "mse_delta_denoised_minus_raw": float(denoised["mse"] - raw["mse"]),
            "mse_relative_change_pct": float((denoised["mse"] - raw["mse"]) / max(raw["mse"], 1e-20) * 100.0),
            "mae_delta_denoised_minus_raw": float(denoised["mae"] - raw["mae"]),
            "mae_relative_change_pct": float((denoised["mae"] - raw["mae"]) / max(raw["mae"], 1e-20) * 100.0),
            "direction_accuracy_delta": float(
                denoised["direction_accuracy_nonzero"] - raw["direction_accuracy_nonzero"]
            ),
            "corr_delta": float(denoised["corr"] - raw["corr"]),
        }
    return out


def write_prediction_head(
    path: Path,
    data: dict[str, object],
    raw_predictions: dict[str, np.ndarray],
    denoised_predictions: dict[str, np.ndarray],
) -> None:
    frames: list[pd.DataFrame] = []
    for split in ["validation", "test", "zero_shot"]:
        n = min(5000, len(data[f"{split}_y"]))
        frames.append(
            pd.DataFrame(
                {
                    "split": split,
                    "actual_next_return": np.asarray(data[f"{split}_y"])[:n],
                    "last_return_baseline": np.asarray(data[f"{split}_last_return"])[:n],
                    "raw_patchtst_prediction": raw_predictions[split][:n],
                    "spectral_denoised_prediction": denoised_predictions[split][:n],
                }
            )
        )
    pd.concat(frames, ignore_index=True).to_csv(path, index=False)


def parse_stock_list(value: str) -> list[int]:
    stocks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not stocks:
        raise ValueError("At least one training stock must be supplied.")
    return stocks


def target_description(scale: str, horizon_steps: int) -> str:
    if scale == "second":
        return f"next {horizon_steps}-second cumulative WAP1 log return inside the same Optiver time_id"
    if scale == "minute":
        return f"next {horizon_steps}-minute cumulative WAP1 log return inside the same Optiver time_id"
    if scale == "hour":
        return f"next {horizon_steps}-time_id cumulative WAP1 log return across sorted Optiver time_ids"
    return f"next {horizon_steps} steps"


def parse_asset_stock_id(asset_name: str) -> int:
    return int(str(asset_name).split("_")[-1])


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
