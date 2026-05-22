from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from scripts.evaluate_vanilla_timescales_additional_stock import metric_dict, select_device  # noqa: E402
from src.baselines.patchtst_lora import (  # noqa: E402
    LoRAConfig,
    PatchTSTForecastConfig,
    PatchTSTLoRA,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PatchTST and FinCast on the local FinCast monthly sample CSV."
    )
    parser.add_argument("--data-path", default=str(WORKSPACE_ROOT / "data" / "raw" / "sample_close_monthly.csv"))
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "fincast_sample_patchtst"))
    parser.add_argument("--context-len", type=int, default=32)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--frequency", default="M")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fincast-backend", choices=["gpu", "cpu"], default="gpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = build_dataset(Path(args.data_path), context_len=args.context_len)
    split = split_dataset(dataset)
    device = select_device(args.device)

    patch_result, patch_pred = train_patchtst(
        split=split,
        context_len=args.context_len,
        patch_len=args.patch_len,
        patch_stride=args.patch_stride,
        device=device,
        seed=args.seed,
        output_dir=output_dir,
    )
    fincast_result, fincast_pred = run_fincast(
        contexts=np.asarray(split["test_price_contexts"], dtype=np.float32),
        actual=np.asarray(split["test_y"], dtype=np.float32),
        last_prices=np.asarray(split["test_last_prices"], dtype=np.float32),
        context_len=args.context_len,
        frequency=args.frequency,
        backend=args.fincast_backend,
    )

    actual = np.asarray(split["test_y"], dtype=np.float32)
    last_return = np.asarray(split["test_return_contexts"], dtype=np.float32)[:, -1]
    baselines = {
        "zero": metric_dict(np.zeros_like(actual), actual),
        "last_return": metric_dict(last_return, actual),
    }

    results = {
        "experiment": "fincast_sample_monthly_C_protocol",
        "data_path": str(Path(args.data_path)),
        "series_names": dataset["series_names"],
        "frequency": args.frequency,
        "context_len": int(args.context_len),
        "patch_len": int(args.patch_len),
        "patch_stride": int(args.patch_stride),
        "target": "next log return",
        "protocol": {
            "patchtst": "return context -> next log return",
            "fincast": "price context -> mean price level -> next log return",
        },
        "split": split["meta"],
        "patchtst": patch_result,
        "fincast_mean": fincast_result,
        "baselines": baselines,
        "caveats": [
            "Local sample_close_monthly.csv has only 128 monotonically increasing monthly close values.",
            "This is an inference smoke-test style sample, not the full FinCast paper test benchmark.",
            "Direction accuracy is not informative here because all test next returns are positive.",
        ],
    }

    rows = [
        summary_row("patchtst", patch_result["test"], split, args),
        summary_row("fincast_mean", fincast_result["test"], split, args),
        summary_row("zero", baselines["zero"], split, args),
        summary_row("last_return", baselines["last_return"], split, args),
    ]
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "summary.csv"
    metrics_path = output_dir / "metrics.json"
    pred_path = output_dir / "predictions.csv"
    trace_path = output_dir / "trace.png"
    summary.to_csv(summary_path, index=False)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    pd.DataFrame(
        {
            "actual_next_log_return": actual,
            "patchtst_pred_next_log_return": patch_pred,
            "fincast_pred_next_log_return": fincast_pred,
            "last_return_baseline": last_return,
        }
    ).to_csv(pred_path, index=False)
    plot_trace(trace_path, actual, patch_pred, fincast_pred, last_return)

    print(f"saved_summary={summary_path}", flush=True)
    print(f"saved_metrics={metrics_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


def build_dataset(data_path: Path, *, context_len: int) -> dict[str, object]:
    df = pd.read_csv(data_path)
    numeric_cols: list[str] = []
    for col in df.columns:
        values = pd.to_numeric(df[col], errors="coerce")
        if values.notna().sum() >= context_len + 2:
            numeric_cols.append(col)
    if not numeric_cols:
        raise ValueError(f"No numeric columns with at least {context_len + 2} values in {data_path}.")

    price_contexts: list[np.ndarray] = []
    return_contexts: list[np.ndarray] = []
    targets: list[float] = []
    last_prices: list[float] = []
    series_ids: list[int] = []
    origins: list[int] = []
    for series_id, col in enumerate(numeric_cols):
        levels = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        levels = levels[np.isfinite(levels)]
        levels = levels[levels > 0]
        log_levels = np.log(levels.astype(np.float64))
        returns = np.diff(log_levels).astype(np.float32)
        for origin in range(context_len, len(levels) - 1):
            price_context = levels[origin - context_len + 1 : origin + 1]
            return_context = returns[origin - context_len : origin]
            target = returns[origin]
            if len(price_context) == context_len and len(return_context) == context_len:
                price_contexts.append(price_context.astype(np.float32))
                return_contexts.append(return_context.astype(np.float32))
                targets.append(float(target))
                last_prices.append(float(levels[origin]))
                series_ids.append(series_id)
                origins.append(origin)

    return {
        "price_contexts": np.asarray(price_contexts, dtype=np.float32),
        "return_contexts": np.asarray(return_contexts, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "last_prices": np.asarray(last_prices, dtype=np.float32),
        "series_ids": np.asarray(series_ids, dtype=np.int64),
        "origins": np.asarray(origins, dtype=np.int64),
        "series_names": numeric_cols,
    }


def split_dataset(dataset: dict[str, object]) -> dict[str, object]:
    targets = np.asarray(dataset["targets"], dtype=np.float32)
    series_ids = np.asarray(dataset["series_ids"], dtype=np.int64)
    origins = np.asarray(dataset["origins"], dtype=np.int64)
    n = len(targets)
    if n < 10:
        raise ValueError(f"Need at least 10 windows for train/validation/test, got {n}.")

    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    for series_id in np.unique(series_ids):
        idx = np.flatnonzero(series_ids == series_id)
        idx = idx[np.argsort(origins[idx])]
        n_series = len(idx)
        n_test = max(1, int(round(n_series * 0.2)))
        n_val = max(1, int(round(n_series * 0.1)))
        n_train = n_series - n_val - n_test
        train_idx.extend(idx[:n_train].tolist())
        val_idx.extend(idx[n_train : n_train + n_val].tolist())
        test_idx.extend(idx[n_train + n_val :].tolist())

    out: dict[str, object] = {
        "meta": {
            "total_windows": int(n),
            "train_windows": int(len(train_idx)),
            "validation_windows": int(len(val_idx)),
            "test_windows": int(len(test_idx)),
        }
    }
    index_map = {"train": np.asarray(train_idx), "validation": np.asarray(val_idx), "test": np.asarray(test_idx)}
    for split_name, idx in index_map.items():
        for key in ["price_contexts", "return_contexts", "targets", "last_prices"]:
            target_key = "y" if key == "targets" else key
            out[f"{split_name}_{target_key}"] = np.asarray(dataset[key])[idx]
    return out


def train_patchtst(
    *,
    split: dict[str, object],
    context_len: int,
    patch_len: int,
    patch_stride: int,
    device: torch.device,
    seed: int,
    output_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    torch.manual_seed(seed)
    train_x = np.asarray(split["train_return_contexts"], dtype=np.float32)[:, :, None]
    val_x = np.asarray(split["validation_return_contexts"], dtype=np.float32)[:, :, None]
    test_x = np.asarray(split["test_return_contexts"], dtype=np.float32)[:, :, None]
    train_y = np.asarray(split["train_y"], dtype=np.float32).reshape(-1, 1)
    val_y = np.asarray(split["validation_y"], dtype=np.float32).reshape(-1, 1)
    test_y = np.asarray(split["test_y"], dtype=np.float32).reshape(-1, 1)
    normalizer = {
        "x_mean": float(train_x.mean()),
        "x_std": float(max(train_x.std(), 1e-8)),
        "y_mean": float(train_y.mean()),
        "y_std": float(max(train_y.std(), 1e-8)),
    }

    config = PatchTSTForecastConfig(
        context_length=int(context_len),
        prediction_length=1,
        input_channels=1,
        patch_length=int(patch_len),
        patch_stride=int(patch_stride),
        d_model=64,
        n_heads=4,
        n_layers=2,
        d_ff=128,
        dropout=0.1,
        lora=LoRAConfig(rank=0, alpha=1.0, dropout=0.0, enabled=False),
    )
    model = PatchTSTLoRA(config).to(device)
    model.set_lora_enabled(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    def nx(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((x - normalizer["x_mean"]) / normalizer["x_std"], dtype=torch.float32)

    def ny(y: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((y - normalizer["y_mean"]) / normalizer["y_std"], dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(nx(train_x), ny(train_y)), batch_size=16, shuffle=True)
    val_loader = DataLoader(TensorDataset(nx(val_x), ny(val_y)), batch_size=32, shuffle=False)
    test_loader = DataLoader(TensorDataset(nx(test_x), ny(test_y)), batch_size=32, shuffle=False)

    best_state = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, 101):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            pred = model(xb).squeeze(1)
            loss = F.mse_loss(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach().cpu()) * xb.shape[0]
            count += xb.shape[0]
        val_metrics, _ = eval_patchtst(model, val_loader, normalizer, device)
        train_loss = total / max(count, 1)
        history.append({"epoch": epoch, "train_loss_scaled": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 25 == 0 or epoch == 100:
            print(
                f"patchtst epoch={epoch} train_loss={train_loss:.6f} "
                f"val_mse={val_metrics['mse']:.8g}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    validation, _ = eval_patchtst(model, val_loader, normalizer, device)
    test, pred = eval_patchtst(model, test_loader, normalizer, device)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "normalizer": normalizer,
            "history": history,
            "test": test,
        },
        output_dir / "patchtst.pt",
    )
    return (
        {
            "model": "vanilla PatchTSTLoRA rank=0",
            "config": asdict(config),
            "normalizer": normalizer,
            "parameters": count_parameters(model),
            "validation": validation,
            "test": test,
        },
        pred,
    )


@torch.no_grad()
def eval_patchtst(
    model: PatchTSTLoRA,
    loader: DataLoader,
    normalizer: dict[str, float],
    device: torch.device,
) -> tuple[dict[str, float], np.ndarray]:
    model.eval()
    preds = []
    actuals = []
    for xb, yb in loader:
        pred_scaled = model(xb.to(device)).squeeze(1).cpu().numpy().reshape(-1)
        pred = pred_scaled * normalizer["y_std"] + normalizer["y_mean"]
        preds.append(pred.astype(np.float32))
        actual = yb.cpu().numpy().reshape(-1) * normalizer["y_std"] + normalizer["y_mean"]
        actuals.append(actual.astype(np.float32))
    pred_arr = np.concatenate(preds)
    actual_arr = np.concatenate(actuals)
    return metric_dict(pred_arr, actual_arr), pred_arr


def run_fincast(
    *,
    contexts: np.ndarray,
    actual: np.ndarray,
    last_prices: np.ndarray,
    context_len: int,
    frequency: str,
    backend: str,
) -> tuple[dict[str, object], np.ndarray]:
    fincast_src = WORKSPACE_ROOT / "FinCast-fts" / "src"
    if str(fincast_src) not in sys.path:
        sys.path.insert(0, str(fincast_src))
    from tools.inference_utils import freq_reader_inference, get_model_api

    api = get_model_api(
        SimpleNamespace(
            model_path=str(WORKSPACE_ROOT / "models" / "FinCast" / "v1.pth"),
            backend=backend,
            horizon_len=1,
            context_len=int(context_len),
            num_experts=4,
            gating_top_n=2,
            load_from_compile=True,
            forecast_mode="mean",
        )
    )
    freq_value = freq_reader_inference(frequency)
    mean, _full = api.forecast([row for row in contexts], freq=[freq_value] * len(contexts))
    raw = np.asarray(mean, dtype=np.float32)[:, 0]
    pred = np.log(np.clip(raw, 1e-12, None) / np.clip(last_prices, 1e-12, None)).astype(np.float32)
    return (
        {
            "model": "frozen FinCast v1 mean forecast",
            "frequency": frequency,
            "freq_token": int(freq_value),
            "context_len": int(context_len),
            "test": metric_dict(pred, actual),
        },
        pred,
    )


def summary_row(
    model: str,
    metrics: dict[str, float],
    split: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "model": model,
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_direction_accuracy_nonzero": metrics["direction_accuracy_nonzero"],
        "test_corr": metrics["corr"],
        "test_windows": split["meta"]["test_windows"],
        "context_len": int(args.context_len),
        "patch_len": int(args.patch_len),
        "patch_stride": int(args.patch_stride),
        "frequency": args.frequency,
    }


def plot_trace(path: Path, actual: np.ndarray, patchtst: np.ndarray, fincast: np.ndarray, last_return: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual, label="actual", marker="o", linewidth=1.2)
    ax.plot(patchtst, label="PatchTST", marker="o", linewidth=1.0)
    ax.plot(fincast, label="FinCast mean", marker="o", linewidth=1.0)
    ax.plot(last_return, label="last-return", marker="o", linewidth=0.9, alpha=0.65)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title("FinCast sample monthly: C protocol")
    ax.set_xlabel("test window")
    ax.set_ylabel("next log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
