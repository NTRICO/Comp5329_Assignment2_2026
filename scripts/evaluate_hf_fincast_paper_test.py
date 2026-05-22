from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import re
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


DEFAULT_DATA_ROOT = (
    WORKSPACE_ROOT
    / "data"
    / "raw"
    / "FinCast-Paper-test"
    / "test_v1_nv_flat"
    / "test_v1_nv_flat"
)

FREQ_TO_FINCAST = {
    "1m": "MIN",
    "1h": "H",
    "1d": "D",
    "1wk": "W",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="C-protocol benchmark on Hugging Face Vincent05R/FinCast-Paper-test CSVs."
    )
    parser.add_argument("--data-root", default=str(DEFAULT_DATA_ROOT))
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "hf_fincast_paper_test_c_protocol"))
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--patch-len", type=int, default=32)
    parser.add_argument("--patch-stride", type=int, default=32)
    parser.add_argument("--train-cap", type=int, default=60_000)
    parser.add_argument("--validation-cap", type=int, default=10_000)
    parser.add_argument("--test-cap", type=int, default=2_048)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--fincast-batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--fincast-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--frequencies",
        default="1m,1h,1d,1wk",
        help="Comma-separated frequency suffixes to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    except Exception:
        pass

    frequencies = [item.strip() for item in args.frequencies.split(",") if item.strip()]
    files_by_frequency = discover_files(data_root, frequencies)
    manifest_rows = [
        {"frequency": freq, "file_count": len(files)}
        for freq, files in sorted(files_by_frequency.items())
    ]
    pd.DataFrame(manifest_rows).to_csv(output_dir / "manifest.csv", index=False)
    print("file manifest", manifest_rows, flush=True)

    device = select_device(args.device)
    results: dict[str, object] = {
        "experiment": "hf_fincast_paper_test_c_protocol_capped",
        "data_root": str(data_root),
        "context_len": int(args.context_len),
        "patch_len": int(args.patch_len),
        "patch_stride": int(args.patch_stride),
        "train_cap": int(args.train_cap),
        "validation_cap": int(args.validation_cap),
        "test_cap": int(args.test_cap),
        "target": "next log return",
        "protocol": {
            "patchtst": "return context -> next log return",
            "fincast": "price context -> mean price level -> next log return",
        },
        "frequencies": {},
        "caveats": [
            "This benchmark pools windows by frequency and trains one vanilla PatchTST per frequency.",
            "The default run is capped for runtime; increase caps for a fuller benchmark.",
            "FinCast is frozen zero-shot; PatchTST is supervised on the earlier windows from the same frequency group.",
        ],
    }
    summary_rows: list[dict[str, object]] = []

    for frequency in frequencies:
        files = files_by_frequency.get(frequency, [])
        if not files:
            print(f"skip frequency={frequency}: no files", flush=True)
            continue
        print(f"\nFREQUENCY {frequency}: files={len(files)}", flush=True)
        dataset = build_frequency_dataset(
            files,
            frequency=frequency,
            context_len=args.context_len,
        )
        split = split_dataset(dataset, args)
        print(f"{frequency} split={split['meta']}", flush=True)

        patch_result, patch_pred = train_patchtst(
            frequency=frequency,
            split=split,
            args=args,
            device=device,
            output_dir=output_dir,
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        fincast_result, fincast_pred = run_fincast(
            frequency=frequency,
            contexts=np.asarray(split["test_price_contexts"], dtype=np.float32),
            actual=np.asarray(split["test_y"], dtype=np.float32),
            last_prices=np.asarray(split["test_last_prices"], dtype=np.float32),
            backend=args.fincast_backend,
            batch_size=max(1, args.fincast_batch_size),
            context_len=args.context_len,
        )

        actual = np.asarray(split["test_y"], dtype=np.float32)
        last_return = np.asarray(split["test_return_contexts"], dtype=np.float32)[:, -1]
        baselines = {
            "zero": metric_dict(np.zeros_like(actual), actual),
            "last_return": metric_dict(last_return, actual),
        }
        results["frequencies"][frequency] = {
            "description": dataset["description"],
            "split": split["meta"],
            "patchtst": patch_result,
            "fincast_mean": fincast_result,
            "baselines": baselines,
        }
        for model_name, metrics in [
            ("patchtst", patch_result["test"]),
            ("fincast_mean", fincast_result["test"]),
            ("zero", baselines["zero"]),
            ("last_return", baselines["last_return"]),
        ]:
            summary_rows.append(summary_row(frequency, model_name, metrics, split, dataset, args))

        write_predictions(
            output_dir / f"{frequency}_predictions_head.csv",
            actual=actual,
            patchtst=patch_pred,
            fincast=fincast_pred,
            last_return=last_return,
        )
        plot_trace(
            output_dir / f"{frequency}_trace.png",
            title=f"HF FinCast paper test {frequency}: C protocol",
            actual=actual,
            patchtst=patch_pred,
            fincast=fincast_pred,
        )

    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    metrics_path = output_dir / "metrics.json"
    summary.to_csv(summary_path, index=False)
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nsaved_summary={summary_path}", flush=True)
    print(f"saved_metrics={metrics_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


def discover_files(data_root: Path, frequencies: list[str]) -> dict[str, list[Path]]:
    files_by_frequency = {freq: [] for freq in frequencies}
    for path in data_root.rglob("*.csv"):
        name = path.stem.lower()
        parent = path.parent.name.lower()
        for frequency in frequencies:
            suffix = f"_{frequency.lower()}"
            if name.endswith(suffix) or parent.endswith(suffix):
                files_by_frequency[frequency].append(path)
                break
    return {freq: sorted(paths) for freq, paths in files_by_frequency.items()}


def build_frequency_dataset(
    files: list[Path],
    *,
    frequency: str,
    context_len: int,
) -> dict[str, object]:
    train_idx: list[int] = []
    val_idx: list[int] = []
    test_idx: list[int] = []
    price_contexts: list[np.ndarray] = []
    return_contexts: list[np.ndarray] = []
    targets: list[float] = []
    last_prices: list[float] = []
    file_ids: list[int] = []
    origins: list[int] = []
    skipped = 0

    for file_id, path in enumerate(files):
        try:
            levels = read_close_series(path)
        except Exception:
            skipped += 1
            continue
        if len(levels) < context_len + 2:
            skipped += 1
            continue
        log_levels = np.log(np.clip(levels.astype(np.float64), 1e-12, None))
        returns = np.diff(log_levels).astype(np.float32)
        local_indices: list[int] = []
        for origin in range(context_len, len(levels) - 1):
            price_context = levels[origin - context_len + 1 : origin + 1]
            return_context = returns[origin - context_len : origin]
            target = returns[origin]
            if (
                len(price_context) == context_len
                and len(return_context) == context_len
                and np.isfinite(price_context).all()
                and np.isfinite(return_context).all()
                and np.isfinite(target)
            ):
                local_indices.append(len(targets))
                price_contexts.append(price_context.astype(np.float32))
                return_contexts.append(return_context.astype(np.float32))
                targets.append(float(target))
                last_prices.append(float(levels[origin]))
                file_ids.append(file_id)
                origins.append(origin)
        if not local_indices:
            skipped += 1
            continue
        n = len(local_indices)
        n_test = max(1, int(round(n * 0.2)))
        n_val = max(1, int(round(n * 0.1)))
        n_train = max(0, n - n_val - n_test)
        if n_train == 0:
            skipped += 1
            # Keep the windows out of the split if there is no train support.
            for _ in local_indices:
                price_contexts.pop()
                return_contexts.pop()
                targets.pop()
                last_prices.pop()
                file_ids.pop()
                origins.pop()
            continue
        train_idx.extend(local_indices[:n_train])
        val_idx.extend(local_indices[n_train : n_train + n_val])
        test_idx.extend(local_indices[n_train + n_val :])

    if not targets:
        raise ValueError(f"No usable windows for frequency={frequency}.")
    return {
        "price_contexts": np.asarray(price_contexts, dtype=np.float32),
        "return_contexts": np.asarray(return_contexts, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "last_prices": np.asarray(last_prices, dtype=np.float32),
        "file_ids": np.asarray(file_ids, dtype=np.int64),
        "origins": np.asarray(origins, dtype=np.int64),
        "train_idx": np.asarray(train_idx, dtype=np.int64),
        "validation_idx": np.asarray(val_idx, dtype=np.int64),
        "test_idx": np.asarray(test_idx, dtype=np.int64),
        "frequency": frequency,
        "description": f"{len(files)} files, skipped={skipped}, context_len={context_len}",
    }


def read_close_series(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    candidates = ["Close", "close", "Adj Close", "adj_close", "Adj_Close"]
    column = next((name for name in candidates if name in df.columns), None)
    if column is None:
        numeric_cols = [col for col in df.columns if pd.to_numeric(df[col], errors="coerce").notna().sum() > 0]
        numeric_cols = [col for col in numeric_cols if col.lower() not in {"date", "datetime", "time"}]
        if not numeric_cols:
            raise ValueError(f"No numeric close-like column in {path}.")
        column = numeric_cols[-1]
    series = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=np.float32)
    series = series[np.isfinite(series)]
    series = series[series > 0]
    return series.astype(np.float32)


def split_dataset(dataset: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    indices = {
        "train": cap_indices(np.asarray(dataset["train_idx"], dtype=np.int64), int(args.train_cap)),
        "validation": cap_indices(np.asarray(dataset["validation_idx"], dtype=np.int64), int(args.validation_cap)),
        "test": cap_indices(np.asarray(dataset["test_idx"], dtype=np.int64), int(args.test_cap)),
    }
    out: dict[str, object] = {
        "meta": {
            "train_total": int(len(dataset["train_idx"])),
            "validation_total": int(len(dataset["validation_idx"])),
            "test_total": int(len(dataset["test_idx"])),
            "train_evaluated": int(len(indices["train"])),
            "validation_evaluated": int(len(indices["validation"])),
            "test_evaluated": int(len(indices["test"])),
        }
    }
    for split_name, idx in indices.items():
        for key in ["price_contexts", "return_contexts", "targets", "last_prices"]:
            target_key = "y" if key == "targets" else key
            out[f"{split_name}_{target_key}"] = np.asarray(dataset[key])[idx]
    return out


def cap_indices(indices: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or cap >= len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, cap).round().astype(np.int64)
    return indices[positions]


def train_patchtst(
    *,
    frequency: str,
    split: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> tuple[dict[str, object], np.ndarray]:
    torch.manual_seed(int(args.seed))
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
        context_length=int(args.context_len),
        prediction_length=1,
        input_channels=1,
        patch_length=int(args.patch_len),
        patch_stride=int(args.patch_stride),
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

    train_loader = DataLoader(TensorDataset(nx(train_x), ny(train_y)), batch_size=int(args.batch_size), shuffle=True)
    val_loader = DataLoader(TensorDataset(nx(val_x), ny(val_y)), batch_size=int(args.batch_size), shuffle=False)
    test_loader = DataLoader(TensorDataset(nx(test_x), ny(test_y)), batch_size=int(args.batch_size), shuffle=False)

    best_state = None
    best_val = float("inf")
    history: list[dict[str, float]] = []
    for epoch in range(1, int(args.epochs) + 1):
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
        print(
            f"{frequency} patchtst epoch={epoch} train_loss={train_loss:.6f} "
            f"val_mse={val_metrics['mse']:.8g} val_dir={val_metrics['direction_accuracy_nonzero']:.4f}",
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
        output_dir / f"{frequency}_patchtst.pt",
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


FINCAST_API_CACHE: dict[tuple[str, int], object] = {}


def run_fincast(
    *,
    frequency: str,
    contexts: np.ndarray,
    actual: np.ndarray,
    last_prices: np.ndarray,
    backend: str,
    batch_size: int,
    context_len: int,
) -> tuple[dict[str, object], np.ndarray]:
    cache_key = (backend, int(context_len))
    if cache_key not in FINCAST_API_CACHE:
        fincast_src = WORKSPACE_ROOT / "FinCast-fts" / "src"
        if str(fincast_src) not in sys.path:
            sys.path.insert(0, str(fincast_src))
        from tools.inference_utils import get_model_api

        print(f"loading frozen FinCast context_len={context_len}", flush=True)
        FINCAST_API_CACHE[cache_key] = get_model_api(
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
    fincast_src = WORKSPACE_ROOT / "FinCast-fts" / "src"
    if str(fincast_src) not in sys.path:
        sys.path.insert(0, str(fincast_src))
    from tools.inference_utils import freq_reader_inference

    freq_token = freq_reader_inference(FREQ_TO_FINCAST[frequency])
    api = FINCAST_API_CACHE[cache_key]
    chunks = []
    for batch_index, start in enumerate(range(0, len(contexts), batch_size), start=1):
        end = min(start + batch_size, len(contexts))
        mean, _full = api.forecast(
            [row for row in contexts[start:end]],
            freq=[freq_token] * (end - start),
        )
        raw = np.asarray(mean, dtype=np.float32)[:, 0]
        pred = np.log(np.clip(raw, 1e-12, None) / np.clip(last_prices[start:end], 1e-12, None))
        chunks.append(pred.astype(np.float32))
        if batch_index == 1 or end == len(contexts) or batch_index % 25 == 0:
            print(f"{frequency} fincast forecasted {end}/{len(contexts)}", flush=True)
    pred_arr = np.concatenate(chunks)
    return (
        {
            "model": "frozen FinCast v1 mean forecast",
            "frequency": frequency,
            "freq_token": int(freq_token),
            "context_len": int(context_len),
            "test": metric_dict(pred_arr, actual),
        },
        pred_arr,
    )


def summary_row(
    frequency: str,
    model: str,
    metrics: dict[str, float],
    split: dict[str, object],
    dataset: dict[str, object],
    args: argparse.Namespace,
) -> dict[str, object]:
    return {
        "frequency": frequency,
        "model": model,
        "test_mse": metrics["mse"],
        "test_rmse": metrics["rmse"],
        "test_mae": metrics["mae"],
        "test_direction_accuracy_nonzero": metrics["direction_accuracy_nonzero"],
        "test_corr": metrics["corr"],
        "test_total_windows": split["meta"]["test_total"],
        "test_evaluated_windows": split["meta"]["test_evaluated"],
        "train_total_windows": split["meta"]["train_total"],
        "train_evaluated_windows": split["meta"]["train_evaluated"],
        "context_len": int(args.context_len),
        "patch_len": int(args.patch_len),
        "patch_stride": int(args.patch_stride),
        "description": dataset["description"],
    }


def write_predictions(path: Path, *, actual: np.ndarray, patchtst: np.ndarray, fincast: np.ndarray, last_return: np.ndarray) -> None:
    n = min(5000, len(actual))
    pd.DataFrame(
        {
            "actual_next_log_return": actual[:n],
            "patchtst_pred_next_log_return": patchtst[:n],
            "fincast_pred_next_log_return": fincast[:n],
            "last_return_baseline": last_return[:n],
        }
    ).to_csv(path, index=False)


def plot_trace(path: Path, *, title: str, actual: np.ndarray, patchtst: np.ndarray, fincast: np.ndarray) -> None:
    n = min(500, len(actual))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(actual[:n], label="actual", linewidth=1.0)
    ax.plot(patchtst[:n], label="PatchTST", linewidth=1.0, alpha=0.85)
    ax.plot(fincast[:n], label="FinCast mean", linewidth=1.0, alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("evaluated test window index")
    ax.set_ylabel("next log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    main()
