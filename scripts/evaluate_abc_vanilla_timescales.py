from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
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
SCRIPT_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_vanilla_timescales_additional_stock import (  # noqa: E402
    ADDITIONAL_ROOT,
    build_stock_hour_price_matrix,
    metric_dict,
    read_stock_names,
    select_device,
)
from src.baselines.patchtst_lora import (  # noqa: E402
    LoRAConfig,
    PatchTSTForecastConfig,
    PatchTSTLoRA,
    count_parameters,
)


INPUT_LENGTHS = {
    "second": 32,
    "minute": 32,
    "hour": 4,
    "day": 21,
}
PATCH_SETTINGS = {
    "second": (32, 32),
    "minute": (32, 32),
    "hour": (4, 4),
    "day": (21, 21),
}
SCALE_ORDER = ("second", "minute", "hour", "day")
FINCAST_API_CACHE: dict[tuple[str, int], object] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run ABC input-format experiments for vanilla PatchTST and frozen "
            "FinCast on Optiver additional data."
        )
    )
    parser.add_argument("--data-root", default=str(ADDITIONAL_ROOT))
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "abc_intraday_patch_stock0"))
    parser.add_argument("--stock-rank", type=int, default=0)
    parser.add_argument("--stock-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--scales", nargs="+", choices=SCALE_ORDER, default=list(SCALE_ORDER))
    parser.add_argument("--day-input-len", type=int, default=INPUT_LENGTHS["day"])
    parser.add_argument("--day-patch-length", type=int, default=None)
    parser.add_argument("--day-patch-stride", type=int, default=None)
    parser.add_argument("--second-train-cap", type=int, default=120_000)
    parser.add_argument("--second-validation-cap", type=int, default=20_000)
    parser.add_argument("--second-test-cap", type=int, default=8_192)
    parser.add_argument("--fincast-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--fincast-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_runtime_settings(args)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    try:
        torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    except Exception:
        pass

    train_frame = pd.read_csv(data_root / "train.csv")
    stock_ids = sorted(train_frame["stock_id"].dropna().astype(int).unique().tolist())
    stock_id = int(args.stock_id) if args.stock_id is not None else int(stock_ids[int(args.stock_rank)])
    stock_names = read_stock_names(data_root / "stock_ids.csv")
    instrument = stock_names.get(stock_id, f"stock_id_{stock_id}")
    print(f"selected_stock_id={stock_id} instrument={instrument}", flush=True)

    time_reference = pd.read_csv(data_root / "time_id_reference.csv")
    time_reference["timestamp"] = pd.to_datetime(time_reference["date"] + " " + time_reference["time"])
    time_reference = time_reference.sort_values("timestamp").reset_index(drop=True)
    price_matrix, metadata = build_stock_hour_price_matrix(
        data_root=data_root,
        stock_id=stock_id,
        valid_time_ids=set(train_frame.loc[train_frame["stock_id"] == stock_id, "time_id"].astype(int).tolist()),
        time_reference=time_reference,
    )
    print(f"price_matrix={price_matrix.shape}", flush=True)

    datasets = build_datasets(price_matrix, metadata)
    selected_scales = set(args.scales)
    datasets = {scale: datasets[scale] for scale in SCALE_ORDER if scale in selected_scales}
    device = select_device(args.device)
    results: dict[str, object] = {
        "experiment": "ABC intraday-patch vanilla PatchTST vs FinCast mean",
        "stock_rank": int(args.stock_rank),
        "stock_id": int(stock_id),
        "instrument": instrument,
        "input_lengths": INPUT_LENGTHS,
        "patchtst_patch_settings": {
            scale: {"patch_length": setting[0], "patch_stride": setting[1]}
            for scale, setting in PATCH_SETTINGS.items()
        },
        "selected_scales": list(datasets.keys()),
        "target": "next log return",
        "abc_definitions": {
            "A_price_to_return": "PatchTST and FinCast both receive same-length price levels; FinCast mean level is converted to log return.",
            "B_return_to_return": "PatchTST and FinCast both receive same-length log returns; FinCast mean is interpreted as next log return.",
            "C_mixed_original": "PatchTST receives same-length log returns; FinCast receives same-length price levels; same origins and targets.",
        },
        "caveats": [
            "Second, minute, and hour windows are built inside a single trading date and never cross overnight gaps.",
            f"Day scale uses a fixed {INPUT_LENGTHS['day']}-trading-day window because PatchTST requires a fixed context_length.",
            "FinCast context_len is chosen per scale as the next multiple of 32, so inputs are padded but not truncated.",
        ],
        "scales": {},
    }
    summary_rows: list[dict[str, object]] = []

    for scale, dataset in datasets.items():
        print(f"\nSCALE {scale}: {dataset['description']}", flush=True)
        split = split_dataset(dataset, args)
        scale_result: dict[str, object] = {
            "description": dataset["description"],
            "split": split["meta"],
            "frequency": dataset["frequency"],
            "input_len": int(dataset["input_len"]),
            "patchtst_patch_length": int(dataset["patch_length"]),
            "patchtst_patch_stride": int(dataset["patch_stride"]),
        }

        patch_price_result, patch_price_pred = train_patchtst(
            scale=scale,
            input_kind="price",
            split=split,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            seed=args.seed,
        )
        patch_return_result, patch_return_pred = train_patchtst(
            scale=scale,
            input_kind="return",
            split=split,
            dataset=dataset,
            device=device,
            output_dir=output_dir,
            seed=args.seed,
        )
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        fincast_price_result, fincast_price_pred = run_fincast(
            scale=scale,
            input_kind="price",
            contexts=np.asarray(split["test_price_contexts"], dtype=np.float32),
            actual=np.asarray(split["test_y"], dtype=np.float32),
            last_prices=np.asarray(split["test_last_prices"], dtype=np.float32),
            frequency=str(dataset["frequency"]),
            backend=args.fincast_backend,
            batch_size=max(1, args.fincast_batch_size),
        )
        fincast_return_result, fincast_return_pred = run_fincast(
            scale=scale,
            input_kind="return",
            contexts=np.asarray(split["test_return_contexts"], dtype=np.float32),
            actual=np.asarray(split["test_y"], dtype=np.float32),
            last_prices=np.asarray(split["test_last_prices"], dtype=np.float32),
            frequency=str(dataset["frequency"]),
            backend=args.fincast_backend,
            batch_size=max(1, args.fincast_batch_size),
        )

        actual = np.asarray(split["test_y"], dtype=np.float32)
        zero_metrics = metric_dict(np.zeros_like(actual), actual)
        last_return_metrics = metric_dict(np.asarray(split["test_return_contexts"], dtype=np.float32)[:, -1], actual)

        experiments = {
            "A_price_to_return": {
                "patchtst": patch_price_result,
                "fincast": fincast_price_result,
                "patchtst_pred": patch_price_pred,
                "fincast_pred": fincast_price_pred,
            },
            "B_return_to_return": {
                "patchtst": patch_return_result,
                "fincast": fincast_return_result,
                "patchtst_pred": patch_return_pred,
                "fincast_pred": fincast_return_pred,
            },
            "C_mixed_original": {
                "patchtst": patch_return_result,
                "fincast": fincast_price_result,
                "patchtst_pred": patch_return_pred,
                "fincast_pred": fincast_price_pred,
            },
        }
        scale_result["experiments"] = experiments_without_predictions(experiments)
        scale_result["baselines"] = {
            "zero": zero_metrics,
            "last_return": last_return_metrics,
        }
        results["scales"][scale] = scale_result

        for exp_name, exp in experiments.items():
            append_summary(summary_rows, scale, exp_name, "patchtst", exp["patchtst"]["test"], split, dataset)
            append_summary(summary_rows, scale, exp_name, "fincast_mean", exp["fincast"]["test"], split, dataset)
            write_predictions(
                output_dir / f"{scale}_{exp_name}_predictions_head.csv",
                actual=actual,
                patchtst=np.asarray(exp["patchtst_pred"], dtype=np.float32),
                fincast=np.asarray(exp["fincast_pred"], dtype=np.float32),
                last_return=np.asarray(split["test_return_contexts"], dtype=np.float32)[:, -1],
            )
            plot_trace(
                output_dir / f"{scale}_{exp_name}_trace.png",
                title=f"{instrument} {scale} {exp_name}",
                actual=actual,
                patchtst=np.asarray(exp["patchtst_pred"], dtype=np.float32),
                fincast=np.asarray(exp["fincast_pred"], dtype=np.float32),
            )
        append_summary(summary_rows, scale, "baseline", "zero", zero_metrics, split, dataset)
        append_summary(summary_rows, scale, "baseline", "last_return", last_return_metrics, split, dataset)

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nsaved_metrics={metrics_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


def configure_runtime_settings(args: argparse.Namespace) -> None:
    day_input_len = int(args.day_input_len)
    if day_input_len < 1:
        raise ValueError("--day-input-len must be positive.")
    day_patch_length = int(args.day_patch_length) if args.day_patch_length is not None else day_input_len
    day_patch_stride = int(args.day_patch_stride) if args.day_patch_stride is not None else day_patch_length
    if day_patch_length < 1 or day_patch_stride < 1:
        raise ValueError("--day-patch-length and --day-patch-stride must be positive.")
    if day_patch_length > day_input_len:
        raise ValueError("--day-patch-length cannot exceed --day-input-len.")
    INPUT_LENGTHS["day"] = day_input_len
    PATCH_SETTINGS["day"] = (day_patch_length, day_patch_stride)


def build_datasets(price_matrix: np.ndarray, metadata: pd.DataFrame) -> dict[str, dict[str, object]]:
    ordered = metadata.assign(row_index=np.arange(len(metadata))).sort_values("timestamp")
    second_segments: list[np.ndarray] = []
    minute_segments: list[np.ndarray] = []
    hour_segments: list[np.ndarray] = []
    for _, group in ordered.groupby("date", sort=True):
        idx = group["row_index"].to_numpy(dtype=np.int64)
        if len(idx) == 0:
            continue
        second_segments.append(price_matrix[idx].reshape(-1).astype(np.float32))
        minute_segments.append(
            np.concatenate([price_matrix[row_idx].reshape(60, 60)[:, -1] for row_idx in idx]).astype(np.float32)
        )
        hour_segments.append(price_matrix[idx, -1].astype(np.float32))

    day_indices = (
        ordered
        .groupby("date", sort=True)["row_index"]
        .last()
        .to_numpy()
    )
    day_series = price_matrix[day_indices, -1].astype(np.float32)
    return {
        "second": make_windows(
            second_segments,
            scale="second",
            frequency="S",
            description="32 seconds -> next second, window kept inside one trading date",
        ),
        "minute": make_windows(
            minute_segments,
            scale="minute",
            frequency="MIN",
            description="32 minutes -> next minute, window kept inside one trading date",
        ),
        "hour": make_windows(
            hour_segments,
            scale="hour",
            frequency="H",
            description="4 market-hour observations -> next market-hour, window kept inside one trading date",
        ),
        "day": make_windows(
            [day_series],
            scale="day",
            frequency="D",
            description=f"{INPUT_LENGTHS['day']} trading days -> next day",
        ),
    }


def make_windows(
    segments: list[np.ndarray],
    *,
    scale: str,
    frequency: str,
    description: str,
) -> dict[str, object]:
    input_len = int(INPUT_LENGTHS[scale])
    patch_length, patch_stride = PATCH_SETTINGS[scale]
    price_contexts: list[np.ndarray] = []
    return_contexts: list[np.ndarray] = []
    targets: list[float] = []
    last_prices: list[float] = []
    segment_ids: list[int] = []
    for segment_id, levels in enumerate(segments):
        levels = np.asarray(levels, dtype=np.float32)
        if len(levels) < input_len + 2:
            continue
        log_levels = np.log(np.clip(levels.astype(np.float64), 1e-12, None))
        returns = np.diff(log_levels).astype(np.float32)
        for origin in range(input_len, len(levels) - 1):
            price_context = levels[origin - input_len + 1 : origin + 1]
            return_context = returns[origin - input_len : origin]
            target = returns[origin]
            if (
                len(price_context) == input_len
                and len(return_context) == input_len
                and np.isfinite(price_context).all()
                and np.isfinite(return_context).all()
                and np.isfinite(target)
            ):
                price_contexts.append(price_context.astype(np.float32))
                return_contexts.append(return_context.astype(np.float32))
                targets.append(float(target))
                last_prices.append(float(levels[origin]))
                segment_ids.append(segment_id)
    return {
        "price_contexts": np.asarray(price_contexts, dtype=np.float32),
        "return_contexts": np.asarray(return_contexts, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "last_prices": np.asarray(last_prices, dtype=np.float32),
        "segment_ids": np.asarray(segment_ids, dtype=np.int64),
        "frequency": frequency,
        "description": description,
        "input_len": int(input_len),
        "patch_length": int(patch_length),
        "patch_stride": int(patch_stride),
    }


def split_dataset(dataset: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    segment_ids = np.asarray(dataset["segment_ids"], dtype=np.int64)
    n = len(segment_ids)
    unique_segments = np.unique(segment_ids)
    if len(unique_segments) > 1:
        n_test_segments = max(1, int(len(unique_segments) * 0.2))
        n_val_segments = max(1, int(len(unique_segments) * 0.1))
        train_segments = set(unique_segments[: len(unique_segments) - n_val_segments - n_test_segments].tolist())
        val_segments = set(unique_segments[len(unique_segments) - n_val_segments - n_test_segments : len(unique_segments) - n_test_segments].tolist())
        test_segments = set(unique_segments[len(unique_segments) - n_test_segments :].tolist())
        masks = {
            "train": np.asarray([sid in train_segments for sid in segment_ids]),
            "validation": np.asarray([sid in val_segments for sid in segment_ids]),
            "test": np.asarray([sid in test_segments for sid in segment_ids]),
        }
    else:
        n_test = max(1, int(n * 0.2))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val - n_test
        idx = np.arange(n)
        masks = {
            "train": idx < n_train,
            "validation": (idx >= n_train) & (idx < n_train + n_val),
            "test": idx >= n_train + n_val,
        }

    caps = {
        "train": args.second_train_cap if dataset["frequency"] == "S" else 0,
        "validation": args.second_validation_cap if dataset["frequency"] == "S" else 0,
        "test": args.second_test_cap if dataset["frequency"] == "S" else 0,
    }
    out: dict[str, object] = {"meta": {"frequency": str(dataset["frequency"])}}
    for split_name, mask in masks.items():
        raw_idx = np.flatnonzero(mask)
        idx = cap_indices(raw_idx, int(caps[split_name]))
        for key in ["price_contexts", "return_contexts", "targets", "last_prices"]:
            target_key = "y" if key == "targets" else key
            out[f"{split_name}_{target_key}"] = np.asarray(dataset[key])[idx]
        out["meta"][f"{split_name}_total"] = int(len(raw_idx))
        out["meta"][f"{split_name}_evaluated"] = int(len(idx))
    return out


def cap_indices(indices: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or cap >= len(indices):
        return indices
    positions = np.linspace(0, len(indices) - 1, cap).round().astype(np.int64)
    return indices[positions]


def train_patchtst(
    *,
    scale: str,
    input_kind: str,
    split: dict[str, object],
    dataset: dict[str, object],
    device: torch.device,
    output_dir: Path,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    torch.manual_seed(seed)
    key = "price_contexts" if input_kind == "price" else "return_contexts"
    train_x = np.asarray(split[f"train_{key}"], dtype=np.float32)[:, :, None]
    val_x = np.asarray(split[f"validation_{key}"], dtype=np.float32)[:, :, None]
    test_x = np.asarray(split[f"test_{key}"], dtype=np.float32)[:, :, None]
    train_y = np.asarray(split["train_y"], dtype=np.float32).reshape(-1, 1)
    val_y = np.asarray(split["validation_y"], dtype=np.float32).reshape(-1, 1)
    test_y = np.asarray(split["test_y"], dtype=np.float32).reshape(-1, 1)

    normalizer = {
        "x_mean": float(train_x.mean()),
        "x_std": float(max(train_x.std(), 1e-8)),
        "y_mean": float(train_y.mean()),
        "y_std": float(max(train_y.std(), 1e-8)),
    }
    input_len = int(dataset["input_len"])
    patch_length = int(dataset["patch_length"])
    patch_stride = int(dataset["patch_stride"])
    batch_size = 512 if scale in {"second", "minute"} else 32
    epochs = {"second": 5, "minute": 8, "hour": 20, "day": 40}[scale]
    config = PatchTSTForecastConfig(
        context_length=input_len,
        prediction_length=1,
        input_channels=1,
        patch_length=patch_length,
        patch_stride=patch_stride,
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

    train_loader = DataLoader(TensorDataset(nx(train_x), ny(train_y)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(nx(val_x), ny(val_y)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(nx(test_x), ny(test_y)), batch_size=batch_size, shuffle=False)

    history: list[dict[str, float]] = []
    best_state = None
    best_val = float("inf")
    for epoch in range(1, epochs + 1):
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
            best_state = {name: param.detach().cpu().clone() for name, param in model.state_dict().items()}
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(
                f"{scale} patchtst_{input_kind} epoch={epoch} "
                f"train_loss={train_loss:.6f} val_mse={val_metrics['mse']:.8g} "
                f"val_dir={val_metrics['direction_accuracy_nonzero']:.4f}",
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
            "input_kind": input_kind,
            "normalizer": normalizer,
            "history": history,
            "test": test,
        },
        output_dir / f"{scale}_patchtst_{input_kind}.pt",
    )
    return (
        {
            "model": "vanilla PatchTSTLoRA rank=0",
            "input_kind": input_kind,
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


def next_multiple_of_32(value: int) -> int:
    return max(32, int(math.ceil(value / 32) * 32))


def run_fincast(
    *,
    scale: str,
    input_kind: str,
    contexts: np.ndarray,
    actual: np.ndarray,
    last_prices: np.ndarray,
    frequency: str,
    backend: str,
    batch_size: int,
) -> tuple[dict[str, object], np.ndarray]:
    input_len = int(contexts.shape[1])
    context_len = next_multiple_of_32(input_len)
    if context_len > 2048:
        raise ValueError(f"FinCast context_len={context_len} exceeds the supported 2048 limit.")
    cache_key = (backend, context_len)
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
                context_len=context_len,
                num_experts=4,
                gating_top_n=2,
                load_from_compile=True,
                forecast_mode="mean",
            )
        )
    from tools.inference_utils import freq_reader_inference

    freq_value = freq_reader_inference(frequency)
    model_api = FINCAST_API_CACHE[cache_key]
    chunks = []
    for batch_index, start in enumerate(range(0, len(contexts), batch_size), start=1):
        end = min(start + batch_size, len(contexts))
        mean, _full = model_api.forecast(
            [row for row in contexts[start:end]],
            freq=[freq_value] * (end - start),
        )
        raw = np.asarray(mean, dtype=np.float32)[:, 0]
        if input_kind == "price":
            pred = np.log(np.clip(raw, 1e-12, None) / np.clip(last_prices[start:end], 1e-12, None))
        else:
            pred = raw
        chunks.append(pred.astype(np.float32))
        if batch_index == 1 or end == len(contexts) or batch_index % 50 == 0:
            print(f"{scale} fincast_{input_kind} forecasted {end}/{len(contexts)}", flush=True)
    pred_arr = np.concatenate(chunks)
    return (
        {
            "model": "frozen FinCast v1 mean forecast",
            "input_kind": input_kind,
            "frequency": frequency,
            "freq_token": int(freq_value),
            "input_len": int(input_len),
            "context_len": int(context_len),
            "test": metric_dict(pred_arr, actual),
        },
        pred_arr,
    )


def experiments_without_predictions(experiments: dict[str, dict[str, object]]) -> dict[str, object]:
    return {
        name: {
            "patchtst": exp["patchtst"],
            "fincast": exp["fincast"],
        }
        for name, exp in experiments.items()
    }


def append_summary(
    rows: list[dict[str, object]],
    scale: str,
    experiment: str,
    model: str,
    metrics: dict[str, float],
    split: dict[str, object],
    dataset: dict[str, object],
) -> None:
    rows.append(
        {
            "scale": scale,
            "experiment": experiment,
            "model": model,
            "test_mse": metrics["mse"],
            "test_rmse": metrics["rmse"],
            "test_mae": metrics["mae"],
            "test_direction_accuracy_nonzero": metrics["direction_accuracy_nonzero"],
            "test_corr": metrics["corr"],
            "test_total_windows": split["meta"]["test_total"],
            "test_evaluated_windows": split["meta"]["test_evaluated"],
            "input_len": int(dataset["input_len"]),
            "patch_length": int(dataset["patch_length"]),
            "patch_stride": int(dataset["patch_stride"]),
            "frequency": dataset["frequency"],
        }
    )


def write_predictions(
    path: Path,
    *,
    actual: np.ndarray,
    patchtst: np.ndarray,
    fincast: np.ndarray,
    last_return: np.ndarray,
) -> None:
    n = min(5000, len(actual))
    pd.DataFrame(
        {
            "actual_next_log_return": actual[:n],
            "patchtst_pred_next_log_return": patchtst[:n],
            "fincast_pred_next_log_return": fincast[:n],
            "last_return_baseline": last_return[:n],
        }
    ).to_csv(path, index=False)


def plot_trace(
    path: Path,
    *,
    title: str,
    actual: np.ndarray,
    patchtst: np.ndarray,
    fincast: np.ndarray,
) -> None:
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
