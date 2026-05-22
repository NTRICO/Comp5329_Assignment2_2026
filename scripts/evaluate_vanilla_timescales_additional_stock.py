from __future__ import annotations

import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace
import xml.etree.ElementTree as ET
import zipfile

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

from src.baselines.patchtst_lora import (  # noqa: E402
    LoRAConfig,
    PatchTSTForecastConfig,
    PatchTSTLoRA,
    count_parameters,
)


ADDITIONAL_ROOT = (
    WORKSPACE_ROOT
    / "data"
    / "high-frequency"
    / "Optiver_additional data"
    / "Optiver_additional data"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate vanilla PatchTST and frozen FinCast on Optiver additional "
            "data for one stock across second/minute/hour/day scales."
        )
    )
    parser.add_argument("--data-root", default=str(ADDITIONAL_ROOT))
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "vanilla_timescales_additional_stock0"))
    parser.add_argument(
        "--stock-rank",
        type=int,
        default=0,
        help="Rank after sorting stock_id values found in train.csv. rank 0 is the first available stock.",
    )
    parser.add_argument("--stock-id", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--second-train-cap", type=int, default=120_000)
    parser.add_argument("--second-validation-cap", type=int, default=20_000)
    parser.add_argument("--second-test-cap", type=int, default=8_192)
    parser.add_argument("--fincast-backend", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--fincast-batch-size", type=int, default=64)
    parser.add_argument("--skip-fincast", action="store_true")
    parser.add_argument("--skip-patchtst", action="store_true")
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

    stock_names = read_stock_names(data_root / "stock_ids.csv")
    train_frame = pd.read_csv(data_root / "train.csv")
    available_stock_ids = sorted(train_frame["stock_id"].dropna().astype(int).unique().tolist())
    stock_id = int(args.stock_id) if args.stock_id is not None else int(available_stock_ids[int(args.stock_rank)])
    instrument = stock_names.get(stock_id, f"stock_id_{stock_id}")
    print(f"selected_stock_id={stock_id} instrument={instrument}", flush=True)

    time_reference = pd.read_csv(data_root / "time_id_reference.csv")
    time_reference["timestamp"] = pd.to_datetime(time_reference["date"] + " " + time_reference["time"])
    time_reference = time_reference.sort_values("time_id").reset_index(drop=True)

    price_matrix, metadata = build_stock_hour_price_matrix(
        data_root=data_root,
        stock_id=stock_id,
        valid_time_ids=set(train_frame.loc[train_frame["stock_id"] == stock_id, "time_id"].astype(int).tolist()),
        time_reference=time_reference,
    )
    print(f"price_matrix={price_matrix.shape}", flush=True)

    scales = build_scale_datasets(price_matrix, metadata)
    results: dict[str, object] = {
        "experiment": "vanilla_patchtst_and_fincast_additional_data_timescales",
        "stock_rank": int(args.stock_rank),
        "stock_id": int(stock_id),
        "instrument": instrument,
        "target": "next-period log return",
        "data_root": str(data_root),
        "caveats": [
            "Optiver additional data has real stock_id values, not stock_id=0; stock0 here means rank 0 after sorting available stock_id values.",
            "Second and minute windows do not cross time_id/hour boundaries.",
            "Hour windows are built within each trading date to avoid overnight leakage.",
            "Day windows use daily close levels from the last available hour of each date.",
            "Frozen FinCast maps S, MIN, H, and D to the same high-frequency token in this checkout; scale differences come from the aggregated input series.",
        ],
        "scales": {},
    }

    summary_rows: list[dict[str, object]] = []
    patch_device = select_device(args.device)

    for name, dataset in scales.items():
        print(f"\nSCALE {name}: {dataset['description']}", flush=True)
        split = split_dataset(dataset, args)
        scale_result: dict[str, object] = {
            "description": dataset["description"],
            "frequency": dataset["frequency"],
            "context_returns": int(dataset["context_returns"]),
            "split": split["meta"],
            "baselines": {},
        }

        eval_level_contexts = split["test_level_contexts"]
        eval_y = split["test_y"]
        last_return = split["test_return_contexts"][:, -1, 0]
        baselines = {
            "zero": metric_dict(np.zeros_like(eval_y), eval_y),
            "last_return": metric_dict(last_return, eval_y),
        }
        scale_result["baselines"] = baselines

        if not args.skip_patchtst:
            patch_result, patch_pred = run_patchtst(
                name=name,
                split=split,
                dataset=dataset,
                output_dir=output_dir,
                device=patch_device,
                seed=args.seed,
            )
            scale_result["patchtst"] = patch_result
            append_summary(summary_rows, name, "patchtst", patch_result["test"], split, dataset)
            plot_trace(
                output_dir / f"{name}_patchtst_trace.png",
                title=f"{instrument} {name}: vanilla PatchTST",
                actual=eval_y,
                prediction=patch_pred,
                last_return=last_return,
            )
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            gc.collect()

        if not args.skip_fincast:
            fincast_result, fincast_pred = run_fincast(
                name=name,
                level_contexts=eval_level_contexts,
                actual_y=eval_y,
                frequency=dataset["frequency"],
                backend=args.fincast_backend,
                batch_size=max(1, args.fincast_batch_size),
            )
            scale_result["fincast"] = fincast_result
            append_summary(summary_rows, name, "fincast", fincast_result["test"], split, dataset)
            plot_trace(
                output_dir / f"{name}_fincast_trace.png",
                title=f"{instrument} {name}: frozen FinCast",
                actual=eval_y,
                prediction=fincast_pred,
                last_return=last_return,
            )

        for baseline_name, baseline_metrics in baselines.items():
            append_summary(summary_rows, name, baseline_name, baseline_metrics, split, dataset)

        write_predictions(
            output_dir / f"{name}_test_predictions_head.csv",
            actual=eval_y,
            last_return=last_return,
            patchtst=np.asarray(scale_result.get("patchtst", {}).get("prediction_head", []), dtype=np.float32),
            fincast=np.asarray(scale_result.get("fincast", {}).get("prediction_head", []), dtype=np.float32),
        )
        results["scales"][name] = scale_result

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    summary = pd.DataFrame(summary_rows)
    summary_path = output_dir / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"\nsaved_metrics={metrics_path}", flush=True)
    print(f"saved_summary={summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)


def read_stock_names(path: Path) -> dict[int, str]:
    if not zipfile.is_zipfile(path):
        return {}
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        shared_strings: list[str] = []
        ns = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        if "xl/sharedStrings.xml" in names:
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in shared_root.findall("a:si", ns):
                shared_strings.append("".join(t.text or "" for t in item.findall(".//a:t", ns)))
        sheet_name = next(name for name in names if name.startswith("xl/worksheets/sheet"))
        sheet_root = ET.fromstring(archive.read(sheet_name))
        rows: list[list[str]] = []
        for row in sheet_root.findall(".//a:row", ns):
            values: list[str] = []
            for cell in row.findall("a:c", ns):
                value = cell.find("a:v", ns)
                if value is None:
                    values.append("")
                elif cell.attrib.get("t") == "s":
                    values.append(shared_strings[int(value.text)])
                else:
                    values.append(value.text or "")
            rows.append(values)
    mapping: dict[int, str] = {}
    for row in rows[1:]:
        if len(row) >= 2:
            try:
                mapping[int(float(row[1]))] = str(row[0])
            except ValueError:
                continue
    return mapping


def build_stock_hour_price_matrix(
    *,
    data_root: Path,
    stock_id: int,
    valid_time_ids: set[int],
    time_reference: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    frames = []
    for filename in ["order_book_feature.csv", "order_book_target.csv"]:
        path = data_root / filename
        print(f"reading {path.name}", flush=True)
        stock_chunks = []
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=[
                "stock_id",
                "time_id",
                "seconds_in_bucket",
                "bid_price1",
                "ask_price1",
                "bid_size1",
                "ask_size1",
            ],
            chunksize=1_000_000,
        ):
            chunk = chunk[chunk["stock_id"] == stock_id].copy()
            if chunk.empty:
                continue
            chunk["seconds_in_bucket"] = chunk["seconds_in_bucket"].astype(int)
            stock_chunks.append(chunk)
        if not stock_chunks:
            raise ValueError(f"No rows for stock_id={stock_id} in {path}.")
        frames.append(pd.concat(stock_chunks, ignore_index=True))

    raw = pd.concat(frames, ignore_index=True)
    raw = raw[raw["time_id"].astype(int).isin(valid_time_ids)].copy()
    raw["wap1"] = (
        raw["bid_price1"] * raw["ask_size1"] + raw["ask_price1"] * raw["bid_size1"]
    ) / np.clip(raw["bid_size1"] + raw["ask_size1"], 1e-12, None)
    raw = raw.sort_values(["time_id", "seconds_in_bucket"]).drop_duplicates(
        ["time_id", "seconds_in_bucket"],
        keep="last",
    )

    matrix_rows: list[np.ndarray] = []
    time_ids: list[int] = []
    for time_id, group in raw.groupby("time_id", sort=True):
        series = group.set_index("seconds_in_bucket")["wap1"].sort_index()
        full = series.reindex(np.arange(3600, dtype=np.int64)).ffill().bfill()
        if full.isna().any():
            continue
        values = full.to_numpy(dtype=np.float32)
        if np.isfinite(values).all() and (values > 0).all():
            matrix_rows.append(values)
            time_ids.append(int(time_id))

    matrix = np.stack(matrix_rows, axis=0)
    meta = pd.DataFrame({"time_id": time_ids})
    meta = meta.merge(time_reference[["time_id", "date", "time", "timestamp"]], on="time_id", how="left")
    meta = meta.sort_values("timestamp").reset_index(drop=True)
    order = meta.index.to_numpy()
    return matrix[order], meta


def build_scale_datasets(price_matrix: np.ndarray, metadata: pd.DataFrame) -> dict[str, dict[str, object]]:
    second_segments = [row for row in price_matrix]
    minute_segments = [row.reshape(60, 60)[:, -1].astype(np.float32) for row in price_matrix]

    hour_segments: list[np.ndarray] = []
    for _, group in metadata.groupby("date", sort=True):
        idx = group.index.to_numpy()
        if len(idx) >= 5:
            hour_segments.append(price_matrix[idx, -1].astype(np.float32))

    day_close = (
        metadata.assign(row_index=np.arange(len(metadata)))
        .sort_values("timestamp")
        .groupby("date", sort=True)["row_index"]
        .last()
        .to_numpy()
    )
    day_series = price_matrix[day_close, -1].astype(np.float32)

    return {
        "second": make_segment_dataset(
            second_segments,
            context_returns=64,
            frequency="S",
            description="next 1-second log return within the same 1-hour time_id",
        ),
        "minute": make_segment_dataset(
            minute_segments,
            context_returns=16,
            frequency="MIN",
            description="next 1-minute log return within the same 1-hour time_id",
        ),
        "hour": make_segment_dataset(
            hour_segments,
            context_returns=3,
            frequency="H",
            description="next 1-hour log return within the same trading date",
        ),
        "day": make_continuous_dataset(
            day_series,
            context_returns=8,
            frequency="D",
            description="next 1-day log return from daily closes",
        ),
    }


def make_segment_dataset(
    segments: list[np.ndarray],
    *,
    context_returns: int,
    frequency: str,
    description: str,
) -> dict[str, object]:
    return_contexts: list[np.ndarray] = []
    level_contexts: list[np.ndarray] = []
    targets: list[float] = []
    segment_ids: list[int] = []
    for segment_id, levels in enumerate(segments):
        log_levels = np.log(np.asarray(levels, dtype=np.float64))
        returns = np.diff(log_levels).astype(np.float32)
        for end in range(context_returns, len(returns)):
            ret_context = returns[end - context_returns : end]
            level_context = np.asarray(levels[end - context_returns : end + 1], dtype=np.float32)
            target = returns[end]
            if np.isfinite(ret_context).all() and np.isfinite(level_context).all() and np.isfinite(target):
                return_contexts.append(ret_context[:, None])
                level_contexts.append(level_context)
                targets.append(float(target))
                segment_ids.append(segment_id)
    return {
        "return_contexts": np.asarray(return_contexts, dtype=np.float32),
        "level_contexts": np.asarray(level_contexts, dtype=np.float32),
        "targets": np.asarray(targets, dtype=np.float32),
        "segment_ids": np.asarray(segment_ids, dtype=np.int64),
        "context_returns": int(context_returns),
        "frequency": frequency,
        "description": description,
    }


def make_continuous_dataset(
    levels: np.ndarray,
    *,
    context_returns: int,
    frequency: str,
    description: str,
) -> dict[str, object]:
    return make_segment_dataset(
        [np.asarray(levels, dtype=np.float32)],
        context_returns=context_returns,
        frequency=frequency,
        description=description,
    )


def split_dataset(dataset: dict[str, object], args: argparse.Namespace) -> dict[str, object]:
    returns = np.asarray(dataset["return_contexts"], dtype=np.float32)
    levels = np.asarray(dataset["level_contexts"], dtype=np.float32)
    targets = np.asarray(dataset["targets"], dtype=np.float32)
    segment_ids = np.asarray(dataset["segment_ids"], dtype=np.int64)
    scale = str(dataset["frequency"])

    if len(np.unique(segment_ids)) > 1:
        unique_segments = np.unique(segment_ids)
        n_test = max(1, int(len(unique_segments) * 0.2))
        n_val = max(1, int(len(unique_segments) * 0.1))
        train_segments = set(unique_segments[: len(unique_segments) - n_val - n_test].tolist())
        val_segments = set(unique_segments[len(unique_segments) - n_val - n_test : len(unique_segments) - n_test].tolist())
        test_segments = set(unique_segments[len(unique_segments) - n_test :].tolist())
        split_masks = {
            "train": np.asarray([sid in train_segments for sid in segment_ids]),
            "validation": np.asarray([sid in val_segments for sid in segment_ids]),
            "test": np.asarray([sid in test_segments for sid in segment_ids]),
        }
    else:
        n = len(targets)
        n_test = max(1, int(n * 0.2))
        n_val = max(1, int(n * 0.1))
        n_train = n - n_val - n_test
        split_masks = {
            "train": np.arange(n) < n_train,
            "validation": (np.arange(n) >= n_train) & (np.arange(n) < n_train + n_val),
            "test": np.arange(n) >= n_train + n_val,
        }

    caps = {
        "train": args.second_train_cap if scale == "S" else 0,
        "validation": args.second_validation_cap if scale == "S" else 0,
        "test": args.second_test_cap if scale == "S" else 0,
    }
    output: dict[str, object] = {"meta": {"scale_frequency": scale}}
    for split_name, mask in split_masks.items():
        idx = np.flatnonzero(mask)
        selected = cap_indices(idx, int(caps[split_name]))
        output[f"{split_name}_return_contexts"] = returns[selected]
        output[f"{split_name}_level_contexts"] = levels[selected]
        output[f"{split_name}_y"] = targets[selected]
        output["meta"][f"{split_name}_total"] = int(len(idx))
        output["meta"][f"{split_name}_evaluated"] = int(len(selected))
    return output


def cap_indices(indices: np.ndarray, cap: int) -> np.ndarray:
    if cap <= 0 or cap >= len(indices):
        return indices
    selected_positions = np.linspace(0, len(indices) - 1, cap).round().astype(np.int64)
    return indices[selected_positions]


def run_patchtst(
    *,
    name: str,
    split: dict[str, object],
    dataset: dict[str, object],
    output_dir: Path,
    device: torch.device,
    seed: int,
) -> tuple[dict[str, object], np.ndarray]:
    torch.manual_seed(seed)
    train_x = np.asarray(split["train_return_contexts"], dtype=np.float32)
    train_y = np.asarray(split["train_y"], dtype=np.float32).reshape(-1, 1)
    val_x = np.asarray(split["validation_return_contexts"], dtype=np.float32)
    val_y = np.asarray(split["validation_y"], dtype=np.float32).reshape(-1, 1)
    test_x = np.asarray(split["test_return_contexts"], dtype=np.float32)
    test_y = np.asarray(split["test_y"], dtype=np.float32).reshape(-1, 1)

    normalizer = {
        "x_mean": float(train_x.mean()),
        "x_std": float(max(train_x.std(), 1e-8)),
        "y_mean": float(train_y.mean()),
        "y_std": float(max(train_y.std(), 1e-8)),
    }

    def norm_x(x: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((x - normalizer["x_mean"]) / normalizer["x_std"], dtype=torch.float32)

    def norm_y(y: np.ndarray) -> torch.Tensor:
        return torch.as_tensor((y - normalizer["y_mean"]) / normalizer["y_std"], dtype=torch.float32)

    batch_size = 512 if name in {"second", "minute"} else 32
    epochs = {"second": 5, "minute": 10, "hour": 40, "day": 60}[name]
    config = PatchTSTForecastConfig(
        context_length=int(dataset["context_returns"]),
        prediction_length=1,
        input_channels=1,
        patch_length={"second": 16, "minute": 4, "hour": 2, "day": 4}[name],
        patch_stride={"second": 8, "minute": 2, "hour": 1, "day": 2}[name],
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

    train_loader = DataLoader(TensorDataset(norm_x(train_x), norm_y(train_y)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(norm_x(val_x), norm_y(val_y)), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(norm_x(test_x), norm_y(test_y)), batch_size=batch_size, shuffle=False)

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
        train_loss = total / max(count, 1)
        val_metrics, _ = evaluate_patchtst(model, val_loader, normalizer, device)
        history.append({"epoch": epoch, "train_loss_scaled": train_loss, **{f"val_{k}": v for k, v in val_metrics.items()}})
        if val_metrics["mse"] < best_val:
            best_val = val_metrics["mse"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            print(
                f"{name} patchtst epoch={epoch} train_loss={train_loss:.6f} "
                f"val_mse={val_metrics['mse']:.8g} val_dir={val_metrics['direction_accuracy_nonzero']:.4f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    val_metrics, _ = evaluate_patchtst(model, val_loader, normalizer, device)
    test_metrics, test_pred = evaluate_patchtst(model, test_loader, normalizer, device)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "normalizer": normalizer,
            "history": history,
            "test": test_metrics,
        },
        output_dir / f"{name}_patchtst.pt",
    )
    return (
        {
            "model": "vanilla PatchTSTLoRA with rank=0",
            "config": asdict(config),
            "normalizer": normalizer,
            "parameters": count_parameters(model),
            "history": history,
            "validation": val_metrics,
            "test": test_metrics,
            "prediction_head": test_pred[:5000].astype(float).tolist(),
        },
        test_pred,
    )


@torch.no_grad()
def evaluate_patchtst(
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
        preds.append(pred)
        actual = yb.cpu().numpy().reshape(-1) * normalizer["y_std"] + normalizer["y_mean"]
        actuals.append(actual)
    pred_arr = np.concatenate(preds, axis=0).astype(np.float32)
    actual_arr = np.concatenate(actuals, axis=0).astype(np.float32)
    return metric_dict(pred_arr, actual_arr), pred_arr


_FINCAST_API = None


def run_fincast(
    *,
    name: str,
    level_contexts: np.ndarray,
    actual_y: np.ndarray,
    frequency: str,
    backend: str,
    batch_size: int,
) -> tuple[dict[str, object], np.ndarray]:
    global _FINCAST_API
    if _FINCAST_API is None:
        fincast_src = WORKSPACE_ROOT / "FinCast-fts" / "src"
        if str(fincast_src) not in sys.path:
            sys.path.insert(0, str(fincast_src))
        from tools.inference_utils import get_model_api

        print("loading frozen FinCast", flush=True)
        _FINCAST_API = get_model_api(
            SimpleNamespace(
                model_path=str(WORKSPACE_ROOT / "models" / "FinCast" / "v1.pth"),
                backend=backend,
                horizon_len=1,
                context_len=128,
                num_experts=4,
                gating_top_n=2,
                load_from_compile=True,
                forecast_mode="mean",
            )
        )

    from tools.inference_utils import freq_reader_inference

    freq_value = freq_reader_inference(frequency)
    pred_chunks = []
    for batch_index, start in enumerate(range(0, len(level_contexts), batch_size), start=1):
        end = min(start + batch_size, len(level_contexts))
        mean, _full = _FINCAST_API.forecast(
            [row for row in level_contexts[start:end]],
            freq=[freq_value] * (end - start),
        )
        pred_level = np.asarray(mean, dtype=np.float32)[:, 0]
        pred_return = np.log(np.clip(pred_level, 1e-12, None) / np.clip(level_contexts[start:end, -1], 1e-12, None))
        pred_chunks.append(pred_return.astype(np.float32))
        if batch_index == 1 or end == len(level_contexts) or batch_index % 50 == 0:
            print(f"{name} fincast forecasted {end}/{len(level_contexts)}", flush=True)
    pred = np.concatenate(pred_chunks, axis=0)
    test_metrics = metric_dict(pred, actual_y)
    return (
        {
            "model": "frozen FinCast v1 mean forecast",
            "frequency": frequency,
            "freq_token": int(freq_value),
            "test": test_metrics,
            "prediction_head": pred[:5000].astype(float).tolist(),
        },
        pred,
    )


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


def append_summary(
    rows: list[dict[str, object]],
    scale: str,
    model: str,
    metrics: dict[str, float],
    split: dict[str, object],
    dataset: dict[str, object],
) -> None:
    rows.append(
        {
            "scale": scale,
            "model": model,
            "test_mse": metrics["mse"],
            "test_rmse": metrics["rmse"],
            "test_mae": metrics["mae"],
            "test_direction_accuracy_nonzero": metrics["direction_accuracy_nonzero"],
            "test_corr": metrics["corr"],
            "test_total_windows": split["meta"]["test_total"],
            "test_evaluated_windows": split["meta"]["test_evaluated"],
            "context_returns": dataset["context_returns"],
            "frequency": dataset["frequency"],
        }
    )


def plot_trace(
    path: Path,
    *,
    title: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    last_return: np.ndarray,
) -> None:
    n = min(500, len(actual))
    fig, ax = plt.subplots(figsize=(12, 4.5))
    ax.plot(actual[:n], label="actual", linewidth=1.0)
    ax.plot(prediction[:n], label="prediction", linewidth=1.0, alpha=0.85)
    ax.plot(last_return[:n], label="last-return baseline", linewidth=0.8, alpha=0.55)
    ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel("evaluated test window index")
    ax.set_ylabel("log return")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def write_predictions(
    path: Path,
    *,
    actual: np.ndarray,
    last_return: np.ndarray,
    patchtst: np.ndarray,
    fincast: np.ndarray,
) -> None:
    n = min(5000, len(actual))
    frame = pd.DataFrame(
        {
            "actual_next_log_return": actual[:n],
            "last_return_baseline": last_return[:n],
        }
    )
    if len(patchtst):
        frame["patchtst_pred_next_log_return"] = patchtst[:n]
    if len(fincast):
        frame["fincast_pred_next_log_return"] = fincast[:n]
    frame.to_csv(path, index=False)


def select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
