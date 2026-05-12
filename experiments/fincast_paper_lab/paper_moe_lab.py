from __future__ import annotations

from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Iterable
import math
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from ffm import FFmHparams
from tools.model_utils import get_model_FFM


def make_model_config(
    *,
    model_path: str | Path,
    backend: str = "gpu",
    context_len: int = 128,
    horizon_len: int = 32,
    per_core_batch_size: int = 32,
    forecast_mode: str = "mean",
    num_experts: int = 4,
    gating_top_n: int = 2,
    threshold_train: float = 0.2,
    threshold_eval: float = 0.2,
    load_from_compile: bool = True,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_path=str(Path(model_path)),
        backend=backend,
        context_len=int(context_len),
        horizon_len=int(horizon_len),
        per_core_batch_size=int(per_core_batch_size),
        forecast_mode=forecast_mode,
        num_experts=int(num_experts),
        gating_top_n=int(gating_top_n),
        threshold_train=float(threshold_train),
        threshold_eval=float(threshold_eval),
        load_from_compile=bool(load_from_compile),
    )


def load_fincast_api(config: SimpleNamespace):
    hparams = FFmHparams(
        backend=config.backend,
        per_core_batch_size=config.per_core_batch_size,
        horizon_len=config.horizon_len,
        context_len=config.context_len,
        use_positional_embedding=False,
        num_experts=config.num_experts,
        gating_top_n=config.gating_top_n,
        threshold_train=config.threshold_train,
        threshold_eval=config.threshold_eval,
        load_from_compile=config.load_from_compile,
        point_forecast_mode=config.forecast_mode,
    )

    model_actual, _, ffm_api = get_model_FFM(config.model_path, hparams)
    ffm_api.model_eval_mode()
    return model_actual, ffm_api


def select_numeric_columns(
    csv_path: str | Path,
    *,
    min_points: int,
) -> list[str]:
    df = pd.read_csv(csv_path)
    selected: list[str] = []

    for col in df.columns:
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().sum() >= min_points:
            selected.append(col)

    return selected


def build_backtest_windows(
    csv_path: str | Path,
    *,
    context_len: int,
    horizon_len: int,
    target_columns: Iterable[str] | None = None,
    windows_per_series: int = 3,
    stride: int | None = None,
    frequency_code: int = 0,
) -> list[dict[str, Any]]:
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    if target_columns is None:
        target_columns = select_numeric_columns(
            csv_path,
            min_points=context_len + horizon_len,
        )

    stride = int(stride or horizon_len)
    windows: list[dict[str, Any]] = []

    for col in target_columns:
        series = pd.to_numeric(df[col], errors="coerce").dropna().to_numpy(dtype=np.float32)
        if len(series) < context_len + horizon_len:
            continue

        valid_end_positions = list(range(context_len, len(series) - horizon_len + 1, stride))
        valid_end_positions = valid_end_positions[-int(windows_per_series) :]

        for local_idx, end_idx in enumerate(valid_end_positions):
            windows.append(
                {
                    "window_id": f"{col}__{local_idx}",
                    "series_name": col,
                    "context": series[end_idx - context_len : end_idx].copy(),
                    "future": series[end_idx : end_idx + horizon_len].copy(),
                    "window_end": int(end_idx),
                    "window_start": int(end_idx - context_len),
                    "frequency_code": int(frequency_code),
                    "csv_path": str(csv_path),
                }
            )

    if not windows:
        raise ValueError(
            "No valid evaluation windows were generated. "
            "Check the csv path, target columns, context_len, and horizon_len."
        )

    return windows


def windows_to_frame(windows: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for item in windows:
        rows.append(
            {
                "window_id": item["window_id"],
                "series_name": item["series_name"],
                "window_start": item["window_start"],
                "window_end": item["window_end"],
                "context_len": len(item["context"]),
                "horizon_len": len(item["future"]),
                "frequency_code": item["frequency_code"],
            }
        )
    return pd.DataFrame(rows)


def get_decoder_layers(model) -> list[tuple[int, Any]]:
    return list(enumerate(model.stacked_transformer.layers))


def summarize_moe_layers(model) -> pd.DataFrame:
    rows = []
    for layer_idx, layer in get_decoder_layers(model):
        gate = layer.moe.moe.gate
        moe = layer.moe.moe
        rows.append(
            {
                "layer_idx": layer_idx,
                "num_experts": int(moe.num_experts),
                "gating_top_n": int(gate.top_n),
                "threshold_train": list(map(float, gate.threshold_train.detach().cpu().tolist()[1:])),
                "threshold_eval": list(map(float, gate.threshold_eval.detach().cpu().tolist()[1:])),
                "capacity_factor_train": float(gate.capacity_factor_train),
                "capacity_factor_eval": float(gate.capacity_factor_eval),
                "balance_loss_coef": float(moe.balance_loss_coef),
                "router_z_loss_coef": float(moe.router_z_loss_coef),
            }
        )
    return pd.DataFrame(rows)


def _normalize_threshold_values(
    value: float | list[float] | tuple[float, ...] | None,
    *,
    target_len: int,
    fallback: list[float],
) -> list[float]:
    if target_len <= 0:
        return []

    if value is None:
        base = list(fallback)
    elif isinstance(value, (list, tuple, np.ndarray)):
        base = [float(v) for v in value]
    else:
        base = [float(value)]

    if not base:
        base = [0.2]

    if len(base) < target_len:
        base = base + [base[-1]] * (target_len - len(base))

    return base[:target_len]


def _selected_layers(model, layer_indices: str | int | Iterable[int] | None):
    layers = get_decoder_layers(model)

    if layer_indices in (None, "all"):
        return layers

    if isinstance(layer_indices, int):
        wanted = {layer_indices}
    else:
        wanted = {int(idx) for idx in layer_indices}

    return [(idx, layer) for idx, layer in layers if idx in wanted]


def apply_moe_runtime_patch(
    model,
    patch: dict[str, Any] | None,
) -> pd.DataFrame:
    if not patch:
        return summarize_moe_layers(model)

    structural_num_experts = patch.get("num_experts")
    if structural_num_experts is not None:
        raise ValueError(
            "Changing num_experts is not hot-swappable for the released checkpoint. "
            "The checkpoint is fixed to 4 experts."
        )

    for layer_idx, layer in _selected_layers(model, patch.get("layer_indices", "all")):
        gate = layer.moe.moe.gate
        moe = layer.moe.moe

        new_top_n = int(patch.get("gating_top_n", gate.top_n))
        if new_top_n < 2 or new_top_n > gate.num_gates:
            raise ValueError(
                f"Invalid gating_top_n={new_top_n} for layer {layer_idx}. "
                f"It must be in [2, {gate.num_gates}]."
            )

        gate.top_n = new_top_n
        threshold_len = new_top_n - 1
        current_train = list(map(float, gate.threshold_train.detach().cpu().tolist()[1:]))
        current_eval = list(map(float, gate.threshold_eval.detach().cpu().tolist()[1:]))

        train_values = _normalize_threshold_values(
            patch.get("threshold_train"),
            target_len=threshold_len,
            fallback=current_train,
        )
        eval_values = _normalize_threshold_values(
            patch.get("threshold_eval"),
            target_len=threshold_len,
            fallback=current_eval,
        )

        eps = float(gate.eps)
        gate.threshold_train = torch.tensor(
            [eps, *train_values],
            dtype=gate.threshold_train.dtype,
            device=gate.threshold_train.device,
        )
        gate.threshold_eval = torch.tensor(
            [eps, *eval_values],
            dtype=gate.threshold_eval.dtype,
            device=gate.threshold_eval.device,
        )

        if patch.get("capacity_factor_train") is not None:
            gate.capacity_factor_train = float(patch["capacity_factor_train"])
        if patch.get("capacity_factor_eval") is not None:
            gate.capacity_factor_eval = float(patch["capacity_factor_eval"])
        if patch.get("balance_loss_coef") is not None:
            moe.balance_loss_coef = float(patch["balance_loss_coef"])
        if patch.get("router_z_loss_coef") is not None:
            moe.router_z_loss_coef = float(patch["router_z_loss_coef"])

    return summarize_moe_layers(model)


def start_moe_routing_capture(model):
    routing_state: dict[int, dict[str, Any]] = {}
    originals: list[tuple[Any, Any]] = []

    for layer_idx, layer in get_decoder_layers(model):
        gate = layer.moe.moe.gate
        routing_state[layer_idx] = {
            "dispatch_sum": torch.zeros(gate.num_gates, dtype=torch.float64),
            "calls": 0,
        }
        original_forward = gate.forward

        def wrapped_forward(
            self,
            x,
            noise_gates: bool = False,
            noise_mult: float = 1.0,
            *,
            _layer_idx: int = layer_idx,
            _original=original_forward,
        ):
            dispatch_tensor, combine_tensor, balance_loss, router_z_loss = _original(
                x,
                noise_gates=noise_gates,
                noise_mult=noise_mult,
            )
            state = routing_state[_layer_idx]
            dispatch_sum = dispatch_tensor.detach().float().sum(dim=(0, 1, 3)).cpu().to(torch.float64)
            state["dispatch_sum"] += dispatch_sum
            state["calls"] += 1
            state["top_n"] = int(self.top_n)
            state["num_gates"] = int(self.num_gates)
            state["threshold_eval"] = list(map(float, self.threshold_eval.detach().cpu().tolist()[1:]))
            state["threshold_train"] = list(map(float, self.threshold_train.detach().cpu().tolist()[1:]))
            return dispatch_tensor, combine_tensor, balance_loss, router_z_loss

        gate.forward = MethodType(wrapped_forward, gate)
        originals.append((gate, original_forward))

    return routing_state, originals


def stop_moe_routing_capture(
    routing_state: dict[int, dict[str, Any]],
    originals: list[tuple[Any, Any]],
) -> pd.DataFrame:
    for gate, original_forward in originals:
        gate.forward = original_forward

    rows = []
    for layer_idx, state in routing_state.items():
        dispatch_sum = state["dispatch_sum"]
        total_assignments = float(dispatch_sum.sum().item())
        for expert_idx, assignments in enumerate(dispatch_sum.tolist()):
            rows.append(
                {
                    "layer_idx": layer_idx,
                    "expert_idx": expert_idx,
                    "assignments": float(assignments),
                    "fraction": float(assignments / total_assignments) if total_assignments > 0 else math.nan,
                    "calls": int(state.get("calls", 0)),
                    "top_n": int(state.get("top_n", 0)),
                    "num_gates": int(state.get("num_gates", 0)),
                    "threshold_eval": state.get("threshold_eval", []),
                    "threshold_train": state.get("threshold_train", []),
                }
            )
    return pd.DataFrame(rows)


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(200.0 * np.mean(np.abs(y_true - y_pred) / denom))


def _point_forecast_from_outputs(
    mean_outputs: np.ndarray,
    full_outputs: np.ndarray,
    *,
    forecast_mode: str,
) -> np.ndarray:
    if forecast_mode == "mean":
        return mean_outputs

    if forecast_mode == "median":
        return full_outputs[:, :, 5]

    raise ValueError(f"Unsupported forecast_mode={forecast_mode}")


def _predictions_to_frame(
    windows: list[dict[str, Any]],
    point_outputs: np.ndarray,
    full_outputs: np.ndarray,
    *,
    experiment_name: str,
) -> pd.DataFrame:
    rows = []
    quantile_count = full_outputs.shape[2] - 1

    for row_idx, meta in enumerate(windows):
        for step in range(point_outputs.shape[1]):
            row = {
                "experiment": experiment_name,
                "window_id": meta["window_id"],
                "series_name": meta["series_name"],
                "horizon_step": step + 1,
                "actual": float(meta["future"][step]),
                "prediction": float(point_outputs[row_idx, step]),
                "mean": float(full_outputs[row_idx, step, 0]),
            }
            for quantile_idx in range(1, quantile_count + 1):
                row[f"q{quantile_idx}"] = float(full_outputs[row_idx, step, quantile_idx])
            rows.append(row)

    return pd.DataFrame(rows)


def _metrics_to_frame(
    windows: list[dict[str, Any]],
    point_outputs: np.ndarray,
    *,
    experiment_name: str,
    runtime_sec: float,
) -> pd.DataFrame:
    rows = []
    for row_idx, meta in enumerate(windows):
        y_true = meta["future"]
        y_pred = point_outputs[row_idx]
        rows.append(
            {
                "experiment": experiment_name,
                "window_id": meta["window_id"],
                "series_name": meta["series_name"],
                "mae": mae(y_true, y_pred),
                "rmse": rmse(y_true, y_pred),
                "smape": smape(y_true, y_pred),
                "runtime_sec_total": float(runtime_sec),
            }
        )
    return pd.DataFrame(rows)


def summarize_metrics(metrics_df: pd.DataFrame) -> pd.DataFrame:
    if metrics_df.empty:
        return pd.DataFrame()

    series_summary = (
        metrics_df.groupby(["experiment", "series_name"], as_index=False)[["mae", "rmse", "smape"]]
        .mean()
        .sort_values(["experiment", "series_name"])
    )

    overall = (
        metrics_df.groupby("experiment", as_index=False)[["mae", "rmse", "smape"]]
        .mean()
        .assign(series_name="__overall__")
    )

    return pd.concat([series_summary, overall], ignore_index=True)


def run_experiment(
    *,
    experiment_name: str,
    model_config: SimpleNamespace,
    windows: list[dict[str, Any]],
    moe_patch: dict[str, Any] | None = None,
    forecast_mode: str = "mean",
    capture_routing: bool = True,
) -> dict[str, Any]:
    model, api = load_fincast_api(model_config)
    moe_summary_before = summarize_moe_layers(model)
    moe_summary_after = apply_moe_runtime_patch(model, moe_patch)

    routing_state = None
    originals = None
    if capture_routing:
        routing_state, originals = start_moe_routing_capture(model)

    contexts = [item["context"] for item in windows]
    freqs = [item["frequency_code"] for item in windows]

    start = time.perf_counter()
    mean_outputs, full_outputs = api.forecast(contexts, freqs)
    runtime_sec = time.perf_counter() - start

    routing_df = pd.DataFrame()
    if capture_routing and routing_state is not None and originals is not None:
        routing_df = stop_moe_routing_capture(routing_state, originals)

    point_outputs = _point_forecast_from_outputs(
        mean_outputs,
        full_outputs,
        forecast_mode=forecast_mode,
    )

    predictions_df = _predictions_to_frame(
        windows,
        point_outputs,
        full_outputs,
        experiment_name=experiment_name,
    )
    metrics_df = _metrics_to_frame(
        windows,
        point_outputs,
        experiment_name=experiment_name,
        runtime_sec=runtime_sec,
    )
    summary_df = summarize_metrics(metrics_df)

    return {
        "name": experiment_name,
        "model_config": model_config,
        "moe_patch": moe_patch,
        "runtime_sec": runtime_sec,
        "windows": windows,
        "point_outputs": point_outputs,
        "full_outputs": full_outputs,
        "predictions_df": predictions_df,
        "metrics_df": metrics_df,
        "summary_df": summary_df,
        "routing_df": routing_df,
        "moe_summary_before": moe_summary_before,
        "moe_summary_after": moe_summary_after,
    }


def compare_experiments(*results: dict[str, Any]) -> pd.DataFrame:
    frames = [item["summary_df"] for item in results if item.get("summary_df") is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def make_delta_table(
    baseline_result: dict[str, Any],
    patched_result: dict[str, Any],
) -> pd.DataFrame:
    left = baseline_result["summary_df"].copy().rename(
        columns={
            "mae": "mae_baseline",
            "rmse": "rmse_baseline",
            "smape": "smape_baseline",
        }
    )
    right = patched_result["summary_df"].copy().rename(
        columns={
            "mae": "mae_patched",
            "rmse": "rmse_patched",
            "smape": "smape_patched",
        }
    )

    merged = left.merge(right, on="series_name", how="inner")
    for metric in ("mae", "rmse", "smape"):
        merged[f"{metric}_delta"] = merged[f"{metric}_patched"] - merged[f"{metric}_baseline"]
    return merged


def plot_window_forecast(
    result: dict[str, Any],
    *,
    series_name: str | None = None,
    window_id: str | None = None,
    quantiles: tuple[int, ...] = (1, 5, 9),
):
    predictions_df = result["predictions_df"]
    windows = result["windows"]

    if window_id is None:
        for item in windows:
            if series_name is None or item["series_name"] == series_name:
                window_id = item["window_id"]
                break

    if window_id is None:
        raise ValueError("No matching window found for plotting.")

    meta = next(item for item in windows if item["window_id"] == window_id)
    plot_df = predictions_df[predictions_df["window_id"] == window_id].copy()

    x_context = np.arange(len(meta["context"]))
    x_future = np.arange(len(meta["context"]), len(meta["context"]) + len(meta["future"]))

    plt.figure(figsize=(10, 4))
    plt.plot(x_context, meta["context"], label="context", linewidth=1.8)
    plt.plot(x_future, meta["future"], label="actual", linewidth=1.8)
    plt.plot(x_future, plot_df["prediction"].to_numpy(), label=result["name"], linewidth=2.0)

    for quantile_idx in quantiles:
        col = f"q{quantile_idx}"
        if col in plot_df.columns:
            plt.plot(
                x_future,
                plot_df[col].to_numpy(),
                linestyle="--",
                linewidth=1.1,
                label=col,
            )

    plt.title(f"{result['name']} | {meta['series_name']} | {window_id}")
    plt.xlabel("relative step")
    plt.ylabel("value")
    plt.grid(alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()


def plot_routing_summary(
    routing_df: pd.DataFrame,
    *,
    value_col: str = "fraction",
):
    if routing_df.empty:
        print("Routing capture is empty.")
        return

    pivot = routing_df.pivot(index="layer_idx", columns="expert_idx", values=value_col).sort_index()
    plt.figure(figsize=(10, max(4, len(pivot) * 0.22)))
    plt.imshow(pivot.to_numpy(), aspect="auto")
    plt.colorbar(label=value_col)
    plt.xticks(range(len(pivot.columns)), pivot.columns)
    plt.yticks(range(len(pivot.index)), pivot.index)
    plt.xlabel("expert_idx")
    plt.ylabel("layer_idx")
    plt.title(f"MoE routing summary ({value_col})")
    plt.tight_layout()
    plt.show()
