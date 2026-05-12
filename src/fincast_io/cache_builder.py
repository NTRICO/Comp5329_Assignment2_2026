from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.datasets.sources import load_close_csv

FORECAST_CHANNEL_NAMES: tuple[str, ...] = (
    "mean",
    "q10",
    "q20",
    "q30",
    "q40",
    "q50",
    "q60",
    "q70",
    "q80",
    "q90",
)


@dataclass(frozen=True)
class RollingForecastSamples:
    contexts: np.ndarray
    last_values: np.ndarray
    realized_returns: np.ndarray
    asset_names: np.ndarray
    window_end_indices: np.ndarray
    dates: np.ndarray


def make_rolling_forecast_samples(
    close_df: pd.DataFrame,
    *,
    context_len: int,
    holding_horizon: int = 1,
    tickers: list[str] | None = None,
    stride: int = 5,
    max_windows_per_asset: int | None = 256,
) -> RollingForecastSamples:
    """Create controlled rolling windows from a wide close-price dataframe."""

    if context_len <= 0:
        raise ValueError("context_len must be positive.")
    if holding_horizon <= 0:
        raise ValueError("holding_horizon must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")

    df = close_df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="raise")
    asset_cols = tickers or [col for col in df.columns if col != "Date"]

    contexts: list[np.ndarray] = []
    last_values: list[float] = []
    realized_returns: list[float] = []
    asset_names: list[str] = []
    window_end_indices: list[int] = []
    dates: list[str] = []

    for asset in asset_cols:
        if asset not in df.columns:
            raise KeyError(f"Ticker column not found: {asset}")
        values = pd.to_numeric(df[asset], errors="coerce").to_numpy(dtype=np.float32)
        valid = np.isfinite(values)
        if not valid.all():
            values = values[valid]
            asset_dates = df.loc[valid, "Date"].reset_index(drop=True)
        else:
            asset_dates = df["Date"].reset_index(drop=True)

        first_end = context_len - 1
        last_end = len(values) - holding_horizon - 1
        if last_end < first_end:
            continue

        end_positions = np.arange(first_end, last_end + 1, stride, dtype=np.int64)
        if max_windows_per_asset is not None and len(end_positions) > max_windows_per_asset:
            keep = np.linspace(0, len(end_positions) - 1, max_windows_per_asset)
            end_positions = end_positions[np.unique(keep.round().astype(np.int64))]

        for end_idx in end_positions:
            start_idx = end_idx - context_len + 1
            last_value = float(values[end_idx])
            future_value = float(values[end_idx + holding_horizon])
            if abs(last_value) < 1e-8:
                continue
            contexts.append(values[start_idx : end_idx + 1].astype(np.float32))
            last_values.append(last_value)
            realized_returns.append(future_value / last_value - 1.0)
            asset_names.append(str(asset))
            window_end_indices.append(int(end_idx))
            dates.append(pd.Timestamp(asset_dates.iloc[end_idx]).strftime("%Y-%m-%d"))

    if not contexts:
        raise ValueError("No rolling samples were created. Check context_len, horizon, and data length.")

    return RollingForecastSamples(
        contexts=np.stack(contexts, axis=0).astype(np.float32),
        last_values=np.asarray(last_values, dtype=np.float32),
        realized_returns=np.asarray(realized_returns, dtype=np.float32),
        asset_names=np.asarray(asset_names, dtype=object),
        window_end_indices=np.asarray(window_end_indices, dtype=np.int64),
    dates=np.asarray(dates, dtype=object),
    )


def _forecast_col(channel_name: str, horizon_index: int) -> str:
    return f"{channel_name}_t+{horizon_index + 1:03d}"


def distribution_cache_to_frame(
    *,
    full_outputs: np.ndarray,
    last_values: np.ndarray,
    realized_returns: np.ndarray,
    asset_names: np.ndarray | None = None,
    dates: np.ndarray | None = None,
    window_end_indices: np.ndarray | None = None,
) -> pd.DataFrame:
    """Flatten [N, H, C] FinCast outputs into a wide CSV-friendly dataframe."""

    full_outputs = np.asarray(full_outputs, dtype=np.float32)
    if full_outputs.ndim != 3:
        raise ValueError(f"full_outputs must be [N, H, C], got {full_outputs.shape}")
    n_samples, horizon_len, channels = full_outputs.shape
    if channels > len(FORECAST_CHANNEL_NAMES):
        raise ValueError(f"Unsupported number of forecast channels: {channels}")

    last_values = np.asarray(last_values, dtype=np.float32).reshape(-1)
    realized_returns = np.asarray(realized_returns, dtype=np.float32).reshape(-1)
    if len(last_values) != n_samples or len(realized_returns) != n_samples:
        raise ValueError("metadata arrays must have one row per forecast sample.")

    data: dict[str, np.ndarray] = {
        "sample_id": np.arange(n_samples, dtype=np.int64),
        "asset": np.asarray(asset_names if asset_names is not None else ["asset"] * n_samples),
        "date": np.asarray(dates if dates is not None else [""] * n_samples),
        "window_end_index": np.asarray(
            window_end_indices if window_end_indices is not None else np.arange(n_samples),
            dtype=np.int64,
        ),
        "last_value": last_values,
        "realized_return": realized_returns,
    }

    channel_names = FORECAST_CHANNEL_NAMES[:channels]
    for h in range(horizon_len):
        for c, channel_name in enumerate(channel_names):
            data[_forecast_col(channel_name, h)] = full_outputs[:, h, c]
    return pd.DataFrame(data)


def save_distribution_cache(
    output_path: str | Path,
    *,
    full_outputs: np.ndarray,
    last_values: np.ndarray,
    realized_returns: np.ndarray,
    asset_names: np.ndarray | None = None,
    dates: np.ndarray | None = None,
    window_end_indices: np.ndarray | None = None,
    context_len: int | None = None,
    horizon_len: int | None = None,
    holding_horizon: int | None = None,
    data_frequency: str | None = None,
) -> Path:
    """Save distribution cache as CSV or NPZ based on output suffix."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        df = distribution_cache_to_frame(
            full_outputs=full_outputs,
            last_values=last_values,
            realized_returns=realized_returns,
            asset_names=asset_names,
            dates=dates,
            window_end_indices=window_end_indices,
        )
        df.to_csv(output_path, index=False)
        return output_path
    if suffix == ".npz":
        np.savez(
            output_path,
            full_outputs=full_outputs,
            last_values=last_values,
            realized_returns=realized_returns,
            asset_names=asset_names,
            window_end_indices=window_end_indices,
            dates=dates,
            context_len=np.asarray(context_len if context_len is not None else -1, dtype=np.int64),
            horizon_len=np.asarray(horizon_len if horizon_len is not None else full_outputs.shape[1], dtype=np.int64),
            holding_horizon=np.asarray(holding_horizon if holding_horizon is not None else -1, dtype=np.int64),
            data_frequency=np.asarray(data_frequency if data_frequency is not None else ""),
        )
        return output_path
    raise ValueError(f"Unsupported cache suffix {suffix!r}; use .csv or .npz.")


def load_distribution_cache(path: str | Path) -> dict[str, np.ndarray]:
    """Load a distribution cache saved as CSV or NPZ."""

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        cache = np.load(path, allow_pickle=True)
        return {
            "full_outputs": cache["full_outputs"].astype(np.float32),
            "last_values": cache["last_values"].astype(np.float32),
            "realized_returns": cache["realized_returns"].astype(np.float32),
            "asset_names": cache["asset_names"] if "asset_names" in cache else np.asarray([]),
            "dates": cache["dates"] if "dates" in cache else np.asarray([]),
        }
    if suffix != ".csv":
        raise ValueError(f"Unsupported cache suffix {suffix!r}; use .csv or .npz.")

    df = pd.read_csv(path)
    required = {"last_value", "realized_return"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV cache is missing required columns: {sorted(missing)}")

    horizon_indices: list[int] = []
    for col in df.columns:
        if col.startswith("mean_t+"):
            horizon_indices.append(int(col.split("+", 1)[1]) - 1)
    if not horizon_indices:
        raise ValueError("CSV cache does not contain mean_t+NNN forecast columns.")
    horizon_len = max(horizon_indices) + 1

    channel_names = [
        name
        for name in FORECAST_CHANNEL_NAMES
        if all(_forecast_col(name, h) in df.columns for h in range(horizon_len))
    ]
    if not channel_names:
        raise ValueError("CSV cache does not contain a complete forecast channel set.")

    full_outputs = np.empty((len(df), horizon_len, len(channel_names)), dtype=np.float32)
    for h in range(horizon_len):
        for c, channel_name in enumerate(channel_names):
            full_outputs[:, h, c] = pd.to_numeric(df[_forecast_col(channel_name, h)], errors="raise")

    return {
        "full_outputs": full_outputs,
        "last_values": pd.to_numeric(df["last_value"], errors="raise").to_numpy(dtype=np.float32),
        "realized_returns": pd.to_numeric(df["realized_return"], errors="raise").to_numpy(dtype=np.float32),
        "asset_names": df["asset"].to_numpy(dtype=object) if "asset" in df.columns else np.asarray([]),
        "dates": df["date"].to_numpy(dtype=object) if "date" in df.columns else np.asarray([]),
    }


def build_fincast_distribution_cache(
    *,
    csv_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    fincast_root: str | Path,
    tickers: list[str] | None,
    context_len: int = 128,
    horizon_len: int = 32,
    holding_horizon: int = 1,
    data_frequency: str = "D",
    stride: int = 5,
    max_windows_per_asset: int | None = 256,
    batch_size: int = 16,
    backend: str = "gpu",
) -> Path:
    """Run frozen FinCast on controlled rolling windows and save a cache."""

    fincast_src = Path(fincast_root) / "src"
    if str(fincast_src) not in sys.path:
        sys.path.insert(0, str(fincast_src))

    from tools.inference_utils import freq_reader_inference, get_model_api

    close_df = load_close_csv(csv_path)
    samples = make_rolling_forecast_samples(
        close_df,
        context_len=context_len,
        holding_horizon=holding_horizon,
        tickers=tickers,
        stride=stride,
        max_windows_per_asset=max_windows_per_asset,
    )

    config = SimpleNamespace(
        model_path=str(model_path),
        backend=backend,
        horizon_len=horizon_len,
        context_len=context_len,
        num_experts=4,
        gating_top_n=2,
        load_from_compile=True,
        forecast_mode="mean",
    )
    model_api = get_model_api(config)
    freq_value = freq_reader_inference(data_frequency)

    full_outputs: list[np.ndarray] = []
    contexts = samples.contexts
    for start in range(0, len(contexts), batch_size):
        end = min(start + batch_size, len(contexts))
        batch_contexts = [row for row in contexts[start:end]]
        freqs = [freq_value] * len(batch_contexts)
        _, full = model_api.forecast(batch_contexts, freq=freqs)
        full_outputs.append(full[:, -horizon_len:, :].astype(np.float32))

    full_outputs_arr = np.concatenate(full_outputs, axis=0)
    return save_distribution_cache(
        output_path,
        full_outputs=full_outputs_arr,
        last_values=samples.last_values,
        realized_returns=samples.realized_returns,
        asset_names=samples.asset_names,
        window_end_indices=samples.window_end_indices,
        dates=samples.dates,
        context_len=context_len,
        horizon_len=horizon_len,
        holding_horizon=holding_horizon,
        data_frequency=data_frequency,
    )
