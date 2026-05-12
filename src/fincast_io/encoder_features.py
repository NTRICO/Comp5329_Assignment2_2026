from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch

from src.datasets.sources import load_close_csv
from src.fincast_io.cache_builder import make_rolling_forecast_samples


PoolMode = Literal["last", "mean"]


def extract_fincast_encoder_features(
    model_api,
    contexts: np.ndarray,
    *,
    freq_value: int,
    batch_size: int = 16,
    pool: PoolMode = "last",
) -> np.ndarray:
    """Extract frozen FinCast transformer states before the forecast head.

    FinCast exposes forecasts through `decode()`, but the model also has an
    internal patched transformer. This helper keeps `FinCast-fts/` unchanged and
    calls that frozen transformer path directly. With `context_len=128` and
    `patch_len=32`, the hidden sequence has 4 patch tokens of width 1280.

    Args:
        model_api: Loaded FinCast API returned by `tools.inference_utils.get_model_api`.
        contexts: Rolling input windows shaped `[N, context_len]`.
        freq_value: FinCast frequency id, e.g. daily -> 0.
        batch_size: Number of contexts to process per call.
        pool: `"last"` returns the final patch token `[N, D]`; `"mean"` averages
            non-padded patch tokens `[N, D]`.

    Returns:
        Encoder features shaped `[N, hidden_dim]`.
    """

    if pool not in {"last", "mean"}:
        raise ValueError("pool must be 'last' or 'mean'.")
    contexts = np.asarray(contexts, dtype=np.float32)
    if contexts.ndim != 2:
        raise ValueError(f"contexts must be [N, context_len], got {contexts.shape}")

    model = model_api._model
    if model is None:
        raise ValueError("FinCast checkpoint is not loaded.")
    model.eval()
    device = next(model.parameters()).device

    features: list[np.ndarray] = []
    for start in range(0, len(contexts), batch_size):
        end = min(start + batch_size, len(contexts))
        batch_contexts = [row for row in contexts[start:end]]
        freqs = [freq_value] * len(batch_contexts)
        input_ts, input_padding, inp_freq, pmap_pad = model_api._preprocess(batch_contexts, freqs)
        real_count = len(batch_contexts)

        t_input_ts = torch.as_tensor(input_ts, dtype=torch.float32, device=device)
        t_input_padding = torch.as_tensor(
            input_padding[:, : t_input_ts.shape[1]],
            dtype=torch.float32,
            device=device,
        )
        t_freq = torch.as_tensor(inp_freq, dtype=torch.long, device=device)

        with torch.no_grad():
            model_input, patched_padding, _, _ = model._preprocess_input(
                input_ts=t_input_ts,
                input_padding=t_input_padding,
            )
            hidden = model_input + model.freq_emb(t_freq)
            hidden, _ = model.stacked_transformer(hidden, patched_padding)

            if pool == "last":
                valid_counts = (1.0 - patched_padding).sum(dim=1).long().clamp_min(1)
                row_idx = torch.arange(hidden.shape[0], device=device)
                pooled = hidden[row_idx, valid_counts - 1]
            else:
                valid_mask = (1.0 - patched_padding).unsqueeze(-1)
                pooled = (hidden * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)

        pooled_np = pooled.detach().cpu().numpy()
        if pmap_pad:
            pooled_np = pooled_np[:real_count]
        features.append(pooled_np.astype(np.float32))

    return np.concatenate(features, axis=0)


def build_fincast_encoder_cache(
    *,
    csv_path: str | Path,
    model_path: str | Path,
    output_path: str | Path,
    fincast_root: str | Path,
    tickers: list[str] | None,
    context_len: int = 128,
    holding_horizon: int = 1,
    data_frequency: str = "D",
    stride: int = 1,
    max_windows_per_asset: int | None = None,
    batch_size: int = 16,
    backend: str = "gpu",
    pool: PoolMode = "last",
) -> Path:
    """Run frozen FinCast encoder on rolling windows and save feature cache."""

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
        horizon_len=32,
        context_len=context_len,
        num_experts=4,
        gating_top_n=2,
        load_from_compile=True,
        forecast_mode="mean",
    )
    model_api = get_model_api(config)
    freq_value = freq_reader_inference(data_frequency)
    features = extract_fincast_encoder_features(
        model_api,
        samples.contexts,
        freq_value=freq_value,
        batch_size=batch_size,
        pool=pool,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        encoder_features=features,
        last_values=samples.last_values,
        realized_returns=samples.realized_returns,
        asset_names=samples.asset_names,
        window_end_indices=samples.window_end_indices,
        dates=samples.dates,
        context_len=np.asarray(context_len, dtype=np.int64),
        holding_horizon=np.asarray(holding_horizon, dtype=np.int64),
        data_frequency=np.asarray(data_frequency),
        pool=np.asarray(pool),
    )
    return output_path
