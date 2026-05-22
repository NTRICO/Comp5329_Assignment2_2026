from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from src.datasets.sources import load_close_csv
from src.fincast_io.cache_builder import make_rolling_forecast_samples, save_distribution_cache


SimpleForecastMode = Literal["random_walk", "rolling_mean", "ewma_vol"]


NORMAL_QUANTILE_Z = np.asarray(
    [
        -1.2815515655446004,
        -0.8416212335729143,
        -0.5244005127080409,
        -0.2533471031357997,
        0.0,
        0.2533471031357997,
        0.5244005127080409,
        0.8416212335729143,
        1.2815515655446004,
    ],
    dtype=np.float32,
)


@dataclass(frozen=True)
class SimpleForecastConfig:
    """Rolling Gaussian baseline with the same cache contract as FinCast."""

    mode: SimpleForecastMode = "random_walk"
    volatility_window: int | None = None
    ewma_alpha: float = 0.06
    min_sigma: float = 1e-6
    eps: float = 1e-8


def simple_gaussian_level_forecasts(
    contexts: np.ndarray,
    *,
    horizon_len: int,
    config: SimpleForecastConfig = SimpleForecastConfig(),
) -> np.ndarray:
    """Create [N, H, 10] level forecasts from rolling historical returns.

    Channel 0 is the arithmetic mean path. Channels 1-9 are q10..q90 under a
    Gaussian return approximation. The output intentionally mirrors FinCast's
    cache shape so downstream position models can run unchanged.
    """

    if horizon_len <= 0:
        raise ValueError("horizon_len must be positive.")
    contexts = np.asarray(contexts, dtype=np.float32)
    if contexts.ndim != 2:
        raise ValueError(f"contexts must be [N, context_len], got {contexts.shape}")
    if contexts.shape[1] < 2:
        raise ValueError("contexts must contain at least two prices.")
    if not np.isfinite(contexts).all():
        raise ValueError("contexts contain non-finite values.")
    if (contexts <= config.eps).any():
        raise ValueError("simple Gaussian forecasts require positive price contexts.")

    returns = contexts[:, 1:] / contexts[:, :-1] - 1.0
    if config.volatility_window is not None:
        if config.volatility_window <= 0:
            raise ValueError("volatility_window must be positive when provided.")
        returns = returns[:, -config.volatility_window :]

    mu, sigma = _estimate_return_moments(returns, config)
    horizon = np.arange(1, horizon_len + 1, dtype=np.float32)[None, :]
    loc = horizon * mu[:, None]
    scale = np.sqrt(horizon) * sigma[:, None]

    return_paths = np.empty((contexts.shape[0], horizon_len, 10), dtype=np.float32)
    return_paths[:, :, 0] = loc
    return_paths[:, :, 1:] = loc[:, :, None] + scale[:, :, None] * NORMAL_QUANTILE_Z

    last_values = contexts[:, -1].astype(np.float32)
    level_paths = last_values[:, None, None] * (1.0 + return_paths)
    return np.maximum(level_paths, config.eps).astype(np.float32)


def build_simple_distribution_cache(
    *,
    csv_path: str | Path,
    output_path: str | Path,
    tickers: list[str] | None,
    context_len: int = 128,
    horizon_len: int = 32,
    holding_horizon: int = 1,
    data_frequency: str = "D",
    stride: int = 1,
    max_windows_per_asset: int | None = None,
    forecast_config: SimpleForecastConfig = SimpleForecastConfig(),
) -> Path:
    """Build a naive Simple-FinCast cache from a wide close-price CSV."""

    close_df = load_close_csv(csv_path)
    samples = make_rolling_forecast_samples(
        close_df,
        context_len=context_len,
        holding_horizon=holding_horizon,
        tickers=tickers,
        stride=stride,
        max_windows_per_asset=max_windows_per_asset,
    )
    full_outputs = simple_gaussian_level_forecasts(
        samples.contexts,
        horizon_len=horizon_len,
        config=forecast_config,
    )
    return save_distribution_cache(
        output_path,
        full_outputs=full_outputs,
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


def _estimate_return_moments(
    returns: np.ndarray,
    config: SimpleForecastConfig,
) -> tuple[np.ndarray, np.ndarray]:
    if config.mode == "random_walk":
        mu = np.zeros(returns.shape[0], dtype=np.float32)
        sigma = _row_std(returns, center=returns.mean(axis=1))
    elif config.mode == "rolling_mean":
        mu = returns.mean(axis=1).astype(np.float32)
        sigma = _row_std(returns, center=mu)
    elif config.mode == "ewma_vol":
        mu = np.zeros(returns.shape[0], dtype=np.float32)
        sigma = _ewma_sigma(returns, alpha=config.ewma_alpha)
    else:
        raise ValueError(f"Unsupported simple forecast mode: {config.mode!r}")
    sigma = np.maximum(sigma, float(config.min_sigma)).astype(np.float32)
    return mu, sigma


def _row_std(returns: np.ndarray, *, center: np.ndarray) -> np.ndarray:
    centered = returns - center[:, None]
    if returns.shape[1] > 1:
        variance = np.sum(centered * centered, axis=1) / float(returns.shape[1] - 1)
    else:
        variance = np.mean(centered * centered, axis=1)
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)


def _ewma_sigma(returns: np.ndarray, *, alpha: float) -> np.ndarray:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("ewma_alpha must be in (0, 1].")
    n = returns.shape[1]
    weights = (1.0 - alpha) ** np.arange(n - 1, -1, -1, dtype=np.float32)
    weights = weights / weights.sum()
    variance = np.sum((returns * returns) * weights[None, :], axis=1)
    return np.sqrt(np.maximum(variance, 0.0)).astype(np.float32)
