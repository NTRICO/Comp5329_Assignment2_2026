from __future__ import annotations

from typing import Literal

import numpy as np
import torch


def _as_tensor(x: np.ndarray | torch.Tensor, *, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype)
    return torch.as_tensor(x, dtype=dtype)


def forecast_to_return_patch(
    forecast_patch: np.ndarray | torch.Tensor,
    last_observed_value: np.ndarray | torch.Tensor | float,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Convert FinCast level forecasts to relative return forecasts.

    Args:
        forecast_patch: Shape [..., horizon, channels]. Channel 0 is mean and
            the remaining channels are quantiles.
        last_observed_value: Last observed price/value before the forecast
            horizon. Broadcastable to forecast_patch without horizon/channel dims.
        eps: Numerical guard for near-zero denominators.

    Returns:
        Tensor with the same shape as forecast_patch, expressed as
        forecast / last_observed_value - 1.
    """

    patch = _as_tensor(forecast_patch)
    last_value = _as_tensor(last_observed_value).to(device=patch.device)
    while last_value.ndim < patch.ndim:
        last_value = last_value.unsqueeze(-1)
    denom = torch.clamp(last_value.abs(), min=eps)
    denom = torch.where(last_value < 0, -denom, denom)
    return patch / denom - 1.0


def make_forward_returns(
    values: np.ndarray | torch.Tensor,
    *,
    horizon: int = 1,
    mode: Literal["point", "mean"] = "point",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Create realized future returns from a 1D price/value series.

    For decision time t, this returns the realized return over the next horizon.
    `mode="point"` uses value[t + horizon] / value[t] - 1.
    `mode="mean"` averages one-step returns across the next horizon.
    """

    x = _as_tensor(values).flatten()
    if horizon <= 0:
        raise ValueError("horizon must be positive.")
    if x.numel() <= horizon:
        raise ValueError("values must be longer than horizon.")

    denom = torch.clamp(x[:-horizon].abs(), min=eps)
    denom = torch.where(x[:-horizon] < 0, -denom, denom)
    denom = torch.where(x[:-horizon] < 0, -denom, denom)

    if mode == "point":
        return x[horizon:] / denom - 1.0
    if mode == "mean":
        one_step_denom = torch.clamp(x[:-1].abs(), min=eps)
        one_step_denom = torch.where(x[:-1] < 0, -one_step_denom, one_step_denom)
        one_step = x[1:] / one_step_denom - 1.0
        windows = one_step.unfold(0, horizon, 1)
        return windows.mean(dim=-1)
    raise ValueError(f"Unsupported mode: {mode}")


def quantile_summary_features(
    return_patch: np.ndarray | torch.Tensor,
    *,
    mean_index: int = 0,
    q10_index: int = 1,
    q50_index: int = 5,
    q90_index: int = 9,
) -> torch.Tensor:
    """Summarize a return-distribution patch for baselines and diagnostics.

    Input shape is [..., horizon, channels]. Output shape is [..., 8].
    """

    patch = _as_tensor(return_patch)
    if patch.shape[-1] <= max(mean_index, q10_index, q50_index, q90_index):
        raise ValueError(
            "return_patch does not contain the requested mean/quantile channels: "
            f"shape={tuple(patch.shape)}"
        )

    mean_path = patch[..., :, mean_index]
    q10 = patch[..., :, q10_index]
    q50 = patch[..., :, q50_index]
    q90 = patch[..., :, q90_index]

    spread = q90 - q10
    upside = q90 - q50
    downside = q50 - q10
    slope = q50[..., -1] - q50[..., 0]

    return torch.stack(
        [
            mean_path.mean(dim=-1),
            mean_path[..., -1],
            q50.mean(dim=-1),
            q50[..., -1],
            spread.mean(dim=-1),
            upside.mean(dim=-1),
            downside.mean(dim=-1),
            slope,
        ],
        dim=-1,
    )
