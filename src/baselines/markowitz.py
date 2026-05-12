from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class MeanVarianceBaselineConfig:
    """Closed-form long-only mean-variance baseline from FinCast quantiles."""

    horizon: int = 5
    risk_aversion: float = 25.0
    min_position: float = 0.0
    max_position: float = 1.0
    max_trade: float | None = 0.25
    round_step: float = 0.01
    mean_index: int = 0
    q10_index: int = 1
    q90_index: int = 9
    q90_minus_q10_normal_width: float = 2.5631031310892007
    eps: float = 1e-8


@dataclass
class MeanVarianceBaselineOutput:
    positions: torch.Tensor
    deltas: torch.Tensor
    expected_returns: torch.Tensor
    variances: torch.Tensor
    unconstrained_positions: torch.Tensor


def estimate_fincast_mean_variance(
    return_patches: np.ndarray | torch.Tensor,
    config: MeanVarianceBaselineConfig = MeanVarianceBaselineConfig(),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate expected return and variance from near-horizon FinCast output.

    Args:
        return_patches: Tensor shaped `[T, H, C]` or `[B, T, H, C]`, where
            channel 0 is the FinCast mean and channels 1/9 are q10/q90.
        config: Baseline configuration. `horizon` selects the first K FinCast
            forecast steps to summarize.

    Returns:
        `(expected_returns, variances)` with sequence shape `[T]` or `[B, T]`.
    """

    patches = _as_tensor(return_patches)
    if patches.ndim not in (3, 4):
        raise ValueError(f"return_patches must be [T,H,C] or [B,T,H,C], got {tuple(patches.shape)}")
    if not 1 <= config.horizon <= patches.shape[-2]:
        raise ValueError(f"horizon must be in [1, {patches.shape[-2]}], got {config.horizon}")
    if patches.shape[-1] <= max(config.mean_index, config.q10_index, config.q90_index):
        raise ValueError(
            "return_patches does not contain the requested mean/q10/q90 channels, "
            f"got {patches.shape[-1]} channels"
        )

    near = patches[..., : config.horizon, :]
    expected_returns = near[..., config.mean_index].mean(dim=-1)
    q10 = near[..., config.q10_index]
    q90 = near[..., config.q90_index]
    sigma = (q90 - q10).abs() / config.q90_minus_q10_normal_width
    variances = sigma.square().mean(dim=-1).clamp_min(config.eps)
    return expected_returns, variances


def mean_variance_positions(
    return_patches: np.ndarray | torch.Tensor,
    config: MeanVarianceBaselineConfig = MeanVarianceBaselineConfig(),
    *,
    initial_position: float | torch.Tensor = 0.0,
) -> MeanVarianceBaselineOutput:
    """Compute constrained mean-variance positions from FinCast return patches.

    The unconstrained single-asset solution is `p = mu / (lambda * sigma^2)`.
    Positions are then clipped to `[min_position, max_position]`, optionally
    limited by `max_trade` relative to the previous position, and rounded to
    `round_step`.
    """

    expected_returns, variances = estimate_fincast_mean_variance(return_patches, config)
    if config.risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")

    raw_positions = expected_returns / (config.risk_aversion * variances + config.eps)
    target_positions = raw_positions.clamp(config.min_position, config.max_position)

    squeeze_batch = False
    if target_positions.ndim == 1:
        target_positions = target_positions.unsqueeze(0)
        raw_positions = raw_positions.unsqueeze(0)
        expected_returns = expected_returns.unsqueeze(0)
        variances = variances.unsqueeze(0)
        squeeze_batch = True
    if target_positions.ndim != 2:
        raise ValueError(
            "mean_variance_positions expects sequence-shaped estimates [T] or [B,T], "
            f"got {tuple(target_positions.shape)}"
        )

    batch_size, seq_len = target_positions.shape
    device = target_positions.device
    dtype = target_positions.dtype
    prev = _initial_position_tensor(initial_position, batch_size, device=device, dtype=dtype)

    positions = []
    deltas = []
    for t in range(seq_len):
        target = target_positions[:, t]
        if config.max_trade is None:
            position = target
        else:
            trade = (target - prev).clamp(-config.max_trade, config.max_trade)
            position = prev + trade
        position = _round_clip(position, config)
        delta = position - prev
        positions.append(position)
        deltas.append(delta)
        prev = position

    out_positions = torch.stack(positions, dim=1)
    out_deltas = torch.stack(deltas, dim=1)
    if squeeze_batch:
        out_positions = out_positions.squeeze(0)
        out_deltas = out_deltas.squeeze(0)
        raw_positions = raw_positions.squeeze(0)
        expected_returns = expected_returns.squeeze(0)
        variances = variances.squeeze(0)

    return MeanVarianceBaselineOutput(
        positions=out_positions,
        deltas=out_deltas,
        expected_returns=expected_returns,
        variances=variances,
        unconstrained_positions=raw_positions,
    )


def _as_tensor(x: np.ndarray | torch.Tensor) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype=torch.float32)
    return torch.as_tensor(x, dtype=torch.float32)


def _initial_position_tensor(
    initial_position: float | torch.Tensor,
    batch_size: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if isinstance(initial_position, torch.Tensor):
        init = initial_position.to(device=device, dtype=dtype)
        if init.numel() == 1:
            return init.reshape(1).expand(batch_size)
        return init.reshape(batch_size)
    return torch.full((batch_size,), float(initial_position), device=device, dtype=dtype)


def _round_clip(position: torch.Tensor, config: MeanVarianceBaselineConfig) -> torch.Tensor:
    if config.round_step > 0:
        position = torch.round(position / config.round_step) * config.round_step
    return position.clamp(config.min_position, config.max_position)
