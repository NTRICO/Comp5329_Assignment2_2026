from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RandomPositionBaselineConfig:
    min_position: float = 0.0
    max_position: float = 1.0
    max_trade: float | None = None
    round_step: float = 0.01
    seed: int = 1234


@dataclass(frozen=True)
class RollingArGarchBaselineConfig:
    """Rolling AR(1)-GARCH-like long-only position rule.

    The rule uses only returns observed earlier in each sequence. At time `t`,
    it estimates an AR(1) one-step expected return from a rolling window and a
    GARCH-like variance forecast from the last shock plus rolling residual
    variance, then maps `mu / (risk_aversion * variance)` to a bounded position.
    """

    lookback: int = 16
    min_history: int = 4
    risk_aversion: float = 25.0
    alpha: float = 0.10
    beta: float = 0.85
    default_variance: float = 1e-4
    min_position: float = 0.0
    max_position: float = 1.0
    max_trade: float | None = 0.05
    round_step: float = 0.01
    eps: float = 1e-8


def random_positions_like(
    realized_returns: torch.Tensor,
    config: RandomPositionBaselineConfig = RandomPositionBaselineConfig(),
    *,
    initial_position: float | torch.Tensor = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = torch.as_tensor(realized_returns, dtype=torch.float32)
    if returns.ndim != 2:
        raise ValueError(f"realized_returns must be [B,T], got {tuple(returns.shape)}")
    if generator is None:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(config.seed))

    target = torch.rand(
        returns.shape,
        generator=generator,
        dtype=returns.dtype,
        device=returns.device,
    )
    target = config.min_position + target * (config.max_position - config.min_position)
    return _apply_trade_rule(
        target,
        initial_position=initial_position,
        min_position=config.min_position,
        max_position=config.max_position,
        max_trade=config.max_trade,
        round_step=config.round_step,
    )


def rolling_ar_garch_positions(
    realized_returns: torch.Tensor,
    config: RollingArGarchBaselineConfig = RollingArGarchBaselineConfig(),
    *,
    initial_position: float | torch.Tensor = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    returns = torch.as_tensor(realized_returns, dtype=torch.float32)
    if returns.ndim != 2:
        raise ValueError(f"realized_returns must be [B,T], got {tuple(returns.shape)}")
    if config.lookback <= 0:
        raise ValueError("lookback must be positive.")
    if config.risk_aversion <= 0:
        raise ValueError("risk_aversion must be positive.")
    if not 0.0 <= config.alpha <= 1.0 or not 0.0 <= config.beta <= 1.0:
        raise ValueError("alpha and beta must be in [0, 1].")

    batch_size, seq_len = returns.shape
    device = returns.device
    dtype = returns.dtype
    prev_position = _initial_position_tensor(initial_position, batch_size, device=device, dtype=dtype)
    prev_variance = torch.full(
        (batch_size,),
        float(config.default_variance),
        dtype=dtype,
        device=device,
    )

    positions = []
    deltas = []
    for t in range(seq_len):
        start = max(0, t - config.lookback)
        history = returns[:, start:t]
        mu, variance = _rolling_ar_garch_forecast(history, prev_variance, config)
        raw_position = mu / (float(config.risk_aversion) * variance.clamp_min(config.eps))
        target = raw_position.clamp(config.min_position, config.max_position)

        if config.max_trade is None:
            position = target
        else:
            trade = (target - prev_position).clamp(-config.max_trade, config.max_trade)
            position = prev_position + trade
        position = _round_clip(position, config.min_position, config.max_position, config.round_step)
        delta = position - prev_position
        positions.append(position)
        deltas.append(delta)
        prev_position = position
        prev_variance = variance

    return torch.stack(positions, dim=1), torch.stack(deltas, dim=1)


def _rolling_ar_garch_forecast(
    history: torch.Tensor,
    prev_variance: torch.Tensor,
    config: RollingArGarchBaselineConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = history.shape[0]
    device = history.device
    dtype = history.dtype
    if history.shape[1] < config.min_history:
        zero_mu = torch.zeros(batch_size, dtype=dtype, device=device)
        default_var = torch.full(
            (batch_size,),
            float(config.default_variance),
            dtype=dtype,
            device=device,
        )
        return zero_mu, default_var

    x = history[:, :-1]
    y = history[:, 1:]
    x_mean = x.mean(dim=1, keepdim=True)
    y_mean = y.mean(dim=1, keepdim=True)
    x_centered = x - x_mean
    y_centered = y - y_mean
    x_var = x_centered.square().mean(dim=1).clamp_min(config.eps)
    phi = (x_centered * y_centered).mean(dim=1) / x_var
    phi = phi.clamp(-0.99, 0.99)
    intercept = y_mean.squeeze(1) - phi * x_mean.squeeze(1)
    mu = intercept + phi * history[:, -1]

    residuals = y - (intercept.unsqueeze(1) + phi.unsqueeze(1) * x)
    long_variance = residuals.square().mean(dim=1).clamp_min(config.eps)
    last_residual = history[:, -1] - mu
    omega_weight = max(0.0, 1.0 - float(config.alpha) - float(config.beta))
    variance = (
        omega_weight * long_variance
        + float(config.alpha) * last_residual.square()
        + float(config.beta) * prev_variance
    )
    variance = torch.maximum(variance, torch.full_like(variance, float(config.default_variance)))
    return mu, variance


def _apply_trade_rule(
    target: torch.Tensor,
    *,
    initial_position: float | torch.Tensor,
    min_position: float,
    max_position: float,
    max_trade: float | None,
    round_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size, seq_len = target.shape
    prev_position = _initial_position_tensor(
        initial_position,
        batch_size,
        device=target.device,
        dtype=target.dtype,
    )
    positions = []
    deltas = []
    for t in range(seq_len):
        desired = target[:, t]
        if max_trade is None:
            position = desired
        else:
            trade = (desired - prev_position).clamp(-float(max_trade), float(max_trade))
            position = prev_position + trade
        position = _round_clip(position, min_position, max_position, round_step)
        delta = position - prev_position
        positions.append(position)
        deltas.append(delta)
        prev_position = position
    return torch.stack(positions, dim=1), torch.stack(deltas, dim=1)


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


def _round_clip(
    position: torch.Tensor,
    min_position: float,
    max_position: float,
    round_step: float,
) -> torch.Tensor:
    if round_step > 0:
        position = torch.round(position / round_step) * round_step
    return position.clamp(float(min_position), float(max_position))
