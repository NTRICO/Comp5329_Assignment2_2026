from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class BacktestMetricsConfig:
    """Metrics knobs for pooled sequence-level smoke backtests."""

    transaction_cost_bps: float = 1.0
    trading_days_per_year: int = 252
    initial_position: float = 0.0
    eps: float = 1e-12


@dataclass(frozen=True)
class StrategyReturnBreakdown:
    gross_returns: torch.Tensor
    transaction_costs: torch.Tensor
    net_returns: torch.Tensor
    turnover: torch.Tensor


@dataclass(frozen=True)
class BacktestMetrics:
    n_steps: int
    mean_return: float
    volatility: float
    sharpe_like: float
    annualized_return: float
    annualized_volatility: float
    sharpe: float
    cumulative_return: float
    max_drawdown: float
    hit_rate: float
    average_position: float
    average_turnover: float
    total_turnover: float
    mean_transaction_cost: float
    exposure_adjusted_return: float
    turnover_adjusted_return: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def compute_position_deltas(
    positions: torch.Tensor,
    *,
    initial_position: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Compute per-step position changes within each independent sequence."""

    positions = torch.as_tensor(positions, dtype=torch.float32)
    if positions.ndim == 1:
        init = _initial_position_like(initial_position, positions[:1])
        previous = torch.cat([init, positions[:-1]], dim=0)
        return positions - previous
    if positions.ndim == 2:
        init = _initial_position_like(initial_position, positions[:, :1])
        previous = torch.cat([init, positions[:, :-1]], dim=1)
        return positions - previous
    raise ValueError(f"positions must be [T] or [B,T], got {tuple(positions.shape)}")


def compute_strategy_returns(
    positions: torch.Tensor,
    realized_returns: torch.Tensor,
    *,
    deltas: torch.Tensor | None = None,
    config: BacktestMetricsConfig = BacktestMetricsConfig(),
) -> StrategyReturnBreakdown:
    """Apply long-only positions to realized returns and subtract trade costs.

    Convention: `positions[..., t]` is chosen at decision time `t` and held over
    `realized_returns[..., t]`. Transaction cost is proportional to absolute
    position change, using `transaction_cost_bps` basis points per unit turnover.
    """

    positions = torch.as_tensor(positions, dtype=torch.float32)
    realized_returns = torch.as_tensor(realized_returns, dtype=torch.float32)
    if positions.shape != realized_returns.shape:
        raise ValueError(
            "positions and realized_returns must have the same shape, "
            f"got {tuple(positions.shape)} and {tuple(realized_returns.shape)}"
        )
    if deltas is None:
        deltas = compute_position_deltas(
            positions,
            initial_position=config.initial_position,
        )
    else:
        deltas = torch.as_tensor(deltas, dtype=torch.float32)
        if deltas.shape != positions.shape:
            raise ValueError(
                "deltas must match positions shape, "
                f"got {tuple(deltas.shape)} and {tuple(positions.shape)}"
            )

    gross_returns = positions * realized_returns
    turnover = deltas.abs()
    transaction_costs = turnover * (float(config.transaction_cost_bps) * 1e-4)
    net_returns = gross_returns - transaction_costs
    return StrategyReturnBreakdown(
        gross_returns=gross_returns,
        transaction_costs=transaction_costs,
        net_returns=net_returns,
        turnover=turnover,
    )


def summarize_backtest(
    positions: torch.Tensor,
    realized_returns: torch.Tensor,
    *,
    deltas: torch.Tensor | None = None,
    config: BacktestMetricsConfig = BacktestMetricsConfig(),
) -> BacktestMetrics:
    """Summarize a strategy over pooled independent sequences."""

    breakdown = compute_strategy_returns(
        positions,
        realized_returns,
        deltas=deltas,
        config=config,
    )
    flat_net = breakdown.net_returns.reshape(-1)
    flat_positions = torch.as_tensor(positions, dtype=torch.float32).reshape(-1)
    flat_turnover = breakdown.turnover.reshape(-1)
    flat_costs = breakdown.transaction_costs.reshape(-1)
    n_steps = int(flat_net.numel())
    if n_steps == 0:
        raise ValueError("Cannot summarize an empty return series.")

    mean_return = float(flat_net.mean().item())
    volatility = float(flat_net.std(unbiased=False).item())
    if volatility <= config.eps:
        sharpe_like = 0.0
    else:
        sharpe_like = mean_return / volatility * (config.trading_days_per_year**0.5)
    annualized_return = float((1.0 + mean_return) ** config.trading_days_per_year - 1.0)
    annualized_volatility = float(volatility * (config.trading_days_per_year**0.5))
    if annualized_volatility <= config.eps:
        sharpe = 0.0
    else:
        sharpe = annualized_return / annualized_volatility
    cumulative_return = float(torch.prod(1.0 + flat_net).item() - 1.0)
    average_position = float(flat_positions.mean().item())
    average_turnover = float(flat_turnover.mean().item())
    exposure_adjusted_return = (
        float(mean_return / abs(average_position))
        if abs(average_position) > config.eps
        else float("nan")
    )
    turnover_adjusted_return = (
        float(mean_return / average_turnover)
        if average_turnover > config.eps
        else float("nan")
    )
    return BacktestMetrics(
        n_steps=n_steps,
        mean_return=mean_return,
        volatility=volatility,
        sharpe_like=float(sharpe_like),
        annualized_return=annualized_return,
        annualized_volatility=annualized_volatility,
        sharpe=float(sharpe),
        cumulative_return=cumulative_return,
        max_drawdown=float(max_drawdown(flat_net).item()),
        hit_rate=float((flat_net > 0).to(torch.float32).mean().item()),
        average_position=average_position,
        average_turnover=average_turnover,
        total_turnover=float(flat_turnover.sum().item()),
        mean_transaction_cost=float(flat_costs.mean().item()),
        exposure_adjusted_return=exposure_adjusted_return,
        turnover_adjusted_return=turnover_adjusted_return,
    )


def max_drawdown(returns: torch.Tensor) -> torch.Tensor:
    """Return positive max drawdown magnitude from a 1D return series."""

    returns = torch.as_tensor(returns, dtype=torch.float32).flatten()
    if returns.numel() == 0:
        raise ValueError("returns must not be empty.")
    equity = torch.cumprod(1.0 + returns, dim=0)
    running_peak = torch.cummax(equity, dim=0).values.clamp_min(1e-12)
    drawdowns = (running_peak - equity) / running_peak
    return drawdowns.max()


def _initial_position_like(
    initial_position: float | torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    if isinstance(initial_position, torch.Tensor):
        init = initial_position.to(device=target.device, dtype=target.dtype)
        if init.numel() == 1:
            return init.reshape(1).expand_as(target)
        return init.reshape_as(target)
    return torch.full_like(target, float(initial_position))
