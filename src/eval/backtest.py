from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Iterable

import torch

from src.baselines import (
    MeanVarianceBaselineConfig,
    RandomPositionBaselineConfig,
    RollingArGarchBaselineConfig,
    mean_variance_positions,
    random_positions_like,
    rolling_ar_garch_positions,
)
from src.eval.metrics import (
    BacktestMetrics,
    BacktestMetricsConfig,
    StrategyReturnBreakdown,
    compute_position_deltas,
    compute_strategy_returns,
    summarize_backtest,
)
from src.training.trainer import move_batch_to_device


@dataclass(frozen=True)
class BacktestResult:
    name: str
    positions: torch.Tensor
    realized_returns: torch.Tensor
    deltas: torch.Tensor
    metrics: BacktestMetrics
    returns: StrategyReturnBreakdown
    metadata: dict[str, object] = field(default_factory=dict)


@torch.no_grad()
def run_policy_backtest(
    *,
    name: str,
    model: torch.nn.Module,
    loader: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    metrics_config: BacktestMetricsConfig = BacktestMetricsConfig(),
) -> BacktestResult:
    """Run a trained sequential policy on a loader of independent sequences."""

    model.eval()
    all_positions: list[torch.Tensor] = []
    all_deltas: list[torch.Tensor] = []
    all_realized: list[torch.Tensor] = []

    for batch in loader:
        batch = move_batch_to_device(batch, device)
        if "patches" in batch:
            model_input = batch["patches"]
        elif "features" in batch:
            model_input = batch["features"]
        else:
            raise KeyError("batch must contain either 'patches' or 'features'.")

        rollout = model(
            model_input,
            initial_position=metrics_config.initial_position,
        )
        all_positions.append(rollout.positions.detach().cpu())
        all_deltas.append(rollout.deltas.detach().cpu())
        all_realized.append(batch["realized_returns"].detach().cpu())

    return _build_result(
        name=name,
        positions=torch.cat(all_positions, dim=0),
        realized_returns=torch.cat(all_realized, dim=0),
        deltas=torch.cat(all_deltas, dim=0),
        metrics_config=metrics_config,
        metadata={"source": "policy"},
    )


@torch.no_grad()
def run_constant_position_backtest(
    *,
    name: str,
    loader: Iterable[dict[str, torch.Tensor]],
    position: float,
    metrics_config: BacktestMetricsConfig = BacktestMetricsConfig(),
) -> BacktestResult:
    """Backtest a constant-position baseline such as cash or buy-and-hold."""

    positions: list[torch.Tensor] = []
    realized: list[torch.Tensor] = []
    for batch in loader:
        returns = batch["realized_returns"].detach().cpu()
        positions.append(torch.full_like(returns, float(position)))
        realized.append(returns)

    all_positions = torch.cat(positions, dim=0)
    return _build_result(
        name=name,
        positions=all_positions,
        realized_returns=torch.cat(realized, dim=0),
        deltas=None,
        metrics_config=metrics_config,
        metadata={"source": "constant", "position": float(position)},
    )


@torch.no_grad()
def run_markowitz_backtest(
    *,
    name: str,
    loader: Iterable[dict[str, torch.Tensor]],
    baseline_config: MeanVarianceBaselineConfig,
    metrics_config: BacktestMetricsConfig = BacktestMetricsConfig(),
) -> BacktestResult:
    """Backtest the closed-form FinCast mean-variance baseline."""

    positions: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    realized: list[torch.Tensor] = []
    for batch in loader:
        if "patches" not in batch:
            raise KeyError("Markowitz baseline requires batch['patches'].")
        out = mean_variance_positions(
            batch["patches"],
            baseline_config,
            initial_position=metrics_config.initial_position,
        )
        positions.append(out.positions.detach().cpu())
        deltas.append(out.deltas.detach().cpu())
        realized.append(batch["realized_returns"].detach().cpu())

    return _build_result(
        name=name,
        positions=torch.cat(positions, dim=0),
        realized_returns=torch.cat(realized, dim=0),
        deltas=torch.cat(deltas, dim=0),
        metrics_config=metrics_config,
        metadata={
            "source": "markowitz",
            "baseline_config": baseline_config.__dict__,
        },
    )


@torch.no_grad()
def run_random_position_backtest(
    *,
    name: str,
    loader: Iterable[dict[str, torch.Tensor]],
    baseline_config: RandomPositionBaselineConfig,
    metrics_config: BacktestMetricsConfig = BacktestMetricsConfig(),
) -> BacktestResult:
    """Backtest a reproducible random long-only position rule."""

    positions: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    realized: list[torch.Tensor] = []
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(baseline_config.seed))
    for batch in loader:
        returns = batch["realized_returns"].detach().cpu()
        random_positions, random_deltas = random_positions_like(
            returns,
            baseline_config,
            initial_position=metrics_config.initial_position,
            generator=generator,
        )
        positions.append(random_positions)
        deltas.append(random_deltas)
        realized.append(returns)

    return _build_result(
        name=name,
        positions=torch.cat(positions, dim=0),
        realized_returns=torch.cat(realized, dim=0),
        deltas=torch.cat(deltas, dim=0),
        metrics_config=metrics_config,
        metadata={
            "source": "random_position",
            "baseline_config": baseline_config.__dict__,
        },
    )


@torch.no_grad()
def run_rolling_ar_garch_backtest(
    *,
    name: str,
    loader: Iterable[dict[str, torch.Tensor]],
    baseline_config: RollingArGarchBaselineConfig,
    metrics_config: BacktestMetricsConfig = BacktestMetricsConfig(),
) -> BacktestResult:
    """Backtest a pure historical-return AR(1)-GARCH-like position rule."""

    positions: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    realized: list[torch.Tensor] = []
    for batch in loader:
        returns = batch["realized_returns"].detach().cpu()
        baseline_positions, baseline_deltas = rolling_ar_garch_positions(
            returns,
            baseline_config,
            initial_position=metrics_config.initial_position,
        )
        positions.append(baseline_positions)
        deltas.append(baseline_deltas)
        realized.append(returns)

    return _build_result(
        name=name,
        positions=torch.cat(positions, dim=0),
        realized_returns=torch.cat(realized, dim=0),
        deltas=torch.cat(deltas, dim=0),
        metrics_config=metrics_config,
        metadata={
            "source": "rolling_ar_garch",
            "baseline_config": baseline_config.__dict__,
        },
    )


@torch.no_grad()
def run_oracle_backtest(
    *,
    name: str,
    loader: Iterable[dict[str, torch.Tensor]],
    metrics_config: BacktestMetricsConfig = BacktestMetricsConfig(),
    min_position: float = 0.0,
    max_position: float = 1.0,
    max_trade: float | None = None,
    round_step: float = 0.01,
) -> BacktestResult:
    """Perfect-foresight long-only upper baseline.

    This intentionally looks at realized future returns. If the next realized
    return is positive it targets `max_position`; otherwise it targets
    `min_position`. Use it only as an upper bound / sanity reference.
    """

    positions: list[torch.Tensor] = []
    deltas: list[torch.Tensor] = []
    realized: list[torch.Tensor] = []
    for batch in loader:
        returns = batch["realized_returns"].detach().cpu()
        target = torch.where(
            returns > 0,
            torch.full_like(returns, float(max_position)),
            torch.full_like(returns, float(min_position)),
        )
        if max_trade is None:
            oracle_positions = _round_clip(target, min_position, max_position, round_step)
            oracle_deltas = compute_position_deltas(
                oracle_positions,
                initial_position=metrics_config.initial_position,
            )
        else:
            oracle_positions, oracle_deltas = _apply_trade_cap(
                target,
                initial_position=metrics_config.initial_position,
                min_position=min_position,
                max_position=max_position,
                max_trade=max_trade,
                round_step=round_step,
            )
        positions.append(oracle_positions)
        deltas.append(oracle_deltas)
        realized.append(returns)

    return _build_result(
        name=name,
        positions=torch.cat(positions, dim=0),
        realized_returns=torch.cat(realized, dim=0),
        deltas=torch.cat(deltas, dim=0),
        metrics_config=metrics_config,
        metadata={
            "source": "oracle",
            "uses_future_returns": True,
            "min_position": min_position,
            "max_position": max_position,
            "max_trade": max_trade,
            "round_step": round_step,
        },
    )


def _build_result(
    *,
    name: str,
    positions: torch.Tensor,
    realized_returns: torch.Tensor,
    deltas: torch.Tensor | None,
    metrics_config: BacktestMetricsConfig,
    metadata: dict[str, object],
) -> BacktestResult:
    returns = compute_strategy_returns(
        positions,
        realized_returns,
        deltas=deltas,
        config=metrics_config,
    )
    if deltas is None:
        deltas = compute_position_deltas(
            positions,
            initial_position=metrics_config.initial_position,
        )
    metrics = summarize_backtest(
        positions,
        realized_returns,
        deltas=deltas,
        config=metrics_config,
    )
    return BacktestResult(
        name=name,
        positions=positions,
        realized_returns=realized_returns,
        deltas=deltas,
        metrics=metrics,
        returns=returns,
        metadata=metadata,
    )


def _apply_trade_cap(
    target: torch.Tensor,
    *,
    initial_position: float | torch.Tensor,
    min_position: float,
    max_position: float,
    max_trade: float,
    round_step: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.ndim != 2:
        raise ValueError(f"target must be [B,T], got {tuple(target.shape)}")
    if max_trade <= 0:
        raise ValueError("max_trade must be positive when provided.")

    batch_size, seq_len = target.shape
    if isinstance(initial_position, torch.Tensor):
        init = initial_position.to(dtype=target.dtype, device=target.device)
        if init.numel() == 1:
            prev = init.reshape(1).expand(batch_size)
        else:
            prev = init.reshape(batch_size)
    else:
        prev = torch.full((batch_size,), float(initial_position), dtype=target.dtype, device=target.device)

    positions = []
    deltas = []
    for t in range(seq_len):
        desired_delta = target[:, t] - prev
        delta = desired_delta.clamp(-float(max_trade), float(max_trade))
        position = _round_clip(prev + delta, min_position, max_position, round_step)
        delta = position - prev
        positions.append(position)
        deltas.append(delta)
        prev = position
    return torch.stack(positions, dim=1), torch.stack(deltas, dim=1)


def _round_clip(
    position: torch.Tensor,
    min_position: float,
    max_position: float,
    round_step: float,
) -> torch.Tensor:
    if round_step > 0:
        position = torch.round(position / round_step) * round_step
    return position.clamp(float(min_position), float(max_position))
