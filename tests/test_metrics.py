import torch

from src.eval.backtest import run_oracle_backtest
from src.eval.metrics import (
    BacktestMetricsConfig,
    compute_position_deltas,
    compute_strategy_returns,
    summarize_backtest,
)


def test_position_deltas_reset_per_sequence() -> None:
    positions = torch.tensor([[0.25, 0.50, 0.25], [0.10, 0.10, 0.00]])

    deltas = compute_position_deltas(positions, initial_position=0.0)

    expected = torch.tensor([[0.25, 0.25, -0.25], [0.10, 0.00, -0.10]])
    assert torch.allclose(deltas, expected)


def test_strategy_returns_apply_transaction_costs() -> None:
    positions = torch.tensor([[1.0, 1.0, 0.5]])
    realized = torch.tensor([[0.01, -0.02, 0.04]])
    config = BacktestMetricsConfig(transaction_cost_bps=10.0, initial_position=0.0)

    out = compute_strategy_returns(positions, realized, config=config)

    expected_net = torch.tensor([[0.009, -0.020, 0.0195]])
    assert torch.allclose(out.net_returns, expected_net)
    assert torch.allclose(out.turnover, torch.tensor([[1.0, 0.0, 0.5]]))


def test_summarize_backtest_shapes() -> None:
    positions = torch.ones(2, 3)
    realized = torch.tensor([[0.01, 0.0, -0.01], [0.02, 0.01, 0.0]])

    metrics = summarize_backtest(positions, realized)

    assert metrics.n_steps == 6
    assert metrics.average_position == 1.0
    assert metrics.max_drawdown >= 0.0


def test_oracle_binary_uses_positive_return_days() -> None:
    loader = [{"realized_returns": torch.tensor([[0.01, -0.02, 0.0, 0.03]])}]

    result = run_oracle_backtest(
        name="oracle_binary",
        loader=loader,
        metrics_config=BacktestMetricsConfig(transaction_cost_bps=0.0),
        max_trade=None,
    )

    assert torch.allclose(result.positions, torch.tensor([[1.0, 0.0, 0.0, 1.0]]))
    assert result.metadata["uses_future_returns"] is True


def test_oracle_trade_cap_respects_daily_trade_limit() -> None:
    loader = [{"realized_returns": torch.tensor([[0.01, 0.01, -0.01, -0.01]])}]

    result = run_oracle_backtest(
        name="oracle_trade_cap",
        loader=loader,
        metrics_config=BacktestMetricsConfig(transaction_cost_bps=0.0),
        max_trade=0.25,
        round_step=0.01,
    )

    assert torch.allclose(result.positions, torch.tensor([[0.25, 0.50, 0.25, 0.00]]))
    assert float(result.deltas.abs().max()) <= 0.25 + 1e-6
