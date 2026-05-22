import torch

from src.baselines import (
    RandomPositionBaselineConfig,
    RollingArGarchBaselineConfig,
    random_positions_like,
    rolling_ar_garch_positions,
)


def test_random_positions_are_reproducible_and_bounded() -> None:
    returns = torch.zeros(3, 5)
    config = RandomPositionBaselineConfig(seed=7, max_trade=0.2, round_step=0.01)

    p1, d1 = random_positions_like(returns, config)
    p2, d2 = random_positions_like(returns, config)

    assert torch.allclose(p1, p2)
    assert torch.allclose(d1, d2)
    assert float(p1.min()) >= 0.0
    assert float(p1.max()) <= 1.0
    assert float(d1.abs().max()) <= 0.2 + 1e-6


def test_rolling_ar_garch_uses_only_past_returns() -> None:
    returns_a = torch.tensor([[0.0, 0.01, 0.02, 0.03, 0.50]])
    returns_b = torch.tensor([[0.0, 0.01, 0.02, 0.03, -0.50]])
    config = RollingArGarchBaselineConfig(
        lookback=4,
        min_history=2,
        risk_aversion=10.0,
        max_trade=0.5,
        round_step=0.01,
    )

    positions_a, _ = rolling_ar_garch_positions(returns_a, config)
    positions_b, _ = rolling_ar_garch_positions(returns_b, config)

    assert torch.allclose(positions_a[:, :-1], positions_b[:, :-1])
    assert float(positions_a.min()) >= 0.0
    assert float(positions_a.max()) <= 1.0
