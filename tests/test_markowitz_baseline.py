import torch

from src.baselines import MeanVarianceBaselineConfig, mean_variance_positions


def test_mean_variance_baseline_shapes_and_constraints() -> None:
    patches = torch.zeros(2, 4, 5, 10)
    patches[..., 0] = 0.01
    patches[..., 1] = -0.02
    patches[..., 9] = 0.02
    config = MeanVarianceBaselineConfig(
        horizon=5,
        risk_aversion=25.0,
        max_trade=0.2,
        round_step=0.01,
    )

    out = mean_variance_positions(patches, config, initial_position=0.0)

    assert out.positions.shape == (2, 4)
    assert out.deltas.shape == (2, 4)
    assert out.expected_returns.shape == (2, 4)
    assert out.variances.shape == (2, 4)
    assert float(out.positions.min()) >= config.min_position
    assert float(out.positions.max()) <= config.max_position
    assert float(out.deltas.abs().max()) <= config.max_trade + 1e-6


def test_mean_variance_baseline_rejects_invalid_horizon() -> None:
    patches = torch.zeros(3, 5, 10)
    config = MeanVarianceBaselineConfig(horizon=6)

    try:
        mean_variance_positions(patches, config)
    except ValueError as exc:
        assert "horizon" in str(exc)
    else:
        raise AssertionError("Expected invalid horizon to raise ValueError.")
