from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PositionControllerConfig:
    """Small, explicit config for the position-aware recurrent controller."""

    horizon_len: int = 32
    forecast_channels: int = 10
    encoder_dim: int = 64
    conv_hidden: int = 64
    conv_layers: int = 2
    kernel_size: int = 3
    state_dim: int = 64
    dropout: float = 0.1
    max_trade: float = 0.25
    min_position: float = 0.0
    max_position: float = 1.0
    round_step: float = 0.01


@dataclass(frozen=True)
class PositionLossConfig:
    """Mean-variance objective for a long-only single-asset policy."""

    lambda_variance: float = 1.0
    lambda_turnover: float = 0.001
    lambda_forecast_risk: float = 0.0
    eps: float = 1e-8
