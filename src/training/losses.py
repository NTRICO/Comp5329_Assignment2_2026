from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.config import PositionLossConfig


@dataclass
class PositionLossBreakdown:
    loss: torch.Tensor
    mean_return: torch.Tensor
    variance: torch.Tensor
    turnover: torch.Tensor
    forecast_risk: torch.Tensor


class MeanVarianceTurnoverLoss(nn.Module):
    """Mean-variance utility loss for sequential long-only positions.

    Convention: position_t is the position chosen at decision time t and held
    over the realized future return r_t in the training sample.
    """

    def __init__(self, config: PositionLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        positions: torch.Tensor,
        realized_returns: torch.Tensor,
        *,
        deltas: torch.Tensor | None = None,
        forecast_risk: torch.Tensor | None = None,
    ) -> PositionLossBreakdown:
        if positions.shape != realized_returns.shape:
            raise ValueError(
                "positions and realized_returns must have the same shape, "
                f"got {tuple(positions.shape)} and {tuple(realized_returns.shape)}"
            )
        portfolio_returns = positions * realized_returns
        mean_return = portfolio_returns.mean()
        variance = portfolio_returns.var(unbiased=False)

        if deltas is None:
            padded = torch.cat([torch.zeros_like(positions[:, :1]), positions], dim=1)
            turnover = (padded[:, 1:] - padded[:, :-1]).abs().mean()
        else:
            turnover = deltas.abs().mean()

        if forecast_risk is None:
            forecast_risk_term = torch.zeros((), dtype=positions.dtype, device=positions.device)
        else:
            if forecast_risk.shape != positions.shape:
                raise ValueError(
                    "forecast_risk must match positions shape, "
                    f"got {tuple(forecast_risk.shape)} and {tuple(positions.shape)}"
                )
            forecast_risk_term = ((positions**2) * forecast_risk).mean()

        loss = (
            -mean_return
            + self.config.lambda_variance * variance
            + self.config.lambda_turnover * turnover
            + self.config.lambda_forecast_risk * forecast_risk_term
        )
        return PositionLossBreakdown(
            loss=loss,
            mean_return=mean_return.detach(),
            variance=variance.detach(),
            turnover=turnover.detach(),
            forecast_risk=forecast_risk_term.detach(),
        )
