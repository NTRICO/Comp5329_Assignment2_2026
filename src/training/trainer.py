from __future__ import annotations

from collections.abc import Iterable

import torch

from .losses import MeanVarianceTurnoverLoss, PositionLossBreakdown
from src.trader.cnn_gru import PositionAwareGRUPolicy, PolicyRollout


def move_batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def policy_loss_on_batch(
    model: PositionAwareGRUPolicy,
    loss_fn: MeanVarianceTurnoverLoss,
    batch: dict[str, torch.Tensor],
    *,
    initial_position: float = 0.0,
) -> tuple[PolicyRollout, PositionLossBreakdown]:
    if "patches" in batch:
        model_input = batch["patches"]
    elif "features" in batch:
        model_input = batch["features"]
    else:
        raise KeyError("batch must contain either 'patches' or 'features'.")
    rollout = model(model_input, initial_position=initial_position)
    loss = loss_fn(
        rollout.positions,
        batch["realized_returns"],
        deltas=rollout.deltas,
        forecast_risk=batch.get("forecast_risk"),
    )
    return rollout, loss


def train_one_epoch(
    model: PositionAwareGRUPolicy,
    loss_fn: MeanVarianceTurnoverLoss,
    loader: Iterable[dict[str, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    initial_position: float = 0.0,
    grad_clip: float | None = 1.0,
) -> dict[str, float]:
    model.train()
    totals = {
        "loss": 0.0,
        "mean_return": 0.0,
        "variance": 0.0,
        "turnover": 0.0,
        "forecast_risk": 0.0,
    }
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        _, breakdown = policy_loss_on_batch(
            model,
            loss_fn,
            batch,
            initial_position=initial_position,
        )
        breakdown.loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        totals["loss"] += float(breakdown.loss.detach().cpu())
        totals["mean_return"] += float(breakdown.mean_return.cpu())
        totals["variance"] += float(breakdown.variance.cpu())
        totals["turnover"] += float(breakdown.turnover.cpu())
        totals["forecast_risk"] += float(breakdown.forecast_risk.cpu())
        count += 1

    return {key: value / max(count, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate_policy(
    model: PositionAwareGRUPolicy,
    loss_fn: MeanVarianceTurnoverLoss,
    loader: Iterable[dict[str, torch.Tensor]],
    *,
    device: torch.device,
    initial_position: float = 0.0,
) -> dict[str, float]:
    model.eval()
    totals = {
        "loss": 0.0,
        "mean_return": 0.0,
        "variance": 0.0,
        "turnover": 0.0,
        "forecast_risk": 0.0,
    }
    count = 0
    for batch in loader:
        batch = move_batch_to_device(batch, device)
        _, breakdown = policy_loss_on_batch(
            model,
            loss_fn,
            batch,
            initial_position=initial_position,
        )
        totals["loss"] += float(breakdown.loss.detach().cpu())
        totals["mean_return"] += float(breakdown.mean_return.cpu())
        totals["variance"] += float(breakdown.variance.cpu())
        totals["turnover"] += float(breakdown.turnover.cpu())
        totals["forecast_risk"] += float(breakdown.forecast_risk.cpu())
        count += 1
    return {key: value / max(count, 1) for key, value in totals.items()}
