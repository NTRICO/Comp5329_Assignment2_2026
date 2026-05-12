from __future__ import annotations

import numpy as np
import torch


def closed_form_markowitz_position(
    expected_return: np.ndarray | torch.Tensor,
    risk: np.ndarray | torch.Tensor,
    *,
    risk_aversion: float = 10.0,
    min_position: float = 0.0,
    max_position: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Long-only single-asset mean-variance position.

    p* = clip(mu / (lambda * sigma^2), min_position, max_position)

    `risk` should be variance-like. If using quantile spread, square it before
    calling this function or treat the spread squared as the variance proxy.
    """

    mu = expected_return if isinstance(expected_return, torch.Tensor) else torch.as_tensor(expected_return)
    sigma2 = risk if isinstance(risk, torch.Tensor) else torch.as_tensor(risk)
    mu = mu.to(dtype=torch.float32)
    sigma2 = sigma2.to(dtype=torch.float32, device=mu.device)
    raw = mu / (risk_aversion * torch.clamp(sigma2, min=eps))
    return torch.clamp(raw, min=min_position, max=max_position)


def buy_and_hold_position(length: int, *, position: float = 1.0) -> torch.Tensor:
    if length < 0:
        raise ValueError("length must be non-negative.")
    return torch.full((length,), float(position), dtype=torch.float32)


def sign_position(
    expected_return: np.ndarray | torch.Tensor,
    *,
    long_position: float = 1.0,
    flat_position: float = 0.0,
) -> torch.Tensor:
    mu = expected_return if isinstance(expected_return, torch.Tensor) else torch.as_tensor(expected_return)
    return torch.where(
        mu.to(dtype=torch.float32) > 0,
        torch.tensor(float(long_position), dtype=torch.float32, device=mu.device),
        torch.tensor(float(flat_position), dtype=torch.float32, device=mu.device),
    )
