from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.trader.cnn_gru import PolicyRollout


@dataclass(frozen=True)
class EncoderFeatureControllerConfig:
    """GRU policy fed directly by frozen FinCast encoder features."""

    feature_dim: int = 1280
    encoder_dim: int = 128
    state_dim: int = 64
    dropout: float = 0.1
    max_trade: float = 0.25
    min_position: float = 0.0
    max_position: float = 1.0
    round_step: float = 0.01


class EncoderFeatureGRUPolicy(nn.Module):
    """Closed-loop policy that consumes frozen FinCast encoder embeddings.

    Input shape is `[B, T, D]` or `[T, D]`, where `D` is usually 1280 from the
    FinCast patched transformer. This is the direct-encoder alternative to the
    current distribution-patch CNN + GRU policy.
    """

    def __init__(self, config: EncoderFeatureControllerConfig) -> None:
        super().__init__()
        self.config = config
        self.feature_projection = nn.Sequential(
            nn.LayerNorm(config.feature_dim),
            nn.Linear(config.feature_dim, config.encoder_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        self.position_projection = nn.Sequential(
            nn.Linear(1, config.encoder_dim),
            nn.GELU(),
        )
        self.core = nn.GRUCell(
            input_size=config.encoder_dim * 2,
            hidden_size=config.state_dim,
        )
        self.target_head = nn.Sequential(
            nn.LayerNorm(config.state_dim),
            nn.Linear(config.state_dim, config.state_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.state_dim, 1),
        )

    def initial_state(self, batch_size: int, *, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.config.state_dim, device=device)

    def step(
        self,
        features: torch.Tensor,
        prev_position: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if features.ndim != 2 or features.shape[-1] != self.config.feature_dim:
            raise ValueError(
                f"features must be [B, {self.config.feature_dim}], got {tuple(features.shape)}"
            )
        if prev_position.ndim == 1:
            prev_position = prev_position.unsqueeze(-1)
        if prev_position.ndim != 2 or prev_position.shape[-1] != 1:
            raise ValueError(
                f"prev_position must be [B] or [B, 1], got {tuple(prev_position.shape)}"
            )
        batch_size = features.shape[0]
        if prev_state is None:
            prev_state = self.initial_state(batch_size, device=features.device)

        z = self.feature_projection(features)
        p_emb = self.position_projection(prev_position.to(dtype=z.dtype, device=z.device))
        state = self.core(torch.cat([z, p_emb], dim=-1), prev_state)
        target_logit = self.target_head(state).squeeze(-1)
        target_position = self.config.min_position + (
            self.config.max_position - self.config.min_position
        ) * torch.sigmoid(target_logit)

        prev_flat = prev_position.squeeze(-1)
        delta = self.config.max_trade * torch.tanh(
            (target_position - prev_flat) / max(self.config.max_trade, 1e-8)
        )
        position = prev_flat + delta
        position = self._ste_round_clip(position)
        delta = position - prev_flat
        return position, delta, state, z

    def _ste_round_clip(self, position: torch.Tensor) -> torch.Tensor:
        step = self.config.round_step
        if step > 0:
            rounded = torch.round(position / step) * step
            position = position + (rounded - position).detach()
        return position.clamp(self.config.min_position, self.config.max_position)

    def forward(
        self,
        features: torch.Tensor,
        *,
        initial_position: torch.Tensor | float | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> PolicyRollout:
        if features.ndim == 2:
            features = features.unsqueeze(0)
        if features.ndim != 3:
            raise ValueError(f"features must be [B,T,D] or [T,D], got {tuple(features.shape)}")
        batch_size, seq_len = features.shape[:2]
        device = features.device

        if initial_position is None:
            prev_position = torch.zeros(batch_size, device=device)
        elif isinstance(initial_position, torch.Tensor):
            init = initial_position.to(device=device, dtype=features.dtype)
            if init.numel() == 1:
                prev_position = init.reshape(1).expand(batch_size)
            else:
                prev_position = init.reshape(batch_size)
        else:
            prev_position = torch.full((batch_size,), float(initial_position), device=device)

        state = initial_state
        positions = []
        deltas = []
        encoded = []
        for t in range(seq_len):
            prev_position, delta, state, z = self.step(
                features[:, t],
                prev_position,
                state,
            )
            positions.append(prev_position)
            deltas.append(delta)
            encoded.append(z)

        return PolicyRollout(
            positions=torch.stack(positions, dim=1),
            deltas=torch.stack(deltas, dim=1),
            encoded=torch.stack(encoded, dim=1),
            final_state=state,
        )
