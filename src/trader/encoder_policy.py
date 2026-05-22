from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class PolicyRollout:
    positions: torch.Tensor
    deltas: torch.Tensor
    encoded: torch.Tensor
    final_state: torch.Tensor


@dataclass(frozen=True)
class EncoderFeatureControllerConfig:
    """Encoder-only policy fed directly by frozen FinCast encoder features."""

    feature_dim: int = 1280
    encoder_dim: int = 128
    hidden_dim: int = 128
    dropout: float = 0.1
    max_trade: float = 0.25
    min_position: float = 0.0
    max_position: float = 1.0
    round_step: float = 0.01


class EncoderFeaturePositionEncoder(nn.Module):
    """Encode FinCast features together with the previous-position token."""

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
        self.encoder = nn.Sequential(
            nn.LayerNorm(config.encoder_dim * 2),
            nn.Linear(config.encoder_dim * 2, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.encoder_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, features: torch.Tensor, prev_position: torch.Tensor) -> torch.Tensor:
        z = self.feature_projection(features)
        p_emb = self.position_projection(prev_position.to(dtype=z.dtype, device=z.device))
        return self.encoder(torch.cat([z, p_emb], dim=-1))


class EncoderFeatureOnlyPolicy(nn.Module):
    """Position-aware encoder policy without a recurrent state."""

    def __init__(self, config: EncoderFeatureControllerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_encoder = EncoderFeaturePositionEncoder(config)
        self.target_head = nn.Sequential(
            nn.LayerNorm(config.encoder_dim),
            nn.Linear(config.encoder_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def step(
        self,
        features: torch.Tensor,
        prev_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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

        z = self.input_encoder(features, prev_position)
        target_logit = self.target_head(z).squeeze(-1)
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
        return position, delta, z

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
        del initial_state
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

        positions = []
        deltas = []
        encoded = []
        for t in range(seq_len):
            prev_position, delta, z = self.step(features[:, t], prev_position)
            positions.append(prev_position)
            deltas.append(delta)
            encoded.append(z)

        encoded_stack = torch.stack(encoded, dim=1)
        return PolicyRollout(
            positions=torch.stack(positions, dim=1),
            deltas=torch.stack(deltas, dim=1),
            encoded=encoded_stack,
            final_state=encoded_stack[:, -1],
        )
