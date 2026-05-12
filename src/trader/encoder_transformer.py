from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.trader.cnn_gru import PolicyRollout


@dataclass(frozen=True)
class EncoderTransformerPolicyConfig:
    """Vanilla TransformerEncoder head on top of frozen FinCast outputs."""

    horizon_len: int = 5
    forecast_channels: int = 10
    model_dim: int = 64
    num_layers: int = 2
    num_heads: int = 4
    ff_dim: int = 128
    dropout: float = 0.1
    max_trade: float = 0.25
    min_position: float = 0.0
    max_position: float = 1.0
    round_step: float = 0.01


class EncoderTransformerPolicy(nn.Module):
    """Frozen FinCast forecast patch -> vanilla encoder -> daily position.

    This is the minimal "encoder after decoder-only FinCast" variant. It uses
    the cached FinCast distribution patch `[H, 10]` as a token sequence, applies
    a vanilla TransformerEncoder over the horizon axis, pools the encoded tokens,
    and outputs a bounded target position. There is no GRU state; the only
    sequential dependency is the previous position used for trade smoothing.
    """

    def __init__(self, config: EncoderTransformerPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.forecast_channels, config.model_dim)
        self.position_embedding = nn.Parameter(torch.zeros(1, config.horizon_len, config.model_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)
        self.output_norm = nn.LayerNorm(config.model_dim)
        self.position_projection = nn.Sequential(
            nn.Linear(1, config.model_dim),
            nn.GELU(),
        )
        self.target_head = nn.Sequential(
            nn.LayerNorm(config.model_dim * 2),
            nn.Linear(config.model_dim * 2, config.model_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.model_dim, 1),
        )

    def encode_patch(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 3:
            raise ValueError(f"patch must be [B,H,C], got {tuple(patch.shape)}")
        if patch.shape[-2] != self.config.horizon_len:
            raise ValueError(
                f"Expected horizon length {self.config.horizon_len}, got {patch.shape[-2]}"
            )
        if patch.shape[-1] != self.config.forecast_channels:
            raise ValueError(
                f"Expected {self.config.forecast_channels} channels, got {patch.shape[-1]}"
            )
        tokens = self.input_projection(patch) + self.position_embedding
        encoded = self.encoder(tokens)
        return self.output_norm(encoded.mean(dim=1))

    def step(
        self,
        patch: torch.Tensor,
        prev_position: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if prev_position.ndim == 1:
            prev_position = prev_position.unsqueeze(-1)
        if prev_position.ndim != 2 or prev_position.shape[-1] != 1:
            raise ValueError(
                f"prev_position must be [B] or [B,1], got {tuple(prev_position.shape)}"
            )

        z = self.encode_patch(patch)
        p_emb = self.position_projection(prev_position.to(dtype=z.dtype, device=z.device))
        target_logit = self.target_head(torch.cat([z, p_emb], dim=-1)).squeeze(-1)
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
        state = torch.empty(0, device=patch.device)
        return position, delta, state, z

    def _ste_round_clip(self, position: torch.Tensor) -> torch.Tensor:
        step = self.config.round_step
        if step > 0:
            rounded = torch.round(position / step) * step
            position = position + (rounded - position).detach()
        return position.clamp(self.config.min_position, self.config.max_position)

    def forward(
        self,
        patches: torch.Tensor,
        *,
        initial_position: torch.Tensor | float | None = None,
        initial_state: torch.Tensor | None = None,
    ) -> PolicyRollout:
        if patches.ndim == 3:
            patches = patches.unsqueeze(0)
        if patches.ndim != 4:
            raise ValueError(f"patches must be [B,T,H,C] or [T,H,C], got {tuple(patches.shape)}")
        batch_size, seq_len = patches.shape[:2]
        device = patches.device

        if initial_position is None:
            prev_position = torch.zeros(batch_size, device=device)
        elif isinstance(initial_position, torch.Tensor):
            init = initial_position.to(device=device, dtype=patches.dtype)
            if init.numel() == 1:
                prev_position = init.reshape(1).expand(batch_size)
            else:
                prev_position = init.reshape(batch_size)
        else:
            prev_position = torch.full((batch_size,), float(initial_position), device=device)

        positions = []
        deltas = []
        encoded = []
        state = torch.empty(0, device=device)
        for t in range(seq_len):
            prev_position, delta, state, z = self.step(
                patches[:, t],
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
