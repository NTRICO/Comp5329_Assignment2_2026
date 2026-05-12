from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from src.utils.config import PositionControllerConfig


class ConvResidualBlock(nn.Module):
    """Small 1D residual block over the forecast horizon axis."""

    def __init__(self, channels: int, *, kernel_size: int = 3, dropout: float = 0.1) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so horizon length is preserved.")
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class DistributionPatchEncoder(nn.Module):
    """Encode one FinCast predictive distribution patch.

    Input shape: [batch, horizon, channels], for example [B, 32, 10].
    Output shape: [batch, encoder_dim].
    """

    def __init__(self, config: PositionControllerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_proj = nn.Conv1d(
            config.forecast_channels,
            config.conv_hidden,
            kernel_size=1,
        )
        self.blocks = nn.ModuleList(
            [
                ConvResidualBlock(
                    config.conv_hidden,
                    kernel_size=config.kernel_size,
                    dropout=config.dropout,
                )
                for _ in range(config.conv_layers)
            ]
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.LayerNorm(config.conv_hidden),
            nn.Linear(config.conv_hidden, config.encoder_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, patch: torch.Tensor) -> torch.Tensor:
        if patch.ndim != 3:
            raise ValueError(f"patch must be [B, H, C], got {tuple(patch.shape)}")
        if patch.shape[-2] != self.config.horizon_len:
            raise ValueError(
                f"Expected horizon length {self.config.horizon_len}, got {patch.shape[-2]}"
            )
        if patch.shape[-1] != self.config.forecast_channels:
            raise ValueError(
                f"Expected {self.config.forecast_channels} forecast channels, got {patch.shape[-1]}"
            )
        x = patch.transpose(1, 2)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.pool(x)
        return self.output(x)


@dataclass
class PolicyRollout:
    positions: torch.Tensor
    deltas: torch.Tensor
    encoded: torch.Tensor
    final_state: torch.Tensor


class PositionAwareGRUPolicy(nn.Module):
    """Closed-loop position controller.

    At each decision step, the model encodes the current FinCast distribution
    patch, combines it with previous position and recurrent state, then outputs
    a bounded position change.
    """

    def __init__(self, config: PositionControllerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = DistributionPatchEncoder(config)
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
        patch: torch.Tensor,
        prev_position: torch.Tensor,
        prev_state: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one Markov decision step.

        Returns:
            position, delta, next_state, encoded_patch.
        """

        if prev_position.ndim == 1:
            prev_position = prev_position.unsqueeze(-1)
        if prev_position.ndim != 2 or prev_position.shape[-1] != 1:
            raise ValueError(
                f"prev_position must be [B] or [B, 1], got {tuple(prev_position.shape)}"
            )
        batch_size = patch.shape[0]
        if prev_state is None:
            prev_state = self.initial_state(batch_size, device=patch.device)

        z = self.encoder(patch)
        p_emb = self.position_projection(prev_position.to(dtype=z.dtype, device=z.device))
        state = self.core(torch.cat([z, p_emb], dim=-1), prev_state)
        target_logit = self.target_head(state).squeeze(-1)
        target_position = self.config.min_position + (
            self.config.max_position - self.config.min_position
        ) * torch.sigmoid(target_logit)

        # Smoothly move toward the target without hard clipping gradients at
        # the position bounds. Since target_position is bounded, the next
        # position remains bounded and moves by at most max_trade.
        prev_flat = prev_position.squeeze(-1)
        delta = self.config.max_trade * torch.tanh(
            (target_position - prev_flat) / max(self.config.max_trade, 1e-8)
        )
        position = prev_flat + delta
        position = self._ste_round_clip(position)
        delta = position - prev_flat
        return position, delta, state, z

    def _ste_round_clip(self, position: torch.Tensor) -> torch.Tensor:
        """Quantize position to `round_step` grid and clip to [min, max].

        Uses a straight-through estimator so the forward pass sees rounded
        values while gradients flow through the unrounded path.
        """
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
        """Roll the controller through a sequence.

        Args:
            patches: [B, T, H, C] or [T, H, C].
            initial_position: scalar, [B], or [B, 1]. Defaults to zero.
            initial_state: optional GRU state [B, state_dim].
        """

        if patches.ndim == 3:
            patches = patches.unsqueeze(0)
        if patches.ndim != 4:
            raise ValueError(f"patches must be [B, T, H, C], got {tuple(patches.shape)}")
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

        state = initial_state
        positions = []
        deltas = []
        encoded = []
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
