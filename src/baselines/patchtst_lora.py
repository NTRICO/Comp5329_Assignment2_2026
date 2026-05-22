from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class LoRAConfig:
    rank: int = 4
    alpha: float = 8.0
    dropout: float = 0.0
    enabled: bool = True


@dataclass(frozen=True)
class PatchTSTForecastConfig:
    context_length: int = 64
    prediction_length: int = 1
    input_channels: int = 1
    patch_length: int = 16
    patch_stride: int = 8
    d_model: int = 64
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 128
    dropout: float = 0.1
    lora: LoRAConfig = LoRAConfig()


class LoRALinear(nn.Module):
    """Linear layer with an optional low-rank trainable update."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        lora: LoRAConfig | None = None,
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.rank = int(lora.rank if lora is not None else 0)
        self.alpha = float(lora.alpha if lora is not None else 1.0)
        self.scaling = self.alpha / self.rank if self.rank > 0 else 0.0
        self.lora_enabled = bool(lora.enabled if lora is not None else False)
        self.lora_dropout = nn.Dropout(float(lora.dropout if lora is not None else 0.0))

        if self.rank > 0:
            self.lora_a = nn.Parameter(torch.empty(self.rank, in_features))
            self.lora_b = nn.Parameter(torch.zeros(out_features, self.rank))
            nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        else:
            self.register_parameter("lora_a", None)
            self.register_parameter("lora_b", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.linear(x)
        if self.rank <= 0 or not self.lora_enabled:
            return out
        update = F.linear(self.lora_dropout(x), self.lora_a)
        update = F.linear(update, self.lora_b)
        return out + update * self.scaling


class PatchTSTLoRA(nn.Module):
    """PatchTST-style one-step forecaster with LoRA adapter hooks.

    The implementation follows PatchTST's useful parts for this project:
    channel-independent patching, shared Transformer weights, and a compact
    forecasting head. It stays self-contained so the project does not need
    extra PEFT/Transformers dependencies just to run the financial baseline.
    """

    def __init__(self, config: PatchTSTForecastConfig) -> None:
        super().__init__()
        self.config = config
        if config.context_length < config.patch_length:
            raise ValueError("context_length must be at least patch_length.")
        if config.patch_stride <= 0:
            raise ValueError("patch_stride must be positive.")
        if config.d_model % config.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads.")

        self.patch_count = 1 + (config.context_length - config.patch_length) // config.patch_stride
        self.patch_projection = nn.Linear(config.patch_length, config.d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.patch_count, config.d_model))
        self.input_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            PatchTSTEncoderLayer(config)
            for _ in range(config.n_layers)
        )
        self.norm = nn.LayerNorm(config.d_model)
        self.head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * self.patch_count, config.prediction_length),
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, past_values: torch.Tensor) -> torch.Tensor:
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        batch_size, context_length, channels = past_values.shape
        if context_length != self.config.context_length:
            raise ValueError(
                f"context_length must be {self.config.context_length}, got {context_length}"
            )
        if channels != self.config.input_channels:
            raise ValueError(f"input_channels must be {self.config.input_channels}, got {channels}")

        x = past_values.transpose(1, 2)
        x = x.unfold(dimension=-1, size=self.config.patch_length, step=self.config.patch_stride)
        x = x.reshape(batch_size * channels, self.patch_count, self.config.patch_length)
        x = self.patch_projection(x)
        x = self.input_dropout(x + self.position_embedding)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        y = self.head(x)
        return y.reshape(batch_size, channels, self.config.prediction_length).transpose(1, 2)

    def set_lora_enabled(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, LoRALinear):
                module.lora_enabled = enabled

    def freeze_base_for_lora(self, *, train_head: bool = True) -> None:
        for name, param in self.named_parameters():
            param.requires_grad = "lora_" in name
        if train_head:
            for param in self.head.parameters():
                param.requires_grad = True


class PatchTSTEncoderLayer(nn.Module):
    def __init__(self, config: PatchTSTForecastConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.d_model)
        self.attn = LoRAMultiheadSelfAttention(
            d_model=config.d_model,
            n_heads=config.n_heads,
            dropout=config.dropout,
            lora=config.lora,
        )
        self.dropout_attn = nn.Dropout(config.dropout)
        self.norm_ff = nn.LayerNorm(config.d_model)
        self.ff = nn.Sequential(
            LoRALinear(config.d_model, config.d_ff, lora=config.lora),
            nn.GELU(),
            nn.Dropout(config.dropout),
            LoRALinear(config.d_ff, config.d_model, lora=config.lora),
        )
        self.dropout_ff = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout_attn(self.attn(self.norm_attn(x)))
        x = x + self.dropout_ff(self.ff(self.norm_ff(x)))
        return x


class LoRAMultiheadSelfAttention(nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        n_heads: int,
        dropout: float,
        lora: LoRAConfig,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q_proj = LoRALinear(d_model, d_model, lora=lora)
        self.k_proj = LoRALinear(d_model, d_model, lora=lora)
        self.v_proj = LoRALinear(d_model, d_model, lora=lora)
        self.out_proj = LoRALinear(d_model, d_model, lora=lora)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x), batch_size, seq_len)
        k = self._split_heads(self.k_proj(x), batch_size, seq_len)
        v = self._split_heads(self.v_proj(x), batch_size, seq_len)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(weights, v)
        context = context.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        return self.out_proj(context)

    def _split_heads(self, x: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor:
        return x.reshape(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)


def count_parameters(model: nn.Module) -> dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for name, p in model.named_parameters() if "lora_" in name)
    return {"total": total, "trainable": trainable, "lora": lora}
