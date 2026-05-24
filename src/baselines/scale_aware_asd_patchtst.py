from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from src.baselines.patchtst_lora import (
    LoRAConfig,
    PatchTSTEncoderLayer,
    PatchTSTForecastConfig,
)


@dataclass(frozen=True)
class ScaleSpec:
    name: str
    scale_id: int
    delta_seconds: float
    context_length: int
    patch_length: int
    patch_stride: int
    prediction_length: int = 1

    @property
    def patch_count(self) -> int:
        return 1 + (self.context_length - self.patch_length) // self.patch_stride


DEFAULT_SCALE_SPECS: dict[str, ScaleSpec] = {
    "second": ScaleSpec(
        name="second",
        scale_id=0,
        delta_seconds=1.0,
        context_length=64,
        patch_length=16,
        patch_stride=8,
    ),
    "minute": ScaleSpec(
        name="minute",
        scale_id=1,
        delta_seconds=60.0,
        context_length=4,
        patch_length=2,
        patch_stride=1,
    ),
    "hour": ScaleSpec(
        name="hour",
        scale_id=2,
        delta_seconds=3600.0,
        context_length=32,
        patch_length=8,
        patch_stride=4,
    ),
}


def default_scale_specs() -> dict[str, ScaleSpec]:
    return dict(DEFAULT_SCALE_SPECS)


class LogScaleEmbedding(nn.Module):
    """Continuous log-time embedding plus a FinCast-style discrete frequency id."""

    def __init__(self, d_model: int, *, n_scales: int = 3, use_id_embedding: bool = True) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.id_embedding = nn.Embedding(n_scales, d_model) if use_id_embedding else None

    def forward(self, delta_seconds: torch.Tensor, scale_id: torch.Tensor | None = None) -> torch.Tensor:
        if delta_seconds.ndim == 1:
            delta_seconds = delta_seconds[:, None]
        log_delta = torch.log(torch.clamp(delta_seconds.float(), min=1.0))
        embedding = self.net(log_delta)
        if self.id_embedding is not None:
            if scale_id is None:
                raise ValueError("scale_id is required when use_id_embedding=True.")
            embedding = embedding + self.id_embedding(scale_id.long().view(-1))
        return embedding


class AdaptiveSpectralDenoising(nn.Module):
    """Scale-conditioned soft spectral shrinkage over the time dimension."""

    def __init__(self, d_model: int, *, init_gate: float = -3.0) -> None:
        super().__init__()
        self.tau_proj = nn.Linear(d_model, 1)
        self.gate_proj = nn.Linear(d_model, 1)
        nn.init.constant_(self.gate_proj.bias, init_gate)

    def forward(
        self,
        x: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,L,C], got {tuple(x.shape)}")
        batch_size, length, _ = x.shape
        x_t = x.transpose(1, 2)
        spectrum = torch.fft.rfft(x_t, dim=-1)
        magnitude = torch.abs(spectrum)
        magnitude_mean = magnitude.mean(dim=-1, keepdim=True)

        tau_scale = F.softplus(self.tau_proj(scale_emb)).view(batch_size, 1, 1)
        tau = tau_scale * magnitude_mean
        shrink = F.relu(magnitude - tau) / (magnitude + 1e-6)
        filtered = torch.fft.irfft(spectrum * shrink, n=length, dim=-1).transpose(1, 2)

        gate = torch.sigmoid(self.gate_proj(scale_emb)).view(batch_size, 1, 1)
        clean = x + gate * (filtered - x)
        if not return_diagnostics:
            return clean
        diagnostics = {
            "gate_mean": gate.mean(),
            "tau_mean": tau.mean(),
            "mean_abs_delta": torch.mean(torch.abs(clean - x)),
        }
        return clean, diagnostics


class StaticSpectralDenoising(nn.Module):
    """Static low-pass denoiser used as the non-scale-aware ASD baseline."""

    def __init__(self, *, keep_ratio: float = 0.1, blend_init: float = 0.15) -> None:
        super().__init__()
        if not 0.0 < keep_ratio <= 1.0:
            raise ValueError("keep_ratio must be in (0, 1].")
        if not 0.0 < blend_init < 1.0:
            raise ValueError("blend_init must be in (0, 1).")
        self.keep_ratio = float(keep_ratio)
        logit = math.log(blend_init / (1.0 - blend_init))
        self.logit_blend = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if x.ndim != 3:
            raise ValueError(f"x must be [B,L,C], got {tuple(x.shape)}")
        length = x.shape[1]
        freq_count = length // 2 + 1
        keep = max(2, min(freq_count, int(math.ceil(freq_count * self.keep_ratio))))
        mask = torch.zeros(1, freq_count, 1, dtype=x.dtype, device=x.device)
        mask[:, :keep, :] = 1.0

        mean = x.mean(dim=1, keepdim=True)
        centered = x - mean
        spectrum = torch.fft.rfft(centered, dim=1)
        filtered = torch.fft.irfft(spectrum * mask, n=length, dim=1) + mean
        blend = torch.sigmoid(self.logit_blend)
        clean = x + blend * (filtered - x)
        if not return_diagnostics:
            return clean
        diagnostics = {
            "gate_mean": blend,
            "tau_mean": torch.zeros((), dtype=x.dtype, device=x.device),
            "mean_abs_delta": torch.mean(torch.abs(clean - x)),
        }
        return clean, diagnostics


class AdaptiveSpectralEncoderBlock(nn.Module):
    """TSLANet-style learnable spectral block over PatchTST token sequences."""

    def __init__(self, d_model: int, max_patch_count: int, *, init_gate: float = -4.0) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.max_patch_count = int(max_patch_count)
        self.max_freq_count = self.max_patch_count // 2 + 1
        self.global_filter_real = nn.Parameter(torch.zeros(1, self.max_freq_count, d_model))
        self.global_filter_imag = nn.Parameter(torch.zeros(1, self.max_freq_count, d_model))
        self.local_filter_real = nn.Parameter(torch.zeros(1, self.max_freq_count, d_model))
        self.local_filter_imag = nn.Parameter(torch.zeros(1, self.max_freq_count, d_model))
        self.tau_proj = nn.Linear(d_model, 1)
        self.gate_proj = nn.Linear(d_model, 1)
        self.log_threshold_sharpness = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))
        nn.init.constant_(self.gate_proj.bias, init_gate)

    def forward(
        self,
        h: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 3:
            raise ValueError(f"h must be [B,N,D], got {tuple(h.shape)}")
        batch_size, token_count, d_model = h.shape
        if d_model != self.d_model:
            raise ValueError(f"d_model must be {self.d_model}, got {d_model}")
        if token_count > self.max_patch_count:
            raise ValueError(f"token_count {token_count} exceeds max_patch_count {self.max_patch_count}")

        spectrum = torch.fft.rfft(h, dim=1)
        freq_count = spectrum.shape[1]
        magnitude = torch.abs(spectrum)
        magnitude_mean = magnitude.mean(dim=(1, 2), keepdim=True)
        tau_scale = F.softplus(self.tau_proj(scale_emb)).view(batch_size, 1, 1)
        tau = tau_scale * magnitude_mean
        sharpness = torch.clamp(torch.exp(self.log_threshold_sharpness), min=1.0, max=50.0)
        local_mask = torch.sigmoid(sharpness * (magnitude - tau))

        one = torch.ones(
            1,
            freq_count,
            d_model,
            dtype=self.global_filter_real.dtype,
            device=h.device,
        )
        global_filter = torch.complex(
            one + self.global_filter_real[:, :freq_count, :],
            self.global_filter_imag[:, :freq_count, :],
        )
        local_filter = torch.complex(
            self.local_filter_real[:, :freq_count, :],
            self.local_filter_imag[:, :freq_count, :],
        )
        filtered_spectrum = spectrum * global_filter + spectrum * local_filter * local_mask
        filtered = torch.fft.irfft(filtered_spectrum, n=token_count, dim=1)

        gate = torch.sigmoid(self.gate_proj(scale_emb)).view(batch_size, 1, 1)
        clean = h + gate * (filtered - h)
        if not return_diagnostics:
            return clean
        global_delta_norm = torch.mean(
            torch.sqrt(
                self.global_filter_real[:, :freq_count, :].pow(2)
                + self.global_filter_imag[:, :freq_count, :].pow(2)
                + 1e-12
            )
        )
        local_filter_norm = torch.mean(
            torch.sqrt(
                self.local_filter_real[:, :freq_count, :].pow(2)
                + self.local_filter_imag[:, :freq_count, :].pow(2)
                + 1e-12
            )
        )
        diagnostics = {
            "gate_mean": gate.mean(),
            "tau_mean": tau.mean(),
            "local_mask_mean": local_mask.mean(),
            "mean_abs_delta": torch.mean(torch.abs(clean - h)),
            "global_filter_norm": global_delta_norm,
            "local_filter_norm": local_filter_norm,
        }
        return clean, diagnostics


class LoRAAdapterExpert(nn.Module):
    """Low-rank residual adapter used as one MoE expert."""

    def __init__(self, d_model: int, rank: int, *, alpha: float, dropout: float) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("rank must be positive.")
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout)
        self.down = nn.Linear(d_model, self.rank, bias=False)
        self.up = nn.Linear(self.rank, d_model, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.up(self.down(self.dropout(h))) * self.scaling


class SharedLoRAAdapter(nn.Module):
    """Single shared LoRA-style residual adapter without MoE routing."""

    def __init__(self, d_model: int, rank: int, *, alpha: float, dropout: float) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.rank = int(rank)
        self.adapter = LoRAAdapterExpert(d_model, rank, alpha=alpha, dropout=dropout)

    def forward(
        self,
        h: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 3:
            raise ValueError(f"h must be [B,N,D], got {tuple(h.shape)}")
        batch_size, _, d_model = h.shape
        if d_model != self.d_model:
            raise ValueError(f"d_model must be {self.d_model}, got {d_model}")
        if scale_emb.shape != (batch_size, d_model):
            raise ValueError(f"scale_emb must be [{batch_size},{d_model}], got {tuple(scale_emb.shape)}")

        update = self.adapter(h)
        out = h + update
        if not return_diagnostics:
            return out
        diagnostics = {
            "mean_abs_delta": torch.mean(torch.abs(out - h)),
            "adapter_rank": torch.tensor(float(self.rank), dtype=h.dtype, device=h.device),
        }
        return out, diagnostics


class MLPAdapterExpert(nn.Module):
    """Bottleneck MLP residual expert used to isolate MoE from LoRA."""

    def __init__(self, d_model: int, bottleneck: int, *, dropout: float) -> None:
        super().__init__()
        if bottleneck <= 0:
            raise ValueError("bottleneck must be positive.")
        self.down = nn.Linear(d_model, int(bottleneck))
        self.up = nn.Linear(int(bottleneck), d_model)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.up(F.gelu(self.down(h))))


class ScaleAwareMLPAdapterMoE(nn.Module):
    """Scale-aware sparse MoE with full MLP bottleneck experts instead of LoRA experts."""

    def __init__(
        self,
        d_model: int,
        *,
        n_experts: int = 4,
        bottleneck: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        if top_k <= 0 or top_k > n_experts:
            raise ValueError("top_k must be in [1, n_experts].")
        self.d_model = int(d_model)
        self.n_experts = int(n_experts)
        self.bottleneck = int(bottleneck)
        self.top_k = int(top_k)
        self.router = nn.Linear(d_model * 2, n_experts)
        self.scale_router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [MLPAdapterExpert(d_model, self.bottleneck, dropout=dropout) for _ in range(n_experts)]
        )
        nn.init.zeros_(self.scale_router.weight)

    def forward(
        self,
        h: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 3:
            raise ValueError(f"h must be [B,N,D], got {tuple(h.shape)}")
        batch_size, token_count, d_model = h.shape
        if d_model != self.d_model:
            raise ValueError(f"d_model must be {self.d_model}, got {d_model}")
        if scale_emb.shape != (batch_size, d_model):
            raise ValueError(f"scale_emb must be [{batch_size},{d_model}], got {tuple(scale_emb.shape)}")

        scale_tokens = scale_emb[:, None, :].expand(batch_size, token_count, d_model)
        scale_logits = self.scale_router(scale_emb)[:, None, :]
        logits = self.router(torch.cat([h, scale_tokens], dim=-1)) + scale_logits
        if self.top_k < self.n_experts:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(-1, top_indices, top_values)
            weights = torch.softmax(masked_logits, dim=-1)
        else:
            weights = torch.softmax(logits, dim=-1)

        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=-2)
        update = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        out = h + update
        if not return_diagnostics:
            return out

        avg_prob = weights.mean(dim=(0, 1))
        uniform = torch.full_like(avg_prob, 1.0 / self.n_experts)
        token_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        diagnostics = {
            "router_entropy": token_entropy / math.log(self.n_experts),
            "router_balance_loss": F.mse_loss(avg_prob, uniform),
            "mean_abs_delta": torch.mean(torch.abs(out - h)),
        }
        scale_prior_prob = torch.softmax(self.scale_router(scale_emb), dim=-1).mean(dim=0)
        for expert_idx in range(self.n_experts):
            diagnostics[f"expert_prob_{expert_idx}"] = avg_prob[expert_idx]
            diagnostics[f"scale_prior_prob_{expert_idx}"] = scale_prior_prob[expert_idx]
        return out, diagnostics


class ScaleAwareLoRAAdapterMoE(nn.Module):
    """Scale-aware sparse MoE over LoRA-style low-rank token adapters."""

    def __init__(
        self,
        d_model: int,
        *,
        n_experts: int = 4,
        rank: int = 4,
        alpha: float = 16.0,
        top_k: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        if top_k <= 0 or top_k > n_experts:
            raise ValueError("top_k must be in [1, n_experts].")
        self.d_model = int(d_model)
        self.n_experts = int(n_experts)
        self.rank = int(rank)
        self.top_k = int(top_k)
        self.router = nn.Linear(d_model * 2, n_experts)
        self.scale_router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                LoRAAdapterExpert(d_model, rank, alpha=alpha, dropout=dropout)
                for _ in range(n_experts)
            ]
        )
        nn.init.zeros_(self.scale_router.weight)

    def routing_logits(self, h: torch.Tensor, scale_emb: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must be [B,N,D], got {tuple(h.shape)}")
        batch_size, token_count, d_model = h.shape
        if d_model != self.d_model:
            raise ValueError(f"d_model must be {self.d_model}, got {d_model}")
        if scale_emb.shape != (batch_size, d_model):
            raise ValueError(f"scale_emb must be [{batch_size},{d_model}], got {tuple(scale_emb.shape)}")
        scale_tokens = scale_emb[:, None, :].expand(batch_size, token_count, d_model)
        scale_logits = self.scale_router(scale_emb)[:, None, :]
        return self.router(torch.cat([h, scale_tokens], dim=-1)) + scale_logits

    def routing_weights(
        self,
        h: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        sparse: bool = True,
    ) -> torch.Tensor:
        logits = self.routing_logits(h, scale_emb)
        if sparse and self.top_k < self.n_experts:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(-1, top_indices, top_values)
            return torch.softmax(masked_logits, dim=-1)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        h: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if h.ndim != 3:
            raise ValueError(f"h must be [B,N,D], got {tuple(h.shape)}")
        batch_size, token_count, d_model = h.shape
        if d_model != self.d_model:
            raise ValueError(f"d_model must be {self.d_model}, got {d_model}")
        if scale_emb.shape != (batch_size, d_model):
            raise ValueError(f"scale_emb must be [{batch_size},{d_model}], got {tuple(scale_emb.shape)}")

        weights = self.routing_weights(h, scale_emb, sparse=True)

        expert_outputs = torch.stack([expert(h) for expert in self.experts], dim=-2)
        update = (weights.unsqueeze(-1) * expert_outputs).sum(dim=-2)
        out = h + update
        if not return_diagnostics:
            return out

        avg_prob = weights.mean(dim=(0, 1))
        uniform = torch.full_like(avg_prob, 1.0 / self.n_experts)
        token_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        diagnostics = {
            "router_entropy": token_entropy / math.log(self.n_experts),
            "router_balance_loss": F.mse_loss(avg_prob, uniform),
            "mean_abs_delta": torch.mean(torch.abs(out - h)),
        }
        scale_prior_prob = torch.softmax(self.scale_router(scale_emb), dim=-1).mean(dim=0)
        for expert_idx in range(self.n_experts):
            diagnostics[f"expert_prob_{expert_idx}"] = avg_prob[expert_idx]
            diagnostics[f"scale_prior_prob_{expert_idx}"] = scale_prior_prob[expert_idx]
        return out, diagnostics


class InteractiveConvolutionBlock(nn.Module):
    """TSLANet-style local interaction block over patch tokens."""

    def __init__(
        self,
        d_model: int,
        *,
        hidden_multiplier: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        hidden = int(d_model) * int(hidden_multiplier)
        padding = int(kernel_size) // 2
        self.branch_pointwise = nn.Conv1d(d_model, hidden, kernel_size=1)
        self.branch_local = nn.Conv1d(d_model, hidden, kernel_size=kernel_size, padding=padding)
        self.project = nn.Conv1d(hidden, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.project.weight)
        nn.init.zeros_(self.project.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.ndim != 3:
            raise ValueError(f"h must be [B,N,D], got {tuple(h.shape)}")
        x = h.transpose(1, 2)
        point = self.branch_pointwise(x)
        local = self.branch_local(x)
        interacted = F.gelu(point) * local + F.gelu(local) * point
        out = self.project(self.dropout(interacted)).transpose(1, 2)
        return h + self.dropout(out)


class TSLANetBlock(nn.Module):
    """Lightweight TSLA block with adaptive spectral filtering and ICB."""

    def __init__(
        self,
        d_model: int,
        max_patch_count: int,
        *,
        icb_hidden_multiplier: int = 2,
        icb_kernel_size: int = 3,
        dropout: float = 0.1,
        spectral_init_gate: float = -4.0,
    ) -> None:
        super().__init__()
        self.norm_spectral = nn.LayerNorm(d_model)
        self.spectral = AdaptiveSpectralEncoderBlock(
            d_model,
            max_patch_count,
            init_gate=spectral_init_gate,
        )
        self.norm_icb = nn.LayerNorm(d_model)
        self.icb = InteractiveConvolutionBlock(
            d_model,
            hidden_multiplier=icb_hidden_multiplier,
            kernel_size=icb_kernel_size,
            dropout=dropout,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        normalized = self.norm_spectral(h)
        spectral_out, diagnostics = self.spectral(normalized, scale_emb, return_diagnostics=True)
        h = h + self.dropout(spectral_out - normalized)
        normalized = self.norm_icb(h)
        h = h + self.dropout(self.icb(normalized) - normalized)
        if return_diagnostics:
            return h, diagnostics
        return h


class TSLANetMultiScaleForecaster(nn.Module):
    """TSLANet-style baseline using shared TSLA blocks across intraday scales."""

    def __init__(
        self,
        scale_specs: dict[str, ScaleSpec],
        *,
        input_channels: int = 1,
        d_model: int = 64,
        n_layers: int = 2,
        dropout: float = 0.1,
        spectral_init_gate: float = -4.0,
        icb_hidden_multiplier: int = 2,
        icb_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if not scale_specs:
            raise ValueError("scale_specs must not be empty.")
        self.scale_specs = dict(scale_specs)
        self.input_channels = int(input_channels)
        self.d_model = int(d_model)
        for spec in self.scale_specs.values():
            if spec.context_length < spec.patch_length:
                raise ValueError(f"{spec.name}: context_length must be at least patch_length.")
            if spec.patch_stride <= 0:
                raise ValueError(f"{spec.name}: patch_stride must be positive.")
        self.scale_embedding = LogScaleEmbedding(d_model)
        self.patch_projection = nn.ModuleDict(
            {name: nn.Linear(spec.patch_length, d_model) for name, spec in self.scale_specs.items()}
        )
        self.max_patch_count = max(spec.patch_count for spec in self.scale_specs.values())
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_patch_count, d_model))
        self.input_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TSLANetBlock(
                    d_model,
                    self.max_patch_count,
                    icb_hidden_multiplier=icb_hidden_multiplier,
                    icb_kernel_size=icb_kernel_size,
                    dropout=dropout,
                    spectral_init_gate=spectral_init_gate,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.heads = nn.ModuleDict(
            {
                name: nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Dropout(dropout),
                    nn.Linear(spec.patch_count * d_model, spec.prediction_length),
                )
                for name, spec in self.scale_specs.items()
            }
        )

    def scale_embedding_for(self, scale_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        spec = self.scale_specs[scale_name]
        delta = torch.full((batch_size,), float(spec.delta_seconds), device=device)
        scale_id = torch.full((batch_size,), int(spec.scale_id), device=device, dtype=torch.long)
        return self.scale_embedding(delta, scale_id)

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.scale_specs)}")
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        batch_size, _, channels = past_values.shape
        spec = self.scale_specs[scale_name]
        x = past_values.transpose(1, 2)
        patches = x.unfold(dimension=-1, size=spec.patch_length, step=spec.patch_stride)
        if patches.shape[2] != spec.patch_count:
            raise ValueError(f"{scale_name}: expected {spec.patch_count} patches, got {patches.shape[2]}")
        patches = patches.reshape(batch_size * channels, spec.patch_count, spec.patch_length)
        tokens = self.patch_projection[scale_name](patches)
        scale_emb = self.scale_embedding_for(scale_name, batch_size, past_values.device)
        scale_emb = scale_emb.repeat_interleave(channels, dim=0)
        tokens = tokens + self.position_embedding[:, : spec.patch_count, :] + scale_emb[:, None, :]
        tokens = self.input_dropout(tokens)

        diagnostics: dict[str, torch.Tensor] = {}
        for idx, block in enumerate(self.blocks):
            tokens, block_diag = block(tokens, scale_emb, return_diagnostics=True)
            if idx == len(self.blocks) - 1:
                diagnostics = {f"tslanet_{key}": value for key, value in block_diag.items()}
        tokens = self.norm(tokens)
        y = self.heads[scale_name](tokens)
        y = y.reshape(batch_size, channels, spec.prediction_length).transpose(1, 2)
        if return_diagnostics:
            return y, diagnostics
        return y


class ForecastHead(nn.Module):
    """Scale-specific forecast head over flattened PatchTST tokens."""

    def __init__(
        self,
        input_dim: int,
        prediction_length: int,
        *,
        dropout: float = 0.1,
        head_type: str = "linear",
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(dropout)
        if head_type == "linear":
            self.net = nn.Linear(input_dim, prediction_length)
        elif head_type == "mlp":
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, prediction_length),
            )
        else:
            raise ValueError("head_type must be 'linear' or 'mlp'.")

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.net(self.dropout(self.flatten(tokens)))


class MultiScalePatchTST(nn.Module):
    """PatchTST-style forecaster with scale-specific patching and shared encoder."""

    def __init__(
        self,
        scale_specs: dict[str, ScaleSpec],
        *,
        input_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        encoder_spectral_mode: str = "none",
        encoder_spectral_init_gate: float = -4.0,
        lora_moe_mode: str = "none",
        lora_moe_rank: int = 4,
        lora_moe_alpha: float = 16.0,
        lora_moe_n_experts: int = 4,
        lora_moe_top_k: int = 2,
        lora_moe_dropout: float = 0.1,
        target_mode: str = "per_channel",
        head_type: str = "linear",
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if encoder_spectral_mode not in {"none", "last1", "after1"}:
            raise ValueError("encoder_spectral_mode must be 'none', 'last1', or 'after1'.")
        if lora_moe_mode not in {"none", "last1", "after1", "lora_only", "mlp_moe"}:
            raise ValueError("lora_moe_mode must be 'none', 'last1', 'after1', 'lora_only', or 'mlp_moe'.")
        if target_mode not in {"per_channel", "all_channels"}:
            raise ValueError("target_mode must be 'per_channel' or 'all_channels'.")
        if head_type not in {"linear", "mlp"}:
            raise ValueError("head_type must be 'linear' or 'mlp'.")
        if not scale_specs:
            raise ValueError("scale_specs must not be empty.")
        for spec in scale_specs.values():
            if spec.context_length < spec.patch_length:
                raise ValueError(f"{spec.name}: context_length must be at least patch_length.")
            if spec.patch_stride <= 0:
                raise ValueError(f"{spec.name}: patch_stride must be positive.")
            if d_model % n_heads != 0:
                raise ValueError("d_model must be divisible by n_heads.")

        self.scale_specs = dict(scale_specs)
        self.input_channels = int(input_channels)
        self.d_model = int(d_model)
        self.encoder_spectral_mode = encoder_spectral_mode
        self.lora_moe_mode = lora_moe_mode
        self.target_mode = target_mode
        self.head_type = head_type
        self.scale_embedding = LogScaleEmbedding(d_model)
        self.patch_projection = nn.ModuleDict(
            {name: nn.Linear(spec.patch_length, d_model) for name, spec in self.scale_specs.items()}
        )
        self.max_patch_count = max(spec.patch_count for spec in self.scale_specs.values())
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_patch_count, d_model))
        self.input_dropout = nn.Dropout(dropout)
        encoder_config = PatchTSTForecastConfig(
            context_length=self.max_patch_count,
            prediction_length=1,
            input_channels=input_channels,
            patch_length=1,
            patch_stride=1,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            lora=LoRAConfig(rank=0, alpha=1.0, dropout=0.0, enabled=False),
        )
        self.layers = nn.ModuleList(PatchTSTEncoderLayer(encoder_config) for _ in range(n_layers))
        self.encoder_spectral = (
            AdaptiveSpectralEncoderBlock(d_model, self.max_patch_count, init_gate=encoder_spectral_init_gate)
            if encoder_spectral_mode != "none"
            else None
        )
        self.lora_moe = None
        if lora_moe_mode in {"last1", "after1"}:
            self.lora_moe = ScaleAwareLoRAAdapterMoE(
                d_model,
                n_experts=lora_moe_n_experts,
                rank=lora_moe_rank,
                alpha=lora_moe_alpha,
                top_k=lora_moe_top_k,
                dropout=lora_moe_dropout,
            )
        elif lora_moe_mode == "lora_only":
            self.lora_moe = SharedLoRAAdapter(
                d_model,
                lora_moe_rank,
                alpha=lora_moe_alpha,
                dropout=lora_moe_dropout,
            )
        elif lora_moe_mode == "mlp_moe":
            self.lora_moe = ScaleAwareMLPAdapterMoE(
                d_model,
                n_experts=lora_moe_n_experts,
                bottleneck=lora_moe_rank,
                top_k=lora_moe_top_k,
                dropout=lora_moe_dropout,
            )
        self.norm = nn.LayerNorm(d_model)
        head_input_multiplier = self.input_channels if target_mode == "all_channels" else 1
        self.heads = nn.ModuleDict(
            {
                name: ForecastHead(
                    d_model * spec.patch_count * head_input_multiplier,
                    spec.prediction_length,
                    dropout=dropout,
                    head_type=head_type,
                    hidden_dim=head_hidden_dim,
                )
                for name, spec in self.scale_specs.items()
            }
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def scale_embedding_for(self, scale_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        spec = self.scale_specs[scale_name]
        delta = torch.full((batch_size,), float(spec.delta_seconds), device=device)
        scale_id = torch.full((batch_size,), int(spec.scale_id), device=device, dtype=torch.long)
        return self.scale_embedding(delta, scale_id)

    def adapter_input_tokens(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        include_encoder_spectral: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int, ScaleSpec]:
        if scale_name not in self.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.scale_specs)}")
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        spec = self.scale_specs[scale_name]
        batch_size, context_length, channels = past_values.shape
        if context_length != spec.context_length:
            raise ValueError(f"{scale_name}: context_length must be {spec.context_length}, got {context_length}")
        if channels != self.input_channels:
            raise ValueError(f"input_channels must be {self.input_channels}, got {channels}")

        x = past_values.transpose(1, 2)
        x = x.unfold(dimension=-1, size=spec.patch_length, step=spec.patch_stride)
        x = x.reshape(batch_size * channels, spec.patch_count, spec.patch_length)
        x = self.patch_projection[scale_name](x)
        scale_emb = self.scale_embedding_for(scale_name, batch_size, past_values.device)
        scale_emb = scale_emb.repeat_interleave(channels, dim=0)
        x = x + self.position_embedding[:, : spec.patch_count, :] + scale_emb[:, None, :]
        x = self.input_dropout(x)
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            if (
                include_encoder_spectral
                and self.encoder_spectral is not None
                and self.encoder_spectral_mode == "after1"
                and layer_idx == 0
            ):
                x = self.encoder_spectral(x, scale_emb)
            if self.lora_moe is not None and self.lora_moe_mode == "after1" and layer_idx == 0:
                x = self.lora_moe(x, scale_emb)
        if (
            include_encoder_spectral
            and self.encoder_spectral is not None
            and self.encoder_spectral_mode == "last1"
        ):
            x = self.encoder_spectral(x, scale_emb)
        return x, scale_emb, batch_size, channels, spec

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.scale_specs)}")
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        spec = self.scale_specs[scale_name]
        batch_size, context_length, channels = past_values.shape
        if context_length != spec.context_length:
            raise ValueError(f"{scale_name}: context_length must be {spec.context_length}, got {context_length}")
        if channels != self.input_channels:
            raise ValueError(f"input_channels must be {self.input_channels}, got {channels}")

        x = past_values.transpose(1, 2)
        x = x.unfold(dimension=-1, size=spec.patch_length, step=spec.patch_stride)
        x = x.reshape(batch_size * channels, spec.patch_count, spec.patch_length)
        x = self.patch_projection[scale_name](x)
        scale_emb = self.scale_embedding_for(scale_name, batch_size, past_values.device)
        scale_emb = scale_emb.repeat_interleave(channels, dim=0)
        x = x + self.position_embedding[:, : spec.patch_count, :] + scale_emb[:, None, :]
        x = self.input_dropout(x)
        diagnostics: dict[str, torch.Tensor] = {}
        for layer_idx, layer in enumerate(self.layers):
            x = layer(x)
            if self.encoder_spectral is not None and self.encoder_spectral_mode == "after1" and layer_idx == 0:
                if return_diagnostics:
                    x, spectral_diagnostics = self.encoder_spectral(x, scale_emb, return_diagnostics=True)
                    diagnostics.update(spectral_diagnostics)
                else:
                    x = self.encoder_spectral(x, scale_emb)
            if self.lora_moe is not None and self.lora_moe_mode == "after1" and layer_idx == 0:
                if return_diagnostics:
                    x, moe_diagnostics = self.lora_moe(x, scale_emb, return_diagnostics=True)
                    diagnostics.update({f"mid_moe_{key}": value for key, value in moe_diagnostics.items()})
                    diagnostics.update(moe_diagnostics)
                else:
                    x = self.lora_moe(x, scale_emb)
        if self.encoder_spectral is not None and self.encoder_spectral_mode == "last1":
            if return_diagnostics:
                x, spectral_diagnostics = self.encoder_spectral(x, scale_emb, return_diagnostics=True)
                diagnostics.update(spectral_diagnostics)
            else:
                x = self.encoder_spectral(x, scale_emb)
        if self.lora_moe is not None and self.lora_moe_mode != "after1":
            x, moe_diagnostics = self.lora_moe(x, scale_emb, return_diagnostics=True)
            diagnostics.update(moe_diagnostics)
        x = self.norm(x)
        if self.target_mode == "all_channels":
            x = x.reshape(batch_size, channels * spec.patch_count, self.d_model)
            y = self.heads[scale_name](x)
            y = y.reshape(batch_size, spec.prediction_length, 1)
        else:
            y = self.heads[scale_name](x)
            y = y.reshape(batch_size, channels, spec.prediction_length).transpose(1, 2)
        if return_diagnostics:
            return y, diagnostics
        return y


class RawMultiScalePatchTST(nn.Module):
    def __init__(self, backbone: MultiScalePatchTST) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        y = self.backbone(past_values, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return y
        if isinstance(y, tuple):
            return y
        zero = torch.zeros((), dtype=past_values.dtype, device=past_values.device)
        return y, {"gate_mean": zero, "tau_mean": zero, "mean_abs_delta": zero}


class ScaleSpecificPatchTST(nn.Module):
    """Use an independent PatchTST backbone for each intraday scale."""

    def __init__(
        self,
        scale_specs: dict[str, ScaleSpec],
        *,
        input_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        head_type: str = "linear",
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.scale_specs = dict(scale_specs)
        self.backbones = nn.ModuleDict(
            {
                name: MultiScalePatchTST(
                    {name: spec},
                    input_channels=input_channels,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    encoder_spectral_mode="none",
                    lora_moe_mode="none",
                    head_type=head_type,
                    head_hidden_dim=head_hidden_dim,
                )
                for name, spec in self.scale_specs.items()
            }
        )

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.backbones:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbones)}")
        out = self.backbones[scale_name](past_values, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            return out
        zero = torch.zeros((), dtype=past_values.dtype, device=past_values.device)
        return out, {"gate_mean": zero, "tau_mean": zero, "mean_abs_delta": zero}


class PaddedFrequencyRouterScaleSpecificASDPatchTST(nn.Module):
    """Pad variable scale windows, route by FFT embedding, then use scale-specific PatchTST backbones."""

    def __init__(
        self,
        scale_specs: dict[str, ScaleSpec],
        *,
        input_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        init_gate: float = -4.0,
        top_k: int = 2,
        scale_prior_strength: float = 1.0,
        head_type: str = "linear",
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if not scale_specs:
            raise ValueError("scale_specs must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.scale_specs = dict(scale_specs)
        self.scale_names = list(self.scale_specs)
        self.input_channels = int(input_channels)
        self.d_model = int(d_model)
        self.max_context_length = max(spec.context_length for spec in self.scale_specs.values())
        self.freq_count = self.max_context_length // 2 + 1
        self.top_k = min(int(top_k), len(self.scale_names))

        self.scale_embedding = LogScaleEmbedding(d_model)
        self.frequency_embedding = nn.Sequential(
            nn.LayerNorm(self.freq_count),
            nn.Linear(self.freq_count, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(self.scale_names)),
        )
        self.scale_router_prior = nn.Embedding(3, len(self.scale_names))
        nn.init.zeros_(self.scale_router_prior.weight)
        with torch.no_grad():
            for expert_idx, spec in enumerate(self.scale_specs.values()):
                if spec.scale_id < self.scale_router_prior.weight.shape[0]:
                    self.scale_router_prior.weight[spec.scale_id, expert_idx] = float(scale_prior_strength)

        self.asd_experts = nn.ModuleDict(
            {
                name: AdaptiveSpectralDenoising(d_model, init_gate=init_gate)
                for name in self.scale_names
            }
        )
        self.backbones = nn.ModuleDict(
            {
                name: MultiScalePatchTST(
                    {name: spec},
                    input_channels=input_channels,
                    d_model=d_model,
                    n_heads=n_heads,
                    n_layers=n_layers,
                    d_ff=d_ff,
                    dropout=dropout,
                    encoder_spectral_mode="none",
                    lora_moe_mode="none",
                    head_type=head_type,
                    head_hidden_dim=head_hidden_dim,
                )
                for name, spec in self.scale_specs.items()
            }
        )

    def scale_embedding_for(self, scale_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        spec = self.scale_specs[scale_name]
        delta = torch.full((batch_size,), float(spec.delta_seconds), device=device)
        scale_id = torch.full((batch_size,), int(spec.scale_id), device=device, dtype=torch.long)
        return self.scale_embedding(delta, scale_id)

    def router_weights(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        scale_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.scale_specs[scale_name]
        length = past_values.shape[1]
        if length > self.max_context_length:
            raise ValueError(f"{scale_name}: length {length} exceeds max padded length {self.max_context_length}.")
        padded = F.pad(past_values, (0, 0, 0, self.max_context_length - length))
        spectrum = torch.fft.rfft(padded.transpose(1, 2), dim=-1)
        magnitude = torch.log1p(torch.abs(spectrum).mean(dim=1))
        freq_emb = self.frequency_embedding(magnitude)
        router_input = torch.cat([freq_emb, scale_emb], dim=-1)
        scale_id = torch.full(
            (past_values.shape[0],),
            int(spec.scale_id),
            dtype=torch.long,
            device=past_values.device,
        )
        logits = self.router(router_input) + self.scale_router_prior(scale_id)
        if self.top_k < logits.shape[-1]:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            masked = torch.full_like(logits, float("-inf"))
            masked.scatter_(-1, top_indices, top_values)
            weights = torch.softmax(masked, dim=-1)
        else:
            weights = torch.softmax(logits, dim=-1)
        return weights, logits, freq_emb

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.scale_specs)}")
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        spec = self.scale_specs[scale_name]
        if past_values.shape[1] != spec.context_length:
            raise ValueError(f"{scale_name}: input length must be {spec.context_length}, got {past_values.shape[1]}")
        if past_values.shape[2] != self.input_channels:
            raise ValueError(f"input_channels must be {self.input_channels}, got {past_values.shape[2]}")

        scale_emb = self.scale_embedding_for(scale_name, past_values.shape[0], past_values.device)
        weights, _, freq_emb = self.router_weights(past_values, scale_name, scale_emb)
        expert_outputs: list[torch.Tensor] = []
        expert_gate_means: list[torch.Tensor] = []
        expert_tau_means: list[torch.Tensor] = []
        for expert_name in self.scale_names:
            clean, asd_diagnostics = self.asd_experts[expert_name](
                past_values,
                scale_emb,
                return_diagnostics=True,
            )
            expert_outputs.append(clean)
            expert_gate_means.append(asd_diagnostics["gate_mean"])
            expert_tau_means.append(asd_diagnostics["tau_mean"])
        stacked = torch.stack(expert_outputs, dim=-1)
        routed = (stacked * weights[:, None, None, :]).sum(dim=-1)
        out = self.backbones[scale_name](routed, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}

        avg_prob = weights.mean(dim=0)
        uniform = torch.full_like(avg_prob, 1.0 / len(self.scale_names))
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        gate_stack = torch.stack(expert_gate_means)
        tau_stack = torch.stack(expert_tau_means)
        diagnostics: dict[str, torch.Tensor] = {
            "router_entropy": entropy,
            "router_balance_loss": F.mse_loss(avg_prob, uniform),
            "asd_router_entropy": entropy,
            "asd_router_balance_loss": F.mse_loss(avg_prob, uniform),
            "frequency_embedding_norm": torch.mean(torch.linalg.vector_norm(freq_emb, dim=-1)),
            "mean_abs_delta": torch.mean(torch.abs(routed - past_values)),
            "asd_gate_mean": torch.sum(avg_prob * gate_stack),
            "asd_tau_mean": torch.sum(avg_prob * tau_stack),
            "padded_context_length": torch.tensor(
                float(self.max_context_length),
                dtype=past_values.dtype,
                device=past_values.device,
            ),
        }
        for expert_idx, expert_name in enumerate(self.scale_names):
            diagnostics[f"expert_prob_{expert_idx}"] = avg_prob[expert_idx]
            diagnostics[f"expert_{expert_name}_prob"] = avg_prob[expert_idx]
            diagnostics[f"expert_{expert_name}_gate_mean"] = expert_gate_means[expert_idx]
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"backbone_{key}": value for key, value in backbone_diagnostics.items()})
        return y, diagnostics


class PaddedFrequencyRouterSharedASDPatchTST(nn.Module):
    """Route padded sequences through ASD experts, crop, then use one shared PatchTST backbone."""

    def __init__(
        self,
        scale_specs: dict[str, ScaleSpec],
        *,
        input_channels: int = 1,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        init_gate: float = -4.0,
        top_k: int = 2,
        scale_prior_strength: float = 1.0,
        include_identity_expert: bool = True,
        backbone_lora_moe_mode: str = "none",
        lora_moe_rank: int = 4,
        lora_moe_alpha: float = 16.0,
        lora_moe_n_experts: int = 4,
        lora_moe_top_k: int = 2,
        lora_moe_dropout: float = 0.1,
        head_type: str = "linear",
        head_hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        if not scale_specs:
            raise ValueError("scale_specs must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.scale_specs = dict(scale_specs)
        self.scale_names = list(self.scale_specs)
        self.expert_names = (["identity"] if include_identity_expert else []) + self.scale_names
        self.input_channels = int(input_channels)
        self.d_model = int(d_model)
        self.max_context_length = max(spec.context_length for spec in self.scale_specs.values())
        self.freq_count = self.max_context_length // 2 + 1
        self.top_k = min(int(top_k), len(self.expert_names))
        self.include_identity_expert = bool(include_identity_expert)

        self.scale_embedding = LogScaleEmbedding(d_model)
        self.frequency_embedding = nn.Sequential(
            nn.LayerNorm(self.freq_count),
            nn.Linear(self.freq_count, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.router = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, len(self.expert_names)),
        )
        self.scale_router_prior = nn.Embedding(3, len(self.expert_names))
        nn.init.zeros_(self.scale_router_prior.weight)
        with torch.no_grad():
            offset = 1 if self.include_identity_expert else 0
            for expert_idx, spec in enumerate(self.scale_specs.values()):
                if spec.scale_id < self.scale_router_prior.weight.shape[0]:
                    self.scale_router_prior.weight[spec.scale_id, expert_idx + offset] = float(scale_prior_strength)

        self.asd_experts = nn.ModuleDict(
            {
                name: AdaptiveSpectralDenoising(d_model, init_gate=init_gate)
                for name in self.scale_names
            }
        )
        self.backbone = MultiScalePatchTST(
            self.scale_specs,
            input_channels=input_channels,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_ff,
            dropout=dropout,
            encoder_spectral_mode="none",
            lora_moe_mode=backbone_lora_moe_mode,
            lora_moe_rank=lora_moe_rank,
            lora_moe_alpha=lora_moe_alpha,
            lora_moe_n_experts=lora_moe_n_experts,
            lora_moe_top_k=lora_moe_top_k,
            lora_moe_dropout=lora_moe_dropout,
            head_type=head_type,
            head_hidden_dim=head_hidden_dim,
        )

    def scale_embedding_for(self, scale_name: str, batch_size: int, device: torch.device) -> torch.Tensor:
        spec = self.scale_specs[scale_name]
        delta = torch.full((batch_size,), float(spec.delta_seconds), device=device)
        scale_id = torch.full((batch_size,), int(spec.scale_id), device=device, dtype=torch.long)
        return self.scale_embedding(delta, scale_id)

    def pad_with_mask(self, past_values: torch.Tensor, scale_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        length = past_values.shape[1]
        if length > self.max_context_length:
            raise ValueError(f"{scale_name}: length {length} exceeds max padded length {self.max_context_length}.")
        padded = F.pad(past_values, (0, 0, 0, self.max_context_length - length))
        mask = torch.zeros(
            past_values.shape[0],
            self.max_context_length,
            1,
            dtype=past_values.dtype,
            device=past_values.device,
        )
        mask[:, :length, :] = 1.0
        return padded, mask

    def router_weights(
        self,
        padded_values: torch.Tensor,
        mask: torch.Tensor,
        scale_name: str,
        scale_emb: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spec = self.scale_specs[scale_name]
        masked_values = padded_values * mask
        spectrum = torch.fft.rfft(masked_values.transpose(1, 2), dim=-1)
        magnitude = torch.log1p(torch.abs(spectrum).mean(dim=1))
        freq_emb = self.frequency_embedding(magnitude)
        router_input = torch.cat([freq_emb, scale_emb], dim=-1)
        scale_id = torch.full(
            (padded_values.shape[0],),
            int(spec.scale_id),
            dtype=torch.long,
            device=padded_values.device,
        )
        logits = self.router(router_input) + self.scale_router_prior(scale_id)
        if self.top_k < logits.shape[-1]:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(-1, top_indices, top_values)
            weights = torch.softmax(masked_logits, dim=-1)
        else:
            weights = torch.softmax(logits, dim=-1)
        return weights, logits, freq_emb

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.scale_specs)}")
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        spec = self.scale_specs[scale_name]
        if past_values.shape[1] != spec.context_length:
            raise ValueError(f"{scale_name}: input length must be {spec.context_length}, got {past_values.shape[1]}")
        if past_values.shape[2] != self.input_channels:
            raise ValueError(f"input_channels must be {self.input_channels}, got {past_values.shape[2]}")

        scale_emb = self.scale_embedding_for(scale_name, past_values.shape[0], past_values.device)
        padded, mask = self.pad_with_mask(past_values, scale_name)
        weights, _, freq_emb = self.router_weights(padded, mask, scale_name, scale_emb)

        expert_outputs: list[torch.Tensor] = []
        expert_gate_means: list[torch.Tensor] = []
        expert_tau_means: list[torch.Tensor] = []
        if self.include_identity_expert:
            expert_outputs.append(padded)
            expert_gate_means.append(torch.zeros((), dtype=past_values.dtype, device=past_values.device))
            expert_tau_means.append(torch.zeros((), dtype=past_values.dtype, device=past_values.device))
        for expert_name in self.scale_names:
            clean, asd_diagnostics = self.asd_experts[expert_name](
                padded,
                scale_emb,
                return_diagnostics=True,
            )
            expert_outputs.append(clean * mask)
            expert_gate_means.append(asd_diagnostics["gate_mean"])
            expert_tau_means.append(asd_diagnostics["tau_mean"])

        stacked = torch.stack(expert_outputs, dim=-1)
        routed_padded = (stacked * weights[:, None, None, :]).sum(dim=-1)
        routed = routed_padded[:, : spec.context_length, :]
        out = self.backbone(routed, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}

        avg_prob = weights.mean(dim=0)
        uniform = torch.full_like(avg_prob, 1.0 / len(self.expert_names))
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        asd_router_balance = F.mse_loss(avg_prob, uniform)
        gate_stack = torch.stack(expert_gate_means)
        tau_stack = torch.stack(expert_tau_means)
        diagnostics: dict[str, torch.Tensor] = {
            "router_entropy": entropy,
            "router_balance_loss": asd_router_balance,
            "asd_router_entropy": entropy,
            "asd_router_balance_loss": asd_router_balance,
            "frequency_embedding_norm": torch.mean(torch.linalg.vector_norm(freq_emb, dim=-1)),
            "mean_abs_delta": torch.mean(torch.abs(routed - past_values)),
            "asd_gate_mean": torch.sum(avg_prob * gate_stack),
            "asd_tau_mean": torch.sum(avg_prob * tau_stack),
            "valid_mask_mean": mask.mean(),
            "padded_context_length": torch.tensor(
                float(self.max_context_length),
                dtype=past_values.dtype,
                device=past_values.device,
            ),
        }
        for expert_idx, expert_name in enumerate(self.expert_names):
            diagnostics[f"expert_prob_{expert_idx}"] = avg_prob[expert_idx]
            diagnostics[f"expert_{expert_name}_prob"] = avg_prob[expert_idx]
        for expert_idx, expert_name in enumerate(self.scale_names):
            diagnostics[f"expert_{expert_name}_gate_mean"] = expert_gate_means[
                expert_idx + (1 if self.include_identity_expert else 0)
            ]
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"backbone_{key}": value for key, value in backbone_diagnostics.items()})
        if "router_balance_loss" in backbone_diagnostics:
            diagnostics["router_balance_loss"] = asd_router_balance + backbone_diagnostics["router_balance_loss"]
            diagnostics["combined_router_balance_loss"] = diagnostics["router_balance_loss"]
        return y, diagnostics


class StaticASDMultiScalePatchTST(nn.Module):
    def __init__(self, backbone: MultiScalePatchTST, *, keep_ratio: float, blend_init: float) -> None:
        super().__init__()
        self.backbone = backbone
        self.denoiser = StaticSpectralDenoising(keep_ratio=keep_ratio, blend_init=blend_init)

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        clean, diagnostics = self.denoiser(past_values, return_diagnostics=True)
        y = self.backbone(clean, scale_name)
        if return_diagnostics:
            return y, diagnostics
        return y


class ScaleAwareASDMultiScalePatchTST(nn.Module):
    def __init__(self, backbone: MultiScalePatchTST, *, init_gate: float = -3.0) -> None:
        super().__init__()
        self.backbone = backbone
        self.denoiser = AdaptiveSpectralDenoising(backbone.d_model, init_gate=init_gate)

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scale_emb = self.backbone.scale_embedding_for(scale_name, past_values.shape[0], past_values.device)
        clean, asd_diagnostics = self.denoiser(past_values, scale_emb, return_diagnostics=True)
        out = self.backbone(clean, scale_name, return_diagnostics=return_diagnostics)
        if return_diagnostics:
            if isinstance(out, tuple):
                y, backbone_diagnostics = out
            else:
                y, backbone_diagnostics = out, {}
            diagnostics: dict[str, torch.Tensor] = dict(asd_diagnostics)
            diagnostics.update({f"asd_{key}": value for key, value in asd_diagnostics.items()})
            diagnostics.update(backbone_diagnostics)
            diagnostics.update({f"moe_{key}": value for key, value in backbone_diagnostics.items()})
            return y, diagnostics
        y = out
        return y


class SideASDFeatureMultiScalePatchTST(nn.Module):
    """Keep the raw input path and append ASD residual as an auxiliary channel."""

    def __init__(self, backbone: MultiScalePatchTST, *, init_gate: float = -4.0) -> None:
        super().__init__()
        if backbone.input_channels != 2 or backbone.target_mode != "all_channels":
            raise ValueError("SideASDFeatureMultiScalePatchTST requires a 2-channel all_channels backbone.")
        self.backbone = backbone
        self.denoiser = AdaptiveSpectralDenoising(backbone.d_model, init_gate=init_gate)

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        scale_emb = self.backbone.scale_embedding_for(scale_name, past_values.shape[0], past_values.device)
        clean, asd_diagnostics = self.denoiser(past_values, scale_emb, return_diagnostics=True)
        residual = clean - past_values
        augmented = torch.cat([past_values, residual], dim=-1)
        out = self.backbone(augmented, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}
        diagnostics: dict[str, torch.Tensor] = dict(asd_diagnostics)
        diagnostics.update({f"asd_{key}": value for key, value in asd_diagnostics.items()})
        diagnostics["side_residual_abs_mean"] = torch.mean(torch.abs(residual))
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"moe_{key}": value for key, value in backbone_diagnostics.items()})
        return y, diagnostics


class ScaleAwareSequenceAdapter(nn.Module):
    """Apply a scale-aware MoE adapter directly to a value sequence before PatchTST."""

    def __init__(
        self,
        d_model: int,
        *,
        input_channels: int = 1,
        adapter_kind: str = "lora_moe",
        n_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        bottleneck: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        init_gate: float = -4.0,
    ) -> None:
        super().__init__()
        if adapter_kind not in {"lora_moe", "mlp_moe"}:
            raise ValueError("adapter_kind must be 'lora_moe' or 'mlp_moe'.")
        self.adapter_kind = adapter_kind
        self.input_channels = int(input_channels)
        self.value_projection = nn.Linear(self.input_channels, d_model)
        self.value_output = nn.Linear(d_model, self.input_channels)
        self.gate_projection = nn.Linear(d_model, self.input_channels)
        nn.init.constant_(self.gate_projection.bias, init_gate)
        if adapter_kind == "lora_moe":
            self.adapter = ScaleAwareLoRAAdapterMoE(
                d_model,
                n_experts=n_experts,
                rank=rank,
                alpha=alpha,
                top_k=top_k,
                dropout=dropout,
            )
        else:
            self.adapter = ScaleAwareMLPAdapterMoE(
                d_model,
                n_experts=n_experts,
                bottleneck=bottleneck,
                top_k=top_k,
                dropout=dropout,
            )

    def forward(
        self,
        values: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if values.ndim != 3:
            raise ValueError(f"values must be [B,L,C], got {tuple(values.shape)}")
        batch_size, _, channels = values.shape
        if channels != self.input_channels:
            raise ValueError(f"input_channels must be {self.input_channels}, got {channels}")
        if scale_emb.shape[0] != batch_size:
            raise ValueError("scale_emb batch size must match values batch size.")

        tokens = self.value_projection(values) + scale_emb[:, None, :]
        adapted_tokens, adapter_diagnostics = self.adapter(tokens, scale_emb, return_diagnostics=True)
        token_delta = adapted_tokens - tokens
        value_delta = self.value_output(token_delta)
        gate = torch.sigmoid(self.gate_projection(scale_emb))[:, None, :]
        adapted_values = values + gate * value_delta
        if not return_diagnostics:
            return adapted_values

        diagnostics: dict[str, torch.Tensor] = {
            "pre_adapter_gate_mean": torch.mean(gate),
            "pre_adapter_mean_abs_delta": torch.mean(torch.abs(adapted_values - values)),
        }
        diagnostics.update({f"pre_adapter_{key}": value for key, value in adapter_diagnostics.items()})
        if "router_balance_loss" in adapter_diagnostics:
            diagnostics["router_balance_loss"] = adapter_diagnostics["router_balance_loss"]
        if "router_entropy" in adapter_diagnostics:
            diagnostics["router_entropy"] = adapter_diagnostics["router_entropy"]
        for key, value in adapter_diagnostics.items():
            if key.startswith("expert_prob_") or key.startswith("scale_prior_prob_"):
                diagnostics[key] = value
        return adapted_values, diagnostics


class CompositeASDAdapterExpert(nn.Module):
    """One routed expert with its own ASD module and lightweight value enhancer."""

    def __init__(
        self,
        d_model: int,
        *,
        input_channels: int = 1,
        adapter_kind: str = "lora",
        rank: int = 8,
        alpha: float = 16.0,
        bottleneck: int = 8,
        dropout: float = 0.1,
        init_gate: float = -4.0,
    ) -> None:
        super().__init__()
        if adapter_kind not in {"lora", "mlp"}:
            raise ValueError("adapter_kind must be 'lora' or 'mlp'.")
        self.adapter_kind = adapter_kind
        self.input_channels = int(input_channels)
        self.denoiser = AdaptiveSpectralDenoising(d_model, init_gate=init_gate)
        self.value_projection = nn.Linear(self.input_channels, d_model)
        self.value_output = nn.Linear(d_model, self.input_channels)
        self.gate_projection = nn.Linear(d_model, self.input_channels)
        nn.init.constant_(self.gate_projection.bias, init_gate)
        if adapter_kind == "lora":
            self.adapter = LoRAAdapterExpert(d_model, rank, alpha=alpha, dropout=dropout)
        else:
            self.adapter = MLPAdapterExpert(d_model, bottleneck, dropout=dropout)

    def forward(
        self,
        values: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if values.ndim != 3:
            raise ValueError(f"values must be [B,L,C], got {tuple(values.shape)}")
        batch_size, _, channels = values.shape
        if channels != self.input_channels:
            raise ValueError(f"input_channels must be {self.input_channels}, got {channels}")
        clean, asd_diagnostics = self.denoiser(values, scale_emb, return_diagnostics=True)
        tokens = self.value_projection(clean) + scale_emb[:, None, :]
        token_update = self.adapter(tokens)
        value_delta = self.value_output(token_update)
        gate = torch.sigmoid(self.gate_projection(scale_emb))[:, None, :]
        out = clean + gate * value_delta
        if not return_diagnostics:
            return out
        diagnostics: dict[str, torch.Tensor] = {
            "asd_gate_mean": asd_diagnostics["gate_mean"],
            "asd_tau_mean": asd_diagnostics["tau_mean"],
            "asd_mean_abs_delta": asd_diagnostics["mean_abs_delta"],
            "enhance_gate_mean": torch.mean(gate),
            "enhance_mean_abs_delta": torch.mean(torch.abs(out - clean)),
            "expert_mean_abs_delta": torch.mean(torch.abs(out - values)),
            "adapter_kind_id": torch.tensor(
                0.0 if self.adapter_kind == "lora" else 1.0,
                dtype=values.dtype,
                device=values.device,
            ),
        }
        return out, diagnostics


class CompositeASDAdapterMoE(nn.Module):
    """Route each value position to composite ASD+adapter experts before PatchTST."""

    def __init__(
        self,
        d_model: int,
        *,
        max_context_length: int,
        input_channels: int = 1,
        expert_kinds: tuple[str, ...] = ("lora", "mlp", "lora", "mlp"),
        rank: int = 8,
        alpha: float = 16.0,
        bottleneck: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        init_gate: float = -4.0,
    ) -> None:
        super().__init__()
        if max_context_length <= 0:
            raise ValueError("max_context_length must be positive.")
        if not expert_kinds:
            raise ValueError("expert_kinds must not be empty.")
        for kind in expert_kinds:
            if kind not in {"lora", "mlp"}:
                raise ValueError("expert_kinds entries must be 'lora' or 'mlp'.")
        n_experts = len(expert_kinds)
        if top_k <= 0 or top_k > n_experts:
            raise ValueError("top_k must be in [1, n_experts].")
        self.d_model = int(d_model)
        self.input_channels = int(input_channels)
        self.max_context_length = int(max_context_length)
        self.n_experts = int(n_experts)
        self.top_k = int(top_k)
        self.value_projection = nn.Linear(self.input_channels, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, self.max_context_length, d_model))
        self.router = nn.Linear(d_model * 2, self.n_experts)
        self.scale_router = nn.Linear(d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                CompositeASDAdapterExpert(
                    d_model,
                    input_channels=self.input_channels,
                    adapter_kind=kind,
                    rank=rank,
                    alpha=alpha,
                    bottleneck=bottleneck,
                    dropout=dropout,
                    init_gate=init_gate,
                )
                for kind in expert_kinds
            ]
        )
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        nn.init.zeros_(self.scale_router.weight)

    def routing_weights(self, values: torch.Tensor, scale_emb: torch.Tensor) -> torch.Tensor:
        if values.ndim != 3:
            raise ValueError(f"values must be [B,L,C], got {tuple(values.shape)}")
        batch_size, length, channels = values.shape
        if channels != self.input_channels:
            raise ValueError(f"input_channels must be {self.input_channels}, got {channels}")
        if length > self.max_context_length:
            raise ValueError(f"context length {length} exceeds max_context_length={self.max_context_length}")
        if scale_emb.shape != (batch_size, self.d_model):
            raise ValueError(f"scale_emb must be [{batch_size},{self.d_model}], got {tuple(scale_emb.shape)}")

        tokens = self.value_projection(values)
        tokens = tokens + self.position_embedding[:, :length, :] + scale_emb[:, None, :]
        scale_tokens = scale_emb[:, None, :].expand(batch_size, length, self.d_model)
        logits = self.router(torch.cat([tokens, scale_tokens], dim=-1))
        logits = logits + self.scale_router(scale_emb)[:, None, :]
        if self.top_k < self.n_experts:
            top_values, top_indices = torch.topk(logits, self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(-1, top_indices, top_values)
            return torch.softmax(masked_logits, dim=-1)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        values: torch.Tensor,
        scale_emb: torch.Tensor,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        weights = self.routing_weights(values, scale_emb)
        expert_outputs = []
        expert_diagnostics: list[dict[str, torch.Tensor]] = []
        for expert in self.experts:
            if return_diagnostics:
                out, diagnostics = expert(values, scale_emb, return_diagnostics=True)
                expert_outputs.append(out)
                expert_diagnostics.append(diagnostics)
            else:
                expert_outputs.append(expert(values, scale_emb))
        stacked = torch.stack(expert_outputs, dim=-2)
        adapted = (weights.unsqueeze(-1) * stacked).sum(dim=-2)
        if not return_diagnostics:
            return adapted

        avg_prob = weights.mean(dim=(0, 1))
        uniform = torch.full_like(avg_prob, 1.0 / self.n_experts)
        token_entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=-1).mean()
        diagnostics: dict[str, torch.Tensor] = {
            "router_entropy": token_entropy / math.log(self.n_experts),
            "router_balance_loss": F.mse_loss(avg_prob, uniform),
            "composite_mean_abs_delta": torch.mean(torch.abs(adapted - values)),
        }
        scale_prior_prob = torch.softmax(self.scale_router(scale_emb), dim=-1).mean(dim=0)
        for expert_idx in range(self.n_experts):
            diagnostics[f"expert_prob_{expert_idx}"] = avg_prob[expert_idx]
            diagnostics[f"scale_prior_prob_{expert_idx}"] = scale_prior_prob[expert_idx]
            if expert_diagnostics:
                diagnostics[f"expert_{expert_idx}_asd_gate_mean"] = expert_diagnostics[expert_idx]["asd_gate_mean"]
                diagnostics[f"expert_{expert_idx}_asd_tau_mean"] = expert_diagnostics[expert_idx]["asd_tau_mean"]
                diagnostics[f"expert_{expert_idx}_enhance_gate_mean"] = expert_diagnostics[expert_idx][
                    "enhance_gate_mean"
                ]
                diagnostics[f"expert_{expert_idx}_kind_id"] = expert_diagnostics[expert_idx]["adapter_kind_id"]
        return adapted, diagnostics


class RoutedCompositeASDAdapterPatchTST(nn.Module):
    """Use routed ASD+LoRA/MLP composite experts with a protected raw residual path."""

    def __init__(
        self,
        backbone: MultiScalePatchTST,
        *,
        init_gate: float = -4.0,
        n_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        bottleneck: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        expert_pattern: str = "one_mlp",
        final_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if backbone.input_channels != 1:
            raise ValueError("RoutedCompositeASDAdapterPatchTST expects one input channel.")
        if n_experts <= 0:
            raise ValueError("n_experts must be positive.")
        if expert_pattern == "alternating":
            expert_kinds = tuple("lora" if idx % 2 == 0 else "mlp" for idx in range(n_experts))
        elif expert_pattern == "lora_first":
            split = max(1, n_experts // 2)
            expert_kinds = tuple("lora" if idx < split else "mlp" for idx in range(n_experts))
        elif expert_pattern == "all_lora":
            expert_kinds = tuple("lora" for _ in range(n_experts))
        elif expert_pattern == "all_mlp":
            expert_kinds = tuple("mlp" for _ in range(n_experts))
        elif expert_pattern == "one_mlp":
            expert_kinds = tuple("mlp" if idx == n_experts - 1 else "lora" for idx in range(n_experts))
        elif expert_pattern == "three_mlp":
            mlp_count = min(3, n_experts)
            expert_kinds = tuple("lora" if idx < n_experts - mlp_count else "mlp" for idx in range(n_experts))
        else:
            raise ValueError(
                "expert_pattern must be one of 'alternating', 'lora_first', "
                "'all_lora', 'all_mlp', 'one_mlp', or 'three_mlp'."
            )
        self.backbone = backbone
        self.composite_moe = CompositeASDAdapterMoE(
            backbone.d_model,
            max_context_length=max(spec.context_length for spec in backbone.scale_specs.values()),
            input_channels=1,
            expert_kinds=expert_kinds,
            rank=rank,
            alpha=alpha,
            bottleneck=bottleneck,
            top_k=top_k,
            dropout=dropout,
            init_gate=init_gate,
        )
        self.final_gate_projection = nn.Linear(backbone.d_model, 1)
        nn.init.zeros_(self.final_gate_projection.weight)
        nn.init.constant_(self.final_gate_projection.bias, final_gate_init)

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.backbone.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbone.scale_specs)}")
        spec = self.backbone.scale_specs[scale_name]
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        if past_values.shape[1] != spec.context_length:
            raise ValueError(f"{scale_name}: input length must be {spec.context_length}, got {past_values.shape[1]}")

        scale_emb = self.backbone.scale_embedding_for(scale_name, past_values.shape[0], past_values.device)
        adapted, moe_diagnostics = self.composite_moe(past_values, scale_emb, return_diagnostics=True)
        final_gate = torch.sigmoid(self.final_gate_projection(scale_emb))[:, None, :]
        patch_input = past_values + final_gate * (adapted - past_values)
        out = self.backbone(patch_input, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}
        diagnostics: dict[str, torch.Tensor] = {
            "final_gate_mean": torch.mean(final_gate),
            "final_mean_abs_delta": torch.mean(torch.abs(patch_input - past_values)),
            "patch_input_abs_mean": torch.mean(torch.abs(patch_input)),
        }
        diagnostics.update(moe_diagnostics)
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"backbone_{key}": value for key, value in backbone_diagnostics.items()})
        return y, diagnostics


class LevelASDMultiScalePatchTST(nn.Module):
    """Apply scale-aware spectral denoising to log-price levels, then difference."""

    def __init__(
        self,
        backbone: MultiScalePatchTST,
        *,
        price_mode: str = "log_price",
        init_gate: float = -4.0,
    ) -> None:
        super().__init__()
        if price_mode not in {"log_price", "raw_price"}:
            raise ValueError("price_mode must be 'log_price' or 'raw_price'.")
        self.backbone = backbone
        self.denoiser = AdaptiveSpectralDenoising(backbone.d_model, init_gate=init_gate)
        self.price_mode = price_mode
        self.return_input_stats: dict[str, tuple[float, float]] = {}

    def set_return_input_stats(self, scale_name: str, *, mean: float, std: float) -> None:
        if scale_name not in self.backbone.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbone.scale_specs)}")
        self.return_input_stats[scale_name] = (float(mean), float(max(std, 1e-12)))

    def forward(
        self,
        past_levels: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.backbone.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbone.scale_specs)}")
        if past_levels.ndim != 3:
            raise ValueError(f"past_levels must be [B,L,C], got {tuple(past_levels.shape)}")
        spec = self.backbone.scale_specs[scale_name]
        expected_length = spec.context_length
        if past_levels.shape[1] != expected_length:
            raise ValueError(f"{scale_name}: level length must be {expected_length}, got {past_levels.shape[1]}")

        scale_emb = self.backbone.scale_embedding_for(scale_name, past_levels.shape[0], past_levels.device)
        clean_levels, asd_diagnostics = self.denoiser(past_levels, scale_emb, return_diagnostics=True)
        clean_returns = torch.zeros_like(clean_levels)
        if self.price_mode == "log_price":
            clean_returns[:, 1:, :] = clean_levels[:, 1:, :] - clean_levels[:, :-1, :]
        else:
            safe_levels = torch.clamp(clean_levels, min=1e-6)
            clean_returns[:, 1:, :] = torch.log(safe_levels[:, 1:, :]) - torch.log(safe_levels[:, :-1, :])
        if scale_name in self.return_input_stats:
            mean, std = self.return_input_stats[scale_name]
            clean_returns = (clean_returns - mean) / std
        out = self.backbone(clean_returns, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}
        diagnostics: dict[str, torch.Tensor] = {
            "gate_mean": asd_diagnostics["gate_mean"],
            "tau_mean": asd_diagnostics["tau_mean"],
            "mean_abs_delta": asd_diagnostics["mean_abs_delta"],
            "level_asd_gate_mean": asd_diagnostics["gate_mean"],
            "level_asd_tau_mean": asd_diagnostics["tau_mean"],
            "level_asd_mean_abs_delta": asd_diagnostics["mean_abs_delta"],
            "clean_return_abs_mean": torch.mean(torch.abs(clean_returns)),
        }
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"moe_{key}": value for key, value in backbone_diagnostics.items()})
        return y, diagnostics


class PreprocessedASDAdapterPatchTST(nn.Module):
    """Run ASD plus a sequence MoE adapter before feeding returns into PatchTST."""

    def __init__(
        self,
        backbone: MultiScalePatchTST,
        *,
        input_mode: str = "return",
        price_mode: str = "log_price",
        adapter_kind: str = "lora_moe",
        init_gate: float = -4.0,
        n_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        bottleneck: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        residual_to_raw: bool = False,
        final_gate_init: float = -2.0,
    ) -> None:
        super().__init__()
        if input_mode not in {"return", "level"}:
            raise ValueError("input_mode must be 'return' or 'level'.")
        if price_mode not in {"log_price", "raw_price"}:
            raise ValueError("price_mode must be 'log_price' or 'raw_price'.")
        if backbone.input_channels != 1:
            raise ValueError("PreprocessedASDAdapterPatchTST currently expects one input channel.")
        self.backbone = backbone
        self.input_mode = input_mode
        self.price_mode = price_mode
        self.residual_to_raw = bool(residual_to_raw)
        self.denoiser = AdaptiveSpectralDenoising(backbone.d_model, init_gate=init_gate)
        self.pre_adapter = ScaleAwareSequenceAdapter(
            backbone.d_model,
            input_channels=1,
            adapter_kind=adapter_kind,
            n_experts=n_experts,
            rank=rank,
            alpha=alpha,
            bottleneck=bottleneck,
            top_k=top_k,
            dropout=dropout,
            init_gate=init_gate,
        )
        self.final_gate_projection = nn.Linear(backbone.d_model, 1)
        nn.init.zeros_(self.final_gate_projection.weight)
        nn.init.constant_(self.final_gate_projection.bias, final_gate_init)
        self.return_input_stats: dict[str, tuple[float, float]] = {}

    def set_return_input_stats(self, scale_name: str, *, mean: float, std: float) -> None:
        if scale_name not in self.backbone.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbone.scale_specs)}")
        self.return_input_stats[scale_name] = (float(mean), float(max(std, 1e-12)))

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.backbone.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbone.scale_specs)}")
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        spec = self.backbone.scale_specs[scale_name]
        if past_values.shape[1] != spec.context_length:
            raise ValueError(f"{scale_name}: input length must be {spec.context_length}, got {past_values.shape[1]}")

        scale_emb = self.backbone.scale_embedding_for(scale_name, past_values.shape[0], past_values.device)
        clean, asd_diagnostics = self.denoiser(past_values, scale_emb, return_diagnostics=True)
        adapted, adapter_diagnostics = self.pre_adapter(clean, scale_emb, return_diagnostics=True)
        final_gate = torch.sigmoid(self.final_gate_projection(scale_emb))[:, None, :]
        if self.residual_to_raw:
            model_input = past_values + final_gate * (adapted - past_values)
        else:
            model_input = adapted

        if self.input_mode == "level":
            patch_input = torch.zeros_like(model_input)
            if self.price_mode == "log_price":
                patch_input[:, 1:, :] = model_input[:, 1:, :] - model_input[:, :-1, :]
            else:
                safe_levels = torch.clamp(model_input, min=1e-6)
                patch_input[:, 1:, :] = torch.log(safe_levels[:, 1:, :]) - torch.log(safe_levels[:, :-1, :])
            if scale_name in self.return_input_stats:
                mean, std = self.return_input_stats[scale_name]
                patch_input = (patch_input - mean) / std
        else:
            patch_input = model_input

        out = self.backbone(patch_input, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}
        diagnostics: dict[str, torch.Tensor] = {
            "gate_mean": asd_diagnostics["gate_mean"],
            "tau_mean": asd_diagnostics["tau_mean"],
            "mean_abs_delta": asd_diagnostics["mean_abs_delta"],
            "asd_gate_mean": asd_diagnostics["gate_mean"],
            "asd_tau_mean": asd_diagnostics["tau_mean"],
            "asd_mean_abs_delta": asd_diagnostics["mean_abs_delta"],
            "patch_input_abs_mean": torch.mean(torch.abs(patch_input)),
            "final_gate_mean": torch.mean(final_gate),
            "final_mean_abs_delta": torch.mean(torch.abs(model_input - past_values)),
        }
        diagnostics.update(adapter_diagnostics)
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"backbone_{key}": value for key, value in backbone_diagnostics.items()})
        return y, diagnostics


class ScaleSpecificGatedPreprocessedASDAdapterPatchTST(nn.Module):
    """Use scale-specific pre-PatchTST adapters with a protected raw residual path."""

    def __init__(
        self,
        backbone: MultiScalePatchTST,
        *,
        init_gate: float = -4.0,
        n_experts: int = 4,
        rank: int = 8,
        alpha: float = 16.0,
        bottleneck: int = 8,
        top_k: int = 2,
        dropout: float = 0.1,
        scale_adapter_kind: dict[str, str] | None = None,
        scale_gate_init: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        if backbone.input_channels != 1:
            raise ValueError("ScaleSpecificGatedPreprocessedASDAdapterPatchTST expects one input channel.")
        self.backbone = backbone
        self.denoiser = AdaptiveSpectralDenoising(backbone.d_model, init_gate=init_gate)
        self.adapters = nn.ModuleDict(
            {
                "lora_moe": ScaleAwareSequenceAdapter(
                    backbone.d_model,
                    input_channels=1,
                    adapter_kind="lora_moe",
                    n_experts=n_experts,
                    rank=rank,
                    alpha=alpha,
                    bottleneck=bottleneck,
                    top_k=top_k,
                    dropout=dropout,
                    init_gate=init_gate,
                ),
                "mlp_moe": ScaleAwareSequenceAdapter(
                    backbone.d_model,
                    input_channels=1,
                    adapter_kind="mlp_moe",
                    n_experts=n_experts,
                    rank=rank,
                    alpha=alpha,
                    bottleneck=bottleneck,
                    top_k=top_k,
                    dropout=dropout,
                    init_gate=init_gate,
                ),
            }
        )
        self.scale_adapter_kind = {
            name: "lora_moe" for name in self.backbone.scale_specs
        }
        self.scale_adapter_kind.update(scale_adapter_kind or {})
        for scale_name, adapter_kind in self.scale_adapter_kind.items():
            if adapter_kind not in self.adapters:
                raise ValueError(f"{scale_name}: unknown adapter kind {adapter_kind!r}")
        default_gate_init = {"second": -6.0, "minute": -2.5, "hour": -1.5}
        default_gate_init.update(scale_gate_init or {})
        self.final_gate_projection = nn.Linear(backbone.d_model, 1)
        nn.init.zeros_(self.final_gate_projection.weight)
        nn.init.zeros_(self.final_gate_projection.bias)
        self.scale_gate_bias = nn.ParameterDict(
            {
                name: nn.Parameter(torch.tensor([float(default_gate_init.get(name, -3.0))]))
                for name in self.backbone.scale_specs
            }
        )

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.backbone.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbone.scale_specs)}")
        scale_emb = self.backbone.scale_embedding_for(scale_name, past_values.shape[0], past_values.device)
        clean, asd_diagnostics = self.denoiser(past_values, scale_emb, return_diagnostics=True)
        adapter_kind = self.scale_adapter_kind[scale_name]
        adapted, adapter_diagnostics = self.adapters[adapter_kind](clean, scale_emb, return_diagnostics=True)
        gate_logits = self.final_gate_projection(scale_emb) + self.scale_gate_bias[scale_name]
        final_gate = torch.sigmoid(gate_logits)[:, None, :]
        patch_input = past_values + final_gate * (adapted - past_values)
        out = self.backbone(patch_input, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}
        diagnostics: dict[str, torch.Tensor] = {
            "gate_mean": asd_diagnostics["gate_mean"],
            "tau_mean": asd_diagnostics["tau_mean"],
            "mean_abs_delta": asd_diagnostics["mean_abs_delta"],
            "asd_gate_mean": asd_diagnostics["gate_mean"],
            "asd_tau_mean": asd_diagnostics["tau_mean"],
            "asd_mean_abs_delta": asd_diagnostics["mean_abs_delta"],
            "final_gate_mean": torch.mean(final_gate),
            "final_mean_abs_delta": torch.mean(torch.abs(patch_input - past_values)),
            "patch_input_abs_mean": torch.mean(torch.abs(patch_input)),
            "adapter_kind_id": torch.tensor(
                0.0 if adapter_kind == "lora_moe" else 1.0,
                dtype=past_values.dtype,
                device=past_values.device,
            ),
        }
        diagnostics.update(adapter_diagnostics)
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"backbone_{key}": value for key, value in backbone_diagnostics.items()})
        return y, diagnostics


class PaddedFrequencyMoEASDPatchTST(nn.Module):
    """Pad variable-scale windows, route them to scale-specific ASD experts, then use PatchTST."""

    def __init__(
        self,
        backbone: MultiScalePatchTST,
        *,
        init_gate: float = -4.0,
        top_k: int = 2,
        router_prior_strength: float = 1.5,
    ) -> None:
        super().__init__()
        if backbone.input_channels != 1:
            raise ValueError("PaddedFrequencyMoEASDPatchTST currently expects one input channel.")
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        self.backbone = backbone
        self.max_context_length = max(spec.context_length for spec in backbone.scale_specs.values())
        self.scale_names = list(backbone.scale_specs)
        self.frequency_embedding = LogScaleEmbedding(backbone.d_model)
        self.denoisers = nn.ModuleDict(
            {
                name: AdaptiveSpectralDenoising(backbone.d_model, init_gate=init_gate)
                for name in self.scale_names
            }
        )
        self.router = nn.Sequential(
            nn.Linear(backbone.d_model + 4, backbone.d_model),
            nn.GELU(),
            nn.Linear(backbone.d_model, len(self.scale_names)),
        )
        self.top_k = min(int(top_k), len(self.scale_names))
        self.router_prior_strength = float(router_prior_strength)

    def _pad_left(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, length, channels = values.shape
        if length > self.max_context_length:
            raise ValueError(
                f"input length {length} exceeds max_context_length={self.max_context_length}."
            )
        pad_length = self.max_context_length - length
        if pad_length:
            pad = values.new_zeros(batch_size, pad_length, channels)
            padded = torch.cat([pad, values], dim=1)
            mask = torch.cat(
                [
                    values.new_zeros(batch_size, pad_length, channels),
                    values.new_ones(batch_size, length, channels),
                ],
                dim=1,
            )
        else:
            padded = values
            mask = values.new_ones(batch_size, length, channels)
        return padded, mask

    def _router_weights(
        self,
        padded: torch.Tensor,
        mask: torch.Tensor,
        scale_name: str,
        freq_emb: torch.Tensor,
    ) -> torch.Tensor:
        valid_count = mask.sum(dim=(1, 2)).clamp_min(1.0)
        mean = (padded * mask).sum(dim=(1, 2), keepdim=False) / valid_count
        centered = (padded - mean[:, None, None]) * mask
        std = torch.sqrt((centered.pow(2).sum(dim=(1, 2)) / valid_count).clamp_min(1e-8))
        abs_mean = (padded.abs() * mask).sum(dim=(1, 2)) / valid_count
        length_ratio = torch.full_like(mean, padded.shape[1] / float(self.max_context_length))
        router_features = torch.cat(
            [freq_emb, mean[:, None], std[:, None], abs_mean[:, None], length_ratio[:, None]],
            dim=-1,
        )
        logits = self.router(router_features)
        scale_idx = self.scale_names.index(scale_name)
        prior = torch.zeros_like(logits)
        prior[:, scale_idx] = self.router_prior_strength
        logits = logits + prior
        if self.top_k < len(self.scale_names):
            top_val, top_idx = torch.topk(logits, self.top_k, dim=-1)
            masked_logits = torch.full_like(logits, float("-inf"))
            masked_logits.scatter_(-1, top_idx, top_val)
            return torch.softmax(masked_logits, dim=-1)
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        past_values: torch.Tensor,
        scale_name: str,
        *,
        return_diagnostics: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if scale_name not in self.backbone.scale_specs:
            raise ValueError(f"unknown scale {scale_name!r}; expected {sorted(self.backbone.scale_specs)}")
        if past_values.ndim != 3:
            raise ValueError(f"past_values must be [B,L,C], got {tuple(past_values.shape)}")
        spec = self.backbone.scale_specs[scale_name]
        if past_values.shape[1] != spec.context_length:
            raise ValueError(f"{scale_name}: input length must be {spec.context_length}, got {past_values.shape[1]}")

        padded, mask = self._pad_left(past_values)
        batch_size = past_values.shape[0]
        device = past_values.device
        scale_spec = self.backbone.scale_specs[scale_name]
        delta = torch.full((batch_size,), float(scale_spec.delta_seconds), device=device)
        scale_id = torch.full((batch_size,), int(scale_spec.scale_id), device=device, dtype=torch.long)
        freq_emb = self.frequency_embedding(delta, scale_id)
        weights = self._router_weights(padded, mask, scale_name, freq_emb)

        expert_outputs: list[torch.Tensor] = []
        expert_diagnostics: list[dict[str, torch.Tensor]] = []
        for expert_name in self.scale_names:
            denoised, diagnostics = self.denoisers[expert_name](
                padded * mask,
                freq_emb,
                return_diagnostics=True,
            )
            expert_outputs.append(denoised)
            expert_diagnostics.append(diagnostics)
        stacked = torch.stack(expert_outputs, dim=-2)
        mixed = (weights[:, None, None, :, None] * stacked).sum(dim=-2)
        mixed = mixed * mask + padded * (1.0 - mask)
        clean = mixed[:, -spec.context_length :, :]

        out = self.backbone(clean, scale_name, return_diagnostics=return_diagnostics)
        if not return_diagnostics:
            return out
        if isinstance(out, tuple):
            y, backbone_diagnostics = out
        else:
            y, backbone_diagnostics = out, {}
        avg_prob = weights.mean(dim=0)
        entropy = -(weights * torch.log(weights.clamp_min(1e-8))).sum(dim=-1).mean()
        uniform = torch.full_like(avg_prob, 1.0 / avg_prob.numel())
        diagnostics: dict[str, torch.Tensor] = {
            "router_entropy": entropy,
            "router_balance_loss": F.mse_loss(avg_prob, uniform),
            "padding_fraction": torch.tensor(
                1.0 - (float(spec.context_length) / float(self.max_context_length)),
                dtype=past_values.dtype,
                device=device,
            ),
            "freq_emb_norm": torch.mean(torch.norm(freq_emb, dim=-1)),
            "mean_abs_delta": torch.mean(torch.abs(clean - past_values)),
        }
        for idx, expert_name in enumerate(self.scale_names):
            diagnostics[f"expert_prob_{idx}"] = avg_prob[idx]
            diagnostics[f"{expert_name}_asd_gate_mean"] = expert_diagnostics[idx]["gate_mean"]
            diagnostics[f"{expert_name}_asd_tau_mean"] = expert_diagnostics[idx]["tau_mean"]
        diagnostics.update(backbone_diagnostics)
        diagnostics.update({f"backbone_{key}": value for key, value in backbone_diagnostics.items()})
        return y, diagnostics


def build_multiscale_patchtst(
    scale_specs: dict[str, ScaleSpec] | None = None,
    *,
    input_channels: int = 1,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    d_ff: int = 128,
    dropout: float = 0.1,
    encoder_spectral_mode: str = "none",
    encoder_spectral_init_gate: float = -4.0,
    lora_moe_mode: str = "none",
    lora_moe_rank: int = 4,
    lora_moe_alpha: float = 16.0,
    lora_moe_n_experts: int = 4,
    lora_moe_top_k: int = 2,
    lora_moe_dropout: float = 0.1,
    target_mode: str = "per_channel",
    head_type: str = "linear",
    head_hidden_dim: int = 128,
) -> MultiScalePatchTST:
    return MultiScalePatchTST(
        default_scale_specs() if scale_specs is None else scale_specs,
        input_channels=input_channels,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        encoder_spectral_mode=encoder_spectral_mode,
        encoder_spectral_init_gate=encoder_spectral_init_gate,
        lora_moe_mode=lora_moe_mode,
        lora_moe_rank=lora_moe_rank,
        lora_moe_alpha=lora_moe_alpha,
        lora_moe_n_experts=lora_moe_n_experts,
        lora_moe_top_k=lora_moe_top_k,
        lora_moe_dropout=lora_moe_dropout,
        target_mode=target_mode,
        head_type=head_type,
        head_hidden_dim=head_hidden_dim,
    )
