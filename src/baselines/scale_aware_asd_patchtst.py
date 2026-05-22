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
    ) -> None:
        super().__init__()
        if encoder_spectral_mode not in {"none", "last1"}:
            raise ValueError("encoder_spectral_mode must be 'none' or 'last1'.")
        if lora_moe_mode not in {"none", "last1", "lora_only", "mlp_moe"}:
            raise ValueError("lora_moe_mode must be 'none', 'last1', 'lora_only', or 'mlp_moe'.")
        if target_mode not in {"per_channel", "all_channels"}:
            raise ValueError("target_mode must be 'per_channel' or 'all_channels'.")
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
            if encoder_spectral_mode == "last1"
            else None
        )
        self.lora_moe = None
        if lora_moe_mode == "last1":
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
                name: nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Dropout(dropout),
                    nn.Linear(d_model * spec.patch_count * head_input_multiplier, spec.prediction_length),
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
        for layer in self.layers:
            x = layer(x)
        diagnostics: dict[str, torch.Tensor] = {}
        if self.encoder_spectral is not None:
            x, diagnostics = self.encoder_spectral(x, scale_emb, return_diagnostics=True)
        if self.lora_moe is not None:
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
    )
