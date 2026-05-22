from __future__ import annotations

import torch

from src.baselines.patchtst_lora import (
    LoRAConfig,
    PatchTSTForecastConfig,
    PatchTSTLoRA,
    count_parameters,
)


def test_patchtst_lora_forward_shape() -> None:
    config = PatchTSTForecastConfig(
        context_length=16,
        patch_length=4,
        patch_stride=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        lora=LoRAConfig(rank=2, alpha=4.0, enabled=True),
    )
    model = PatchTSTLoRA(config)
    x = torch.randn(5, 16, 1)

    y = model(x)

    assert y.shape == (5, 1, 1)


def test_freeze_base_for_lora_keeps_adapter_and_head_trainable() -> None:
    config = PatchTSTForecastConfig(
        context_length=16,
        patch_length=4,
        patch_stride=4,
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        lora=LoRAConfig(rank=2, alpha=4.0, enabled=True),
    )
    model = PatchTSTLoRA(config)

    model.freeze_base_for_lora(train_head=True)
    params = count_parameters(model)

    assert params["lora"] > 0
    assert params["trainable"] > params["lora"]
    assert params["trainable"] < params["total"]
    assert all(
        param.requires_grad
        for name, param in model.named_parameters()
        if "lora_" in name
    )
