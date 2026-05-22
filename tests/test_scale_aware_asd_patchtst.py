from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import torch

from scripts.evaluate_scale_aware_asd_patchtst import (
    PATCH_PRESETS,
    SCALE_ORDER,
    TARGETED_ASD_LORA_MOE_INIT_GATE,
    TARGETED_ASD_LORA_MOE_PATCH_PRESET,
    TARGETED_ASD_LORA_MOE_RANK,
    TARGETED_ASD_LORA_MOE_REGIME,
    apply_training_regime,
    evaluate_asd_lora_moe_final_decision,
    load_raw_backbone_checkpoint,
    make_scale_specs,
    targeted_asd_lora_moe_robustness_configs,
)
from src.baselines.scale_aware_asd_patchtst import (
    AdaptiveSpectralEncoderBlock,
    DEFAULT_SCALE_SPECS,
    AdaptiveSpectralDenoising,
    RawMultiScalePatchTST,
    ScaleAwareMLPAdapterMoE,
    ScaleAwareLoRAAdapterMoE,
    ScaleAwareASDMultiScalePatchTST,
    SharedLoRAAdapter,
    StaticASDMultiScalePatchTST,
    TSLANetMultiScaleForecaster,
    build_multiscale_patchtst,
)


def test_default_scale_specs_are_intraday_only() -> None:
    assert tuple(DEFAULT_SCALE_SPECS) == ("second", "minute", "hour")
    assert "day" not in DEFAULT_SCALE_SPECS
    assert [DEFAULT_SCALE_SPECS[name].scale_id for name in DEFAULT_SCALE_SPECS] == [0, 1, 2]


def test_runner_fincast_adapted_patch_preset_is_scale_specific() -> None:
    args = SimpleNamespace(patch_preset="fincast_adapted", scales=list(SCALE_ORDER))
    for scale in SCALE_ORDER:
        setattr(args, f"{scale}_context_length", None)
        setattr(args, f"{scale}_patch_length", None)
        setattr(args, f"{scale}_patch_stride", None)

    specs = make_scale_specs(args)

    assert specs["second"].patch_length == 32
    assert specs["minute"].patch_length == 4
    assert specs["hour"].patch_length == 8


def test_ablation_patch_presets_are_intraday_and_legal() -> None:
    assert set(PATCH_PRESETS) >= {"compact", "fincast_adapted", "short_second", "long_context"}
    for preset in PATCH_PRESETS:
        args = SimpleNamespace(patch_preset=preset, scales=list(SCALE_ORDER))
        for scale in SCALE_ORDER:
            setattr(args, f"{scale}_context_length", None)
            setattr(args, f"{scale}_patch_length", None)
            setattr(args, f"{scale}_patch_stride", None)
        specs = make_scale_specs(args)
        assert tuple(specs) == SCALE_ORDER
        assert all(spec.patch_length <= spec.context_length for spec in specs.values())
        assert all(spec.patch_stride > 0 for spec in specs.values())


def test_targeted_asd_lora_moe_robustness_config_is_fixed_intraday() -> None:
    assert TARGETED_ASD_LORA_MOE_PATCH_PRESET == "short_second"
    assert tuple(PATCH_PRESETS[TARGETED_ASD_LORA_MOE_PATCH_PRESET]) == SCALE_ORDER
    configs = targeted_asd_lora_moe_robustness_configs()

    assert [config["training_regime"] for config in configs] == [
        "raw_joint",
        "asd_frozen_encoder_train_head",
        "lora_moe_frozen_base_train_moe_head",
        TARGETED_ASD_LORA_MOE_REGIME,
    ]
    combined = configs[-1]
    assert combined["adapter_rank"] == TARGETED_ASD_LORA_MOE_RANK == 8
    assert combined["init_gate"] == TARGETED_ASD_LORA_MOE_INIT_GATE == -4.0


def test_asd_lora_moe_final_decision_gate_uses_robustness_mean() -> None:
    rows = []
    for scale in SCALE_ORDER:
        for model in [
            "raw_joint",
            "asd_frozen_encoder_train_head",
            "lora_moe_frozen_base_train_moe_head",
            TARGETED_ASD_LORA_MOE_REGIME,
        ]:
            rows.append(
                {
                    "patch_preset": TARGETED_ASD_LORA_MOE_PATCH_PRESET,
                    "model": model,
                    "init_gate": TARGETED_ASD_LORA_MOE_INIT_GATE if model == TARGETED_ASD_LORA_MOE_REGIME else None,
                    "adapter_rank": TARGETED_ASD_LORA_MOE_RANK if "lora_moe" in model else None,
                    "scale": scale,
                    "mse_mean": 1.0,
                    "mse_std": 0.01,
                    "nmse_mean": 1.0,
                    "nmse_std": 0.01,
                    "mae_mean": 1.0,
                    "mae_std": 0.01,
                    "direction_accuracy_nonzero_mean": 0.50,
                    "direction_accuracy_nonzero_std": 0.01,
                    "corr_mean": 0.10,
                    "corr_std": 0.01,
                }
            )
    frame = pd.DataFrame(rows)
    combined_mask = frame["model"] == TARGETED_ASD_LORA_MOE_REGIME
    frame.loc[combined_mask & (frame["scale"] == "second"), "mse_mean"] = 1.005
    frame.loc[combined_mask & (frame["scale"] == "second"), "direction_accuracy_nonzero_mean"] = 0.495
    frame.loc[combined_mask & (frame["scale"] == "minute"), "mse_mean"] = 1.01
    frame.loc[combined_mask & (frame["scale"] == "minute"), "direction_accuracy_nonzero_mean"] = 0.505
    frame.loc[combined_mask & (frame["scale"] == "hour"), "nmse_mean"] = 0.99

    decision = evaluate_asd_lora_moe_final_decision(frame)

    assert decision["quality_pass"] is True
    assert set(decision["scale_table"]["scale"]) == set(SCALE_ORDER)


def test_multiscale_patchtst_forwards_each_scale() -> None:
    backbone = build_multiscale_patchtst(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
    )
    model = RawMultiScalePatchTST(backbone)

    for scale, spec in DEFAULT_SCALE_SPECS.items():
        x = torch.randn(3, spec.context_length, 1)
        y = model(x, scale)
        assert y.shape == (3, 1, 1)


def test_static_and_scale_aware_asd_diagnostics_are_finite() -> None:
    static_model = StaticASDMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0),
        keep_ratio=0.5,
        blend_init=0.15,
    )
    scale_aware_model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0),
        init_gate=-3.0,
    )
    spec = DEFAULT_SCALE_SPECS["second"]
    x = torch.randn(2, spec.context_length, 1)

    for model in [static_model, scale_aware_model]:
        y, diagnostics = model(x, "second", return_diagnostics=True)
        assert y.shape == (2, 1, 1)
        assert diagnostics
        assert all(torch.isfinite(value) for value in diagnostics.values())


def test_scale_aware_asd_starts_near_identity() -> None:
    denoiser = AdaptiveSpectralDenoising(d_model=8, init_gate=-3.0)
    x = torch.randn(4, 16, 1)
    scale_emb = torch.randn(4, 8)

    clean, diagnostics = denoiser(x, scale_emb, return_diagnostics=True)

    assert clean.shape == x.shape
    assert torch.isfinite(clean).all()
    assert 0.0 < float(diagnostics["gate_mean"].detach()) < 0.1


def test_adaptive_spectral_encoder_block_starts_near_identity() -> None:
    block = AdaptiveSpectralEncoderBlock(d_model=8, max_patch_count=16, init_gate=-4.0)
    h = torch.randn(4, 9, 8)
    scale_emb = torch.randn(4, 8)

    clean, diagnostics = block(h, scale_emb, return_diagnostics=True)

    assert clean.shape == h.shape
    assert torch.isfinite(clean).all()
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert float(diagnostics["mean_abs_delta"].detach()) < 1e-6


def test_lora_moe_adapter_starts_near_identity() -> None:
    module = ScaleAwareLoRAAdapterMoE(
        d_model=8,
        n_experts=4,
        rank=4,
        alpha=16.0,
        top_k=2,
        dropout=0.0,
    )
    h = torch.randn(4, 9, 8)
    scale_emb = torch.randn(4, 8)

    out, diagnostics = module(h, scale_emb, return_diagnostics=True)

    assert out.shape == h.shape
    assert torch.isfinite(out).all()
    assert all(torch.isfinite(value) for value in diagnostics.values())
    assert float(diagnostics["mean_abs_delta"].detach()) < 1e-6
    assert "router_entropy" in diagnostics
    assert "router_balance_loss" in diagnostics
    assert "scale_prior_prob_0" in diagnostics


def test_lora_only_and_mlp_moe_adapters_start_near_identity() -> None:
    h = torch.randn(4, 9, 8)
    scale_emb = torch.randn(4, 8)
    lora = SharedLoRAAdapter(d_model=8, rank=4, alpha=16.0, dropout=0.0)
    mlp_moe = ScaleAwareMLPAdapterMoE(
        d_model=8,
        n_experts=4,
        bottleneck=4,
        top_k=2,
        dropout=0.0,
    )

    for module in [lora, mlp_moe]:
        out, diagnostics = module(h, scale_emb, return_diagnostics=True)
        assert out.shape == h.shape
        assert torch.isfinite(out).all()
        assert all(torch.isfinite(value) for value in diagnostics.values())
        assert float(diagnostics["mean_abs_delta"].detach()) < 1e-6

    _, moe_diagnostics = mlp_moe(h, scale_emb, return_diagnostics=True)
    assert "router_entropy" in moe_diagnostics
    assert "router_balance_loss" in moe_diagnostics


def test_multiscale_patchtst_with_encoder_asb_forwards_each_scale() -> None:
    backbone = build_multiscale_patchtst(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        encoder_spectral_mode="last1",
        encoder_spectral_init_gate=-4.0,
    )
    model = RawMultiScalePatchTST(backbone)

    for scale, spec in DEFAULT_SCALE_SPECS.items():
        x = torch.randn(3, spec.context_length, 1)
        y, diagnostics = model(x, scale, return_diagnostics=True)
        assert y.shape == (3, 1, 1)
        assert "local_mask_mean" in diagnostics
        assert all(torch.isfinite(value) for value in diagnostics.values())


def test_tslanet_multiscale_baseline_forwards_each_scale() -> None:
    model = TSLANetMultiScaleForecaster(
        DEFAULT_SCALE_SPECS,
        d_model=16,
        n_layers=1,
        dropout=0.0,
        spectral_init_gate=-4.0,
    )

    for scale, spec in DEFAULT_SCALE_SPECS.items():
        x = torch.randn(3, spec.context_length, 1)
        y, diagnostics = model(x, scale, return_diagnostics=True)
        assert y.shape == (3, 1, 1)
        assert "tslanet_gate_mean" in diagnostics
        assert all(torch.isfinite(value) for value in diagnostics.values())


def test_multiscale_patchtst_with_lora_moe_forwards_each_scale() -> None:
    backbone = build_multiscale_patchtst(
        d_model=16,
        n_heads=4,
        n_layers=1,
        d_ff=32,
        dropout=0.0,
        lora_moe_mode="last1",
        lora_moe_rank=4,
        lora_moe_alpha=16.0,
        lora_moe_n_experts=4,
        lora_moe_top_k=2,
        lora_moe_dropout=0.0,
    )
    model = RawMultiScalePatchTST(backbone)

    for scale, spec in DEFAULT_SCALE_SPECS.items():
        x = torch.randn(3, spec.context_length, 1)
        y, diagnostics = model(x, scale, return_diagnostics=True)
        assert y.shape == (3, 1, 1)
        assert "router_entropy" in diagnostics
        assert all(torch.isfinite(value) for value in diagnostics.values())


def test_multiscale_patchtst_with_lora_only_and_mlp_moe_forward_each_scale() -> None:
    for mode, expected_key in [("lora_only", "adapter_rank"), ("mlp_moe", "router_entropy")]:
        backbone = build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode=mode,
            lora_moe_rank=4,
            lora_moe_alpha=16.0,
            lora_moe_n_experts=4,
            lora_moe_top_k=2,
            lora_moe_dropout=0.0,
        )
        model = RawMultiScalePatchTST(backbone)

        for scale, spec in DEFAULT_SCALE_SPECS.items():
            x = torch.randn(3, spec.context_length, 1)
            y, diagnostics = model(x, scale, return_diagnostics=True)
            assert y.shape == (3, 1, 1)
            assert expected_key in diagnostics
            assert all(torch.isfinite(value) for value in diagnostics.values())


def test_combined_asd_lora_moe_forwards_each_scale() -> None:
    model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="last1",
            lora_moe_rank=4,
            lora_moe_alpha=16.0,
            lora_moe_n_experts=4,
            lora_moe_top_k=2,
            lora_moe_dropout=0.0,
        ),
        init_gate=-4.0,
    )

    for scale, spec in DEFAULT_SCALE_SPECS.items():
        x = torch.randn(3, spec.context_length, 1)
        y, diagnostics = model(x, scale, return_diagnostics=True)
        assert y.shape == (3, 1, 1)
        assert "asd_gate_mean" in diagnostics
        assert "router_balance_loss" in diagnostics
        assert "moe_router_entropy" in diagnostics
        assert all(torch.isfinite(value) for value in diagnostics.values())


def test_freeze_regimes_train_expected_parameters() -> None:
    model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0),
        init_gate=-3.0,
    )
    info = apply_training_regime(model, "asd_only_frozen_backbone")
    assert info["trainable_parameters"] > 0
    assert all(not parameter.requires_grad for parameter in model.backbone.parameters())
    assert all(parameter.requires_grad for parameter in model.denoiser.parameters())

    model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0),
        init_gate=-3.0,
    )
    apply_training_regime(model, "asd_frozen_encoder_train_head")
    assert all(parameter.requires_grad for parameter in model.denoiser.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.backbone.named_parameters()
        if not name.startswith("heads.")
    )

    model = RawMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0)
    )
    apply_training_regime(model, "raw_frozen_base_train_head")
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.heads.")
    )

    model = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            encoder_spectral_mode="last1",
        )
    )
    apply_training_regime(model, "asb_encoder_frozen_base_train_asb_only")
    assert all(parameter.requires_grad for parameter in model.backbone.encoder_spectral.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.encoder_spectral.")
    )

    model = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            encoder_spectral_mode="last1",
        )
    )
    apply_training_regime(model, "asb_encoder_frozen_base_train_asb_head")
    assert all(parameter.requires_grad for parameter in model.backbone.encoder_spectral.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.encoder_spectral.") and not name.startswith("backbone.heads.")
    )

    model = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="last1",
            lora_moe_rank=4,
        )
    )
    apply_training_regime(model, "lora_moe_frozen_base_train_moe_only")
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.lora_moe.")
    )

    model = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="last1",
            lora_moe_rank=4,
        )
    )
    apply_training_regime(model, "lora_moe_frozen_base_train_moe_head")
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.lora_moe.") and not name.startswith("backbone.heads.")
    )

    model = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="lora_only",
            lora_moe_rank=4,
        )
    )
    apply_training_regime(model, "lora_only_frozen_base_train_adapter_head")
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.lora_moe.") and not name.startswith("backbone.heads.")
    )

    model = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="mlp_moe",
            lora_moe_rank=4,
        )
    )
    apply_training_regime(model, "mlp_moe_frozen_base_train_moe_only")
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("backbone.lora_moe.")
    )

    model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="last1",
            lora_moe_rank=4,
        ),
        init_gate=-4.0,
    )
    apply_training_regime(model, "asd_lora_moe_frozen_base_train_adapters_only")
    assert all(parameter.requires_grad for parameter in model.denoiser.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("denoiser.") and not name.startswith("backbone.lora_moe.")
    )

    model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="last1",
            lora_moe_rank=4,
        ),
        init_gate=-4.0,
    )
    apply_training_regime(model, "asd_lora_moe_frozen_base_train_adapters_head")
    assert all(parameter.requires_grad for parameter in model.denoiser.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("denoiser.")
        and not name.startswith("backbone.lora_moe.")
        and not name.startswith("backbone.heads.")
    )

    model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="lora_only",
            lora_moe_rank=4,
        ),
        init_gate=-4.0,
    )
    apply_training_regime(model, "asd_lora_only_frozen_base_train_adapter_head")
    assert all(parameter.requires_grad for parameter in model.denoiser.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("denoiser.")
        and not name.startswith("backbone.lora_moe.")
        and not name.startswith("backbone.heads.")
    )

    model = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="mlp_moe",
            lora_moe_rank=4,
        ),
        init_gate=-4.0,
    )
    apply_training_regime(model, "asd_mlp_moe_frozen_base_train_moe_head")
    assert all(parameter.requires_grad for parameter in model.denoiser.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.lora_moe.parameters())
    assert all(parameter.requires_grad for parameter in model.backbone.heads.parameters())


def test_raw_checkpoint_loads_into_asd_backbone(tmp_path) -> None:
    raw = RawMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0)
    )
    asd = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0),
        init_gate=-3.0,
    )
    with torch.no_grad():
        raw.backbone.position_embedding.fill_(0.123)
    checkpoint_path = tmp_path / "raw.pt"
    torch.save({"model": raw.state_dict()}, checkpoint_path)

    info = load_raw_backbone_checkpoint(asd, checkpoint_path)

    assert info["loaded_backbone_tensors"] > 0
    assert torch.allclose(asd.backbone.position_embedding, raw.backbone.position_embedding)


def test_raw_checkpoint_loads_into_asb_backbone(tmp_path) -> None:
    raw = RawMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0)
    )
    asb = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            encoder_spectral_mode="last1",
        )
    )
    with torch.no_grad():
        raw.backbone.position_embedding.fill_(0.321)
    checkpoint_path = tmp_path / "raw.pt"
    torch.save({"model": raw.state_dict()}, checkpoint_path)

    info = load_raw_backbone_checkpoint(asb, checkpoint_path)

    assert info["loaded_backbone_tensors"] > 0
    assert torch.allclose(asb.backbone.position_embedding, raw.backbone.position_embedding)


def test_raw_checkpoint_loads_into_lora_moe_backbone(tmp_path) -> None:
    raw = RawMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0)
    )
    lora_moe = RawMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="last1",
            lora_moe_rank=4,
        )
    )
    with torch.no_grad():
        raw.backbone.position_embedding.fill_(0.456)
    checkpoint_path = tmp_path / "raw.pt"
    torch.save({"model": raw.state_dict()}, checkpoint_path)

    info = load_raw_backbone_checkpoint(lora_moe, checkpoint_path)

    assert info["loaded_backbone_tensors"] > 0
    assert torch.allclose(lora_moe.backbone.position_embedding, raw.backbone.position_embedding)


def test_raw_checkpoint_loads_into_combined_asd_lora_moe_backbone(tmp_path) -> None:
    raw = RawMultiScalePatchTST(
        build_multiscale_patchtst(d_model=16, n_heads=4, n_layers=1, d_ff=32, dropout=0.0)
    )
    combined = ScaleAwareASDMultiScalePatchTST(
        build_multiscale_patchtst(
            d_model=16,
            n_heads=4,
            n_layers=1,
            d_ff=32,
            dropout=0.0,
            lora_moe_mode="last1",
            lora_moe_rank=4,
        ),
        init_gate=-4.0,
    )
    with torch.no_grad():
        raw.backbone.position_embedding.fill_(0.654)
    checkpoint_path = tmp_path / "raw.pt"
    torch.save({"model": raw.state_dict()}, checkpoint_path)

    info = load_raw_backbone_checkpoint(combined, checkpoint_path)

    assert info["loaded_backbone_tensors"] > 0
    assert torch.allclose(combined.backbone.position_embedding, raw.backbone.position_embedding)
