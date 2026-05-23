from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from evaluate_optiver_spectral_denoise_patchtst import (  # noqa: E402
    build_datasets as build_single_scale_datasets,
    fit_normalizer,
    metric_dict,
    select_device,
    set_seed,
)
from src.baselines.patchtst_lora import count_parameters  # noqa: E402
from src.baselines.scale_aware_asd_patchtst import (  # noqa: E402
    AdaptiveSpectralEncoderBlock,
    DEFAULT_SCALE_SPECS,
    RawMultiScalePatchTST,
    ScaleAwareASDMultiScalePatchTST,
    ScaleSpec,
    ScaleAwareLoRAAdapterMoE,
    StaticASDMultiScalePatchTST,
    TSLANetMultiScaleForecaster,
    build_multiscale_patchtst,
)


SCALE_ORDER = ("second", "minute", "hour")
PATCH_PRESETS = {
    "compact": {
        "second": {"context_length": 64, "patch_length": 16, "patch_stride": 8},
        "minute": {"context_length": 4, "patch_length": 2, "patch_stride": 1},
        "hour": {"context_length": 32, "patch_length": 8, "patch_stride": 4},
    },
    "fincast_adapted": {
        # FinCast uses frequency-aware context choices and a configurable
        # patch_len; these values keep that spirit while respecting the short
        # Optiver minute/hour validation segments in this project.
        "second": {"context_length": 128, "patch_length": 32, "patch_stride": 16},
        "minute": {"context_length": 8, "patch_length": 4, "patch_stride": 2},
        "hour": {"context_length": 32, "patch_length": 8, "patch_stride": 4},
    },
    "short_second": {
        "second": {"context_length": 64, "patch_length": 8, "patch_stride": 4},
        "minute": {"context_length": 8, "patch_length": 4, "patch_stride": 2},
        "hour": {"context_length": 32, "patch_length": 8, "patch_stride": 4},
    },
    "long_context": {
        "second": {"context_length": 256, "patch_length": 32, "patch_stride": 16},
        "minute": {"context_length": 8, "patch_length": 4, "patch_stride": 2},
        "hour": {"context_length": 32, "patch_length": 8, "patch_stride": 4},
    },
}
TARGET_HORIZON_STEPS = {
    "second": 30,
    "minute": 1,
    "hour": 1,
}
FULL_REFERENCE_RUNS = {
    "second": "30s_second_512t",
    "minute": "minute_512t",
    "hour": "hour_512t",
}
FULL_REFERENCE_METRICS = {
    "second": WORKSPACE_ROOT
    / "outputs"
    / "optiver_spectral_denoise_patchtst_h30_512t_keep010_blend015"
    / "metrics.json",
    "minute": WORKSPACE_ROOT
    / "outputs"
    / "optiver_spectral_denoise_patchtst_minute_512t_keep010_blend015"
    / "metrics.json",
    "hour": WORKSPACE_ROOT
    / "outputs"
    / "optiver_spectral_denoise_patchtst_hour_512t_keep010_blend015"
    / "metrics.json",
}
MODEL_ORDER = ("zero", "raw_patchtst", "static_asd_patchtst", "scale_aware_asd_patchtst")
ABLATION_TRAINING_REGIMES = (
    "raw_joint",
    "asd_joint",
    "asd_only_frozen_backbone",
    "asd_frozen_encoder_train_head",
)
ASB_TRAINING_REGIMES = (
    "raw_joint",
    "asd_frozen_encoder_train_head",
    "asb_encoder_joint",
    "asb_encoder_frozen_base_train_asb_only",
    "asb_encoder_frozen_base_train_asb_head",
)
LORA_MOE_TRAINING_REGIMES = (
    "raw_joint",
    "asd_frozen_encoder_train_head",
    "lora_moe_joint",
    "lora_moe_frozen_base_train_moe_only",
    "lora_moe_frozen_base_train_moe_head",
)
ADAPTER_ABLATION_TRAINING_REGIMES = (
    "raw_joint",
    "raw_frozen_base_train_head",
    "asd_frozen_encoder_train_head",
    "lora_only_joint",
    "lora_only_frozen_base_train_adapter_only",
    "lora_only_frozen_base_train_adapter_head",
    "mlp_moe_joint",
    "mlp_moe_frozen_base_train_moe_only",
    "mlp_moe_frozen_base_train_moe_head",
    "lora_moe_frozen_base_train_moe_head",
    "asd_lora_only_frozen_base_train_adapter_head",
    "asd_mlp_moe_frozen_base_train_moe_head",
    "asd_lora_moe_frozen_base_train_adapters_head",
)
ASD_LORA_MOE_TRAINING_REGIMES = (
    "raw_joint",
    "asd_frozen_encoder_train_head",
    "lora_moe_frozen_base_train_moe_head",
    "asd_lora_moe_joint",
    "asd_lora_moe_frozen_base_train_adapters_only",
    "asd_lora_moe_frozen_base_train_adapters_head",
)
TSLANET_TRAINING_REGIMES = (
    "raw_joint",
    "asd_frozen_encoder_train_head",
    "tslanet_joint",
)
DEFAULT_ABLATION_PATCH_PRESETS = ("compact", "fincast_adapted", "short_second")
DEFAULT_ASB_PATCH_PRESETS = ("compact", "short_second")
DEFAULT_LORA_MOE_PATCH_PRESETS = ("compact", "short_second")
DEFAULT_ADAPTER_ABLATION_PATCH_PRESETS = ("short_second",)
DEFAULT_ASD_LORA_MOE_PATCH_PRESETS = ("compact", "short_second")
DEFAULT_TSLANET_PATCH_PRESETS = ("compact", "short_second")
DEFAULT_ASD_INIT_GATES = (-4.0, -3.0, -2.0)
DEFAULT_ASB_INIT_GATES = (-4.0, -3.0)
DEFAULT_ASD_LORA_MOE_INIT_GATES = (-4.0, -3.0)
DEFAULT_LORA_MOE_RANKS = (4, 8)
TARGETED_ASD_LORA_MOE_PATCH_PRESET = "short_second"
TARGETED_ASD_LORA_MOE_INIT_GATE = -4.0
TARGETED_ASD_LORA_MOE_RANK = 8
TARGETED_ASD_LORA_MOE_REGIME = "asd_lora_moe_frozen_base_train_adapters_head"
TARGETED_ASD_LORA_MOE_OUTPUT_SUBDIR = "round3_robustness_short_second_rank8"
TARGETED_ADAPTER_OUTPUT_SUBDIR = "targeted_robustness_short_second_rank8"


@dataclass
class ScaleData:
    name: str
    spec: ScaleSpec
    arrays: dict[str, Any]
    normalizer: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run small-first intraday scale-aware ASD PatchTST experiments. "
            "The auto mode runs small, writes full-data estimates, applies the "
            "quality gate, and then runs full if the gate passes."
        )
    )
    parser.add_argument(
        "--mode",
        choices=[
            "small",
            "full",
            "auto",
            "ablation",
            "asb_ablation",
            "lora_moe_ablation",
            "adapter_ablation",
            "adapter_targeted_robustness",
            "asd_lora_moe_ablation",
            "asd_lora_moe_targeted_robustness",
            "tslanet_baseline",
        ],
        default="small",
    )
    parser.add_argument(
        "--small-cache",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "cache"
            / "position_optiver_hf_second_feature_cache_11stocks_512t.npz"
        ),
    )
    parser.add_argument(
        "--full-cache",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "cache"
            / "position_optiver_hf_second_feature_cache_11stocks_512t.npz"
        ),
    )
    parser.add_argument("--output-dir", default=str(WORKSPACE_ROOT / "outputs" / "scale_aware_asd_patchtst"))
    parser.add_argument("--report-path", default=str(WORKSPACE_ROOT / "report" / "scale_aware_asd_patchtst_experiment.md"))
    parser.add_argument(
        "--ablation-output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "scale_aware_asd_patchtst_ablation"),
    )
    parser.add_argument(
        "--ablation-report-path",
        default=str(WORKSPACE_ROOT / "report" / "scale_aware_asd_patchtst_ablation.md"),
    )
    parser.add_argument(
        "--asb-output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "scale_aware_asb_encoder_patchtst"),
    )
    parser.add_argument(
        "--asb-report-path",
        default=str(WORKSPACE_ROOT / "report" / "scale_aware_asb_encoder_patchtst_experiment.md"),
    )
    parser.add_argument(
        "--lora-moe-output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "scale_aware_lora_moe_patchtst"),
    )
    parser.add_argument(
        "--lora-moe-report-path",
        default=str(WORKSPACE_ROOT / "report" / "scale_aware_lora_moe_patchtst_experiment.md"),
    )
    parser.add_argument(
        "--adapter-ablation-output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "scale_aware_adapter_ablation_patchtst"),
    )
    parser.add_argument(
        "--adapter-ablation-report-path",
        default=str(WORKSPACE_ROOT / "report" / "scale_aware_adapter_ablation_patchtst.md"),
    )
    parser.add_argument(
        "--adapter-targeted-report-path",
        default=str(WORKSPACE_ROOT / "report" / "scale_aware_adapter_targeted_robustness.md"),
    )
    parser.add_argument(
        "--asd-lora-moe-output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "scale_aware_asd_lora_moe_patchtst"),
    )
    parser.add_argument(
        "--asd-lora-moe-report-path",
        default=str(WORKSPACE_ROOT / "report" / "scale_aware_asd_lora_moe_patchtst_experiment.md"),
    )
    parser.add_argument(
        "--asd-lora-moe-final-report-path",
        default=str(WORKSPACE_ROOT / "report" / "scale_aware_asd_lora_moe_final_decision.md"),
    )
    parser.add_argument(
        "--tslanet-output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "tslanet_intraday_baseline"),
    )
    parser.add_argument(
        "--tslanet-report-path",
        default=str(WORKSPACE_ROOT / "report" / "tslanet_intraday_baseline.md"),
    )
    parser.add_argument("--full-reference-summary", default=str(WORKSPACE_ROOT / "outputs" / "optiver_spectral_denoise_patchtst_scale_summary.csv"))
    parser.add_argument("--train-stocks", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--zero-shot-stock", type=int, default=10)
    parser.add_argument("--scales", nargs="+", choices=SCALE_ORDER, default=list(SCALE_ORDER))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-preset", choices=sorted(PATCH_PRESETS), default="fincast_adapted")
    for scale in SCALE_ORDER:
        parser.add_argument(f"--{scale}-context-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-length", type=int, default=None)
        parser.add_argument(f"--{scale}-patch-stride", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--small-epochs", type=int, default=3)
    parser.add_argument("--full-epochs", type=int, default=5)
    parser.add_argument("--small-steps-per-epoch", type=int, default=12)
    parser.add_argument("--full-steps-per-epoch", type=int, default=250)
    parser.add_argument("--small-train-cap", type=int, default=4096)
    parser.add_argument("--small-validation-cap", type=int, default=1024)
    parser.add_argument("--small-test-cap", type=int, default=1024)
    parser.add_argument("--small-zero-shot-cap", type=int, default=1024)
    parser.add_argument("--full-train-cap", type=int, default=0)
    parser.add_argument("--full-validation-cap", type=int, default=0)
    parser.add_argument("--full-test-cap", type=int, default=0)
    parser.add_argument("--full-zero-shot-cap", type=int, default=0)
    parser.add_argument("--static-keep-ratio", type=float, default=0.1)
    parser.add_argument("--static-blend-init", type=float, default=0.15)
    parser.add_argument("--scale-aware-init-gate", type=float, default=-3.0)
    parser.add_argument("--encoder-spectral-mode", choices=["none", "last1"], default="none")
    parser.add_argument("--encoder-spectral-init-gate", type=float, default=-4.0)
    parser.add_argument("--lora-moe-rank", type=int, default=4)
    parser.add_argument("--lora-moe-alpha", type=float, default=16.0)
    parser.add_argument("--lora-moe-n-experts", type=int, default=4)
    parser.add_argument("--lora-moe-top-k", type=int, default=2)
    parser.add_argument("--lora-moe-dropout", type=float, default=0.1)
    parser.add_argument("--router-balance-weight", type=float, default=1e-3)
    parser.add_argument(
        "--training-regimes",
        nargs="+",
        choices=ABLATION_TRAINING_REGIMES,
        default=list(ABLATION_TRAINING_REGIMES),
    )
    parser.add_argument(
        "--patch-presets",
        nargs="+",
        choices=sorted(PATCH_PRESETS),
        default=list(DEFAULT_ABLATION_PATCH_PRESETS),
    )
    parser.add_argument("--asd-init-gates", nargs="+", type=float, default=list(DEFAULT_ASD_INIT_GATES))
    parser.add_argument(
        "--asb-training-regimes",
        nargs="+",
        choices=ASB_TRAINING_REGIMES,
        default=list(ASB_TRAINING_REGIMES),
    )
    parser.add_argument(
        "--asb-patch-presets",
        nargs="+",
        choices=sorted(PATCH_PRESETS),
        default=list(DEFAULT_ASB_PATCH_PRESETS),
    )
    parser.add_argument("--asb-init-gates", nargs="+", type=float, default=list(DEFAULT_ASB_INIT_GATES))
    parser.add_argument("--asb-top-k-full", type=int, default=2)
    parser.add_argument("--skip-asb-full", action="store_true")
    parser.add_argument(
        "--lora-moe-training-regimes",
        nargs="+",
        choices=LORA_MOE_TRAINING_REGIMES,
        default=list(LORA_MOE_TRAINING_REGIMES),
    )
    parser.add_argument(
        "--lora-moe-patch-presets",
        nargs="+",
        choices=sorted(PATCH_PRESETS),
        default=list(DEFAULT_LORA_MOE_PATCH_PRESETS),
    )
    parser.add_argument("--lora-moe-ranks", nargs="+", type=int, default=list(DEFAULT_LORA_MOE_RANKS))
    parser.add_argument("--lora-moe-top-k-full", type=int, default=2)
    parser.add_argument("--skip-lora-moe-full", action="store_true")
    parser.add_argument(
        "--adapter-ablation-training-regimes",
        nargs="+",
        choices=ADAPTER_ABLATION_TRAINING_REGIMES,
        default=[
            "raw_joint",
            "raw_frozen_base_train_head",
            "asd_frozen_encoder_train_head",
            "lora_only_frozen_base_train_adapter_only",
            "lora_only_frozen_base_train_adapter_head",
            "mlp_moe_frozen_base_train_moe_only",
            "mlp_moe_frozen_base_train_moe_head",
            "lora_moe_frozen_base_train_moe_head",
            "asd_lora_only_frozen_base_train_adapter_head",
            "asd_mlp_moe_frozen_base_train_moe_head",
            "asd_lora_moe_frozen_base_train_adapters_head",
        ],
    )
    parser.add_argument(
        "--adapter-ablation-patch-presets",
        nargs="+",
        choices=sorted(PATCH_PRESETS),
        default=list(DEFAULT_ADAPTER_ABLATION_PATCH_PRESETS),
    )
    parser.add_argument("--adapter-ablation-ranks", nargs="+", type=int, default=list(DEFAULT_LORA_MOE_RANKS))
    parser.add_argument("--run-adapter-ablation-full", action="store_true")
    parser.add_argument(
        "--asd-lora-moe-training-regimes",
        nargs="+",
        choices=ASD_LORA_MOE_TRAINING_REGIMES,
        default=list(ASD_LORA_MOE_TRAINING_REGIMES),
    )
    parser.add_argument(
        "--asd-lora-moe-patch-presets",
        nargs="+",
        choices=sorted(PATCH_PRESETS),
        default=list(DEFAULT_ASD_LORA_MOE_PATCH_PRESETS),
    )
    parser.add_argument("--asd-lora-moe-ranks", nargs="+", type=int, default=list(DEFAULT_LORA_MOE_RANKS))
    parser.add_argument("--asd-lora-moe-init-gates", nargs="+", type=float, default=list(DEFAULT_ASD_LORA_MOE_INIT_GATES))
    parser.add_argument("--asd-lora-moe-top-k-full", type=int, default=2)
    parser.add_argument("--skip-asd-lora-moe-full", action="store_true")
    parser.add_argument(
        "--tslanet-training-regimes",
        nargs="+",
        choices=TSLANET_TRAINING_REGIMES,
        default=list(TSLANET_TRAINING_REGIMES),
    )
    parser.add_argument(
        "--tslanet-patch-presets",
        nargs="+",
        choices=sorted(PATCH_PRESETS),
        default=list(DEFAULT_TSLANET_PATCH_PRESETS),
    )
    parser.add_argument("--skip-tslanet-full", action="store_true")
    parser.add_argument("--ablation-top-k-full", type=int, default=2)
    parser.add_argument("--robustness-seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--skip-ablation-full", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--skip-long-context-auto", action="store_true")
    parser.add_argument("--router-free-note", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if "day" in args.scales:
        raise ValueError("This runner intentionally supports only second/minute/hour.")
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.report_path).parent.mkdir(parents=True, exist_ok=True)
    device = select_device(args.device)
    print(f"device={device}", flush=True)
    print(f"mode={args.mode} scales={args.scales}", flush=True)

    if args.mode == "ablation":
        run_ablation(args, device=device)
        print(f"saved_report={Path(args.ablation_report_path)}", flush=True)
        return
    if args.mode == "asb_ablation":
        run_asb_ablation(args, device=device)
        print(f"saved_report={Path(args.asb_report_path)}", flush=True)
        return
    if args.mode == "lora_moe_ablation":
        run_lora_moe_ablation(args, device=device)
        print(f"saved_report={Path(args.lora_moe_report_path)}", flush=True)
        return
    if args.mode == "adapter_ablation":
        run_adapter_ablation(args, device=device)
        print(f"saved_report={Path(args.adapter_ablation_report_path)}", flush=True)
        return
    if args.mode == "adapter_targeted_robustness":
        run_adapter_targeted_robustness(args, device=device)
        print(f"saved_report={Path(args.adapter_targeted_report_path)}", flush=True)
        return
    if args.mode == "asd_lora_moe_ablation":
        run_asd_lora_moe_ablation(args, device=device)
        print(f"saved_report={Path(args.asd_lora_moe_report_path)}", flush=True)
        return
    if args.mode == "asd_lora_moe_targeted_robustness":
        run_asd_lora_moe_targeted_robustness(args, device=device)
        print(f"saved_report={Path(args.asd_lora_moe_final_report_path)}", flush=True)
        return
    if args.mode == "tslanet_baseline":
        run_tslanet_baseline(args, device=device)
        print(f"saved_report={Path(args.tslanet_report_path)}", flush=True)
        return

    small_result = None
    full_result = None
    full_estimate = None
    quality_gate = None

    if args.mode in {"small", "auto"}:
        small_result = run_preset(args, preset="small", device=device)
        full_estimate = estimate_full_from_small(args, small_result)
        quality_gate = evaluate_quality_gate(small_result, args.scales)
        write_report(
            args=args,
            small_result=small_result,
            full_estimate=full_estimate,
            quality_gate=quality_gate,
            full_result=None,
        )
        print(f"quality_gate={quality_gate['passed']} reasons={quality_gate['reasons']}", flush=True)
        if args.mode == "auto" and quality_gate["passed"]:
            full_result = run_preset(args, preset="full", device=device)
            write_report(
                args=args,
                small_result=small_result,
                full_estimate=full_estimate,
                quality_gate=quality_gate,
                full_result=full_result,
            )
    elif args.mode == "full":
        full_result = run_preset(args, preset="full", device=device)
        write_report(
            args=args,
            small_result=None,
            full_estimate=None,
            quality_gate=None,
            full_result=full_result,
        )

    print(f"saved_report={Path(args.report_path)}", flush=True)


def run_preset(args: argparse.Namespace, *, preset: str, device: torch.device) -> dict[str, Any]:
    preset_dir = Path(args.output_dir) / preset
    preset_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.small_cache if preset == "small" else args.full_cache)
    caps = caps_for_preset(args, preset)
    epochs = int(args.small_epochs if preset == "small" else args.full_epochs)
    requested_steps = int(args.small_steps_per_epoch if preset == "small" else args.full_steps_per_epoch)
    scale_specs = make_scale_specs(args)

    print(f"\nPRESET {preset} cache={cache_path}", flush=True)
    print(f"patch_preset={args.patch_preset} specs={ {name: asdict(scale_specs[name]) for name in args.scales} }", flush=True)
    set_seed(args.seed)
    scale_data = load_scale_data(args, cache_path=cache_path, caps=caps, scale_specs=scale_specs)
    steps_per_epoch = resolve_steps_per_epoch(scale_data, args.batch_size, requested_steps)
    loaders = make_all_loaders(scale_data, batch_size=args.batch_size, device=device)
    results: dict[str, Any] = {
        "preset": preset,
        "cache": str(cache_path),
        "scales": list(scale_data.keys()),
        "caps": caps,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": int(args.batch_size),
        "patch_preset": args.patch_preset,
        "fincast_reference": {
            "local_files": [
                "FinCast-fts/src/data_tools/TSdataset.py",
                "FinCast-fts/src/ffm/pytorch_patched_decoder_MOE.py",
            ],
            "note": "FinCast passes a frequency id and uses frequency-aware context-length candidates; this runner keeps patch/context explicit per intraday scale.",
        },
        "scale_specs": {name: asdict(data.spec) for name, data in scale_data.items()},
        "data_meta": {name: data.arrays["meta"] for name, data in scale_data.items()},
        "models": {},
    }
    summary_rows: list[dict[str, Any]] = []
    append_baseline_rows(summary_rows, preset, scale_data)

    for model_name in ["raw_patchtst", "static_asd_patchtst", "scale_aware_asd_patchtst"]:
        set_seed(args.seed)
        model = build_model(model_name, args, {name: data.spec for name, data in scale_data.items()}).to(device)
        print(f"training {preset}/{model_name}", flush=True)
        train_result = train_model(
            model=model,
            model_name=model_name,
            scale_data=scale_data,
            loaders=loaders,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=device,
            output_dir=preset_dir,
        )
        results["models"][model_name] = train_result
        append_model_rows(summary_rows, preset, model_name, train_result)

    summary = pd.DataFrame(summary_rows)
    summary = summary.sort_values(["preset", "split", "scale", "model"], key=sort_summary_key)
    summary_path = preset_dir / "summary.csv"
    metrics_path = preset_dir / "metrics.json"
    summary.to_csv(summary_path, index=False)
    metrics_path.write_text(json.dumps(to_jsonable(results), indent=2), encoding="utf-8")
    results["summary_path"] = str(summary_path)
    results["metrics_path"] = str(metrics_path)
    results["summary"] = summary
    print(f"saved_summary={summary_path}", flush=True)
    print(summary.to_string(index=False), flush=True)
    return results


def run_ablation(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    output_root = Path(args.ablation_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.ablation_report_path).parent.mkdir(parents=True, exist_ok=True)

    patch_presets = unique_list(args.patch_presets)
    training_regimes = unique_list(args.training_regimes)
    init_gates = [float(value) for value in args.asd_init_gates]
    if "day" in args.scales:
        raise ValueError("The ablation runner intentionally supports only second/minute/hour.")
    for preset_name in patch_presets:
        validate_patch_preset_intraday(preset_name)

    print(
        f"ablation patch_presets={patch_presets} regimes={training_regimes} "
        f"init_gates={init_gates}",
        flush=True,
    )

    round1_rows: list[dict[str, Any]] = []
    round1_records: list[dict[str, Any]] = []
    for patch_preset in patch_presets:
        rows, records = run_ablation_patch_matrix(
            args,
            preset="small",
            round_name="round1_small",
            patch_preset=patch_preset,
            training_regimes=training_regimes,
            init_gates=init_gates,
            seed=int(args.seed),
            device=device,
            output_dir=output_root / "round1_small" / patch_preset,
        )
        round1_rows.extend(rows)
        round1_records.extend(records)

    round1_summary = save_summary(round1_rows, output_root / "round1_small_summary.csv")
    if (
        not args.skip_long_context_auto
        and "long_context" not in patch_presets
        and needs_long_context(round1_summary)
    ):
        print("second scale is still weak in Round 1; adding long_context patch preset.", flush=True)
        rows, records = run_ablation_patch_matrix(
            args,
            preset="small",
            round_name="round1_small",
            patch_preset="long_context",
            training_regimes=training_regimes,
            init_gates=init_gates,
            seed=int(args.seed),
            device=device,
            output_dir=output_root / "round1_small" / "long_context",
        )
        round1_rows.extend(rows)
        round1_records.extend(records)
        round1_summary = save_summary(round1_rows, output_root / "round1_small_summary.csv")

    selection = select_top_ablation_configs(round1_summary, top_k=int(args.ablation_top_k_full))
    selection_path = output_root / "round1_selection.csv"
    pd.DataFrame(selection).to_csv(selection_path, index=False)

    round2_summary = pd.DataFrame()
    round2_records: list[dict[str, Any]] = []
    if selection and not args.skip_ablation_full:
        full_rows: list[dict[str, Any]] = []
        for patch_preset in sorted({str(item["patch_preset"]) for item in selection}):
            patch_configs = [
                {
                    "training_regime": str(item["training_regime"]),
                    "init_gate": item["init_gate"],
                }
                for item in selection
                if str(item["patch_preset"]) == patch_preset
            ]
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round2_full_confirm",
                patch_preset=patch_preset,
                configs=patch_configs,
                seed=int(args.seed),
                device=device,
                output_dir=output_root / "round2_full_confirm" / patch_preset,
            )
            full_rows.extend(rows)
            round2_records.extend(records)
        round2_summary = save_summary(full_rows, output_root / "round2_full_summary.csv")

    robustness_summary = pd.DataFrame()
    robustness_aggregate = pd.DataFrame()
    robustness_records: list[dict[str, Any]] = []
    best_config = select_best_confirmed_config(round2_summary, selection)
    if best_config and not args.skip_robustness:
        robustness_rows: list[dict[str, Any]] = []
        for seed in unique_list(args.robustness_seeds):
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round3_robustness",
                patch_preset=str(best_config["patch_preset"]),
                configs=[
                    {
                        "training_regime": str(best_config["training_regime"]),
                        "init_gate": best_config["init_gate"],
                    }
                ],
                seed=int(seed),
                device=device,
                output_dir=output_root / "round3_robustness" / f"seed_{seed}",
            )
            robustness_rows.extend(rows)
            robustness_records.extend(records)
        robustness_summary = save_summary(robustness_rows, output_root / "round3_robustness_summary.csv")
        robustness_aggregate = aggregate_robustness(robustness_summary)
        robustness_aggregate.to_csv(output_root / "round3_robustness_aggregate.csv", index=False)

    diagnostics = diagnostics_frame(round1_records + round2_records + robustness_records)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    metrics_payload = {
        "round1_summary": round1_summary,
        "selection": selection,
        "round2_summary": round2_summary,
        "robustness_summary": robustness_summary,
        "robustness_aggregate": robustness_aggregate,
        "diagnostics": diagnostics,
    }
    (output_root / "ablation_metrics.json").write_text(
        json.dumps(to_jsonable(metrics_payload), indent=2),
        encoding="utf-8",
    )
    write_ablation_report(
        args=args,
        output_root=output_root,
        round1_summary=round1_summary,
        selection=pd.DataFrame(selection),
        round2_summary=round2_summary,
        robustness_aggregate=robustness_aggregate,
        diagnostics=diagnostics,
        best_config=best_config,
    )
    return metrics_payload


def run_asb_ablation(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    output_root = Path(args.asb_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.asb_report_path).parent.mkdir(parents=True, exist_ok=True)
    patch_presets = unique_list(args.asb_patch_presets)
    training_regimes = unique_list(args.asb_training_regimes)
    init_gates = [float(value) for value in args.asb_init_gates]
    for preset_name in patch_presets:
        validate_patch_preset_intraday(preset_name)

    print(
        f"asb_ablation patch_presets={patch_presets} regimes={training_regimes} "
        f"init_gates={init_gates}",
        flush=True,
    )
    round1_rows: list[dict[str, Any]] = []
    round1_records: list[dict[str, Any]] = []
    for patch_preset in patch_presets:
        configs = make_asb_config_matrix(training_regimes, init_gates)
        rows, records = run_selected_ablation_configs(
            args,
            preset="small",
            round_name="round1_small",
            patch_preset=patch_preset,
            configs=configs,
            seed=int(args.seed),
            device=device,
            output_dir=output_root / "round1_small" / patch_preset,
        )
        round1_rows.extend(rows)
        round1_records.extend(records)

    round1_summary = save_summary(round1_rows, output_root / "round1_small_summary.csv")
    selection = select_top_asb_configs(round1_summary, top_k=int(args.asb_top_k_full))
    pd.DataFrame(selection).to_csv(output_root / "round1_selection.csv", index=False)

    round2_summary = pd.DataFrame()
    round2_records: list[dict[str, Any]] = []
    if selection and not args.skip_asb_full:
        full_rows: list[dict[str, Any]] = []
        for patch_preset in sorted({str(item["patch_preset"]) for item in selection}):
            patch_configs = [{"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0}]
            patch_configs.extend(
                {
                    "training_regime": str(item["training_regime"]),
                    "init_gate": item["init_gate"],
                }
                for item in selection
                if str(item["patch_preset"]) == patch_preset
            )
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round2_full_confirm",
                patch_preset=patch_preset,
                configs=patch_configs,
                seed=int(args.seed),
                device=device,
                output_dir=output_root / "round2_full_confirm" / patch_preset,
            )
            full_rows.extend(rows)
            round2_records.extend(records)
        round2_summary = save_summary(full_rows, output_root / "round2_full_summary.csv")

    robustness_summary = pd.DataFrame()
    robustness_aggregate = pd.DataFrame()
    robustness_records: list[dict[str, Any]] = []
    best_config = select_best_confirmed_asb_config(round2_summary, selection)
    if best_config and not args.skip_robustness:
        robustness_rows: list[dict[str, Any]] = []
        for seed in unique_list(args.robustness_seeds):
            configs = [
                {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0},
                {
                    "training_regime": str(best_config["training_regime"]),
                    "init_gate": best_config["init_gate"],
                },
            ]
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round3_robustness",
                patch_preset=str(best_config["patch_preset"]),
                configs=configs,
                seed=int(seed),
                device=device,
                output_dir=output_root / "round3_robustness" / f"seed_{seed}",
            )
            robustness_rows.extend(rows)
            robustness_records.extend(records)
        robustness_summary = save_summary(robustness_rows, output_root / "round3_robustness_summary.csv")
        robustness_aggregate = aggregate_robustness(robustness_summary)
        robustness_aggregate.to_csv(output_root / "round3_robustness_aggregate.csv", index=False)

    diagnostics = diagnostics_frame(round1_records + round2_records + robustness_records)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    payload = {
        "round1_summary": round1_summary,
        "selection": selection,
        "round2_summary": round2_summary,
        "robustness_summary": robustness_summary,
        "robustness_aggregate": robustness_aggregate,
        "diagnostics": diagnostics,
    }
    (output_root / "asb_metrics.json").write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    write_asb_report(
        args=args,
        output_root=output_root,
        round1_summary=round1_summary,
        selection=pd.DataFrame(selection),
        round2_summary=round2_summary,
        robustness_aggregate=robustness_aggregate,
        diagnostics=diagnostics,
        best_config=best_config,
    )
    return payload


def run_lora_moe_ablation(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    output_root = Path(args.lora_moe_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.lora_moe_report_path).parent.mkdir(parents=True, exist_ok=True)
    patch_presets = unique_list(args.lora_moe_patch_presets)
    training_regimes = unique_list(args.lora_moe_training_regimes)
    ranks = [int(value) for value in args.lora_moe_ranks]
    for preset_name in patch_presets:
        validate_patch_preset_intraday(preset_name)
    if any(rank <= 0 for rank in ranks):
        raise ValueError("LoRA-MoE ranks must be positive.")

    print(
        f"lora_moe_ablation patch_presets={patch_presets} regimes={training_regimes} "
        f"ranks={ranks}",
        flush=True,
    )
    round1_rows: list[dict[str, Any]] = []
    round1_records: list[dict[str, Any]] = []
    for patch_preset in patch_presets:
        configs = make_lora_moe_config_matrix(training_regimes, ranks)
        rows, records = run_selected_ablation_configs(
            args,
            preset="small",
            round_name="round1_small",
            patch_preset=patch_preset,
            configs=configs,
            seed=int(args.seed),
            device=device,
            output_dir=output_root / "round1_small" / patch_preset,
        )
        round1_rows.extend(rows)
        round1_records.extend(records)

    round1_summary = save_summary(round1_rows, output_root / "round1_small_summary.csv")
    selection = select_top_lora_moe_configs(round1_summary, top_k=int(args.lora_moe_top_k_full))
    pd.DataFrame(selection).to_csv(output_root / "round1_selection.csv", index=False)

    round2_summary = pd.DataFrame()
    round2_records: list[dict[str, Any]] = []
    if selection and not args.skip_lora_moe_full:
        full_rows: list[dict[str, Any]] = []
        for patch_preset in sorted({str(item["patch_preset"]) for item in selection}):
            patch_configs = [
                {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None}
            ]
            patch_configs.extend(
                {
                    "training_regime": str(item["training_regime"]),
                    "init_gate": item["init_gate"],
                    "adapter_rank": item.get("adapter_rank"),
                }
                for item in selection
                if str(item["patch_preset"]) == patch_preset
            )
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round2_full_confirm",
                patch_preset=patch_preset,
                configs=patch_configs,
                seed=int(args.seed),
                device=device,
                output_dir=output_root / "round2_full_confirm" / patch_preset,
            )
            full_rows.extend(rows)
            round2_records.extend(records)
        round2_summary = save_summary(full_rows, output_root / "round2_full_summary.csv")

    robustness_summary = pd.DataFrame()
    robustness_aggregate = pd.DataFrame()
    robustness_records: list[dict[str, Any]] = []
    best_config = select_best_confirmed_lora_moe_config(round2_summary, selection)
    if best_config and not args.skip_robustness:
        robustness_rows: list[dict[str, Any]] = []
        for seed in unique_list(args.robustness_seeds):
            configs = [
                {"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None},
                {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None},
                {
                    "training_regime": str(best_config["training_regime"]),
                    "init_gate": best_config.get("init_gate"),
                    "adapter_rank": best_config.get("adapter_rank"),
                },
            ]
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round3_robustness",
                patch_preset=str(best_config["patch_preset"]),
                configs=configs,
                seed=int(seed),
                device=device,
                output_dir=output_root / "round3_robustness" / f"seed_{seed}",
            )
            robustness_rows.extend(rows)
            robustness_records.extend(records)
        robustness_summary = save_summary(robustness_rows, output_root / "round3_robustness_summary.csv")
        robustness_aggregate = aggregate_robustness(robustness_summary)
        robustness_aggregate.to_csv(output_root / "round3_robustness_aggregate.csv", index=False)

    diagnostics = diagnostics_frame(round1_records + round2_records + robustness_records)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    oracle = build_per_scale_oracle(round2_summary, output_root)
    payload = {
        "round1_summary": round1_summary,
        "selection": selection,
        "round2_summary": round2_summary,
        "robustness_summary": robustness_summary,
        "robustness_aggregate": robustness_aggregate,
        "diagnostics": diagnostics,
        "per_scale_oracle": oracle,
    }
    (output_root / "lora_moe_metrics.json").write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")
    write_lora_moe_report(
        args=args,
        output_root=output_root,
        round1_summary=round1_summary,
        selection=pd.DataFrame(selection),
        round2_summary=round2_summary,
        robustness_aggregate=robustness_aggregate,
        diagnostics=diagnostics,
        oracle=oracle,
        best_config=best_config,
    )
    return payload


def run_adapter_ablation(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    output_root = Path(args.adapter_ablation_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.adapter_ablation_report_path).parent.mkdir(parents=True, exist_ok=True)
    patch_presets = unique_list(args.adapter_ablation_patch_presets)
    training_regimes = unique_list(args.adapter_ablation_training_regimes)
    ranks = [int(value) for value in args.adapter_ablation_ranks]
    for preset_name in patch_presets:
        validate_patch_preset_intraday(preset_name)
    if any(rank <= 0 for rank in ranks):
        raise ValueError("adapter ablation ranks must be positive.")

    print(
        f"adapter_ablation patch_presets={patch_presets} regimes={training_regimes} ranks={ranks}",
        flush=True,
    )
    round1_rows: list[dict[str, Any]] = []
    round1_records: list[dict[str, Any]] = []
    for patch_preset in patch_presets:
        configs = make_adapter_ablation_config_matrix(training_regimes, ranks)
        rows, records = run_selected_ablation_configs(
            args,
            preset="small",
            round_name="round1_small",
            patch_preset=patch_preset,
            configs=configs,
            seed=int(args.seed),
            device=device,
            output_dir=output_root / "round1_small" / patch_preset,
        )
        round1_rows.extend(rows)
        round1_records.extend(records)

    round1_summary = save_summary(round1_rows, output_root / "round1_small_summary.csv")
    round2_summary = pd.DataFrame()
    round2_records: list[dict[str, Any]] = []
    if args.run_adapter_ablation_full:
        full_rows: list[dict[str, Any]] = []
        for patch_preset in patch_presets:
            configs = make_adapter_ablation_config_matrix(training_regimes, ranks)
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round2_full",
                patch_preset=patch_preset,
                configs=configs,
                seed=int(args.seed),
                device=device,
                output_dir=output_root / "round2_full" / patch_preset,
            )
            full_rows.extend(rows)
            round2_records.extend(records)
        round2_summary = save_summary(full_rows, output_root / "round2_full_summary.csv")

    diagnostics = diagnostics_frame(round1_records + round2_records)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    payload = {
        "round1_summary": round1_summary,
        "round2_summary": round2_summary,
        "diagnostics": diagnostics,
    }
    (output_root / "adapter_ablation_metrics.json").write_text(
        json.dumps(to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    write_adapter_ablation_report(
        args=args,
        output_root=output_root,
        round1_summary=round1_summary,
        round2_summary=round2_summary,
        diagnostics=diagnostics,
    )
    return payload


def targeted_adapter_robustness_configs() -> list[dict[str, Any]]:
    return [
        {"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None},
        {"training_regime": "raw_frozen_base_train_head", "init_gate": None, "adapter_rank": None},
        {
            "training_regime": "lora_moe_frozen_base_train_moe_head",
            "init_gate": None,
            "adapter_rank": TARGETED_ASD_LORA_MOE_RANK,
        },
        {
            "training_regime": TARGETED_ASD_LORA_MOE_REGIME,
            "init_gate": TARGETED_ASD_LORA_MOE_INIT_GATE,
            "adapter_rank": TARGETED_ASD_LORA_MOE_RANK,
        },
    ]


def run_adapter_targeted_robustness(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    output_root = Path(args.adapter_ablation_output_dir)
    target_root = output_root / TARGETED_ADAPTER_OUTPUT_SUBDIR
    target_root.mkdir(parents=True, exist_ok=True)
    Path(args.adapter_targeted_report_path).parent.mkdir(parents=True, exist_ok=True)
    validate_patch_preset_intraday(TARGETED_ASD_LORA_MOE_PATCH_PRESET)

    target_args = clone_args(args, scales=list(SCALE_ORDER), patch_preset=TARGETED_ASD_LORA_MOE_PATCH_PRESET)
    configs = targeted_adapter_robustness_configs()
    print(
        "adapter_targeted_robustness "
        f"patch_preset={TARGETED_ASD_LORA_MOE_PATCH_PRESET} "
        f"rank={TARGETED_ASD_LORA_MOE_RANK} "
        f"init_gate={TARGETED_ASD_LORA_MOE_INIT_GATE} "
        f"seeds={list(unique_list(args.robustness_seeds))}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for seed in unique_list(args.robustness_seeds):
        seed_rows, seed_records = run_selected_ablation_configs(
            target_args,
            preset="small",
            round_name="targeted_robustness",
            patch_preset=TARGETED_ASD_LORA_MOE_PATCH_PRESET,
            configs=configs,
            seed=int(seed),
            device=device,
            output_dir=target_root / f"seed_{seed}",
        )
        rows.extend(seed_rows)
        records.extend(seed_records)

    summary = save_summary(rows, target_root / "targeted_robustness_summary.csv")
    aggregate = aggregate_robustness(summary)
    aggregate.to_csv(target_root / "targeted_robustness_aggregate.csv", index=False)
    diagnostics = diagnostics_frame(records)
    diagnostics.to_csv(target_root / "targeted_robustness_diagnostics.csv", index=False)
    router_usage = summarize_router_usage(diagnostics)
    router_usage.to_csv(target_root / "router_usage_by_scale.csv", index=False)
    asd_stats = summarize_asd_gate_tau(diagnostics)
    asd_stats.to_csv(target_root / "asd_gate_tau_by_scale.csv", index=False)
    payload = {
        "summary": summary,
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "router_usage": router_usage,
        "asd_gate_tau": asd_stats,
    }
    (target_root / "adapter_targeted_robustness_metrics.json").write_text(
        json.dumps(to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    write_adapter_targeted_robustness_report(
        args=args,
        target_root=target_root,
        aggregate=aggregate,
        router_usage=router_usage,
        asd_stats=asd_stats,
    )
    return payload


def run_asd_lora_moe_ablation(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    output_root = Path(args.asd_lora_moe_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.asd_lora_moe_report_path).parent.mkdir(parents=True, exist_ok=True)
    patch_presets = unique_list(args.asd_lora_moe_patch_presets)
    training_regimes = unique_list(args.asd_lora_moe_training_regimes)
    ranks = [int(value) for value in args.asd_lora_moe_ranks]
    init_gates = [float(value) for value in args.asd_lora_moe_init_gates]
    for preset_name in patch_presets:
        validate_patch_preset_intraday(preset_name)
    if any(rank <= 0 for rank in ranks):
        raise ValueError("ASD+LoRA-MoE ranks must be positive.")

    print(
        f"asd_lora_moe_ablation patch_presets={patch_presets} regimes={training_regimes} "
        f"ranks={ranks} init_gates={init_gates}",
        flush=True,
    )
    round1_rows: list[dict[str, Any]] = []
    round1_records: list[dict[str, Any]] = []
    for patch_preset in patch_presets:
        configs = make_asd_lora_moe_config_matrix(training_regimes, ranks, init_gates)
        rows, records = run_selected_ablation_configs(
            args,
            preset="small",
            round_name="round1_small",
            patch_preset=patch_preset,
            configs=configs,
            seed=int(args.seed),
            device=device,
            output_dir=output_root / "round1_small" / patch_preset,
        )
        round1_rows.extend(rows)
        round1_records.extend(records)

    round1_summary = save_summary(round1_rows, output_root / "round1_small_summary.csv")
    selection = select_top_asd_lora_moe_configs(round1_summary, top_k=int(args.asd_lora_moe_top_k_full))
    pd.DataFrame(selection).to_csv(output_root / "round1_selection.csv", index=False)

    round2_summary = pd.DataFrame()
    round2_records: list[dict[str, Any]] = []
    if selection and any(bool(item.get("quality_pass")) for item in selection) and not args.skip_asd_lora_moe_full:
        full_rows: list[dict[str, Any]] = []
        for patch_preset in sorted({str(item["patch_preset"]) for item in selection}):
            patch_configs: list[dict[str, Any]] = [
                {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None}
            ]
            for item in selection:
                if str(item["patch_preset"]) != patch_preset:
                    continue
                rank = item.get("adapter_rank")
                if rank is not None:
                    patch_configs.append(
                        {
                            "training_regime": "lora_moe_frozen_base_train_moe_head",
                            "init_gate": None,
                            "adapter_rank": rank,
                        }
                    )
                patch_configs.append(
                    {
                        "training_regime": str(item["training_regime"]),
                        "init_gate": item.get("init_gate"),
                        "adapter_rank": rank,
                    }
                )
            patch_configs = dedupe_configs(patch_configs)
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round2_full_confirm",
                patch_preset=patch_preset,
                configs=patch_configs,
                seed=int(args.seed),
                device=device,
                output_dir=output_root / "round2_full_confirm" / patch_preset,
            )
            full_rows.extend(rows)
            round2_records.extend(records)
        round2_summary = save_summary(full_rows, output_root / "round2_full_summary.csv")

    robustness_summary = pd.DataFrame()
    robustness_aggregate = pd.DataFrame()
    robustness_records: list[dict[str, Any]] = []
    best_config = select_best_confirmed_asd_lora_moe_config(round2_summary, selection)
    if best_config and not args.skip_robustness:
        robustness_rows: list[dict[str, Any]] = []
        for seed in unique_list(args.robustness_seeds):
            configs = [
                {"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None},
                {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None},
                {
                    "training_regime": "lora_moe_frozen_base_train_moe_head",
                    "init_gate": None,
                    "adapter_rank": best_config.get("adapter_rank"),
                },
                {
                    "training_regime": str(best_config["training_regime"]),
                    "init_gate": best_config.get("init_gate"),
                    "adapter_rank": best_config.get("adapter_rank"),
                },
            ]
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round3_robustness",
                patch_preset=str(best_config["patch_preset"]),
                configs=dedupe_configs(configs),
                seed=int(seed),
                device=device,
                output_dir=output_root / "round3_robustness" / f"seed_{seed}",
            )
            robustness_rows.extend(rows)
            robustness_records.extend(records)
        robustness_summary = save_summary(robustness_rows, output_root / "round3_robustness_summary.csv")
        robustness_aggregate = aggregate_robustness(robustness_summary)
        robustness_aggregate.to_csv(output_root / "round3_robustness_aggregate.csv", index=False)

    diagnostics = diagnostics_frame(round1_records + round2_records + robustness_records)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    oracle = build_per_scale_oracle(
        round2_summary,
        output_root,
        current_source="asd_lora_moe_current_full",
    )
    payload = {
        "round1_summary": round1_summary,
        "selection": selection,
        "round2_summary": round2_summary,
        "robustness_summary": robustness_summary,
        "robustness_aggregate": robustness_aggregate,
        "diagnostics": diagnostics,
        "per_scale_oracle": oracle,
    }
    (output_root / "asd_lora_moe_metrics.json").write_text(
        json.dumps(to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    write_asd_lora_moe_report(
        args=args,
        output_root=output_root,
        round1_summary=round1_summary,
        selection=pd.DataFrame(selection),
        round2_summary=round2_summary,
        robustness_aggregate=robustness_aggregate,
        diagnostics=diagnostics,
        oracle=oracle,
        best_config=best_config,
    )
    return payload


def targeted_asd_lora_moe_robustness_configs() -> list[dict[str, Any]]:
    return [
        {"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None},
        {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None},
        {
            "training_regime": "lora_moe_frozen_base_train_moe_head",
            "init_gate": None,
            "adapter_rank": TARGETED_ASD_LORA_MOE_RANK,
        },
        {
            "training_regime": TARGETED_ASD_LORA_MOE_REGIME,
            "init_gate": TARGETED_ASD_LORA_MOE_INIT_GATE,
            "adapter_rank": TARGETED_ASD_LORA_MOE_RANK,
        },
    ]


def run_asd_lora_moe_targeted_robustness(
    args: argparse.Namespace,
    *,
    device: torch.device,
) -> dict[str, Any]:
    output_root = Path(args.asd_lora_moe_output_dir)
    target_root = output_root / TARGETED_ASD_LORA_MOE_OUTPUT_SUBDIR
    target_root.mkdir(parents=True, exist_ok=True)
    Path(args.asd_lora_moe_final_report_path).parent.mkdir(parents=True, exist_ok=True)
    validate_patch_preset_intraday(TARGETED_ASD_LORA_MOE_PATCH_PRESET)

    target_args = clone_args(args, scales=list(SCALE_ORDER), patch_preset=TARGETED_ASD_LORA_MOE_PATCH_PRESET)
    configs = targeted_asd_lora_moe_robustness_configs()
    print(
        "asd_lora_moe_targeted_robustness "
        f"patch_preset={TARGETED_ASD_LORA_MOE_PATCH_PRESET} "
        f"regime={TARGETED_ASD_LORA_MOE_REGIME} "
        f"rank={TARGETED_ASD_LORA_MOE_RANK} "
        f"init_gate={TARGETED_ASD_LORA_MOE_INIT_GATE} "
        f"seeds={list(unique_list(args.robustness_seeds))}",
        flush=True,
    )

    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for seed in unique_list(args.robustness_seeds):
        seed_rows, seed_records = run_selected_ablation_configs(
            target_args,
            preset="full",
            round_name="round3_short_second_rank8",
            patch_preset=TARGETED_ASD_LORA_MOE_PATCH_PRESET,
            configs=dedupe_configs(configs),
            seed=int(seed),
            device=device,
            output_dir=target_root / f"seed_{seed}",
        )
        rows.extend(seed_rows)
        records.extend(seed_records)

    summary = save_summary(rows, target_root / "round3_short_second_rank8_summary.csv")
    aggregate = aggregate_robustness(summary)
    aggregate.to_csv(target_root / "round3_short_second_rank8_aggregate.csv", index=False)
    diagnostics = diagnostics_frame(records)
    diagnostics.to_csv(target_root / "short_second_rank8_diagnostics.csv", index=False)
    validate_targeted_robustness_outputs(summary, diagnostics)

    existing_full_summary = load_csv_or_empty(output_root / "round2_full_summary.csv")
    compact_robustness = load_csv_or_empty(output_root / "round3_robustness_aggregate.csv")
    oracle_input = pd.concat(
        [frame for frame in [existing_full_summary, summary] if not frame.empty],
        ignore_index=True,
        sort=False,
    ) if not existing_full_summary.empty or not summary.empty else pd.DataFrame()
    oracle = build_per_scale_oracle(
        oracle_input,
        target_root,
        current_source="asd_lora_moe_targeted_robustness",
    )
    decision = evaluate_asd_lora_moe_final_decision(aggregate)
    payload = {
        "summary": summary,
        "aggregate": aggregate,
        "diagnostics": diagnostics,
        "compact_robustness": compact_robustness,
        "existing_full_summary": existing_full_summary,
        "per_scale_oracle": oracle,
        "decision": decision,
    }
    (target_root / "asd_lora_moe_short_second_rank8_metrics.json").write_text(
        json.dumps(to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    write_asd_lora_moe_final_decision_report(
        args=args,
        output_root=output_root,
        target_root=target_root,
        existing_full_summary=existing_full_summary,
        compact_robustness=compact_robustness,
        short_second_summary=summary,
        short_second_robustness=aggregate,
        diagnostics=diagnostics,
        oracle=oracle,
        decision=decision,
    )
    return payload


def run_tslanet_baseline(args: argparse.Namespace, *, device: torch.device) -> dict[str, Any]:
    output_root = Path(args.tslanet_output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    Path(args.tslanet_report_path).parent.mkdir(parents=True, exist_ok=True)
    patch_presets = unique_list(args.tslanet_patch_presets)
    training_regimes = unique_list(args.tslanet_training_regimes)
    if "day" in args.scales:
        raise ValueError("The TSLANet baseline intentionally supports only second/minute/hour.")
    for preset_name in patch_presets:
        validate_patch_preset_intraday(preset_name)
    print(
        f"tslanet_baseline patch_presets={patch_presets} regimes={training_regimes}",
        flush=True,
    )

    round1_rows: list[dict[str, Any]] = []
    round1_records: list[dict[str, Any]] = []
    configs = make_tslanet_config_matrix(training_regimes)
    for patch_preset in patch_presets:
        rows, records = run_selected_ablation_configs(
            args,
            preset="small",
            round_name="round1_small",
            patch_preset=patch_preset,
            configs=configs,
            seed=int(args.seed),
            device=device,
            output_dir=output_root / "round1_small" / patch_preset,
        )
        round1_rows.extend(rows)
        round1_records.extend(records)
    round1_summary = save_summary(round1_rows, output_root / "round1_small_summary.csv")

    round2_summary = pd.DataFrame()
    round2_records: list[dict[str, Any]] = []
    if not args.skip_tslanet_full:
        full_rows: list[dict[str, Any]] = []
        for patch_preset in patch_presets:
            rows, records = run_selected_ablation_configs(
                args,
                preset="full",
                round_name="round2_full",
                patch_preset=patch_preset,
                configs=configs,
                seed=int(args.seed),
                device=device,
                output_dir=output_root / "round2_full" / patch_preset,
            )
            full_rows.extend(rows)
            round2_records.extend(records)
        round2_summary = save_summary(full_rows, output_root / "round2_full_summary.csv")

    diagnostics = diagnostics_frame(round1_records + round2_records)
    diagnostics.to_csv(output_root / "diagnostics.csv", index=False)
    payload = {
        "round1_summary": round1_summary,
        "round2_summary": round2_summary,
        "diagnostics": diagnostics,
    }
    (output_root / "tslanet_baseline_metrics.json").write_text(
        json.dumps(to_jsonable(payload), indent=2),
        encoding="utf-8",
    )
    write_tslanet_report(
        args=args,
        output_root=output_root,
        round1_summary=round1_summary,
        round2_summary=round2_summary,
        diagnostics=diagnostics,
    )
    return payload


def make_tslanet_config_matrix(training_regimes: list[str]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if "raw_joint" in training_regimes:
        configs.append({"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None})
    if "asd_frozen_encoder_train_head" in training_regimes:
        configs.append(
            {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None}
        )
    if "tslanet_joint" in training_regimes:
        configs.append({"training_regime": "tslanet_joint", "init_gate": None, "adapter_rank": None})
    return dedupe_configs(configs)


def make_asb_config_matrix(training_regimes: list[str], init_gates: list[float]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if "raw_joint" in training_regimes:
        configs.append({"training_regime": "raw_joint", "init_gate": None})
    if "asd_frozen_encoder_train_head" in training_regimes:
        configs.append({"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0})
    for regime in training_regimes:
        if not regime.startswith("asb_encoder_"):
            continue
        for init_gate in init_gates:
            configs.append({"training_regime": regime, "init_gate": float(init_gate)})
    return configs


def make_lora_moe_config_matrix(training_regimes: list[str], ranks: list[int]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if "raw_joint" in training_regimes:
        configs.append({"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None})
    if "asd_frozen_encoder_train_head" in training_regimes:
        configs.append(
            {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None}
        )
    for regime in training_regimes:
        if not regime.startswith("lora_moe_"):
            continue
        for rank in ranks:
            configs.append({"training_regime": regime, "init_gate": None, "adapter_rank": int(rank)})
    return configs


def make_adapter_ablation_config_matrix(training_regimes: list[str], ranks: list[int]) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if "raw_joint" in training_regimes:
        configs.append({"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None})
    if "raw_frozen_base_train_head" in training_regimes:
        configs.append({"training_regime": "raw_frozen_base_train_head", "init_gate": None, "adapter_rank": None})
    if "asd_frozen_encoder_train_head" in training_regimes:
        configs.append(
            {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None}
        )
    for regime in training_regimes:
        if not (
            regime.startswith("lora_only_")
            or regime.startswith("mlp_moe_")
            or regime.startswith("lora_moe_")
            or regime.startswith("asd_lora_only_")
            or regime.startswith("asd_mlp_moe_")
            or regime.startswith("asd_lora_moe_")
        ):
            continue
        for rank in ranks:
            init_gate = -4.0 if regime.startswith("asd_") else None
            configs.append({"training_regime": regime, "init_gate": init_gate, "adapter_rank": int(rank)})
    return dedupe_configs(configs)


def make_asd_lora_moe_config_matrix(
    training_regimes: list[str],
    ranks: list[int],
    init_gates: list[float],
) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    if "raw_joint" in training_regimes:
        configs.append({"training_regime": "raw_joint", "init_gate": None, "adapter_rank": None})
    if "asd_frozen_encoder_train_head" in training_regimes:
        configs.append(
            {"training_regime": "asd_frozen_encoder_train_head", "init_gate": -4.0, "adapter_rank": None}
        )
    if "lora_moe_frozen_base_train_moe_head" in training_regimes:
        for rank in ranks:
            configs.append(
                {
                    "training_regime": "lora_moe_frozen_base_train_moe_head",
                    "init_gate": None,
                    "adapter_rank": int(rank),
                }
            )
    for regime in training_regimes:
        if not regime.startswith("asd_lora_moe_"):
            continue
        for rank in ranks:
            for init_gate in init_gates:
                configs.append(
                    {
                        "training_regime": regime,
                        "init_gate": float(init_gate),
                        "adapter_rank": int(rank),
                    }
                )
    return dedupe_configs(configs)


def dedupe_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[str, str, str]] = set()
    out: list[dict[str, Any]] = []
    for config in configs:
        key = (
            str(config.get("training_regime")),
            "none" if config.get("init_gate") is None else f"{float(config['init_gate']):.6f}",
            "none" if config.get("adapter_rank") is None else str(int(config["adapter_rank"])),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(config)
    return out


def run_ablation_patch_matrix(
    args: argparse.Namespace,
    *,
    preset: str,
    round_name: str,
    patch_preset: str,
    training_regimes: list[str],
    init_gates: list[float],
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    configs: list[dict[str, Any]] = []
    if "raw_joint" in training_regimes:
        configs.append({"training_regime": "raw_joint", "init_gate": None})
    for regime in training_regimes:
        if regime == "raw_joint":
            continue
        for init_gate in init_gates:
            configs.append({"training_regime": regime, "init_gate": float(init_gate)})
    return run_selected_ablation_configs(
        args,
        preset=preset,
        round_name=round_name,
        patch_preset=patch_preset,
        configs=configs,
        seed=seed,
        device=device,
        output_dir=output_dir,
    )


def run_selected_ablation_configs(
    args: argparse.Namespace,
    *,
    preset: str,
    round_name: str,
    patch_preset: str,
    configs: list[dict[str, Any]],
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    context = prepare_ablation_context(
        args,
        preset=preset,
        round_name=round_name,
        patch_preset=patch_preset,
        seed=seed,
        device=device,
        output_dir=output_dir,
    )
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    baseline_extra = context["row_extra"] | {
        "training_regime": "baseline",
        "init_gate": None,
        "adapter_rank": None,
        "checkpoint": "",
    }
    append_baseline_rows(rows, preset, context["scale_data"], extra=baseline_extra)

    config_regimes = {str(config["training_regime"]) for config in configs}
    needs_raw = True
    raw_checkpoint: str | None = None
    if needs_raw:
        raw_result, raw_record = train_ablation_config(
            context,
            training_regime="raw_joint",
            init_gate=None,
            adapter_rank=None,
            raw_checkpoint=None,
        )
        raw_checkpoint = str(raw_result["checkpoint"])
        if "raw_joint" in config_regimes or needs_raw:
            append_model_rows(rows, preset, "raw_joint", raw_result, extra=raw_record["row_extra"])
            records.append(raw_record)

    for config in configs:
        regime = str(config["training_regime"])
        if regime == "raw_joint":
            continue
        init_gate = None if config.get("init_gate") is None else float(config["init_gate"])
        adapter_rank = None if config.get("adapter_rank") is None else int(config["adapter_rank"])
        result, record = train_ablation_config(
            context,
            training_regime=regime,
            init_gate=init_gate,
            adapter_rank=adapter_rank,
            raw_checkpoint=raw_checkpoint,
        )
        append_model_rows(rows, preset, regime, result, extra=record["row_extra"])
        records.append(record)
    return rows, records


def prepare_ablation_context(
    args: argparse.Namespace,
    *,
    preset: str,
    round_name: str,
    patch_preset: str,
    seed: int,
    device: torch.device,
    output_dir: Path,
) -> dict[str, Any]:
    run_args = clone_args(args, patch_preset=patch_preset, seed=seed)
    cache_path = Path(run_args.small_cache if preset == "small" else run_args.full_cache)
    caps = caps_for_preset(run_args, preset)
    epochs = int(run_args.small_epochs if preset == "small" else run_args.full_epochs)
    requested_steps = int(run_args.small_steps_per_epoch if preset == "small" else run_args.full_steps_per_epoch)
    scale_specs = make_scale_specs(run_args)

    print(f"\n{round_name} preset={preset} patch_preset={patch_preset} seed={seed}", flush=True)
    print(f"cache={cache_path}", flush=True)
    print(f"specs={ {name: asdict(scale_specs[name]) for name in run_args.scales} }", flush=True)
    set_seed(seed)
    scale_data = load_scale_data(run_args, cache_path=cache_path, caps=caps, scale_specs=scale_specs)
    steps_per_epoch = resolve_steps_per_epoch(scale_data, run_args.batch_size, requested_steps)
    loaders = make_all_loaders(scale_data, batch_size=run_args.batch_size, device=device)
    row_extra = {
        "round": round_name,
        "patch_preset": patch_preset,
        "seed": seed,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
    }
    return {
        "args": run_args,
        "preset": preset,
        "round_name": round_name,
        "patch_preset": patch_preset,
        "seed": seed,
        "cache_path": cache_path,
        "caps": caps,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "scale_specs": scale_specs,
        "scale_data": scale_data,
        "loaders": loaders,
        "device": device,
        "output_dir": output_dir,
        "row_extra": row_extra,
    }


def train_ablation_config(
    context: dict[str, Any],
    *,
    training_regime: str,
    init_gate: float | None,
    raw_checkpoint: str | None,
    adapter_rank: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    args = context["args"]
    device = context["device"]
    if training_regime in {"raw_joint", "raw_frozen_base_train_head"}:
        model_name = "raw_patchtst"
    elif training_regime == "tslanet_joint":
        model_name = "tslanet"
    elif training_regime.startswith("asb_encoder_"):
        model_name = "asb_encoder_patchtst"
    elif training_regime.startswith("asd_lora_moe_"):
        model_name = "asd_lora_moe_patchtst"
    elif training_regime.startswith("asd_lora_only_"):
        model_name = "asd_lora_adapter_patchtst"
    elif training_regime.startswith("asd_mlp_moe_"):
        model_name = "asd_mlp_moe_patchtst"
    elif training_regime.startswith("lora_only_"):
        model_name = "lora_adapter_patchtst"
    elif training_regime.startswith("mlp_moe_"):
        model_name = "mlp_moe_patchtst"
    elif training_regime.startswith("lora_moe_"):
        model_name = "lora_moe_patchtst"
    else:
        model_name = "scale_aware_asd_patchtst"
    effective_rank = int(adapter_rank if adapter_rank is not None else args.lora_moe_rank)
    run_args = clone_args(
        args,
        scale_aware_init_gate=(-3.0 if init_gate is None else float(init_gate)),
        encoder_spectral_mode=("last1" if model_name == "asb_encoder_patchtst" else "none"),
        encoder_spectral_init_gate=(-4.0 if init_gate is None else float(init_gate)),
        lora_moe_mode={
            "lora_moe_patchtst": "last1",
            "asd_lora_moe_patchtst": "last1",
            "lora_adapter_patchtst": "lora_only",
            "mlp_moe_patchtst": "mlp_moe",
            "asd_lora_adapter_patchtst": "lora_only",
            "asd_mlp_moe_patchtst": "mlp_moe",
        }.get(model_name, "none"),
        lora_moe_rank=effective_rank,
    )
    set_seed(int(context["seed"]))
    model = build_model(model_name, run_args, context["scale_specs"]).to(device)
    load_info: dict[str, Any] = {}
    if training_regime in {
        "raw_frozen_base_train_head",
        "asd_only_frozen_backbone",
        "asd_frozen_encoder_train_head",
        "asb_encoder_frozen_base_train_asb_only",
        "asb_encoder_frozen_base_train_asb_head",
        "lora_only_frozen_base_train_adapter_only",
        "lora_only_frozen_base_train_adapter_head",
        "mlp_moe_frozen_base_train_moe_only",
        "mlp_moe_frozen_base_train_moe_head",
        "lora_moe_frozen_base_train_moe_only",
        "lora_moe_frozen_base_train_moe_head",
        "asd_lora_only_frozen_base_train_adapter_head",
        "asd_mlp_moe_frozen_base_train_moe_head",
        "asd_lora_moe_frozen_base_train_adapters_only",
        "asd_lora_moe_frozen_base_train_adapters_head",
    }:
        if raw_checkpoint is None:
            raise RuntimeError(f"{training_regime} requires a raw PatchTST checkpoint.")
        load_info = load_raw_backbone_checkpoint(model, Path(raw_checkpoint))
    freeze_info = apply_training_regime(model, training_regime)
    validate_training_regime(model, training_regime)
    label = safe_run_label(
        context["round_name"],
        context["patch_preset"],
        training_regime,
        gate_label(init_gate),
        rank_label(adapter_rank),
        f"seed_{context['seed']}",
    )
    print(
        f"training {label} trainable={freeze_info['trainable_parameters']} "
        f"total={freeze_info['total_parameters']}",
        flush=True,
    )
    result = train_model(
        model=model,
        model_name=training_regime,
        scale_data=context["scale_data"],
        loaders=context["loaders"],
        epochs=context["epochs"],
        steps_per_epoch=context["steps_per_epoch"],
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=device,
        output_dir=context["output_dir"],
        checkpoint_name=label,
        router_balance_weight=float(args.router_balance_weight),
    )
    result["training_regime"] = training_regime
    result["init_gate"] = init_gate
    result["adapter_rank"] = adapter_rank
    result["load_info"] = load_info
    result["freeze_info"] = freeze_info
    result["patch_preset"] = context["patch_preset"]
    result["seed"] = context["seed"]
    row_extra = context["row_extra"] | {
        "training_regime": training_regime,
        "init_gate": init_gate,
        "adapter_rank": adapter_rank,
        "checkpoint": result["checkpoint"],
        "elapsed_seconds": result["elapsed_seconds"],
        "trainable_parameters": freeze_info["trainable_parameters"],
    }
    record = {
        "round": context["round_name"],
        "preset": context["preset"],
        "patch_preset": context["patch_preset"],
        "training_regime": training_regime,
        "init_gate": init_gate,
        "adapter_rank": adapter_rank,
        "seed": context["seed"],
        "row_extra": row_extra,
        "result": result,
    }
    return result, record


def clone_args(args: argparse.Namespace, **updates: Any) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(updates)
    return argparse.Namespace(**values)


def unique_list(values: Any) -> list[Any]:
    out: list[Any] = []
    for value in values:
        if value not in out:
            out.append(value)
    return out


def validate_patch_preset_intraday(preset_name: str) -> None:
    preset = PATCH_PRESETS[preset_name]
    if set(preset) - set(SCALE_ORDER):
        raise ValueError(f"{preset_name}: patch preset contains non-intraday scales.")
    for scale in SCALE_ORDER:
        values = preset[scale]
        spec = ScaleSpec(
            name=scale,
            scale_id=DEFAULT_SCALE_SPECS[scale].scale_id,
            delta_seconds=DEFAULT_SCALE_SPECS[scale].delta_seconds,
            context_length=int(values["context_length"]),
            patch_length=int(values["patch_length"]),
            patch_stride=int(values["patch_stride"]),
            prediction_length=DEFAULT_SCALE_SPECS[scale].prediction_length,
        )
        validate_scale_spec(spec)


def save_summary(rows: list[dict[str, Any]], path: Path) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    if not frame.empty:
        sort_columns = [
            column
            for column in ["round", "preset", "patch_preset", "split", "scale", "model", "init_gate", "adapter_rank"]
            if column in frame
        ]
        frame = frame.sort_values(sort_columns, key=sort_summary_key).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"saved_summary={path}", flush=True)
    return frame


def load_raw_backbone_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, Any]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    raw_state = checkpoint.get("model", checkpoint)
    model_state = model.state_dict()
    compatible: dict[str, torch.Tensor] = {}
    skipped = 0
    for key, value in raw_state.items():
        if key.startswith("backbone.") and key in model_state and tuple(model_state[key].shape) == tuple(value.shape):
            compatible[key] = value
        elif key.startswith("backbone."):
            skipped += 1
    if not compatible:
        raise RuntimeError(f"No compatible backbone tensors found in {checkpoint_path}.")
    model_state.update(compatible)
    model.load_state_dict(model_state)
    return {
        "source_checkpoint": str(checkpoint_path),
        "loaded_backbone_tensors": len(compatible),
        "skipped_backbone_tensors": skipped,
    }


def apply_training_regime(model: torch.nn.Module, training_regime: str) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = True

    if training_regime == "raw_joint":
        pass
    elif training_regime == "raw_frozen_base_train_head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    elif training_regime == "asd_joint":
        pass
    elif training_regime == "asb_encoder_joint":
        pass
    elif training_regime == "tslanet_joint":
        pass
    elif training_regime == "lora_moe_joint":
        pass
    elif training_regime == "lora_only_joint":
        pass
    elif training_regime == "mlp_moe_joint":
        pass
    elif training_regime == "asd_lora_moe_joint":
        pass
    elif training_regime == "asd_only_frozen_backbone":
        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.denoiser.parameters():
            parameter.requires_grad = True
    elif training_regime == "asd_frozen_encoder_train_head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.denoiser.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    elif training_regime == "asb_encoder_frozen_base_train_asb_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "encoder_spectral", None) is None:
            raise RuntimeError("ASB training regime requires backbone.encoder_spectral.")
        for parameter in model.backbone.encoder_spectral.parameters():
            parameter.requires_grad = True
    elif training_regime == "asb_encoder_frozen_base_train_asb_head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "encoder_spectral", None) is None:
            raise RuntimeError("ASB training regime requires backbone.encoder_spectral.")
        for parameter in model.backbone.encoder_spectral.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    elif training_regime == "lora_moe_frozen_base_train_moe_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("LoRA-MoE training regime requires backbone.lora_moe.")
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
    elif training_regime == "lora_moe_frozen_base_train_moe_head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("LoRA-MoE training regime requires backbone.lora_moe.")
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    elif training_regime == "lora_only_frozen_base_train_adapter_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("LoRA-only training regime requires backbone.lora_moe.")
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
    elif training_regime == "lora_only_frozen_base_train_adapter_head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("LoRA-only training regime requires backbone.lora_moe.")
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    elif training_regime == "mlp_moe_frozen_base_train_moe_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("MLP-MoE training regime requires backbone.lora_moe.")
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
    elif training_regime == "mlp_moe_frozen_base_train_moe_head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("MLP-MoE training regime requires backbone.lora_moe.")
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    elif training_regime == "asd_lora_moe_frozen_base_train_adapters_only":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if not hasattr(model, "denoiser"):
            raise RuntimeError("ASD+LoRA-MoE training regime requires model.denoiser.")
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("ASD+LoRA-MoE training regime requires backbone.lora_moe.")
        for parameter in model.denoiser.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
    elif training_regime == "asd_lora_moe_frozen_base_train_adapters_head":
        for parameter in model.parameters():
            parameter.requires_grad = False
        if not hasattr(model, "denoiser"):
            raise RuntimeError("ASD+LoRA-MoE training regime requires model.denoiser.")
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError("ASD+LoRA-MoE training regime requires backbone.lora_moe.")
        for parameter in model.denoiser.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    elif training_regime in {
        "asd_lora_only_frozen_base_train_adapter_head",
        "asd_mlp_moe_frozen_base_train_moe_head",
    }:
        for parameter in model.parameters():
            parameter.requires_grad = False
        if not hasattr(model, "denoiser"):
            raise RuntimeError(f"{training_regime} requires model.denoiser.")
        if getattr(model.backbone, "lora_moe", None) is None:
            raise RuntimeError(f"{training_regime} requires backbone.lora_moe.")
        for parameter in model.denoiser.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.lora_moe.parameters():
            parameter.requires_grad = True
        for parameter in model.backbone.heads.parameters():
            parameter.requires_grad = True
    else:
        raise ValueError(f"unknown training_regime={training_regime!r}")

    counts = count_parameters(model)
    return {
        "total_parameters": int(counts["total"]),
        "trainable_parameters": int(counts["trainable"]),
    }


def validate_training_regime(model: torch.nn.Module, training_regime: str) -> None:
    named = dict(model.named_parameters())
    if training_regime == "raw_joint":
        if not all(parameter.requires_grad for parameter in named.values()):
            raise RuntimeError("raw_joint should train all raw PatchTST parameters.")
        return
    if training_regime == "raw_frozen_base_train_head":
        head_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.heads.")
        ]
        non_head_frozen = all(
            not parameter.requires_grad
            for name, parameter in named.items()
            if not name.startswith("backbone.heads.")
        )
        if not head_trainable or not all(head_trainable) or not non_head_frozen:
            raise RuntimeError("raw_frozen_base_train_head: only scale heads should train.")
        return
    if training_regime == "asd_joint":
        if not all(parameter.requires_grad for parameter in named.values()):
            raise RuntimeError("asd_joint should train ASD and PatchTST jointly.")
        return
    if training_regime == "asb_encoder_joint":
        spectral_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.encoder_spectral.")
        ]
        if not spectral_trainable or not all(parameter.requires_grad for parameter in named.values()):
            raise RuntimeError("asb_encoder_joint should train ASB and PatchTST jointly.")
        return
    if training_regime == "tslanet_joint":
        spectral_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if ".spectral." in name
        ]
        icb_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if ".icb." in name
        ]
        if not spectral_trainable or not icb_trainable or not all(parameter.requires_grad for parameter in named.values()):
            raise RuntimeError("tslanet_joint should train all TSLANet-style baseline parameters.")
        return
    if training_regime == "lora_moe_joint":
        moe_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.lora_moe.")
        ]
        if not moe_trainable or not all(parameter.requires_grad for parameter in named.values()):
            raise RuntimeError("lora_moe_joint should train LoRA-MoE and PatchTST jointly.")
        return
    if training_regime in {"lora_only_joint", "mlp_moe_joint"}:
        adapter_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.lora_moe.")
        ]
        if not adapter_trainable or not all(parameter.requires_grad for parameter in named.values()):
            raise RuntimeError(f"{training_regime} should train the adapter and PatchTST jointly.")
        return
    if training_regime == "asd_lora_moe_joint":
        denoiser_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("denoiser.")
        ]
        moe_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.lora_moe.")
        ]
        if not denoiser_trainable or not moe_trainable or not all(parameter.requires_grad for parameter in named.values()):
            raise RuntimeError("asd_lora_moe_joint should train ASD, LoRA-MoE, and PatchTST jointly.")
        return
    if training_regime in {"asb_encoder_frozen_base_train_asb_only", "asb_encoder_frozen_base_train_asb_head"}:
        spectral_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.encoder_spectral.")
        ]
        if not spectral_trainable or not all(spectral_trainable):
            raise RuntimeError(f"{training_regime}: ASB parameters must be trainable.")
        if training_regime == "asb_encoder_frozen_base_train_asb_only":
            non_asb_frozen = all(
                not parameter.requires_grad
                for name, parameter in named.items()
                if not name.startswith("backbone.encoder_spectral.")
            )
            if not non_asb_frozen:
                raise RuntimeError("asb_encoder_frozen_base_train_asb_only: only ASB should train.")
            return
        head_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.heads.")
        ]
        non_asb_head_frozen = all(
            not parameter.requires_grad
            for name, parameter in named.items()
            if not name.startswith("backbone.encoder_spectral.") and not name.startswith("backbone.heads.")
        )
        if not head_trainable or not all(head_trainable) or not non_asb_head_frozen:
            raise RuntimeError("asb_encoder_frozen_base_train_asb_head: only ASB and heads should train.")
        return
    if training_regime in {"lora_moe_frozen_base_train_moe_only", "lora_moe_frozen_base_train_moe_head"}:
        moe_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.lora_moe.")
        ]
        if not moe_trainable or not all(moe_trainable):
            raise RuntimeError(f"{training_regime}: LoRA-MoE parameters must be trainable.")
        if training_regime == "lora_moe_frozen_base_train_moe_only":
            non_moe_frozen = all(
                not parameter.requires_grad
                for name, parameter in named.items()
                if not name.startswith("backbone.lora_moe.")
            )
            if not non_moe_frozen:
                raise RuntimeError("lora_moe_frozen_base_train_moe_only: only LoRA-MoE should train.")
            return
        head_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.heads.")
        ]
        non_moe_head_frozen = all(
            not parameter.requires_grad
            for name, parameter in named.items()
            if not name.startswith("backbone.lora_moe.") and not name.startswith("backbone.heads.")
        )
        if not head_trainable or not all(head_trainable) or not non_moe_head_frozen:
            raise RuntimeError("lora_moe_frozen_base_train_moe_head: only LoRA-MoE and heads should train.")
        return
    if training_regime in {
        "lora_only_frozen_base_train_adapter_only",
        "lora_only_frozen_base_train_adapter_head",
        "mlp_moe_frozen_base_train_moe_only",
        "mlp_moe_frozen_base_train_moe_head",
    }:
        adapter_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.lora_moe.")
        ]
        if not adapter_trainable or not all(adapter_trainable):
            raise RuntimeError(f"{training_regime}: adapter parameters must be trainable.")
        trains_head = training_regime.endswith("_head")
        if not trains_head:
            non_adapter_frozen = all(
                not parameter.requires_grad
                for name, parameter in named.items()
                if not name.startswith("backbone.lora_moe.")
            )
            if not non_adapter_frozen:
                raise RuntimeError(f"{training_regime}: only the adapter should train.")
            return
        head_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.heads.")
        ]
        non_adapter_head_frozen = all(
            not parameter.requires_grad
            for name, parameter in named.items()
            if not name.startswith("backbone.lora_moe.") and not name.startswith("backbone.heads.")
        )
        if not head_trainable or not all(head_trainable) or not non_adapter_head_frozen:
            raise RuntimeError(f"{training_regime}: only the adapter and heads should train.")
        return
    if training_regime in {
        "asd_lora_moe_frozen_base_train_adapters_only",
        "asd_lora_moe_frozen_base_train_adapters_head",
        "asd_lora_only_frozen_base_train_adapter_head",
        "asd_mlp_moe_frozen_base_train_moe_head",
    }:
        denoiser_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("denoiser.")
        ]
        moe_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.lora_moe.")
        ]
        if not denoiser_trainable or not all(denoiser_trainable):
            raise RuntimeError(f"{training_regime}: ASD parameters must be trainable.")
        if not moe_trainable or not all(moe_trainable):
            raise RuntimeError(f"{training_regime}: LoRA-MoE parameters must be trainable.")
        if training_regime == "asd_lora_moe_frozen_base_train_adapters_only":
            only_adapters = all(
                parameter.requires_grad
                for name, parameter in named.items()
                if name.startswith("denoiser.") or name.startswith("backbone.lora_moe.")
            ) and all(
                not parameter.requires_grad
                for name, parameter in named.items()
                if not name.startswith("denoiser.") and not name.startswith("backbone.lora_moe.")
            )
            if not only_adapters:
                raise RuntimeError("asd_lora_moe_frozen_base_train_adapters_only: only ASD and LoRA-MoE should train.")
            return
        head_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.heads.")
        ]
        only_adapters_head = all(
            not parameter.requires_grad
            for name, parameter in named.items()
            if not name.startswith("denoiser.")
            and not name.startswith("backbone.lora_moe.")
            and not name.startswith("backbone.heads.")
        )
        if not head_trainable or not all(head_trainable) or not only_adapters_head:
            raise RuntimeError("asd_lora_moe_frozen_base_train_adapters_head: only ASD, LoRA-MoE, and heads should train.")
        return
    denoiser_trainable = [
        parameter.requires_grad
        for name, parameter in named.items()
        if name.startswith("denoiser.")
    ]
    if not denoiser_trainable or not all(denoiser_trainable):
        raise RuntimeError(f"{training_regime}: ASD parameters must be trainable.")
    if training_regime == "asd_only_frozen_backbone":
        frozen_ok = all(
            not parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.")
        )
        if not frozen_ok:
            raise RuntimeError("asd_only_frozen_backbone: backbone parameters must be frozen.")
        return
    if training_regime == "asd_frozen_encoder_train_head":
        head_trainable = [
            parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.heads.")
        ]
        non_head_backbone_frozen = all(
            not parameter.requires_grad
            for name, parameter in named.items()
            if name.startswith("backbone.") and not name.startswith("backbone.heads.")
        )
        if not head_trainable or not all(head_trainable) or not non_head_backbone_frozen:
            raise RuntimeError("asd_frozen_encoder_train_head: only ASD and scale heads should train.")
        return
    raise ValueError(f"unknown training_regime={training_regime!r}")


def safe_run_label(*parts: Any) -> str:
    label = "__".join(str(part) for part in parts if part is not None and str(part))
    for old, new in [("-", "m"), (".", "p"), (" ", "_"), ("/", "_"), ("\\", "_")]:
        label = label.replace(old, new)
    return label


def gate_label(init_gate: float | None) -> str:
    if init_gate is None:
        return "gate_none"
    return f"gate_{float(init_gate):.1f}"


def rank_label(adapter_rank: int | None) -> str:
    if adapter_rank is None:
        return "rank_none"
    return f"rank_{int(adapter_rank)}"


def needs_long_context(summary: pd.DataFrame) -> bool:
    if summary.empty or "long_context" in set(summary.get("patch_preset", [])):
        return False
    candidate_rows = summary[
        (summary["split"] == "validation")
        & (~summary["model"].isin(["zero", "last_return", "raw_joint"]))
        & (summary["scale"] == "second")
    ]
    for _, candidate in candidate_rows.iterrows():
        raw = matching_summary_row(summary, candidate, model="raw_joint", scale="second", split="validation")
        if raw is None:
            continue
        if float(candidate["mse"]) <= float(raw["mse"]) * 1.01 and float(candidate["direction_accuracy_nonzero"]) >= float(raw["direction_accuracy_nonzero"]) - 0.02:
            return False
    return True


def matching_summary_row(
    summary: pd.DataFrame,
    candidate: pd.Series | dict[str, Any],
    *,
    model: str,
    scale: str,
    split: str,
) -> pd.Series | None:
    rows = summary[
        (summary["patch_preset"] == candidate["patch_preset"])
        & (summary["seed"] == candidate["seed"])
        & (summary["split"] == split)
        & (summary["scale"] == scale)
        & (summary["model"] == model)
    ]
    candidate_model = candidate.get("model", candidate.get("training_regime", None))
    if (
        model == candidate_model
        and "init_gate" in summary.columns
        and "init_gate" in candidate
        and not pd.isna(candidate["init_gate"])
    ):
        rows = rows[rows["init_gate"].astype(float).round(6) == float(candidate["init_gate"])]
    if (
        model == candidate_model
        and "adapter_rank" in summary.columns
        and "adapter_rank" in candidate
        and not pd.isna(candidate["adapter_rank"])
    ):
        rows = rows[rows["adapter_rank"].astype(float).round(6) == float(candidate["adapter_rank"])]
    if rows.empty:
        return None
    return rows.iloc[0]


def select_top_ablation_configs(summary: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    candidates = summary[
        (summary["split"] == "validation")
        & (~summary["model"].isin(["zero", "last_return", "raw_joint"]))
    ][["patch_preset", "model", "init_gate", "seed"]].drop_duplicates()
    scored: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        key_filter = (
            (summary["patch_preset"] == candidate["patch_preset"])
            & (summary["model"] == candidate["model"])
            & (summary["seed"] == candidate["seed"])
            & (summary["split"] == "validation")
        )
        if not pd.isna(candidate["init_gate"]):
            key_filter &= summary["init_gate"].astype(float).round(6) == float(candidate["init_gate"])
        model_rows = summary[key_filter]
        if set(model_rows["scale"]) < set(SCALE_ORDER):
            continue
        metrics = {str(row["scale"]): row for _, row in model_rows.iterrows()}
        raw = {
            scale: matching_summary_row(summary, candidate, model="raw_joint", scale=scale, split="validation")
            for scale in SCALE_ORDER
        }
        zero = {
            scale: matching_summary_row(summary, candidate, model="zero", scale=scale, split="validation")
            for scale in SCALE_ORDER
        }
        if any(value is None for value in raw.values()) or zero["hour"] is None:
            continue
        second_mse_ratio = float(metrics["second"]["mse"]) / max(float(raw["second"]["mse"]), 1e-20)
        second_dir_delta = float(metrics["second"]["direction_accuracy_nonzero"]) - float(raw["second"]["direction_accuracy_nonzero"])
        minute_mse_ratio = float(metrics["minute"]["mse"]) / max(float(raw["minute"]["mse"]), 1e-20)
        minute_dir_delta = float(metrics["minute"]["direction_accuracy_nonzero"]) - float(raw["minute"]["direction_accuracy_nonzero"])
        minute_corr_delta = float(metrics["minute"]["corr"]) - float(raw["minute"]["corr"])
        hour_nmse_delta = float(raw["hour"]["nmse"]) - float(metrics["hour"]["nmse"])
        hour_vs_zero_delta = float(zero["hour"]["nmse"]) - float(metrics["hour"]["nmse"])
        second_ok = second_mse_ratio <= 1.01 and second_dir_delta >= -0.02
        minute_ok = minute_mse_ratio <= 1.03 or minute_dir_delta >= 0.0 or minute_corr_delta >= 0.0
        hour_ok = hour_nmse_delta > 0.0 and hour_vs_zero_delta > 0.0
        selection_score = (
            2.0 * hour_nmse_delta
            + 0.75 * minute_dir_delta
            + 0.25 * minute_corr_delta
            - 0.5 * max(second_mse_ratio - 1.0, 0.0)
            - 0.2 * max(minute_mse_ratio - 1.0, 0.0)
        )
        scored.append(
            {
                "patch_preset": str(candidate["patch_preset"]),
                "training_regime": str(candidate["model"]),
                "init_gate": None if pd.isna(candidate["init_gate"]) else float(candidate["init_gate"]),
                "seed": int(candidate["seed"]),
                "selection_score": float(selection_score),
                "quality_pass": bool(second_ok and minute_ok and hour_ok),
                "second_mse_over_raw": float(second_mse_ratio),
                "second_dir_delta": float(second_dir_delta),
                "minute_mse_over_raw": float(minute_mse_ratio),
                "minute_dir_delta": float(minute_dir_delta),
                "minute_corr_delta": float(minute_corr_delta),
                "hour_nmse_delta_raw_minus_model": float(hour_nmse_delta),
                "hour_nmse_delta_zero_minus_model": float(hour_vs_zero_delta),
            }
        )
    scored.sort(
        key=lambda item: (
            bool(item["quality_pass"]),
            float(item["selection_score"]),
            float(item["hour_nmse_delta_raw_minus_model"]),
        ),
        reverse=True,
    )
    return scored[: max(1, top_k)]


def select_top_asb_configs(summary: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    candidates = summary[
        (summary["split"] == "validation")
        & (summary["model"].astype(str).str.startswith("asb_encoder_"))
    ][["patch_preset", "model", "init_gate", "seed"]].drop_duplicates()
    scored: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        key_filter = (
            (summary["patch_preset"] == candidate["patch_preset"])
            & (summary["model"] == candidate["model"])
            & (summary["seed"] == candidate["seed"])
            & (summary["split"] == "validation")
        )
        if not pd.isna(candidate["init_gate"]):
            key_filter &= summary["init_gate"].astype(float).round(6) == float(candidate["init_gate"])
        model_rows = summary[key_filter]
        if set(model_rows["scale"]) < set(SCALE_ORDER):
            continue
        metrics = {str(row["scale"]): row for _, row in model_rows.iterrows()}
        raw = {
            scale: matching_summary_row(summary, candidate, model="raw_joint", scale=scale, split="validation")
            for scale in SCALE_ORDER
        }
        zero_hour = matching_summary_row(summary, candidate, model="zero", scale="hour", split="validation")
        asd_hour = matching_summary_row(
            summary,
            candidate,
            model="asd_frozen_encoder_train_head",
            scale="hour",
            split="validation",
        )
        if any(value is None for value in raw.values()) or zero_hour is None:
            continue
        second_mse_ratio = float(metrics["second"]["mse"]) / max(float(raw["second"]["mse"]), 1e-20)
        second_dir_delta = float(metrics["second"]["direction_accuracy_nonzero"]) - float(raw["second"]["direction_accuracy_nonzero"])
        minute_mse_ratio = float(metrics["minute"]["mse"]) / max(float(raw["minute"]["mse"]), 1e-20)
        minute_dir_delta = float(metrics["minute"]["direction_accuracy_nonzero"]) - float(raw["minute"]["direction_accuracy_nonzero"])
        minute_corr_delta = float(metrics["minute"]["corr"]) - float(raw["minute"]["corr"])
        hour_nmse_delta = float(raw["hour"]["nmse"]) - float(metrics["hour"]["nmse"])
        hour_vs_zero_delta = float(zero_hour["nmse"]) - float(metrics["hour"]["nmse"])
        hour_vs_asd_delta = (
            float(asd_hour["nmse"]) - float(metrics["hour"]["nmse"])
            if asd_hour is not None
            else float("nan")
        )
        second_ok = second_mse_ratio <= 1.01 and second_dir_delta >= -0.01
        minute_ok = minute_mse_ratio <= 1.02 and (minute_dir_delta >= 0.0 or minute_corr_delta >= 0.0)
        hour_ok = hour_nmse_delta > 0.0 and hour_vs_zero_delta > 0.0
        selection_score = (
            2.0 * hour_nmse_delta
            + 0.75 * np.nan_to_num(hour_vs_asd_delta, nan=0.0)
            + 0.5 * minute_dir_delta
            + 0.25 * minute_corr_delta
            - 0.75 * max(second_mse_ratio - 1.0, 0.0)
            - 0.3 * max(minute_mse_ratio - 1.0, 0.0)
        )
        scored.append(
            {
                "patch_preset": str(candidate["patch_preset"]),
                "training_regime": str(candidate["model"]),
                "init_gate": None if pd.isna(candidate["init_gate"]) else float(candidate["init_gate"]),
                "seed": int(candidate["seed"]),
                "selection_score": float(selection_score),
                "quality_pass": bool(second_ok and minute_ok and hour_ok),
                "second_mse_over_raw": float(second_mse_ratio),
                "second_dir_delta": float(second_dir_delta),
                "minute_mse_over_raw": float(minute_mse_ratio),
                "minute_dir_delta": float(minute_dir_delta),
                "minute_corr_delta": float(minute_corr_delta),
                "hour_nmse_delta_raw_minus_model": float(hour_nmse_delta),
                "hour_nmse_delta_asd_minus_model": float(hour_vs_asd_delta),
                "hour_nmse_delta_zero_minus_model": float(hour_vs_zero_delta),
            }
        )
    scored.sort(
        key=lambda item: (
            bool(item["quality_pass"]),
            float(item["selection_score"]),
            float(item["hour_nmse_delta_raw_minus_model"]),
        ),
        reverse=True,
    )
    return scored[: max(1, top_k)]


def select_top_lora_moe_configs(summary: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    candidate_columns = ["patch_preset", "model", "seed"]
    if "adapter_rank" in summary.columns:
        candidate_columns.append("adapter_rank")
    candidates = summary[
        (summary["split"] == "validation")
        & (summary["model"].astype(str).str.startswith("lora_moe_"))
    ][candidate_columns].drop_duplicates()
    scored: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        key_filter = (
            (summary["patch_preset"] == candidate["patch_preset"])
            & (summary["model"] == candidate["model"])
            & (summary["seed"] == candidate["seed"])
            & (summary["split"] == "validation")
        )
        if "adapter_rank" in candidate and not pd.isna(candidate["adapter_rank"]):
            key_filter &= summary["adapter_rank"].astype(float).round(6) == float(candidate["adapter_rank"])
        model_rows = summary[key_filter]
        if set(model_rows["scale"]) < set(SCALE_ORDER):
            continue
        metrics = {str(row["scale"]): row for _, row in model_rows.iterrows()}
        raw = {
            scale: matching_summary_row(summary, candidate, model="raw_joint", scale=scale, split="validation")
            for scale in SCALE_ORDER
        }
        zero_hour = matching_summary_row(summary, candidate, model="zero", scale="hour", split="validation")
        asd_hour = matching_summary_row(
            summary,
            candidate,
            model="asd_frozen_encoder_train_head",
            scale="hour",
            split="validation",
        )
        if any(value is None for value in raw.values()) or zero_hour is None:
            continue
        second_mse_ratio = float(metrics["second"]["mse"]) / max(float(raw["second"]["mse"]), 1e-20)
        second_dir_delta = float(metrics["second"]["direction_accuracy_nonzero"]) - float(raw["second"]["direction_accuracy_nonzero"])
        minute_mse_ratio = float(metrics["minute"]["mse"]) / max(float(raw["minute"]["mse"]), 1e-20)
        minute_dir_delta = float(metrics["minute"]["direction_accuracy_nonzero"]) - float(raw["minute"]["direction_accuracy_nonzero"])
        minute_corr_delta = float(metrics["minute"]["corr"]) - float(raw["minute"]["corr"])
        hour_nmse_delta = float(raw["hour"]["nmse"]) - float(metrics["hour"]["nmse"])
        hour_vs_zero_delta = float(zero_hour["nmse"]) - float(metrics["hour"]["nmse"])
        hour_vs_asd_delta = (
            float(asd_hour["nmse"]) - float(metrics["hour"]["nmse"])
            if asd_hour is not None
            else float("nan")
        )
        second_ok = second_mse_ratio <= 1.01 and second_dir_delta >= -0.01
        minute_ok = minute_mse_ratio <= 1.02 and (minute_dir_delta >= 0.0 or minute_corr_delta >= 0.0)
        hour_ok = hour_nmse_delta > 0.0 and hour_vs_zero_delta > 0.0
        selection_score = (
            2.0 * hour_nmse_delta
            + 0.75 * np.nan_to_num(hour_vs_asd_delta, nan=0.0)
            + 0.5 * minute_dir_delta
            + 0.25 * minute_corr_delta
            - 0.75 * max(second_mse_ratio - 1.0, 0.0)
            - 0.3 * max(minute_mse_ratio - 1.0, 0.0)
        )
        scored.append(
            {
                "patch_preset": str(candidate["patch_preset"]),
                "training_regime": str(candidate["model"]),
                "init_gate": None,
                "adapter_rank": None
                if "adapter_rank" not in candidate or pd.isna(candidate["adapter_rank"])
                else int(candidate["adapter_rank"]),
                "seed": int(candidate["seed"]),
                "selection_score": float(selection_score),
                "quality_pass": bool(second_ok and minute_ok and hour_ok),
                "strong_pass": bool(second_ok and minute_ok and hour_ok and hour_vs_asd_delta > 0.0),
                "second_mse_over_raw": float(second_mse_ratio),
                "second_dir_delta": float(second_dir_delta),
                "minute_mse_over_raw": float(minute_mse_ratio),
                "minute_dir_delta": float(minute_dir_delta),
                "minute_corr_delta": float(minute_corr_delta),
                "hour_nmse_delta_raw_minus_model": float(hour_nmse_delta),
                "hour_nmse_delta_asd_minus_model": float(hour_vs_asd_delta),
                "hour_nmse_delta_zero_minus_model": float(hour_vs_zero_delta),
            }
        )
    scored.sort(
        key=lambda item: (
            bool(item["quality_pass"]),
            bool(item["strong_pass"]),
            float(item["selection_score"]),
            float(item["hour_nmse_delta_raw_minus_model"]),
        ),
        reverse=True,
    )
    return scored[: max(1, top_k)]


def select_top_asd_lora_moe_configs(summary: pd.DataFrame, *, top_k: int) -> list[dict[str, Any]]:
    if summary.empty:
        return []
    candidate_columns = ["patch_preset", "model", "seed", "init_gate"]
    if "adapter_rank" in summary.columns:
        candidate_columns.append("adapter_rank")
    candidates = summary[
        (summary["split"] == "validation")
        & (summary["model"].astype(str).str.startswith("asd_lora_moe_"))
    ][candidate_columns].drop_duplicates()
    scored: list[dict[str, Any]] = []
    for _, candidate in candidates.iterrows():
        key_filter = (
            (summary["patch_preset"] == candidate["patch_preset"])
            & (summary["model"] == candidate["model"])
            & (summary["seed"] == candidate["seed"])
            & (summary["split"] == "validation")
        )
        if not pd.isna(candidate["init_gate"]):
            key_filter &= summary["init_gate"].astype(float).round(6) == float(candidate["init_gate"])
        if "adapter_rank" in candidate and not pd.isna(candidate["adapter_rank"]):
            key_filter &= summary["adapter_rank"].astype(float).round(6) == float(candidate["adapter_rank"])
        model_rows = summary[key_filter]
        if set(model_rows["scale"]) < set(SCALE_ORDER):
            continue
        metrics = {str(row["scale"]): row for _, row in model_rows.iterrows()}
        raw = {
            scale: matching_summary_row(summary, candidate, model="raw_joint", scale=scale, split="validation")
            for scale in SCALE_ORDER
        }
        zero_hour = matching_summary_row(summary, candidate, model="zero", scale="hour", split="validation")
        asd_hour = matching_summary_row(
            summary,
            candidate,
            model="asd_frozen_encoder_train_head",
            scale="hour",
            split="validation",
        )
        if any(value is None for value in raw.values()) or zero_hour is None or asd_hour is None:
            continue
        second_mse_ratio = float(metrics["second"]["mse"]) / max(float(raw["second"]["mse"]), 1e-20)
        second_dir_delta = float(metrics["second"]["direction_accuracy_nonzero"]) - float(raw["second"]["direction_accuracy_nonzero"])
        minute_mse_ratio = float(metrics["minute"]["mse"]) / max(float(raw["minute"]["mse"]), 1e-20)
        minute_dir_delta = float(metrics["minute"]["direction_accuracy_nonzero"]) - float(raw["minute"]["direction_accuracy_nonzero"])
        minute_corr_delta = float(metrics["minute"]["corr"]) - float(raw["minute"]["corr"])
        hour_nmse_delta = float(raw["hour"]["nmse"]) - float(metrics["hour"]["nmse"])
        hour_vs_zero_delta = float(zero_hour["nmse"]) - float(metrics["hour"]["nmse"])
        hour_vs_asd_delta = float(asd_hour["nmse"]) - float(metrics["hour"]["nmse"])
        hour_over_asd = float(metrics["hour"]["nmse"]) / max(float(asd_hour["nmse"]), 1e-20)
        second_ok = second_mse_ratio <= 1.01 and second_dir_delta >= -0.01
        minute_ok = minute_mse_ratio <= 1.02 and (minute_dir_delta >= 0.0 or minute_corr_delta >= 0.0)
        hour_ok = hour_nmse_delta > 0.0 and hour_vs_zero_delta > 0.0 and hour_over_asd <= 1.01
        selection_score = (
            2.0 * hour_nmse_delta
            + 0.75 * hour_vs_asd_delta
            + 0.5 * minute_dir_delta
            + 0.25 * minute_corr_delta
            - 0.75 * max(second_mse_ratio - 1.0, 0.0)
            - 0.3 * max(minute_mse_ratio - 1.0, 0.0)
            - 0.5 * max(hour_over_asd - 1.0, 0.0)
        )
        scored.append(
            {
                "patch_preset": str(candidate["patch_preset"]),
                "training_regime": str(candidate["model"]),
                "init_gate": None if pd.isna(candidate["init_gate"]) else float(candidate["init_gate"]),
                "adapter_rank": None
                if "adapter_rank" not in candidate or pd.isna(candidate["adapter_rank"])
                else int(candidate["adapter_rank"]),
                "seed": int(candidate["seed"]),
                "selection_score": float(selection_score),
                "quality_pass": bool(second_ok and minute_ok and hour_ok),
                "strong_pass": bool(second_ok and minute_ok and hour_ok and hour_vs_asd_delta > 0.0),
                "second_mse_over_raw": float(second_mse_ratio),
                "second_dir_delta": float(second_dir_delta),
                "minute_mse_over_raw": float(minute_mse_ratio),
                "minute_dir_delta": float(minute_dir_delta),
                "minute_corr_delta": float(minute_corr_delta),
                "hour_mse_over_asd": float(hour_over_asd),
                "hour_nmse_delta_raw_minus_model": float(hour_nmse_delta),
                "hour_nmse_delta_asd_minus_model": float(hour_vs_asd_delta),
                "hour_nmse_delta_zero_minus_model": float(hour_vs_zero_delta),
            }
        )
    scored.sort(
        key=lambda item: (
            bool(item["quality_pass"]),
            bool(item["strong_pass"]),
            float(item["selection_score"]),
            float(item["hour_nmse_delta_raw_minus_model"]),
        ),
        reverse=True,
    )
    return scored[: max(1, top_k)] if scored else []


def select_best_confirmed_config(round2_summary: pd.DataFrame, selection: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not round2_summary.empty:
        full_selection = select_top_ablation_configs(round2_summary, top_k=1)
        if full_selection:
            return full_selection[0]
    return selection[0] if selection else None


def select_best_confirmed_asb_config(round2_summary: pd.DataFrame, selection: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not round2_summary.empty:
        full_selection = select_top_asb_configs(round2_summary, top_k=1)
        if full_selection:
            return full_selection[0]
    return selection[0] if selection else None


def select_best_confirmed_lora_moe_config(
    round2_summary: pd.DataFrame,
    selection: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not round2_summary.empty:
        full_selection = select_top_lora_moe_configs(round2_summary, top_k=1)
        if full_selection:
            return full_selection[0]
    return selection[0] if selection else None


def select_best_confirmed_asd_lora_moe_config(
    round2_summary: pd.DataFrame,
    selection: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if not round2_summary.empty:
        full_selection = select_top_asd_lora_moe_configs(round2_summary, top_k=1)
        if full_selection:
            return full_selection[0]
    passed = [item for item in selection if bool(item.get("quality_pass"))]
    return passed[0] if passed else None


def aggregate_robustness(summary: pd.DataFrame) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    rows = summary[
        (summary["split"] == "test")
        & (~summary["model"].isin(["zero", "last_return"]))
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    group_cols = [
        column
        for column in ["patch_preset", "model", "init_gate", "adapter_rank", "scale"]
        if column in rows.columns
    ]
    metrics = ["mse", "nmse", "mae", "direction_accuracy_nonzero", "corr"]
    grouped = rows.groupby(group_cols, dropna=False)[metrics].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    return grouped


def summarize_router_usage(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = diagnostics[
        diagnostics["training_regime"].astype(str).isin(
            [
                "lora_moe_frozen_base_train_moe_head",
                TARGETED_ASD_LORA_MOE_REGIME,
            ]
        )
    ].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["router_entropy_used"] = rows.get("moe_router_entropy", pd.Series(index=rows.index, dtype=float)).combine_first(
        rows.get("router_entropy", pd.Series(index=rows.index, dtype=float))
    )
    rows["router_balance_loss_used"] = rows.get(
        "moe_router_balance_loss",
        pd.Series(index=rows.index, dtype=float),
    ).combine_first(rows.get("router_balance_loss", pd.Series(index=rows.index, dtype=float)))
    for expert_idx in range(4):
        rows[f"expert_{expert_idx}_used"] = rows.get(
            f"moe_expert_prob_{expert_idx}",
            pd.Series(index=rows.index, dtype=float),
        ).combine_first(rows.get(f"expert_prob_{expert_idx}", pd.Series(index=rows.index, dtype=float)))

    metric_cols = ["router_entropy_used", "router_balance_loss_used"] + [
        f"expert_{expert_idx}_used" for expert_idx in range(4)
    ]
    grouped = rows.groupby(["training_regime", "adapter_rank", "scale"], dropna=False)[metric_cols].mean().reset_index()
    return grouped.rename(
        columns={
            "router_entropy_used": "router_entropy",
            "router_balance_loss_used": "router_balance_loss",
            **{f"expert_{idx}_used": f"expert_{idx}" for idx in range(4)},
        }
    )


def summarize_asd_gate_tau(diagnostics: pd.DataFrame) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    rows = diagnostics[diagnostics["training_regime"].astype(str).str.startswith("asd_")].copy()
    if rows.empty:
        return pd.DataFrame()
    rows["gate_used"] = rows.get("asd_gate_mean", pd.Series(index=rows.index, dtype=float)).combine_first(
        rows.get("gate_mean", pd.Series(index=rows.index, dtype=float))
    )
    rows["tau_used"] = rows.get("asd_tau_mean", pd.Series(index=rows.index, dtype=float)).combine_first(
        rows.get("tau_mean", pd.Series(index=rows.index, dtype=float))
    )
    rows["mean_abs_delta_used"] = rows.get(
        "asd_mean_abs_delta",
        pd.Series(index=rows.index, dtype=float),
    ).combine_first(rows.get("mean_abs_delta", pd.Series(index=rows.index, dtype=float)))
    grouped = rows.groupby(["training_regime", "adapter_rank", "scale"], dropna=False)[
        ["gate_used", "tau_used", "mean_abs_delta_used"]
    ].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in grouped.columns
    ]
    return grouped.rename(
        columns={
            "gate_used_mean": "gate_mean",
            "gate_used_std": "gate_std",
            "tau_used_mean": "tau_mean",
            "tau_used_std": "tau_std",
            "mean_abs_delta_used_mean": "mean_abs_delta_mean",
            "mean_abs_delta_used_std": "mean_abs_delta_std",
        }
    )


def diagnostics_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        diagnostics = record["result"].get("diagnostics", {})
        for scale, values in diagnostics.items():
            rows.append(
                {
                    "round": record["round"],
                    "preset": record["preset"],
                    "patch_preset": record["patch_preset"],
                    "training_regime": record["training_regime"],
                    "init_gate": record["init_gate"],
                    "adapter_rank": record.get("adapter_rank"),
                    "seed": record["seed"],
                    "scale": scale,
                    "gate_mean": values.get("gate_mean", float("nan")),
                    "tau_mean": values.get("tau_mean", float("nan")),
                    "local_mask_mean": values.get("local_mask_mean", float("nan")),
                    "mean_abs_delta": values.get("mean_abs_delta", float("nan")),
                    "asd_gate_mean": values.get("asd_gate_mean", float("nan")),
                    "asd_tau_mean": values.get("asd_tau_mean", float("nan")),
                    "asd_mean_abs_delta": values.get("asd_mean_abs_delta", float("nan")),
                    "global_filter_norm": values.get("global_filter_norm", float("nan")),
                    "local_filter_norm": values.get("local_filter_norm", float("nan")),
                    "router_entropy": values.get("router_entropy", float("nan")),
                    "router_balance_loss": values.get("router_balance_loss", float("nan")),
                    "expert_prob_0": values.get("expert_prob_0", float("nan")),
                    "expert_prob_1": values.get("expert_prob_1", float("nan")),
                    "expert_prob_2": values.get("expert_prob_2", float("nan")),
                    "expert_prob_3": values.get("expert_prob_3", float("nan")),
                    "scale_prior_prob_0": values.get("scale_prior_prob_0", float("nan")),
                    "scale_prior_prob_1": values.get("scale_prior_prob_1", float("nan")),
                    "scale_prior_prob_2": values.get("scale_prior_prob_2", float("nan")),
                    "scale_prior_prob_3": values.get("scale_prior_prob_3", float("nan")),
                    "moe_router_entropy": values.get("moe_router_entropy", float("nan")),
                    "moe_router_balance_loss": values.get("moe_router_balance_loss", float("nan")),
                    "moe_expert_prob_0": values.get("moe_expert_prob_0", float("nan")),
                    "moe_expert_prob_1": values.get("moe_expert_prob_1", float("nan")),
                    "moe_expert_prob_2": values.get("moe_expert_prob_2", float("nan")),
                    "moe_expert_prob_3": values.get("moe_expert_prob_3", float("nan")),
                    "moe_scale_prior_prob_0": values.get("moe_scale_prior_prob_0", float("nan")),
                    "moe_scale_prior_prob_1": values.get("moe_scale_prior_prob_1", float("nan")),
                    "moe_scale_prior_prob_2": values.get("moe_scale_prior_prob_2", float("nan")),
                    "moe_scale_prior_prob_3": values.get("moe_scale_prior_prob_3", float("nan")),
                    "moe_mean_abs_delta": values.get("moe_mean_abs_delta", float("nan")),
                    "tslanet_gate_mean": values.get("tslanet_gate_mean", float("nan")),
                    "tslanet_tau_mean": values.get("tslanet_tau_mean", float("nan")),
                    "tslanet_local_mask_mean": values.get("tslanet_local_mask_mean", float("nan")),
                    "tslanet_mean_abs_delta": values.get("tslanet_mean_abs_delta", float("nan")),
                    "tslanet_global_filter_norm": values.get("tslanet_global_filter_norm", float("nan")),
                    "tslanet_local_filter_norm": values.get("tslanet_local_filter_norm", float("nan")),
                    "checkpoint": record["result"].get("checkpoint", ""),
                }
            )
    return pd.DataFrame(rows)


def build_per_scale_oracle(
    lora_summary: pd.DataFrame,
    output_root: Path,
    *,
    current_source: str = "lora_moe_current_full",
) -> pd.DataFrame:
    candidates: list[pd.DataFrame] = []
    if not lora_summary.empty:
        current = lora_summary[
            (lora_summary["split"] == "test")
            & (
                lora_summary["model"].isin(["raw_joint", "asd_frozen_encoder_train_head"])
                | lora_summary["model"].astype(str).str.startswith("lora_moe_")
                | lora_summary["model"].astype(str).str.startswith("lora_only_")
                | lora_summary["model"].astype(str).str.startswith("mlp_moe_")
                | lora_summary["model"].astype(str).str.startswith("asd_lora_moe_")
            )
        ].copy()
        if not current.empty:
            current["source"] = current_source
            candidates.append(current)

    asb_path = WORKSPACE_ROOT / "outputs" / "scale_aware_asb_encoder_patchtst" / "round2_full_summary.csv"
    if asb_path.exists():
        asb = pd.read_csv(asb_path)
        asb = asb[
            (asb["split"] == "test")
            & (asb["model"].isin(["raw_joint", "asd_frozen_encoder_train_head"]) | asb["model"].astype(str).str.startswith("asb_encoder_"))
        ].copy()
        if not asb.empty:
            asb["source"] = "asb_previous_full"
            candidates.append(asb)

    previous_lora_path = WORKSPACE_ROOT / "outputs" / "scale_aware_lora_moe_patchtst" / "round2_full_summary.csv"
    if previous_lora_path.exists() and output_root != previous_lora_path.parent:
        previous_lora = pd.read_csv(previous_lora_path)
        previous_lora = previous_lora[
            (previous_lora["split"] == "test")
            & (
                previous_lora["model"].isin(["raw_joint", "asd_frozen_encoder_train_head"])
                | previous_lora["model"].astype(str).str.startswith("lora_moe_")
            )
        ].copy()
        if not previous_lora.empty:
            previous_lora["source"] = "lora_moe_previous_full"
            candidates.append(previous_lora)

    if not candidates:
        oracle = pd.DataFrame()
    else:
        all_candidates = pd.concat(candidates, ignore_index=True, sort=False)
        oracle_rows: list[pd.Series] = []
        for scale in SCALE_ORDER:
            rows = all_candidates[all_candidates["scale"] == scale].copy()
            if rows.empty:
                continue
            oracle_rows.append(rows.sort_values("mse").iloc[0])
        oracle = pd.DataFrame(oracle_rows).reset_index(drop=True)
    oracle.to_csv(output_root / "per_scale_oracle.csv", index=False)
    return oracle


def load_csv_or_empty(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def validate_targeted_robustness_outputs(summary: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    if summary.empty:
        raise RuntimeError("targeted robustness produced an empty summary.")
    if set(summary["scale"].astype(str)) - set(SCALE_ORDER):
        raise RuntimeError("targeted robustness summary contains non-intraday scales.")
    required_models = {
        "raw_joint",
        "asd_frozen_encoder_train_head",
        "lora_moe_frozen_base_train_moe_head",
        TARGETED_ASD_LORA_MOE_REGIME,
    }
    for split in ["validation", "test", "zero_shot"]:
        for scale in SCALE_ORDER:
            rows = summary[(summary["split"] == split) & (summary["scale"] == scale)]
            missing = required_models - set(rows["model"].astype(str))
            if missing:
                raise RuntimeError(f"targeted robustness missing {split}/{scale}: {sorted(missing)}")
    metric_columns = ["mse", "rmse", "mae", "nmse", "direction_accuracy_nonzero"]
    metric_values = summary[metric_columns].to_numpy(dtype=float)
    if not np.isfinite(metric_values).all():
        raise RuntimeError("targeted robustness summary contains NaN/Inf metrics.")
    model_rows = summary[summary["model"].astype(str).isin(required_models)]
    corr_values = model_rows["corr"].to_numpy(dtype=float)
    if not np.isfinite(corr_values).all():
        raise RuntimeError("targeted robustness model corr contains NaN/Inf values.")
    if diagnostics.empty:
        raise RuntimeError("targeted robustness produced empty diagnostics.")
    if np.isinf(diagnostics.select_dtypes(include=[np.number]).to_numpy(dtype=float)).any():
        raise RuntimeError("targeted robustness diagnostics contain Inf values.")
    combined = diagnostics[diagnostics["training_regime"].astype(str) == TARGETED_ASD_LORA_MOE_REGIME]
    required_diag = [
        "asd_gate_mean",
        "asd_tau_mean",
        "asd_mean_abs_delta",
        "moe_router_entropy",
        "moe_router_balance_loss",
        "moe_expert_prob_0",
        "moe_expert_prob_1",
        "moe_expert_prob_2",
        "moe_expert_prob_3",
        "moe_mean_abs_delta",
    ]
    if combined.empty:
        raise RuntimeError("targeted robustness missing combined diagnostics.")
    combined_values = combined[required_diag].to_numpy(dtype=float)
    if not np.isfinite(combined_values).all():
        raise RuntimeError("targeted robustness combined diagnostics contain NaN/Inf values.")


def evaluate_asd_lora_moe_final_decision(robustness: pd.DataFrame) -> dict[str, Any]:
    if robustness.empty:
        return {
            "quality_pass": False,
            "strong_pass": False,
            "reason": "missing robustness aggregate",
            "scale_table": pd.DataFrame(),
            "fallback_unified_model": None,
            "per_scale_recommendation": pd.DataFrame(),
        }

    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for scale in SCALE_ORDER:
        raw = robustness_metric_row(robustness, "raw_joint", scale)
        asd = robustness_metric_row(robustness, "asd_frozen_encoder_train_head", scale)
        combined = robustness_metric_row(robustness, TARGETED_ASD_LORA_MOE_REGIME, scale)
        if raw is None or asd is None or combined is None:
            missing.append(scale)
            continue
        second_rule = minute_rule = hour_rule = True
        if scale == "second":
            second_rule = (
                float(combined["mse_mean"]) <= float(raw["mse_mean"]) * 1.01
                and float(combined["direction_accuracy_nonzero_mean"])
                >= float(raw["direction_accuracy_nonzero_mean"]) - 0.01
            )
            scale_pass = second_rule
        elif scale == "minute":
            minute_rule = (
                float(combined["mse_mean"]) <= float(raw["mse_mean"]) * 1.02
                and (
                    float(combined["direction_accuracy_nonzero_mean"])
                    >= float(raw["direction_accuracy_nonzero_mean"])
                    or float(combined["corr_mean"]) >= float(raw["corr_mean"])
                )
            )
            scale_pass = minute_rule
        else:
            hour_rule = (
                float(combined["nmse_mean"]) < float(raw["nmse_mean"])
                and float(combined["nmse_mean"]) <= float(asd["nmse_mean"]) * 1.01
            )
            scale_pass = hour_rule
        rows.append(
            {
                "scale": scale,
                "pass": bool(scale_pass),
                "raw_mse_mean": float(raw["mse_mean"]),
                "combined_mse_mean": float(combined["mse_mean"]),
                "combined_mse_over_raw": safe_ratio(float(combined["mse_mean"]), float(raw["mse_mean"])),
                "raw_nmse_mean": float(raw["nmse_mean"]),
                "asd_nmse_mean": float(asd["nmse_mean"]),
                "combined_nmse_mean": float(combined["nmse_mean"]),
                "combined_nmse_std": float(combined.get("nmse_std", float("nan"))),
                "combined_nmse_delta_raw_minus_model": float(raw["nmse_mean"]) - float(combined["nmse_mean"]),
                "combined_nmse_delta_asd_minus_model": float(asd["nmse_mean"]) - float(combined["nmse_mean"]),
                "raw_direction_mean": float(raw["direction_accuracy_nonzero_mean"]),
                "combined_direction_mean": float(combined["direction_accuracy_nonzero_mean"]),
                "combined_direction_delta": (
                    float(combined["direction_accuracy_nonzero_mean"])
                    - float(raw["direction_accuracy_nonzero_mean"])
                ),
                "raw_corr_mean": float(raw["corr_mean"]),
                "combined_corr_mean": float(combined["corr_mean"]),
                "combined_corr_delta": float(combined["corr_mean"]) - float(raw["corr_mean"]),
            }
        )

    scale_table = pd.DataFrame(rows)
    quality_pass = bool(not missing and not scale_table.empty and scale_table["pass"].all())
    hour_row = scale_table[scale_table["scale"] == "hour"].iloc[0] if "hour" in set(scale_table.get("scale", [])) else None
    second_or_minute_dir_gain = bool(
        not scale_table.empty
        and (
            scale_table[scale_table["scale"].isin(["second", "minute"])]["combined_direction_delta"] > 0.0
        ).any()
    )
    seed_std_warning = False
    if hour_row is not None:
        hour_margin = abs(float(hour_row["combined_nmse_delta_raw_minus_model"]))
        asd_margin = abs(float(hour_row["combined_nmse_delta_asd_minus_model"]))
        hour_std = float(hour_row["combined_nmse_std"])
        seed_std_warning = bool(np.isfinite(hour_std) and hour_std > max(min(hour_margin, asd_margin), 1e-20))
    strong_pass = bool(
        quality_pass
        and hour_row is not None
        and float(hour_row["combined_nmse_delta_asd_minus_model"]) > 0.0
        and second_or_minute_dir_gain
        and not seed_std_warning
    )
    fallback, per_scale = robustness_recommendations(robustness)
    if missing:
        reason = f"missing scale rows: {missing}"
    elif quality_pass and strong_pass:
        reason = "ASD+LoRA-MoE passes the unified gate and strong-pass criteria."
    elif quality_pass:
        reason = "ASD+LoRA-MoE passes the unified gate, but strong-pass criteria are not fully met."
    else:
        reason = "ASD+LoRA-MoE does not pass the unified three-scale gate."
    return {
        "quality_pass": quality_pass,
        "strong_pass": strong_pass,
        "seed_std_warning": seed_std_warning,
        "reason": reason,
        "scale_table": scale_table,
        "fallback_unified_model": fallback,
        "per_scale_recommendation": per_scale,
    }


def robustness_metric_row(robustness: pd.DataFrame, model: str, scale: str) -> pd.Series | None:
    rows = robustness[(robustness["model"] == model) & (robustness["scale"] == scale)]
    if model == TARGETED_ASD_LORA_MOE_REGIME and "adapter_rank" in rows:
        rows = rows[rows["adapter_rank"].astype(float).round(6) == float(TARGETED_ASD_LORA_MOE_RANK)]
    if rows.empty:
        return None
    return rows.iloc[0]


def robustness_recommendations(robustness: pd.DataFrame) -> tuple[str | None, pd.DataFrame]:
    rows = robustness[~robustness["model"].isin(["zero", "last_return"])].copy()
    if rows.empty:
        return None, pd.DataFrame()
    mean_nmse = rows.groupby("model", dropna=False)["nmse_mean"].mean().sort_values()
    non_combined = mean_nmse[~mean_nmse.index.astype(str).str.startswith("asd_lora_moe_")]
    fallback = str(non_combined.index[0]) if len(non_combined) else str(mean_nmse.index[0])
    per_scale_rows: list[pd.Series] = []
    for scale in SCALE_ORDER:
        scale_rows = rows[rows["scale"] == scale].sort_values("nmse_mean")
        if not scale_rows.empty:
            per_scale_rows.append(scale_rows.iloc[0])
    per_scale = pd.DataFrame(per_scale_rows).reset_index(drop=True) if per_scale_rows else pd.DataFrame()
    return fallback, per_scale


def caps_for_preset(args: argparse.Namespace, preset: str) -> dict[str, int]:
    prefix = "small" if preset == "small" else "full"
    return {
        "train": int(getattr(args, f"{prefix}_train_cap")),
        "validation": int(getattr(args, f"{prefix}_validation_cap")),
        "test": int(getattr(args, f"{prefix}_test_cap")),
        "zero_shot": int(getattr(args, f"{prefix}_zero_shot_cap")),
    }


def load_scale_data(
    args: argparse.Namespace,
    *,
    cache_path: Path,
    caps: dict[str, int],
    scale_specs: dict[str, ScaleSpec],
) -> dict[str, ScaleData]:
    train_stocks = parse_stock_list(args.train_stocks)
    out: dict[str, ScaleData] = {}
    for scale in args.scales:
        spec = scale_specs[scale]
        data = build_single_scale_datasets(
            cache_path=cache_path,
            scale=scale,
            train_stocks=train_stocks,
            zero_shot_stock=int(args.zero_shot_stock),
            feature_name="wap1_log_return_1s",
            target_horizon_steps=TARGET_HORIZON_STEPS[scale],
            context_length=spec.context_length,
            train_fraction=0.8,
            validation_fraction=0.1,
            caps=caps,
            seed=args.seed,
        )
        normalizer = fit_normalizer(data["train_x"], data["train_y"])
        out[scale] = ScaleData(name=scale, spec=spec, arrays=data, normalizer=normalizer)
        print(
            f"{scale}: train={len(data['train_y'])} validation={len(data['validation_y'])} "
            f"test={len(data['test_y'])} zero_shot={len(data['zero_shot_y'])}",
            flush=True,
        )
    return out


def make_scale_specs(args: argparse.Namespace) -> dict[str, ScaleSpec]:
    preset = PATCH_PRESETS[args.patch_preset]
    specs: dict[str, ScaleSpec] = {}
    for scale in args.scales:
        values = dict(preset[scale])
        for key in ["context_length", "patch_length", "patch_stride"]:
            override = getattr(args, f"{scale}_{key}")
            if override is not None:
                values[key] = int(override)
        base = DEFAULT_SCALE_SPECS[scale]
        spec = ScaleSpec(
            name=scale,
            scale_id=base.scale_id,
            delta_seconds=base.delta_seconds,
            context_length=int(values["context_length"]),
            patch_length=int(values["patch_length"]),
            patch_stride=int(values["patch_stride"]),
            prediction_length=base.prediction_length,
        )
        validate_scale_spec(spec)
        specs[scale] = spec
    return specs


def validate_scale_spec(spec: ScaleSpec) -> None:
    if spec.context_length <= 0 or spec.patch_length <= 0 or spec.patch_stride <= 0:
        raise ValueError(f"{spec.name}: context, patch, and stride must be positive.")
    if spec.patch_length > spec.context_length:
        raise ValueError(f"{spec.name}: patch_length cannot exceed context_length.")
    if spec.patch_count <= 0:
        raise ValueError(f"{spec.name}: patch_count must be positive.")


def build_model(
    model_name: str,
    args: argparse.Namespace,
    scale_specs: dict[str, ScaleSpec],
) -> torch.nn.Module:
    if model_name == "tslanet":
        return TSLANetMultiScaleForecaster(
            scale_specs,
            input_channels=1,
            d_model=args.d_model,
            n_layers=args.n_layers,
            dropout=args.dropout,
            spectral_init_gate=args.encoder_spectral_init_gate,
        )
    encoder_spectral_mode = "last1" if model_name == "asb_encoder_patchtst" else args.encoder_spectral_mode
    lora_moe_mode = {
        "lora_moe_patchtst": "last1",
        "asd_lora_moe_patchtst": "last1",
        "lora_adapter_patchtst": "lora_only",
        "mlp_moe_patchtst": "mlp_moe",
        "asd_lora_adapter_patchtst": "lora_only",
        "asd_mlp_moe_patchtst": "mlp_moe",
    }.get(model_name, "none")
    backbone = build_multiscale_patchtst(
        scale_specs,
        input_channels=1,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        encoder_spectral_mode=encoder_spectral_mode,
        encoder_spectral_init_gate=args.encoder_spectral_init_gate,
        lora_moe_mode=lora_moe_mode,
        lora_moe_rank=args.lora_moe_rank,
        lora_moe_alpha=args.lora_moe_alpha,
        lora_moe_n_experts=args.lora_moe_n_experts,
        lora_moe_top_k=args.lora_moe_top_k,
        lora_moe_dropout=args.lora_moe_dropout,
        head_type=getattr(args, "head_type", "linear"),
        head_hidden_dim=getattr(args, "head_hidden_dim", 128),
    )
    if model_name == "raw_patchtst":
        return RawMultiScalePatchTST(backbone)
    if model_name == "asb_encoder_patchtst":
        return RawMultiScalePatchTST(backbone)
    if model_name == "lora_moe_patchtst":
        return RawMultiScalePatchTST(backbone)
    if model_name == "lora_adapter_patchtst":
        return RawMultiScalePatchTST(backbone)
    if model_name == "mlp_moe_patchtst":
        return RawMultiScalePatchTST(backbone)
    if model_name == "asd_lora_moe_patchtst":
        return ScaleAwareASDMultiScalePatchTST(backbone, init_gate=args.scale_aware_init_gate)
    if model_name == "asd_lora_adapter_patchtst":
        return ScaleAwareASDMultiScalePatchTST(backbone, init_gate=args.scale_aware_init_gate)
    if model_name == "asd_mlp_moe_patchtst":
        return ScaleAwareASDMultiScalePatchTST(backbone, init_gate=args.scale_aware_init_gate)
    if model_name == "static_asd_patchtst":
        return StaticASDMultiScalePatchTST(
            backbone,
            keep_ratio=args.static_keep_ratio,
            blend_init=args.static_blend_init,
        )
    if model_name == "scale_aware_asd_patchtst":
        return ScaleAwareASDMultiScalePatchTST(backbone, init_gate=args.scale_aware_init_gate)
    raise ValueError(f"unknown model_name={model_name!r}")


def make_all_loaders(
    scale_data: dict[str, ScaleData],
    *,
    batch_size: int,
    device: torch.device,
) -> dict[str, dict[str, DataLoader]]:
    return {
        scale: {
            split: make_loader(
                data.arrays[f"{split}_x"],
                data.arrays[f"{split}_y"],
                data.normalizer,
                batch_size=batch_size,
                shuffle=(split == "train"),
                device=device,
            )
            for split in ["train", "validation", "test", "zero_shot"]
        }
        for scale, data in scale_data.items()
    }


def make_loader(
    x: np.ndarray,
    y: np.ndarray,
    normalizer: dict[str, float],
    *,
    batch_size: int,
    shuffle: bool,
    device: torch.device,
) -> DataLoader:
    x_norm = (np.asarray(x, dtype=np.float32) - normalizer["x_mean"]) / normalizer["x_std"]
    y_norm = (np.asarray(y, dtype=np.float32).reshape(-1, 1) - normalizer["y_mean"]) / normalizer["y_std"]
    dataset = TensorDataset(torch.as_tensor(x_norm, dtype=torch.float32), torch.as_tensor(y_norm, dtype=torch.float32))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )


def resolve_steps_per_epoch(
    scale_data: dict[str, ScaleData],
    batch_size: int,
    requested_steps: int,
) -> int:
    if requested_steps > 0:
        return requested_steps
    return max(1, max(math.ceil(len(data.arrays["train_y"]) / batch_size) for data in scale_data.values()))


def train_model(
    *,
    model: torch.nn.Module,
    model_name: str,
    scale_data: dict[str, ScaleData],
    loaders: dict[str, dict[str, DataLoader]],
    epochs: int,
    steps_per_epoch: int,
    learning_rate: float,
    weight_decay: float,
    device: torch.device,
    output_dir: Path,
    checkpoint_name: str | None = None,
    router_balance_weight: float = 0.0,
) -> dict[str, Any]:
    trainable_parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable_parameters:
        raise RuntimeError(f"{model_name}: no trainable parameters.")
    optimizer = torch.optim.AdamW(trainable_parameters, lr=learning_rate, weight_decay=weight_decay)
    best_state: dict[str, torch.Tensor] | None = None
    best_validation_nmse = float("inf")
    history: list[dict[str, float]] = []
    start = perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        train_iters = {scale: iter(loaders[scale]["train"]) for scale in scale_data}
        running_loss = 0.0
        for _ in range(steps_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            losses: list[torch.Tensor] = []
            router_losses: list[torch.Tensor] = []
            for scale in scale_data:
                xb, yb, train_iters[scale] = next_batch(loaders[scale]["train"], train_iters[scale])
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                out = model(xb, scale, return_diagnostics=True)
                if isinstance(out, tuple):
                    pred, diagnostics = out
                else:
                    pred, diagnostics = out, {}
                pred = pred.squeeze(1)
                loss = F.huber_loss(pred, yb, delta=1.0, reduction="mean")
                losses.append(loss)
                if router_balance_weight > 0.0 and "router_balance_loss" in diagnostics:
                    router_losses.append(diagnostics["router_balance_loss"])
            loss = torch.stack(losses).mean()
            if router_losses:
                loss = loss + float(router_balance_weight) * torch.stack(router_losses).mean()
            if not torch.isfinite(loss):
                raise RuntimeError(f"{model_name}: non-finite loss encountered.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 1.0)
            optimizer.step()
            running_loss += float(loss.detach().cpu())

        validation_metrics = evaluate_all_scales(model, scale_data, loaders, split="validation", device=device)
        validation_nmse = float(np.mean([metrics["nmse"] for metrics in validation_metrics.values()]))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss_scaled": running_loss / max(steps_per_epoch, 1),
                "validation_mean_nmse": validation_nmse,
            }
        )
        if validation_nmse < best_validation_nmse:
            best_validation_nmse = validation_nmse
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        print(
            f"{model_name} epoch={epoch}/{epochs} "
            f"train_loss={history[-1]['train_loss_scaled']:.6f} val_nmse={validation_nmse:.6f}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    elapsed = perf_counter() - start
    split_metrics = {
        split: evaluate_all_scales(model, scale_data, loaders, split=split, device=device)
        for split in ["validation", "test", "zero_shot"]
    }
    diagnostics = collect_diagnostics(model, loaders, device)
    checkpoint_path = output_dir / f"{checkpoint_name or model_name}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "model_name": model_name,
            "history": history,
            "split_metrics": split_metrics,
            "diagnostics": diagnostics,
        },
        checkpoint_path,
    )
    return {
        "model": model_name,
        "parameters": count_parameters(model),
        "checkpoint": str(checkpoint_path),
        "elapsed_seconds": float(elapsed),
        "history": history,
        "best_validation_mean_nmse": best_validation_nmse,
        "split_metrics": split_metrics,
        "diagnostics": diagnostics,
    }


def next_batch(
    loader: DataLoader,
    iterator: Any,
) -> tuple[torch.Tensor, torch.Tensor, Any]:
    try:
        xb, yb = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        xb, yb = next(iterator)
    return xb, yb, iterator


def forward_prediction(model: torch.nn.Module, xb: torch.Tensor, scale: str) -> torch.Tensor:
    out = model(xb, scale)
    if isinstance(out, tuple):
        return out[0]
    return out


@torch.no_grad()
def evaluate_all_scales(
    model: torch.nn.Module,
    scale_data: dict[str, ScaleData],
    loaders: dict[str, dict[str, DataLoader]],
    *,
    split: str,
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    return {
        scale: evaluate_single_scale(
            model,
            scale=scale,
            loader=loaders[scale][split],
            normalizer=data.normalizer,
            device=device,
        )
        for scale, data in scale_data.items()
    }


@torch.no_grad()
def evaluate_single_scale(
    model: torch.nn.Module,
    *,
    scale: str,
    loader: DataLoader,
    normalizer: dict[str, float],
    device: torch.device,
) -> dict[str, float]:
    preds: list[np.ndarray] = []
    actuals: list[np.ndarray] = []
    for xb, yb in loader:
        pred_scaled = forward_prediction(model, xb.to(device, non_blocking=True), scale).squeeze(1)
        pred = pred_scaled.detach().cpu().numpy().reshape(-1) * normalizer["y_std"] + normalizer["y_mean"]
        actual = yb.numpy().reshape(-1) * normalizer["y_std"] + normalizer["y_mean"]
        preds.append(pred.astype(np.float32))
        actuals.append(actual.astype(np.float32))
    pred_arr = np.concatenate(preds)
    actual_arr = np.concatenate(actuals)
    return metric_with_nmse(pred_arr, actual_arr)


@torch.no_grad()
def collect_diagnostics(
    model: torch.nn.Module,
    loaders: dict[str, dict[str, DataLoader]],
    device: torch.device,
) -> dict[str, dict[str, float]]:
    model.eval()
    diagnostics: dict[str, dict[str, float]] = {}
    for scale, split_loaders in loaders.items():
        xb, _ = next(iter(split_loaders["validation"]))
        out = model(xb.to(device, non_blocking=True), scale, return_diagnostics=True)
        diag = out[1] if isinstance(out, tuple) else {}
        diagnostics[scale] = {
            key: float(value.detach().cpu())
            for key, value in diag.items()
        }
    return diagnostics


def append_baseline_rows(
    rows: list[dict[str, Any]],
    preset: str,
    scale_data: dict[str, ScaleData],
    extra: dict[str, Any] | None = None,
) -> None:
    for scale, data in scale_data.items():
        for split in ["validation", "test", "zero_shot"]:
            actual = np.asarray(data.arrays[f"{split}_y"], dtype=np.float32)
            last_return = np.asarray(data.arrays[f"{split}_last_return"], dtype=np.float32)
            baselines = {
                "zero": metric_with_nmse(np.zeros_like(actual), actual),
                "last_return": metric_with_nmse(last_return, actual),
            }
            for model, metrics in baselines.items():
                rows.append(metric_row(preset, split, scale, model, metrics, extra=extra))


def append_model_rows(
    rows: list[dict[str, Any]],
    preset: str,
    model_name: str,
    result: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    for split, scale_metrics in result["split_metrics"].items():
        for scale, metrics in scale_metrics.items():
            rows.append(metric_row(preset, split, scale, model_name, metrics, extra=extra))


def metric_with_nmse(prediction: np.ndarray, actual: np.ndarray) -> dict[str, float]:
    metrics = metric_dict(prediction, actual)
    variance = float(np.var(np.asarray(actual, dtype=np.float64).reshape(-1)))
    metrics["nmse"] = float(metrics["mse"] / max(variance, 1e-20))
    metrics["n"] = float(len(np.asarray(actual).reshape(-1)))
    return metrics


def metric_row(
    preset: str,
    split: str,
    scale: str,
    model: str,
    metrics: dict[str, float],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    row = {
        "preset": preset,
        "split": split,
        "scale": scale,
        "model": model,
        "n": int(metrics["n"]),
        "mse": metrics["mse"],
        "rmse": metrics["rmse"],
        "mae": metrics["mae"],
        "nmse": metrics["nmse"],
        "direction_accuracy_nonzero": metrics["direction_accuracy_nonzero"],
        "corr": metrics["corr"],
    }
    if extra:
        row.update(extra)
    return row


def estimate_full_from_small(args: argparse.Namespace, small_result: dict[str, Any]) -> dict[str, Any]:
    small_summary: pd.DataFrame = small_result["summary"]
    reference = load_full_reference(Path(args.full_reference_summary))
    sample_counts = load_full_sample_counts()
    rows: list[dict[str, Any]] = []
    for split in ["validation", "test", "zero_shot"]:
        for scale in args.scales:
            small_raw = select_metric_row(small_summary, split, scale, "raw_patchtst")
            small_sa = select_metric_row(small_summary, split, scale, "scale_aware_asd_patchtst")
            ref = reference.get((scale, split))
            if ref and args.patch_preset == "compact":
                source = "existing_full_raw_reference"
            elif ref:
                source = "existing_full_raw_reference_context_proxy"
            else:
                source = "small_proxy_no_full_reference"
            raw_mse = ref["raw_mse"] if ref else small_raw["mse"]
            raw_mae = ref["raw_mae"] if ref else small_raw["mae"]
            raw_dir = ref["raw_dir"] if ref else small_raw["direction_accuracy_nonzero"]
            raw_corr = ref["raw_corr"] if ref else small_raw["corr"]
            mse_ratio = safe_ratio(small_sa["mse"], small_raw["mse"])
            mae_ratio = safe_ratio(small_sa["mae"], small_raw["mae"])
            rows.append(
                {
                    "split": split,
                    "scale": scale,
                    "source": source,
                    "full_reference_raw_mse": raw_mse,
                    "estimated_scale_aware_mse": raw_mse * mse_ratio,
                    "full_reference_raw_mae": raw_mae,
                    "estimated_scale_aware_mae": raw_mae * mae_ratio,
                    "estimated_direction_accuracy_nonzero": raw_dir
                    + (small_sa["direction_accuracy_nonzero"] - small_raw["direction_accuracy_nonzero"]),
                    "estimated_corr": raw_corr + (small_sa["corr"] - small_raw["corr"]),
                    "small_mse_ratio_scale_aware_over_raw": mse_ratio,
                    "small_mae_ratio_scale_aware_over_raw": mae_ratio,
                    "full_sample_count": sample_counts.get((scale, split), float("nan")),
                }
            )
    small_elapsed = small_result["models"]["scale_aware_asd_patchtst"]["elapsed_seconds"]
    small_total_steps = max(1, int(args.small_epochs) * int(small_result["steps_per_epoch"]))
    full_total_steps = max(1, int(args.full_epochs) * int(args.full_steps_per_epoch))
    runtime_estimate = float(small_elapsed * full_total_steps / small_total_steps)
    estimate = {
        "rows": rows,
        "estimated_full_runtime_seconds": runtime_estimate,
        "runtime_formula": "small_scale_aware_elapsed * full_train_steps / small_train_steps",
    }
    estimate_path = Path(args.output_dir) / "small" / "full_estimate.csv"
    pd.DataFrame(rows).to_csv(estimate_path, index=False)
    (Path(args.output_dir) / "small" / "full_estimate.json").write_text(
        json.dumps(to_jsonable(estimate), indent=2),
        encoding="utf-8",
    )
    return estimate


def load_full_reference(path: Path) -> dict[tuple[str, str], dict[str, float]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    out: dict[tuple[str, str], dict[str, float]] = {}
    for scale, run in FULL_REFERENCE_RUNS.items():
        rows = frame[frame["run"] == run]
        for _, row in rows.iterrows():
            out[(scale, str(row["split"]))] = {
                "raw_mse": float(row["raw_mse"]),
                "raw_mae": float(row["raw_mae"]),
                "raw_dir": float(row["raw_dir"]),
                "raw_corr": float(row["raw_corr"]),
            }
    return out


def load_full_sample_counts() -> dict[tuple[str, str], int]:
    counts: dict[tuple[str, str], int] = {}
    for scale, path in FULL_REFERENCE_METRICS.items():
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        meta = metrics.get("data_meta", {})
        for split in ["validation", "test", "zero_shot"]:
            key = f"{split}_evaluated_windows"
            if key in meta:
                counts[(scale, split)] = int(meta[key])
    return counts


def evaluate_quality_gate(small_result: dict[str, Any], scales: list[str]) -> dict[str, Any]:
    summary: pd.DataFrame = small_result["summary"]
    reasons: list[str] = []
    warnings: list[str] = []
    required = {(split, scale) for split in ["validation", "test", "zero_shot"] for scale in scales}
    present = {
        (str(row["split"]), str(row["scale"]))
        for _, row in summary[summary["model"] == "scale_aware_asd_patchtst"].iterrows()
    }
    missing = sorted(required - present)
    if missing:
        reasons.append(f"missing scale-aware metrics: {missing}")

    numeric = summary[["mse", "rmse", "mae", "nmse", "direction_accuracy_nonzero"]].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        reasons.append("summary contains non-finite numeric metrics")

    diagnostics = small_result["models"]["scale_aware_asd_patchtst"]["diagnostics"]
    for scale, diag in diagnostics.items():
        values = np.asarray(list(diag.values()), dtype=np.float64)
        if len(values) and not np.isfinite(values).all():
            reasons.append(f"{scale} ASD diagnostics contain non-finite values")

    worse_scales: list[str] = []
    for scale in scales:
        raw = select_metric_row(summary, "validation", scale, "raw_patchtst")
        scale_aware = select_metric_row(summary, "validation", scale, "scale_aware_asd_patchtst")
        if scale_aware["mse"] > raw["mse"] * 1.10:
            worse_scales.append(scale)
    if worse_scales:
        warnings.append(f"validation MSE is >10% worse than raw on: {', '.join(worse_scales)}")
    if len(worse_scales) == len(scales):
        reasons.append("scale-aware ASD is >10% worse than raw PatchTST on all validation scales")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "warnings": warnings,
        "worse_validation_scales": worse_scales,
    }


def write_ablation_report(
    *,
    args: argparse.Namespace,
    output_root: Path,
    round1_summary: pd.DataFrame,
    selection: pd.DataFrame,
    round2_summary: pd.DataFrame,
    robustness_aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    best_config: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append("# 后续实验：训练机制、Patch Scaling 与 ASD 归因")
    lines.append("")
    lines.append("本轮只包含 second / minute / hour，不包含 day，也不加入 MoE 或 LoRA。没有新增 smoke test；small experiment 本身是第一道运行验证。")
    lines.append("")
    lines.append("训练方式固定为 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，三个 scale loss 平均后更新一次。")
    lines.append("")
    lines.append("## 1. Round 1 Small Selection")
    lines.append("")
    lines.append(
        f"数据 cache: `{args.small_cache}`；epochs={args.small_epochs}；"
        f"balanced steps/epoch={args.small_steps_per_epoch}；训练股票 `{args.train_stocks}`，zero-shot stock `{args.zero_shot_stock}`。"
    )
    lines.append("")
    lines.append(f"完整 small summary: `{output_root / 'round1_small_summary.csv'}`")
    lines.append("")
    if selection.empty:
        lines.append("没有选出可进入 full confirm 的配置。")
    else:
        lines.append("### Selection Ranking")
        lines.extend(
            frame_to_markdown(
                selection[
                    [
                        "patch_preset",
                        "training_regime",
                        "init_gate",
                        "quality_pass",
                        "selection_score",
                        "second_mse_over_raw",
                        "second_dir_delta",
                        "minute_mse_over_raw",
                        "minute_dir_delta",
                        "minute_corr_delta",
                        "hour_nmse_delta_raw_minus_model",
                    ]
                ]
            )
        )
        lines.append("")
        lines.append("### Top Config Test Metrics")
        lines.extend(top_config_metric_lines(round1_summary, selection, split="test"))
    lines.append("")
    lines.append("## 2. Round 2 Full Confirm")
    lines.append("")
    if round2_summary.empty:
        lines.append("本次未运行 full confirm，或没有可确认的配置。")
    else:
        lines.append(f"完整 full summary: `{output_root / 'round2_full_summary.csv'}`")
        lines.append("")
        lines.extend(top_config_metric_lines(round2_summary, selection, split="test"))
    lines.append("")
    lines.append("## 3. Round 3 Robustness")
    lines.append("")
    if robustness_aggregate.empty:
        lines.append("本次未运行 seed robustness，或没有可聚合结果。")
    else:
        lines.append(f"robustness 配置（按 full validation selection score 选择）: `{best_config}`")
        lines.append("")
        lines.extend(frame_to_markdown(robustness_aggregate))
    lines.append("")
    lines.append("## 4. ASD Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("没有可用 ASD diagnostics。")
    else:
        diag = diagnostics[diagnostics["training_regime"] != "raw_joint"].copy()
        if diag.empty:
            lines.append("只有 raw_joint 结果，因此没有 ASD gate/threshold 诊断。")
        else:
            diag_table = (
                diag.groupby(["round", "patch_preset", "training_regime", "init_gate", "scale"], dropna=False)[
                    ["gate_mean", "tau_mean", "mean_abs_delta"]
                ]
                .mean()
                .reset_index()
            )
            lines.extend(frame_to_markdown(diag_table))
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- `asd_only_frozen_backbone` 用 raw checkpoint 初始化并冻结 PatchTST backbone、patch projection、scale embedding 和 heads，只训练 ASD gate/threshold。")
    lines.append("- `asd_frozen_encoder_train_head` 用 raw checkpoint 初始化，冻结 shared encoder 与 patch/scale embedding，只训练 ASD 和 scale-specific heads。")
    lines.append("- second 判断为 MSE 不劣于 raw 且 direction 不明显下降；minute 允许 MSE 小幅不占优，但 direction/corr 要稳定；hour 要求 MSE/NMSE 明确优于 raw 和 zero。")
    lines.append("- hour test 的 n 较小，报告中保留 `n`，避免过度解释。")
    Path(args.ablation_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_asb_report(
    *,
    args: argparse.Namespace,
    output_root: Path,
    round1_summary: pd.DataFrame,
    selection: pd.DataFrame,
    round2_summary: pd.DataFrame,
    robustness_aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    best_config: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append("# ASB-Style Encoder PatchTST 实验报告")
    lines.append("")
    lines.append("本轮只包含 second / minute / hour，不包含 day、MoE 或 LoRA。ASB 放在 PatchTST encoder 最后一层后，对 patch-token 序列做 learnable spectral filtering。")
    lines.append("")
    lines.append("训练方式保持 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，三个 scale loss 平均后更新一次。")
    lines.append("")
    lines.append("## 1. Round 1 Small Selection")
    lines.append("")
    lines.append(
        f"cache: `{args.small_cache}`；epochs={args.small_epochs}；"
        f"balanced steps/epoch={args.small_steps_per_epoch}；patch presets={list(args.asb_patch_presets)}；"
        f"ASB init gates={list(args.asb_init_gates)}。"
    )
    lines.append("")
    lines.append(f"完整 small summary: `{output_root / 'round1_small_summary.csv'}`")
    lines.append("")
    if selection.empty:
        lines.append("没有选出可进入 full confirm 的 ASB 配置。")
    else:
        lines.append("### Selection Ranking")
        lines.extend(
            frame_to_markdown(
                selection[
                    [
                        "patch_preset",
                        "training_regime",
                        "init_gate",
                        "quality_pass",
                        "selection_score",
                        "second_mse_over_raw",
                        "second_dir_delta",
                        "minute_mse_over_raw",
                        "minute_dir_delta",
                        "minute_corr_delta",
                        "hour_nmse_delta_raw_minus_model",
                        "hour_nmse_delta_asd_minus_model",
                    ]
                ]
            )
        )
        lines.append("")
        lines.append("### Small Test Comparison")
        lines.extend(comparison_metric_lines(round1_summary, selection, split="test"))
    lines.append("")
    lines.append("## 2. Round 2 Full Confirm")
    lines.append("")
    if round2_summary.empty:
        lines.append("本次未运行 full confirm，或没有可确认的 ASB 配置。")
    else:
        lines.append(f"完整 full summary: `{output_root / 'round2_full_summary.csv'}`")
        lines.append("")
        lines.extend(comparison_metric_lines(round2_summary, selection, split="test"))
    lines.append("")
    lines.append("## 3. Round 3 Robustness")
    lines.append("")
    if robustness_aggregate.empty:
        lines.append("本次未运行 seed robustness，或没有可聚合结果。")
    else:
        lines.append(f"robustness 配置（按 full validation ASB selection score 选择）: `{best_config}`")
        lines.append("")
        lines.extend(frame_to_markdown(robustness_aggregate))
    lines.append("")
    lines.append("## 4. ASB Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("没有可用 diagnostics。")
    else:
        diag = diagnostics[diagnostics["training_regime"].astype(str).str.startswith("asb_encoder_")].copy()
        if diag.empty:
            lines.append("没有 ASB diagnostics。")
        else:
            diag_table = (
                diag.groupby(["round", "patch_preset", "training_regime", "init_gate", "scale"], dropna=False)[
                    [
                        "gate_mean",
                        "tau_mean",
                        "local_mask_mean",
                        "mean_abs_delta",
                        "global_filter_norm",
                        "local_filter_norm",
                    ]
                ]
                .mean()
                .reset_index()
            )
            lines.extend(frame_to_markdown(diag_table))
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- raw PatchTST 是全部训练的 baseline；当前输入端 ASD 对照为 `asd_frozen_encoder_train_head`。")
    lines.append("- `asb_encoder_joint` 训练 ASB + PatchTST；`asb_encoder_frozen_base_train_asb_only` 只训练 ASB；`asb_encoder_frozen_base_train_asb_head` 只训练 ASB + scale heads。")
    lines.append("- second 要求 MSE 不差于 raw 超过 1%，direction 不下降超过 1 个百分点；minute 要求 MSE 不差于 raw 超过 2%，且 direction/corr 至少一个不低于 raw；hour 要求 NMSE 优于 raw。")
    lines.append("- 若 ASB 只改善 hour 但损伤 second，则只作为低频 scale 特化模块，不作为跨尺度主模型。")
    Path(args.asb_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_adapter_ablation_report(
    *,
    args: argparse.Namespace,
    output_root: Path,
    round1_summary: pd.DataFrame,
    round2_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# PatchTST Adapter Ablation")
    lines.append("")
    lines.append(
        "本轮只做后接 adapter 消融：所有 adapter 都插在 PatchTST encoder 后、scale-specific head 前；"
        "无 ASD 的 LoRA-only/MLP-MoE/LoRA-MoE 前面没有 ASD；`asd_*` 行用于测试加 ASD 后结论是否改变。"
    )
    lines.append("")
    lines.append(
        f"small setting: patch presets={list(args.adapter_ablation_patch_presets)}, "
        f"ranks={list(args.adapter_ablation_ranks)}, epochs={args.small_epochs}, "
        f"steps/epoch={args.small_steps_per_epoch}."
    )
    lines.append("")
    lines.append("## 1. Small Result")
    lines.append("")
    lines.append(f"summary: `{output_root / 'round1_small_summary.csv'}`")
    lines.append("")
    lines.extend(adapter_ablation_table_lines(round1_summary, split="test"))
    lines.append("")
    lines.append("## 2. Full Result")
    lines.append("")
    if round2_summary.empty:
        lines.append("本次没有运行 full；如 small 结果出现明确候选，再用 `--run-adapter-ablation-full` 补跑。")
    else:
        lines.append(f"summary: `{output_root / 'round2_full_summary.csv'}`")
        lines.append("")
        lines.extend(adapter_ablation_table_lines(round2_summary, split="test"))
    lines.append("")
    lines.append("## 3. Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("没有 diagnostics。")
    else:
        diag = diagnostics[
            diagnostics["training_regime"]
            .astype(str)
            .str.startswith(("lora_only_", "mlp_moe_", "lora_moe_", "asd_lora_only_", "asd_mlp_moe_", "asd_lora_moe_"))
        ].copy()
        if diag.empty:
            lines.append("没有 adapter diagnostics。")
        else:
            group_cols = ["round", "patch_preset", "training_regime", "adapter_rank", "scale"]
            metric_cols = [
                "mean_abs_delta",
                "router_entropy",
                "router_balance_loss",
                "expert_prob_0",
                "expert_prob_1",
                "expert_prob_2",
                "expert_prob_3",
            ]
            available = [column for column in metric_cols if column in diag.columns]
            table = diag.groupby(group_cols, dropna=False)[available].mean().reset_index()
            lines.extend(frame_to_markdown(table))
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- `raw_frozen_base_train_head` 测的是 head recalibration 本身，不含 ASD/LoRA/MoE。")
    lines.append("- `lora_only_*` 测的是共享低秩金融域适配，不含 MoE routing。")
    lines.append("- `mlp_moe_*` 测的是 MoE routing + 更强 MLP expert，不含 LoRA low-rank 约束。")
    lines.append("- `lora_moe_*` 是已有 LoRA expert + MoE router；`asd_*` 版本是在同一 adapter 前加入 ASD。")
    lines.append("- 如果 `asd_*` 没有超过对应无 ASD 行，说明 ASD 不改变 adapter 消融结论。")
    Path(args.adapter_ablation_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_adapter_targeted_robustness_report(
    *,
    args: argparse.Namespace,
    target_root: Path,
    aggregate: pd.DataFrame,
    router_usage: pd.DataFrame,
    asd_stats: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# LoRA-MoE Targeted Robustness")
    lines.append("")
    lines.append(
        "本轮只确认 4 个核心模型：raw PatchTST、head-only、LoRA-MoE + head、ASD + LoRA-MoE + head。"
    )
    lines.append(
        f"配置固定为 patch preset `{TARGETED_ASD_LORA_MOE_PATCH_PRESET}`，rank={TARGETED_ASD_LORA_MOE_RANK}，"
        f"ASD init gate={TARGETED_ASD_LORA_MOE_INIT_GATE}，seeds={list(unique_list(args.robustness_seeds))}。"
    )
    lines.append("")
    lines.append("## 1. Mean/Std NMSE")
    lines.append("")
    if aggregate.empty:
        lines.append("没有 aggregate 结果。")
    else:
        keep = [
            "patch_preset",
            "model",
            "adapter_rank",
            "scale",
            "nmse_mean",
            "nmse_std",
            "mse_mean",
            "mse_std",
            "direction_accuracy_nonzero_mean",
            "direction_accuracy_nonzero_std",
            "corr_mean",
            "corr_std",
        ]
        lines.extend(frame_to_markdown(aggregate[[column for column in keep if column in aggregate.columns]]))
    lines.append("")
    lines.append("## 2. Router Expert Usage By Scale")
    lines.append("")
    if router_usage.empty:
        lines.append("没有 router diagnostics。")
    else:
        keep = [
            "training_regime",
            "adapter_rank",
            "scale",
            "expert_0",
            "expert_1",
            "expert_2",
            "expert_3",
            "router_entropy",
            "router_balance_loss",
        ]
        lines.extend(frame_to_markdown(router_usage[[column for column in keep if column in router_usage.columns]]))
    lines.append("")
    lines.append("## 3. ASD Gate/Tau By Scale")
    lines.append("")
    if asd_stats.empty:
        lines.append("没有 ASD diagnostics。")
    else:
        keep = [
            "training_regime",
            "adapter_rank",
            "scale",
            "gate_mean",
            "gate_std",
            "tau_mean",
            "tau_std",
            "mean_abs_delta_mean",
            "mean_abs_delta_std",
        ]
        lines.extend(frame_to_markdown(asd_stats[[column for column in keep if column in asd_stats.columns]]))
    lines.append("")
    lines.append("## Files")
    lines.append("")
    lines.append(f"- summary: `{target_root / 'targeted_robustness_summary.csv'}`")
    lines.append(f"- aggregate: `{target_root / 'targeted_robustness_aggregate.csv'}`")
    lines.append(f"- router usage: `{target_root / 'router_usage_by_scale.csv'}`")
    lines.append(f"- ASD diagnostics: `{target_root / 'asd_gate_tau_by_scale.csv'}`")
    Path(args.adapter_targeted_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def adapter_ablation_table_lines(summary: pd.DataFrame, *, split: str) -> list[str]:
    if summary.empty:
        return ["无可用结果。"]
    rows = summary[
        (summary["split"] == split)
        & (
            summary["model"].isin(["zero", "raw_joint", "raw_frozen_base_train_head", "asd_frozen_encoder_train_head"])
            | summary["model"].astype(str).str.startswith("lora_only_")
            | summary["model"].astype(str).str.startswith("mlp_moe_")
            | summary["model"].astype(str).str.startswith("lora_moe_")
            | summary["model"].astype(str).str.startswith("asd_lora_only_")
            | summary["model"].astype(str).str.startswith("asd_mlp_moe_")
            | summary["model"].astype(str).str.startswith("asd_lora_moe_")
        )
    ].copy()
    if rows.empty:
        return ["无可用结果。"]
    keep = [
        "patch_preset",
        "model",
        "adapter_rank",
        "scale",
        "n",
        "mse",
        "mae",
        "nmse",
        "direction_accuracy_nonzero",
        "corr",
    ]
    rows = rows.sort_values(["patch_preset", "scale", "model", "adapter_rank"], key=sort_summary_key)
    return frame_to_markdown(rows[[column for column in keep if column in rows.columns]])


def write_lora_moe_report(
    *,
    args: argparse.Namespace,
    output_root: Path,
    round1_summary: pd.DataFrame,
    selection: pd.DataFrame,
    round2_summary: pd.DataFrame,
    robustness_aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    oracle: pd.DataFrame,
    best_config: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append("# LoRA-MoE PatchTST 跨尺度金融预测实验报告")
    lines.append("")
    lines.append(
        "本轮只包含 second / minute / hour，不包含 day。方法定位是：shared PatchTST 保留通用时间序列预测能力，"
        "LoRA-style low-rank adapter 负责把模型适配到金融 intraday return 分布，MoE router 负责不同 temporal scale "
        "之间的专家选择。ASD/ASB 只作为对照和 oracle 候选。"
    )
    lines.append("")
    lines.append(
        "训练仍采用 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，"
        "主 loss 平均后，对 LoRA-MoE 加 `router_balance_weight * router_balance_loss`。"
    )
    lines.append("")
    lines.append(
        "与本地 FinCast 代码保持同一个思路：数据侧保留 freq/scale id；模型侧把 frequency embedding 加到 token 表示，"
        "并用 sparse top-k MoE 与 balance loss 让不同频率样本走不同专家。本实现同时使用 `log(delta_seconds)` "
        "和离散 scale id embedding，并在 router 中加入 scale-conditioned expert prior。"
    )
    lines.append("")
    lines.append("## 1. Small Selection")
    lines.append("")
    lines.append(
        f"cache: `{args.small_cache}`; epochs={args.small_epochs}; "
        f"balanced steps/epoch={args.small_steps_per_epoch}; patch presets={list(args.lora_moe_patch_presets)}; "
        f"ranks={list(args.lora_moe_ranks)}."
    )
    lines.append("")
    lines.append(f"完整 small summary: `{output_root / 'round1_small_summary.csv'}`")
    lines.append("")
    if selection.empty:
        lines.append("没有选出可进入 full confirm 的 LoRA-MoE 配置。")
    else:
        lines.append("### Selection Ranking")
        keep = [
            "patch_preset",
            "training_regime",
            "adapter_rank",
            "quality_pass",
            "strong_pass",
            "selection_score",
            "second_mse_over_raw",
            "second_dir_delta",
            "minute_mse_over_raw",
            "minute_dir_delta",
            "minute_corr_delta",
            "hour_nmse_delta_raw_minus_model",
            "hour_nmse_delta_asd_minus_model",
        ]
        lines.extend(frame_to_markdown(selection[[column for column in keep if column in selection.columns]]))
        lines.append("")
        lines.append("### Small Test Comparison")
        lines.extend(comparison_metric_lines(round1_summary, selection, split="test"))
    lines.append("")
    lines.append("## 2. Full Confirm")
    lines.append("")
    if round2_summary.empty:
        lines.append("本次未运行 full confirm，或 small selection 没有可确认配置。")
    else:
        lines.append(f"完整 full summary: `{output_root / 'round2_full_summary.csv'}`")
        lines.append("")
        lines.extend(comparison_metric_lines(round2_summary, selection, split="test"))
    lines.append("")
    lines.append("## 3. Robustness")
    lines.append("")
    if robustness_aggregate.empty:
        lines.append("本次未运行 seed robustness，或没有可聚合结果。")
    else:
        lines.append(f"robustness 配置: `{best_config}`")
        lines.append("")
        lines.extend(frame_to_markdown(robustness_aggregate))
    lines.append("")
    lines.append("## 4. Per-Scale Oracle")
    lines.append("")
    lines.append(
        "oracle 表允许每个 scale 从 raw / ASD / ASB / LoRA-MoE 中选 test MSE 最低者，"
        "只作为实用上界，不作为单一主模型 claim。"
    )
    lines.append("")
    if oracle.empty:
        lines.append("没有可用 oracle 候选。")
    else:
        keep = [
            "scale",
            "source",
            "patch_preset",
            "model",
            "adapter_rank",
            "n",
            "mse",
            "mae",
            "nmse",
            "direction_accuracy_nonzero",
            "corr",
        ]
        lines.extend(frame_to_markdown(oracle[[column for column in keep if column in oracle.columns]]))
    lines.append("")
    lines.append("## 5. Router Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("没有可用 router diagnostics。")
    else:
        diag = diagnostics[diagnostics["training_regime"].astype(str).str.startswith("lora_moe_")].copy()
        if diag.empty:
            lines.append("没有 LoRA-MoE diagnostics。")
        else:
            group_cols = ["round", "patch_preset", "training_regime", "adapter_rank", "scale"]
            metric_cols = [
                "router_entropy",
                "router_balance_loss",
                "expert_prob_0",
                "expert_prob_1",
                "expert_prob_2",
                "expert_prob_3",
                "scale_prior_prob_0",
                "scale_prior_prob_1",
                "scale_prior_prob_2",
                "scale_prior_prob_3",
                "mean_abs_delta",
            ]
            diag_table = diag.groupby(group_cols, dropna=False)[metric_cols].mean().reset_index()
            lines.extend(frame_to_markdown(diag_table))
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- `lora_moe_joint` 训练 PatchTST + LoRA-MoE 全部参数。")
    lines.append("- `lora_moe_frozen_base_train_moe_only` 从 raw checkpoint 加载 shared backbone 和 heads，只训练 LoRA-MoE。")
    lines.append("- `lora_moe_frozen_base_train_moe_head` 从 raw checkpoint 加载并冻结 encoder/patch/scale embedding，只训练 LoRA-MoE + scale heads，是本轮推荐主候选。")
    lines.append("- 如果没有统一 LoRA-MoE 通过三尺度 gate，结论应写为单一跨尺度最优模型未形成，并使用 oracle 表说明 per-scale 上界。")
    Path(args.lora_moe_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_asd_lora_moe_report(
    *,
    args: argparse.Namespace,
    output_root: Path,
    round1_summary: pd.DataFrame,
    selection: pd.DataFrame,
    round2_summary: pd.DataFrame,
    robustness_aggregate: pd.DataFrame,
    diagnostics: pd.DataFrame,
    oracle: pd.DataFrame,
    best_config: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append("# ASD + PatchTST + LoRA-MoE 跨尺度金融预测实验报告")
    lines.append("")
    lines.append(
        "本轮只包含 second / minute / hour，不包含 day。方法定位固定为：ASD 做 scale-aware denoising，"
        "shared PatchTST 保留通用时间序列表征，LoRA-style adapter 做金融域低秩适配，MoE router 做 "
        "second/minute/hour 的 scale/frequency specialization。"
    )
    lines.append("")
    lines.append(
        "训练采用 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，"
        "主 loss 平均后，对 LoRA-MoE 加 `router_balance_weight * router_balance_loss`。"
    )
    lines.append("")
    lines.append("## 1. Small Selection")
    lines.append("")
    lines.append(
        f"cache: `{args.small_cache}`; epochs={args.small_epochs}; "
        f"balanced steps/epoch={args.small_steps_per_epoch}; patch presets={list(args.asd_lora_moe_patch_presets)}; "
        f"ranks={list(args.asd_lora_moe_ranks)}; ASD init gates={list(args.asd_lora_moe_init_gates)}."
    )
    lines.append("")
    lines.append(f"完整 small summary: `{output_root / 'round1_small_summary.csv'}`")
    lines.append("")
    if selection.empty:
        lines.append("没有 combined ASD+LoRA-MoE 配置通过 small selection；按 gate 规则不直接进入 full confirm。")
    else:
        lines.append("### Selection Ranking")
        keep = [
            "patch_preset",
            "training_regime",
            "init_gate",
            "adapter_rank",
            "quality_pass",
            "strong_pass",
            "selection_score",
            "second_mse_over_raw",
            "second_dir_delta",
            "minute_mse_over_raw",
            "minute_dir_delta",
            "minute_corr_delta",
            "hour_mse_over_asd",
            "hour_nmse_delta_raw_minus_model",
            "hour_nmse_delta_asd_minus_model",
        ]
        lines.extend(frame_to_markdown(selection[[column for column in keep if column in selection.columns]]))
        lines.append("")
        lines.append("### Small Test Comparison")
        lines.extend(comparison_metric_lines(round1_summary, selection, split="test"))
    lines.append("")
    lines.append("## 2. Full Confirm")
    lines.append("")
    if round2_summary.empty:
        lines.append("本次未运行 full confirm：没有 small quality pass，或显式设置了 skip。")
    else:
        lines.append(f"完整 full summary: `{output_root / 'round2_full_summary.csv'}`")
        lines.append("")
        lines.extend(comparison_metric_lines(round2_summary, selection, split="test"))
    lines.append("")
    lines.append("## 3. Robustness")
    lines.append("")
    if robustness_aggregate.empty:
        lines.append("本次未运行 seed robustness，或没有可聚合结果。")
    else:
        lines.append(f"robustness 配置: `{best_config}`")
        lines.append("")
        lines.extend(frame_to_markdown(robustness_aggregate))
    lines.append("")
    lines.append("## 4. Per-Scale Oracle")
    lines.append("")
    lines.append(
        "oracle 表允许每个 scale 从 raw / ASD / ASB / LoRA-MoE / ASD+LoRA-MoE 中选 test MSE 最低者，"
        "只作为实用上界，不作为单一主模型 claim。"
    )
    lines.append("")
    if oracle.empty:
        lines.append("没有可用 oracle 候选。")
    else:
        keep = [
            "scale",
            "source",
            "patch_preset",
            "model",
            "init_gate",
            "adapter_rank",
            "n",
            "mse",
            "mae",
            "nmse",
            "direction_accuracy_nonzero",
            "corr",
        ]
        lines.extend(frame_to_markdown(oracle[[column for column in keep if column in oracle.columns]]))
    lines.append("")
    lines.append("## 5. Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("没有可用 diagnostics。")
    else:
        diag = diagnostics[
            diagnostics["training_regime"].astype(str).str.startswith("asd_lora_moe_")
        ].copy()
        if diag.empty:
            lines.append("没有 combined ASD+LoRA-MoE diagnostics。")
        else:
            group_cols = ["round", "patch_preset", "training_regime", "init_gate", "adapter_rank", "scale"]
            metric_cols = [
                "asd_gate_mean",
                "asd_tau_mean",
                "asd_mean_abs_delta",
                "router_entropy",
                "router_balance_loss",
                "expert_prob_0",
                "expert_prob_1",
                "expert_prob_2",
                "expert_prob_3",
                "scale_prior_prob_0",
                "scale_prior_prob_1",
                "scale_prior_prob_2",
                "scale_prior_prob_3",
                "moe_mean_abs_delta",
            ]
            available_metrics = [column for column in metric_cols if column in diag.columns]
            diag_table = diag.groupby(group_cols, dropna=False)[available_metrics].mean().reset_index()
            lines.extend(frame_to_markdown(diag_table))
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- `asd_lora_moe_joint` 训练 ASD + PatchTST + LoRA-MoE 全部参数。")
    lines.append("- `asd_lora_moe_frozen_base_train_adapters_only` 从 raw checkpoint 加载，只训练 ASD + LoRA-MoE。")
    lines.append("- `asd_lora_moe_frozen_base_train_adapters_head` 从 raw checkpoint 加载，只训练 ASD + LoRA-MoE + scale heads，是本轮主候选。")
    lines.append("- hour test 的 `n` 很小，约 200 个窗口；即使 full confirm 有提升，也需要 robustness 和 oracle 一起解释。")
    lines.append("- 若 combined 没有通过三尺度 gate，结论应写为组合模块未形成稳定统一主模型，而不是模块完全无效。")
    Path(args.asd_lora_moe_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_asd_lora_moe_final_decision_report(
    *,
    args: argparse.Namespace,
    output_root: Path,
    target_root: Path,
    existing_full_summary: pd.DataFrame,
    compact_robustness: pd.DataFrame,
    short_second_summary: pd.DataFrame,
    short_second_robustness: pd.DataFrame,
    diagnostics: pd.DataFrame,
    oracle: pd.DataFrame,
    decision: dict[str, Any],
) -> None:
    lines: list[str] = []
    lines.append("# ASD + PatchTST + LoRA-MoE 最终决策报告")
    lines.append("")
    lines.append(
        "本报告只确认 `short_second + rank=8 + ASD init gate=-4.0` 的多 seed 稳定性；"
        "没有新增 ASB、attention-level LoRA、MoE 层数或 day 数据。"
    )
    lines.append("")
    lines.append(
        "训练仍采用 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，"
        "平均主 loss 后加入 `router_balance_weight * router_balance_loss`。"
    )
    lines.append("")
    lines.append("## 1. 已有 Full 结果摘要")
    lines.append("")
    if existing_full_summary.empty:
        lines.append("未找到已有 `round2_full_summary.csv`，本节只保留 targeted robustness 结论。")
    else:
        target_selection = pd.DataFrame(
            [
                {
                    "patch_preset": TARGETED_ASD_LORA_MOE_PATCH_PRESET,
                    "training_regime": TARGETED_ASD_LORA_MOE_REGIME,
                    "init_gate": TARGETED_ASD_LORA_MOE_INIT_GATE,
                    "adapter_rank": TARGETED_ASD_LORA_MOE_RANK,
                    "seed": int(args.seed),
                }
            ]
        )
        lines.append(f"已有 full summary: `{output_root / 'round2_full_summary.csv'}`")
        lines.append("")
        lines.extend(comparison_metric_lines(existing_full_summary, target_selection, split="test"))
    lines.append("")
    lines.append("## 2. Compact Robustness")
    lines.append("")
    if compact_robustness.empty:
        lines.append("未找到已有 compact robustness aggregate。")
    else:
        lines.append(f"已有 compact robustness: `{output_root / 'round3_robustness_aggregate.csv'}`")
        lines.append("")
        lines.extend(frame_to_markdown(compact_robustness))
    lines.append("")
    lines.append("## 3. Short-Second Robustness")
    lines.append("")
    lines.append(f"targeted summary: `{target_root / 'round3_short_second_rank8_summary.csv'}`")
    lines.append("")
    lines.append(f"targeted aggregate: `{target_root / 'round3_short_second_rank8_aggregate.csv'}`")
    lines.append("")
    if short_second_robustness.empty:
        lines.append("targeted robustness 没有可聚合结果。")
    else:
        lines.extend(frame_to_markdown(short_second_robustness))
    lines.append("")
    hour_rows = short_second_summary[
        (short_second_summary["split"] == "test")
        & (short_second_summary["scale"] == "hour")
        & (short_second_summary["model"] == TARGETED_ASD_LORA_MOE_REGIME)
    ]
    if hour_rows.empty:
        lines.append("hour test n: 未找到。")
    else:
        lines.append(f"hour test n: `{int(round(float(hour_rows['n'].mean())))}`，因此 hour 结果需要按低样本量解释。")
    lines.append("")
    lines.append("## 4. 最终模型选择")
    lines.append("")
    quality = bool(decision.get("quality_pass"))
    strong = bool(decision.get("strong_pass"))
    if strong:
        lines.append("结论：`ASD+PatchTST+LoRA-MoE` 通过 strong pass，可作为当前统一三尺度主模型。")
    elif quality:
        lines.append("结论：`ASD+PatchTST+LoRA-MoE` 通过三尺度 gate，但 strong pass 不充分；可作为候选主模型，报告中需要保留不确定性。")
    else:
        lines.append("结论：组合模块没有形成稳定统一主模型；`ASD+LoRA-MoE` 应保留为 exploratory / oracle 候选。")
    lines.append("")
    lines.append(f"decision reason: {decision.get('reason')}")
    if decision.get("seed_std_warning"):
        lines.append("seed std warning: hour NMSE 的 seed 波动接近或超过模型间 margin。")
    fallback = decision.get("fallback_unified_model")
    if fallback:
        lines.append(f"fallback unified recommendation: `{fallback}`，按 robustness mean NMSE 在非 combined 模型中选择。")
    lines.append("")
    scale_table = decision.get("scale_table", pd.DataFrame())
    if isinstance(scale_table, pd.DataFrame) and not scale_table.empty:
        lines.extend(frame_to_markdown(scale_table))
    lines.append("")
    lines.append("### Robustness Per-Scale Recommendation")
    lines.append("")
    per_scale = decision.get("per_scale_recommendation", pd.DataFrame())
    if isinstance(per_scale, pd.DataFrame) and not per_scale.empty:
        keep = [
            "scale",
            "model",
            "init_gate",
            "adapter_rank",
            "mse_mean",
            "mse_std",
            "nmse_mean",
            "nmse_std",
            "direction_accuracy_nonzero_mean",
            "corr_mean",
        ]
        lines.extend(frame_to_markdown(per_scale[[column for column in keep if column in per_scale.columns]]))
    else:
        lines.append("无可用 per-scale recommendation。")
    lines.append("")
    lines.append("## 5. Per-Scale Oracle")
    lines.append("")
    lines.append(
        "oracle 允许每个 scale 从 raw / ASD / ASB / LoRA-MoE / ASD+LoRA-MoE 中选 test MSE 最低者，"
        "只作为实用上界，不作为单一主模型 claim。"
    )
    lines.append("")
    if oracle.empty:
        lines.append("没有可用 oracle 候选。")
    else:
        keep = [
            "scale",
            "source",
            "patch_preset",
            "model",
            "init_gate",
            "adapter_rank",
            "n",
            "mse",
            "mae",
            "nmse",
            "direction_accuracy_nonzero",
            "corr",
        ]
        lines.extend(frame_to_markdown(oracle[[column for column in keep if column in oracle.columns]]))
    lines.append("")
    lines.append("## 6. Diagnostics 解读")
    lines.append("")
    lines.append(f"diagnostics: `{target_root / 'short_second_rank8_diagnostics.csv'}`")
    lines.append("")
    if diagnostics.empty:
        lines.append("没有 diagnostics。")
    else:
        diag = diagnostics[diagnostics["training_regime"].astype(str) == TARGETED_ASD_LORA_MOE_REGIME].copy()
        if diag.empty:
            lines.append("没有 combined ASD+LoRA-MoE diagnostics。")
        else:
            metric_cols = [
                "asd_gate_mean",
                "asd_tau_mean",
                "asd_mean_abs_delta",
                "moe_router_entropy",
                "moe_router_balance_loss",
                "moe_expert_prob_0",
                "moe_expert_prob_1",
                "moe_expert_prob_2",
                "moe_expert_prob_3",
                "moe_mean_abs_delta",
            ]
            available = [column for column in metric_cols if column in diag.columns]
            table = diag.groupby(["scale"], dropna=False)[available].mean().reset_index()
            lines.extend(frame_to_markdown(table))
    lines.append("")
    lines.append("## Guardrails")
    lines.append("")
    lines.append("- 本报告的最终判断基于 seeds `42, 43, 44` 的 mean/std，而不是单 seed full test。")
    lines.append("- 若 combined 失败，解释应写成“组合模块不稳定”，不是“ASD 或 LoRA-MoE 一定无效”。")
    lines.append("- hour test 样本量小，必须结合 robustness 和 oracle 一起解释。")
    Path(args.asd_lora_moe_final_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_tslanet_report(
    *,
    args: argparse.Namespace,
    output_root: Path,
    round1_summary: pd.DataFrame,
    round2_summary: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# TSLANet-Style Intraday Baseline")
    lines.append("")
    lines.append(
        "本报告加入一个轻量 TSLANet-style baseline，用 ASB + ICB blocks 替换 PatchTST Transformer encoder。"
        "它复用当前 second/minute/hour 数据协议、scale-specific patch projection 和 scale-specific heads。"
    )
    lines.append("")
    lines.append(
        "说明：这是本地轻量复现，用于和当前 PatchTST/ASD 线做公平工程对比；不是官方 TSLANet 代码的逐行移植。"
    )
    lines.append("")
    lines.append(
        f"small setting: epochs={args.small_epochs}, steps/epoch={args.small_steps_per_epoch}; "
        f"full setting: epochs={args.full_epochs}, steps/epoch={args.full_steps_per_epoch}; "
        f"patch presets={list(args.tslanet_patch_presets)}."
    )
    lines.append("")
    lines.append("## 1. Small Result")
    lines.append("")
    lines.append(f"summary: `{output_root / 'round1_small_summary.csv'}`")
    lines.append("")
    lines.extend(tslanet_comparison_lines(round1_summary, split="test"))
    lines.append("")
    lines.append("## 2. Full Result")
    lines.append("")
    if round2_summary.empty:
        lines.append("本次跳过 full run。")
    else:
        lines.append(f"summary: `{output_root / 'round2_full_summary.csv'}`")
        lines.append("")
        lines.extend(tslanet_comparison_lines(round2_summary, split="test"))
    lines.append("")
    lines.append("## 3. Diagnostics")
    lines.append("")
    if diagnostics.empty:
        lines.append("没有 diagnostics。")
    else:
        diag = diagnostics[diagnostics["training_regime"].astype(str) == "tslanet_joint"].copy()
        if diag.empty:
            lines.append("没有 TSLANet diagnostics。")
        else:
            metric_cols = [
                "tslanet_gate_mean",
                "tslanet_tau_mean",
                "tslanet_local_mask_mean",
                "tslanet_mean_abs_delta",
                "tslanet_global_filter_norm",
                "tslanet_local_filter_norm",
            ]
            available = [column for column in metric_cols if column in diag.columns]
            table = diag.groupby(["round", "patch_preset", "scale"], dropna=False)[available].mean().reset_index()
            lines.extend(frame_to_markdown(table))
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- `tslanet_joint` 是 ASB + ICB 的轻量时间序列 baseline，不包含 LoRA/MoE。")
    lines.append("- 对比重点是 raw PatchTST、ASD+PatchTST 和 TSLANet-style baseline 在同一数据协议下的 test metrics。")
    lines.append("- hour test 的 `n` 约 200，低频结果需要谨慎解释。")
    Path(args.tslanet_report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def tslanet_comparison_lines(summary: pd.DataFrame, *, split: str) -> list[str]:
    if summary.empty:
        return ["无可用结果。"]
    rows = summary[
        (summary["split"] == split)
        & (summary["model"].isin(["zero", "raw_joint", "asd_frozen_encoder_train_head", "tslanet_joint"]))
    ].copy()
    if rows.empty:
        return ["无可用结果。"]
    keep = [
        "patch_preset",
        "model",
        "scale",
        "n",
        "mse",
        "mae",
        "nmse",
        "direction_accuracy_nonzero",
        "corr",
    ]
    rows = rows.sort_values(["patch_preset", "scale", "model"], key=sort_summary_key)
    return frame_to_markdown(rows[[column for column in keep if column in rows.columns]])


def comparison_metric_lines(summary: pd.DataFrame, selection: pd.DataFrame, *, split: str) -> list[str]:
    if summary.empty or selection.empty:
        return ["无可用结果。"]
    selected_rows: list[pd.DataFrame] = []
    for _, config in selection.iterrows():
        base_model_mask = summary["model"].isin(["zero", "raw_joint", "asd_frozen_encoder_train_head"])
        if str(config["training_regime"]).startswith("asd_lora_moe_"):
            base_model_mask = base_model_mask | (summary["model"] == "lora_moe_frozen_base_train_moe_head")
        base = summary[
            (summary["split"] == split)
            & (summary["patch_preset"] == config["patch_preset"])
            & base_model_mask
            & (summary["seed"] == config["seed"])
        ].copy()
        if (
            str(config["training_regime"]).startswith("asd_lora_moe_")
            and "adapter_rank" in base
            and "adapter_rank" in config
            and not pd.isna(config["adapter_rank"])
        ):
            base = base[
                (~base["model"].astype(str).str.startswith("lora_moe_"))
                | (base["adapter_rank"].astype(float).round(6) == float(config["adapter_rank"]))
            ]
        candidate = summary[
            (summary["split"] == split)
            & (summary["patch_preset"] == config["patch_preset"])
            & (summary["model"] == config["training_regime"])
            & (summary["seed"] == config["seed"])
        ].copy()
        if "init_gate" in candidate and not pd.isna(config["init_gate"]):
            candidate = candidate[candidate["init_gate"].astype(float).round(6) == float(config["init_gate"])]
        if "adapter_rank" in candidate and "adapter_rank" in config and not pd.isna(config["adapter_rank"]):
            candidate = candidate[candidate["adapter_rank"].astype(float).round(6) == float(config["adapter_rank"])]
        selected_rows.extend([base, candidate])
    table = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if table.empty:
        return ["无可用结果。"]
    subset = [column for column in ["patch_preset", "model", "init_gate", "adapter_rank", "scale", "seed"] if column in table]
    table = table.drop_duplicates(subset=subset)
    table = table.sort_values(["patch_preset", "scale", "model", "init_gate"], key=sort_summary_key)
    keep_columns = [
        "patch_preset",
        "model",
        "init_gate",
        "adapter_rank",
        "scale",
        "n",
        "mse",
        "mae",
        "nmse",
        "direction_accuracy_nonzero",
        "corr",
    ]
    return frame_to_markdown(table[[column for column in keep_columns if column in table.columns]])


def top_config_metric_lines(summary: pd.DataFrame, selection: pd.DataFrame, *, split: str) -> list[str]:
    if summary.empty or selection.empty:
        return ["无可用结果。"]
    selected_rows: list[pd.DataFrame] = []
    for _, config in selection.iterrows():
        selected = summary[
            (summary["split"] == split)
            & (summary["patch_preset"] == config["patch_preset"])
            & (summary["model"] == config["training_regime"])
            & (summary["seed"] == config["seed"])
        ].copy()
        if "init_gate" in selected and not pd.isna(config["init_gate"]):
            selected = selected[selected["init_gate"].astype(float).round(6) == float(config["init_gate"])]
        if "adapter_rank" in selected and "adapter_rank" in config and not pd.isna(config["adapter_rank"]):
            selected = selected[selected["adapter_rank"].astype(float).round(6) == float(config["adapter_rank"])]
        raw = summary[
            (summary["split"] == split)
            & (summary["patch_preset"] == config["patch_preset"])
            & (summary["model"].isin(["zero", "raw_joint"]))
            & (summary["seed"] == config["seed"])
        ].copy()
        selected_rows.extend([raw, selected])
    table = pd.concat(selected_rows, ignore_index=True) if selected_rows else pd.DataFrame()
    if table.empty:
        return ["无可用结果。"]
    subset = [column for column in ["patch_preset", "model", "init_gate", "adapter_rank", "scale", "seed"] if column in table]
    table = table.drop_duplicates(subset=subset)
    table = table.sort_values(["patch_preset", "scale", "model", "init_gate"], key=sort_summary_key)
    keep_columns = [
        "patch_preset",
        "model",
        "init_gate",
        "adapter_rank",
        "scale",
        "n",
        "mse",
        "mae",
        "nmse",
        "direction_accuracy_nonzero",
        "corr",
    ]
    return frame_to_markdown(table[[column for column in keep_columns if column in table.columns]])


def write_report(
    *,
    args: argparse.Namespace,
    small_result: dict[str, Any] | None,
    full_estimate: dict[str, Any] | None,
    quality_gate: dict[str, Any] | None,
    full_result: dict[str, Any] | None,
) -> None:
    lines: list[str] = []
    lines.append("# Scale-Aware ASD PatchTST 小数据优先实验报告")
    lines.append("")
    lines.append("本报告只包含 second / minute / hour 三个 intraday scale，不包含 day 数据。")
    lines.append("")
    lines.append("## 1. 小数据真实结果")
    lines.append("")
    if small_result is None:
        lines.append("本次未运行 small preset。")
    else:
        lines.append(
            f"small cache: `{small_result['cache']}`；epochs={small_result['epochs']}；"
            f"balanced steps/epoch={small_result['steps_per_epoch']}；"
            f"patch preset=`{small_result['patch_preset']}`。"
        )
        lines.append("")
        lines.extend(summary_table_lines(small_result["summary"], preset="small", split="test"))
        lines.append("")
        lines.append("### Quality Gate")
        if quality_gate is None:
            lines.append("未执行 quality gate。")
        else:
            status = "PASS" if quality_gate["passed"] else "FAIL"
            lines.append(f"- gate: **{status}**")
            if quality_gate["warnings"]:
                lines.extend([f"- warning: {item}" for item in quality_gate["warnings"]])
            if quality_gate["reasons"]:
                lines.extend([f"- reason: {item}" for item in quality_gate["reasons"]])
            if not quality_gate["warnings"] and not quality_gate["reasons"]:
                lines.append("- diagnostics finite; required splits and scales are present.")
    lines.append("")
    lines.append("## 2. 全量数据预估水平")
    lines.append("")
    if full_estimate is None:
        lines.append("本次未生成 full-data estimate。")
    else:
        lines.append(
            "以下是 full actual 运行前预估，不作为最终实验结论。性能预估使用 "
            "`existing_full_raw_metric * small(scale-aware/raw)`。"
        )
        lines.append("")
        estimate_frame = pd.DataFrame(full_estimate["rows"])
        lines.extend(estimate_table_lines(estimate_frame, split="test"))
        lines.append("")
        lines.append(
            f"预计 full scale-aware 训练时间约 {format_seconds(full_estimate['estimated_full_runtime_seconds'])} "
            f"（{full_estimate['runtime_formula']}）。"
        )
    lines.append("")
    lines.append("## 3. 全量实际结果")
    lines.append("")
    if full_result is None:
        lines.append("full actual 尚未运行，或 quality gate 未通过。")
    else:
        lines.append(
            f"full cache: `{full_result['cache']}`；epochs={full_result['epochs']}；"
            f"balanced steps/epoch={full_result['steps_per_epoch']}；"
            f"patch preset=`{full_result['patch_preset']}`。"
        )
        lines.append("")
        lines.extend(summary_table_lines(full_result["summary"], preset="full", split="test"))
        if full_estimate is not None:
            lines.append("")
            lines.append("### 预估 vs 实际")
            lines.extend(estimate_vs_actual_lines(pd.DataFrame(full_estimate["rows"]), full_result["summary"]))
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- hour scale 沿用当前项目的 intraday hour/proxy 构造，不写成严格自然小时。")
    lines.append("- patch/context 不固定为 32；默认 preset 参考 FinCast 的 frequency-aware context 思路，并按 Optiver 可用窗口长度调整。")
    lines.append("- full preset 使用 balanced training；不会强制每个 epoch 穷尽 second 全部 windows。")
    lines.append("- 第一版不包含 Adapter-MoE 或 LoRA。")
    Path(args.report_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def summary_table_lines(summary: pd.DataFrame, *, preset: str, split: str) -> list[str]:
    rows = summary[(summary["preset"] == preset) & (summary["split"] == split)]
    rows = rows[rows["model"].isin(MODEL_ORDER)]
    rows = rows.sort_values(["scale", "model"], key=sort_summary_key)
    table = rows[
        [
            "scale",
            "model",
            "n",
            "mse",
            "mae",
            "nmse",
            "direction_accuracy_nonzero",
            "corr",
        ]
    ].copy()
    return frame_to_markdown(table)


def estimate_table_lines(frame: pd.DataFrame, *, split: str) -> list[str]:
    rows = frame[frame["split"] == split].copy()
    table = rows[
        [
            "scale",
            "source",
            "full_sample_count",
            "full_reference_raw_mse",
            "estimated_scale_aware_mse",
            "estimated_scale_aware_mae",
            "estimated_direction_accuracy_nonzero",
        ]
    ]
    return frame_to_markdown(table)


def estimate_vs_actual_lines(estimate: pd.DataFrame, full_summary: pd.DataFrame) -> list[str]:
    rows: list[dict[str, Any]] = []
    for scale in SCALE_ORDER:
        est = estimate[(estimate["split"] == "test") & (estimate["scale"] == scale)]
        actual = full_summary[
            (full_summary["split"] == "test")
            & (full_summary["scale"] == scale)
            & (full_summary["model"] == "scale_aware_asd_patchtst")
        ]
        if est.empty or actual.empty:
            continue
        rows.append(
            {
                "scale": scale,
                "estimated_mse": float(est.iloc[0]["estimated_scale_aware_mse"]),
                "actual_mse": float(actual.iloc[0]["mse"]),
                "actual_minus_estimated_mse": float(actual.iloc[0]["mse"] - est.iloc[0]["estimated_scale_aware_mse"]),
                "estimated_mae": float(est.iloc[0]["estimated_scale_aware_mae"]),
                "actual_mae": float(actual.iloc[0]["mae"]),
            }
        )
    if not rows:
        return ["没有可比的 full estimate 与 actual 结果。"]
    return frame_to_markdown(pd.DataFrame(rows))


def frame_to_markdown(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["无可用结果。"]
    formatted = frame.copy()
    for column in formatted.columns:
        if pd.api.types.is_numeric_dtype(formatted[column]):
            formatted[column] = formatted[column].map(format_number)
    headers = [str(column) for column in formatted.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in formatted.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in formatted.columns) + " |")
    return lines


def format_number(value: Any) -> str:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(x):
        return "nan"
    if abs(x) >= 1000:
        return f"{x:,.0f}"
    if abs(x) >= 0.01:
        return f"{x:.4f}"
    return f"{x:.4e}"


def format_seconds(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f} 秒"
    if seconds < 3600:
        return f"{seconds / 60:.1f} 分钟"
    return f"{seconds / 3600:.2f} 小时"


def sort_summary_key(series: pd.Series) -> pd.Series:
    if series.name == "scale":
        order = {name: idx for idx, name in enumerate(SCALE_ORDER)}
        return series.map(order).fillna(99)
    if series.name == "model":
        order = {name: idx for idx, name in enumerate(("zero", "last_return", *MODEL_ORDER[1:]))}
        return series.map(order).fillna(99)
    return series


def select_metric_row(summary: pd.DataFrame, split: str, scale: str, model: str) -> dict[str, float]:
    rows = summary[(summary["split"] == split) & (summary["scale"] == scale) & (summary["model"] == model)]
    if rows.empty:
        raise KeyError(f"missing row split={split} scale={scale} model={model}")
    row = rows.iloc[0]
    return {key: float(row[key]) for key in ["mse", "mae", "direction_accuracy_nonzero", "corr"]}


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(denominator, 1e-20))


def parse_stock_list(value: str) -> list[int]:
    stocks = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not stocks:
        raise ValueError("At least one training stock must be supplied.")
    return stocks


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [to_jsonable(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


if __name__ == "__main__":
    main()
