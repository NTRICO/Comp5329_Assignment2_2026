# COMP5329 PatchTST Financial Forecasting

This repository now focuses on one assignment direction: improving PatchTST for
multi-scale intraday financial return forecasting.

The active data protocol uses Optiver-style intraday windows at three scales:

- `second`
- `minute`
- `hour`

Day/week trader and FinCast-position-control experiments were removed from the
tracked project history so teammates can read this checkout as a PatchTST
optimization codebase.

## Current Best Model

The current robust main model is:

```text
return window
-> per-scale normalization
-> scale-aware ASD denoising
-> scale-aware LoRA-MoE sequence adapter
-> gated residual back to raw return input
-> scale-specific patch embedding
-> shared PatchTST encoder
-> scale-specific linear head
-> future return prediction
```

In code this is the `gated_pre_return_asd_lora_moe_patchtst` configuration in
`scripts/evaluate_prepatch_asd_adapter_patchtst.py`, wrapped by the 32-stock
multi-seed runner:

```powershell
.\.conda-fincast\python.exe scripts\evaluate_gated_pre_asd_32stock_multiseed.py
```

The strongest follow-up candidate is the 15-channel variant:

```text
multi-channel intraday features
-> scale-aware ASD
-> shared PatchTST
-> LoRA-MoE
-> scale-specific head
```

Its multi-seed confirmation runner is:

```powershell
.\.conda-fincast\python.exe scripts\evaluate_multichannel_patchtst_multiseed.py
```

## Best Config Files

- `configs/recommended_patchtst_main.json`: current robust main model.
- `configs/multichannel_candidate.json`: higher-potential 15-channel candidate.

## Main Files

```text
src/baselines/patchtst_lora.py
    Self-contained PatchTST baseline and LoRA primitives.

src/baselines/scale_aware_asd_patchtst.py
    Multi-scale PatchTST, ASD, LoRA-MoE, ASB, pre-PatchTST adapters,
    multi-channel support, and experimental variants.

scripts/evaluate_gated_pre_asd_32stock_multiseed.py
    Main 32-stock multi-seed confirmation for the selected robust model.

scripts/evaluate_multichannel_patchtst.py
scripts/evaluate_multichannel_patchtst_multiseed.py
    15-channel raw / ASD / LoRA-MoE experiments.

scripts/evaluate_prepatch_asd_adapter_patchtst.py
    Pre-PatchTST ASD + adapter ablations.

scripts/evaluate_scale_aware_asd_patchtst.py
    General ASD / LoRA-MoE / ASB ablation runner.

scripts/build_optiver_second_feature_cache.py
scripts/build_optiver_feature_cache.py
    Cache builders for the local Optiver intraday data.
```

## Reports For Teammates

Start here:

```text
report/TEAMMATE_HANDOFF.md
report/README.md
```

The report files intentionally summarize results as percentage changes against
the relevant raw PatchTST baseline. Large CSV outputs and checkpoints are not
tracked by Git.

## Local Artifacts

These are intentionally ignored:

```text
.conda-fincast/
models/
data/cache/
data/high-frequency/
outputs/
FinCast-fts/
third_party/PatchTST/
```

The local interpreter used for all current experiments is:

```powershell
.\.conda-fincast\python.exe
```

If a teammate does not have the local cache, they need the Optiver cache file
under `data/cache/` or must rebuild it with the cache builder scripts.

## Quick Validation

```powershell
.\.conda-fincast\python.exe -m py_compile `
  src/baselines/patchtst_lora.py `
  src/baselines/scale_aware_asd_patchtst.py `
  scripts/evaluate_gated_pre_asd_32stock_multiseed.py `
  scripts/evaluate_multichannel_patchtst_multiseed.py
```

Focused tests:

```powershell
.\.conda-fincast\python.exe -m pytest tests/test_patchtst_lora.py tests/test_scale_aware_asd_patchtst.py
```
