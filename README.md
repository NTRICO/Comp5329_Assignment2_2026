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

The same runner also includes
`gated_pre_return_asd_lora_moe_joint_patchtst`, which loads the raw PatchTST
checkpoint and then unfreezes ASD, LoRA-MoE, PatchTST, and heads together. A
focused 3-seed check found this joint route slightly better for hour but weaker
for second/minute, so it is kept as an exploratory variant rather than the
default model.

The current true-hour target protocol uses a 60/45/24 context layout and a
10-second cumulative target for the second scale:

```text
second -> context 60 seconds, predict next 10-second cumulative return
minute -> context 45 minutes, predict next 1-minute return
hour   -> context 24 true-hour time_ids, predict next true-hour return
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
    Main multi-seed confirmation for the selected robust model. The current
    defaults use the additional-data true-hour cache and 60/45/24 context.

scripts/evaluate_multichannel_patchtst.py
scripts/evaluate_multichannel_patchtst_multiseed.py
    15-channel raw / ASD / LoRA-MoE experiments.

scripts/evaluate_prepatch_asd_adapter_patchtst.py
    Pre-PatchTST ASD + adapter ablations.

scripts/evaluate_scale_aware_asd_patchtst.py
    General ASD / LoRA-MoE / ASB ablation runner.

scripts/build_optiver_second_feature_cache.py
scripts/build_optiver_additional_second_feature_cache.py
scripts/build_optiver_feature_cache.py
    Cache builders for the local Optiver intraday data.
```

## Additional Data True-Hour Cut

The original Optiver cache used by most existing reports is based on anonymous
600-second buckets. The additional Optiver data is cleaner for the scale story:
`time_id` is sequential and one `time_id` represents one hour. Its order book is
split into two half-hour files:

```text
order_book_feature.csv -> seconds_in_bucket 0-1799
order_book_target.csv  -> seconds_in_bucket 1800-3599
```

Use this builder to combine those halves into one 3600-second true-hour episode
per stock/time_id:

```powershell
.\.conda-fincast\python.exe scripts\build_optiver_additional_second_feature_cache.py
```

By default it writes:

```text
data/cache/position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz
```

The builder maps raw additional-data stock ids to dense `stock_0`, `stock_1`,
... names. The local additional dataset has 10 usable stocks, so the current
default split is `--train-stocks 0-8` and `--zero-shot-stock 9`. The cache stores `seconds_per_bucket=3600`, so the
minute scale now aggregates 60 one-minute levels for this cache, while the old
600-second cache still aggregates 10 one-minute levels.

The active recommended task uses patch preset `balanced_60_45_24`:

```text
second: context 60, patch 10, stride 5
minute: context 45, patch 9, stride 4
hour:   context 24, patch 4, stride 2
```

## Reports For Teammates

Start here:

```text
report/TEAMMATE_HANDOFF.md
report/data_handoff_for_teammate.md
report/context_sweep_true_hour_summary.md
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
