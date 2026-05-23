# Current Project Progress

## Main Direction

The project now focuses on **PatchTST optimization for multi-scale intraday financial return forecasting**.

The active candidate model is:

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

Module interpretation:

- ASD: scale-aware spectral denoising.
- LoRA-MoE: financial-domain adaptation and scale/frequency specialization before PatchTST.
- PatchTST: shared temporal representation.
- Scale-specific head: map the shared representation to each scale target.

## Data Protocol

The previous cache used anonymous 600-second Optiver buckets. That made the `hour` scale a low-frequency proxy rather than a real hour.

The current protocol uses Optiver additional data:

```text
order_book_feature.csv: seconds_in_bucket 0-1799
order_book_target.csv:  seconds_in_bucket 1800-3599
        -> one 3600-second true-hour episode
```

Builder:

```text
scripts/build_optiver_additional_second_feature_cache.py
```

Default cache:

```text
data/cache/position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz
```

The local additional data has 10 usable dense stocks:

```text
train stocks: 0-8
zero-shot:    9
```

## Current Recommended Lengths

After a small context sweep, the current recommended preset is:

```text
balanced_60_45_24
```

| scale | input context | target |
| --- | ---: | --- |
| second | 60 seconds | next 10-second cumulative return |
| minute | 45 minutes | next 1-minute return |
| hour | 24 true-hour steps | next true-hour return |

Patch settings:

| scale | context | patch | stride |
| --- | ---: | ---: | ---: |
| second | 60 | 10 | 5 |
| minute | 45 | 9 | 4 |
| hour | 24 | 4 | 2 |

## Confirmation Run

Current confirmation setting:

```text
seeds: 42, 43, 44
epochs: 5
steps/epoch: 30
train cap: 4096 per scale
eval cap: 1024 per scale
models: raw PatchTST vs gated pre-ASD + LoRA-MoE
```

Relative test improvement over raw PatchTST:

| preset | second MSE | minute MSE | hour MSE | note |
| --- | ---: | ---: | ---: | --- |
| A: 60/30/10 | -0.03% +/- 0.43% | -0.37% +/- 0.23% | -0.46% +/- 1.54% | not useful |
| B: 60/45/24 | -0.41% +/- 0.31% | +1.29% +/- 0.07% | +0.31% +/- 0.56% | current main candidate |
| C: 120/58/48 | +0.21% +/- 0.22% | +0.40% +/- 1.84% | -10.24% +/- 6.41% | hour test has only 36 windows |

Relative zero-shot improvement over raw PatchTST:

| preset | second MSE | minute MSE | hour MSE | note |
| --- | ---: | ---: | ---: | --- |
| A: 60/30/10 | -0.37% +/- 0.48% | -0.12% +/- 0.53% | -0.23% +/- 1.29% | not useful |
| B: 60/45/24 | -0.67% +/- 0.68% | +1.65% +/- 0.29% | +0.74% +/- 0.68% | best balanced candidate |
| C: 120/58/48 | +1.30% +/- 1.28% | +1.05% +/- 0.36% | +2.69% +/- 1.27% | good zero-shot but weak test reliability |

## Interpretation

The original `60/30/10` setting was too short for minute/hour. Extending it to `60/45/24` improves minute and hour without creating the severe sample-size collapse seen in `120/58/48`.

The current main candidate should therefore be:

```text
balanced_60_45_24 + gated pre-ASD + LoRA-MoE
```

This is still a small capped confirmation, not a final full-data result. The next run should use the same preset with a larger training budget if time allows.

## Output Files

```text
outputs/prepatch_asd_adapter_patchtst/gated_pre_asd_true_hour_60_30_10_h10_confirm_10stocks_3seed/
outputs/context_sweep_true_hour/B_60_45_24_confirm_10stocks_3seed/
outputs/context_sweep_true_hour/C_120_58_48_confirm_10stocks_3seed/
outputs/context_sweep_true_hour/context_sweep_relative_improvement_aggregate.csv
```
