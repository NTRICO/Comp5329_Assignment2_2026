# True-Hour Context Sweep Summary

## Setup

Data:

```text
data/cache/position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz
```

Split:

```text
train stocks: 0-8
zero-shot:    9
```

Training:

```text
seeds: 42, 43, 44
epochs: 5
steps/epoch: 30
train cap: 4096 per scale
eval cap: 1024 per scale
models: raw PatchTST vs gated pre-ASD + LoRA-MoE
```

Metrics below are relative improvement over raw PatchTST. Positive means the candidate is better.

## Presets

| preset | second | minute | hour |
| --- | --- | --- | --- |
| A | 60 sec | 30 min | 10 hours |
| B | 60 sec | 45 min | 24 hours |
| C | 120 sec | 58 min | 48 hours |

The originally proposed `minute=60` is not valid under the current no-cross-hour dataset builder because each true-hour episode only provides 60 minute levels and the task also needs a next-minute target.

## Test Result

| preset | second MSE | minute MSE | hour MSE | second direction | minute direction | hour direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: 60/30/10 | -0.03% +/- 0.43% | -0.37% +/- 0.23% | -0.46% +/- 1.54% | +0.89 pp | -0.65 pp | -1.15 pp |
| B: 60/45/24 | -0.41% +/- 0.31% | +1.29% +/- 0.07% | +0.31% +/- 0.56% | -0.13 pp | +3.78 pp | -1.06 pp |
| C: 120/58/48 | +0.21% +/- 0.22% | +0.40% +/- 1.84% | -10.24% +/- 6.41% | +0.03 pp | -0.53 pp | -6.48 pp |

## Zero-Shot Result

| preset | second MSE | minute MSE | hour MSE | second direction | minute direction | hour direction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A: 60/30/10 | -0.37% +/- 0.48% | -0.12% +/- 0.53% | -0.23% +/- 1.29% | -0.20 pp | +0.13 pp | -0.80 pp |
| B: 60/45/24 | -0.67% +/- 0.68% | +1.65% +/- 0.29% | +0.74% +/- 0.68% | -0.49 pp | -0.36 pp | +0.96 pp |
| C: 120/58/48 | +1.30% +/- 1.28% | +1.05% +/- 0.36% | +2.69% +/- 1.27% | +0.94 pp | +1.14 pp | +2.23 pp |

## Decision

Use **B: `balanced_60_45_24`** as the current main preset.

Reason:

- A is almost flat or slightly worse.
- B improves minute and hour on test, and also improves minute and hour on zero-shot.
- C has attractive zero-shot numbers, but its hour test set drops to only 36 windows because the 48-hour context is too long for the held-out split. Its hour test result is therefore not reliable and is visibly worse.

Current main candidate:

```text
balanced_60_45_24
+ gated pre-ASD
+ LoRA-MoE sequence adapter
+ shared PatchTST
```

## Output CSV

```text
outputs/context_sweep_true_hour/context_sweep_relative_improvement_by_seed.csv
outputs/context_sweep_true_hour/context_sweep_relative_improvement_aggregate.csv
```
