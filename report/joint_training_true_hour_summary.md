# Joint Training True-Hour Summary

This run answers one question: should the current module stack be trained
jointly with the PatchTST backbone, instead of freezing the raw PatchTST
checkpoint and only training adapters plus heads?

## Setup

- Data: Optiver additional true-hour cache, train stocks `0-8`, zero-shot stock `9`.
- Preset: `balanced_60_45_24`.
- Target: second predicts the next 10-second cumulative return; minute/hour predict the next 1 unit.
- Seeds: `42, 43, 44`.
- Budget: 5 epochs, 30 balanced multi-scale steps per epoch.
- Frozen route: freeze PatchTST and train pre-ASD, LoRA-MoE, and scale heads.
- Joint route: load the raw PatchTST checkpoint, then unfreeze pre-ASD, LoRA-MoE, PatchTST, and heads.

## Test Relative To Raw PatchTST

| model | scale | n | MSE improvement vs raw | MAE improvement vs raw | Direction delta | Corr delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen ASD+LoRA-MoE+head | second | 1024 | -0.41% +/- 0.31% | -0.34% +/- 0.19% | -0.13 pp +/- 2.29 pp | +0.0042 +/- 0.0094 |
| frozen ASD+LoRA-MoE+head | minute | 1024 | +1.29% +/- 0.07% | +1.17% +/- 0.16% | +3.78 pp +/- 0.95 pp | -0.0008 +/- 0.0045 |
| frozen ASD+LoRA-MoE+head | hour | 252 | +0.31% +/- 0.56% | +0.18% +/- 1.26% | -1.06 pp +/- 3.30 pp | -0.0088 +/- 0.0109 |
| joint ASD+LoRA-MoE+PatchTST+head | second | 1024 | -0.67% +/- 0.50% | -0.55% +/- 0.42% | -0.39 pp +/- 0.51 pp | +0.0042 +/- 0.0113 |
| joint ASD+LoRA-MoE+PatchTST+head | minute | 1024 | +1.08% +/- 0.55% | +0.99% +/- 0.54% | +3.03 pp +/- 2.78 pp | -0.0017 +/- 0.0142 |
| joint ASD+LoRA-MoE+PatchTST+head | hour | 252 | +0.71% +/- 1.08% | +0.66% +/- 1.52% | -1.19 pp +/- 1.59 pp | +0.0141 +/- 0.0293 |

## Zero-Shot Relative To Raw PatchTST

| model | scale | n | MSE improvement vs raw | MAE improvement vs raw | Direction delta | Corr delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| frozen ASD+LoRA-MoE+head | second | 1024 | -0.67% +/- 0.68% | -0.64% +/- 0.36% | -0.49 pp +/- 1.92 pp | -0.0048 +/- 0.0282 |
| frozen ASD+LoRA-MoE+head | minute | 1024 | +1.65% +/- 0.29% | +1.18% +/- 0.37% | -0.36 pp +/- 1.24 pp | -0.0018 +/- 0.0078 |
| frozen ASD+LoRA-MoE+head | hour | 488 | +0.74% +/- 0.68% | +0.74% +/- 1.37% | +0.96 pp +/- 2.56 pp | -0.0050 +/- 0.0046 |
| joint ASD+LoRA-MoE+PatchTST+head | second | 1024 | -0.61% +/- 1.32% | -0.51% +/- 0.61% | +0.75 pp +/- 0.84 pp | +0.0006 +/- 0.0402 |
| joint ASD+LoRA-MoE+PatchTST+head | minute | 1024 | +1.81% +/- 1.01% | +1.28% +/- 0.62% | +0.39 pp +/- 1.48 pp | +0.0085 +/- 0.0070 |
| joint ASD+LoRA-MoE+PatchTST+head | hour | 488 | +0.92% +/- 1.13% | +1.26% +/- 1.82% | +1.84 pp +/- 2.48 pp | +0.0003 +/- 0.0140 |

## Joint vs Frozen

Positive values mean the joint route has a larger MSE improvement than the
frozen route; negative values mean the frozen route is better.

| split | scale | joint MSE advantage over frozen |
| --- | --- | ---: |
| test | second | -0.26 pp +/- 0.78 pp |
| test | minute | -0.20 pp +/- 0.62 pp |
| test | hour | +0.40 pp +/- 1.03 pp |
| validation | second | -0.17 pp +/- 0.47 pp |
| validation | minute | +0.19 pp +/- 0.41 pp |
| validation | hour | +0.49 pp +/- 0.67 pp |
| zero_shot | second | +0.07 pp +/- 0.73 pp |
| zero_shot | minute | +0.15 pp +/- 0.74 pp |
| zero_shot | hour | +0.18 pp +/- 0.45 pp |

## Diagnostics

| model | scale | asd_gate_mean | asd_tau_mean | final_gate_mean | final_mean_abs_delta | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| frozen ASD+LoRA-MoE+head | second | 0.0105 | 2.7701 | 0.1484 | 0.0213 | 0.4884 | 0.0373 | 0.0701 | 0.2908 | 0.2008 | 0.4383 |
| frozen ASD+LoRA-MoE+head | minute | 0.0255 | 1.2334 | 0.0595 | 0.0009 | 0.4878 | 0.0193 | 0.3146 | 0.3234 | 0.2144 | 0.1477 |
| frozen ASD+LoRA-MoE+head | hour | 0.0250 | 0.7431 | 0.0409 | 0.0002 | 0.4919 | 0.0125 | 0.3422 | 0.2033 | 0.2871 | 0.1674 |
| joint ASD+LoRA-MoE+PatchTST+head | second | 0.0041 | 2.0641 | 0.1522 | 0.0370 | 0.4637 | 0.0315 | 0.2635 | 0.3780 | 0.1781 | 0.1803 |
| joint ASD+LoRA-MoE+PatchTST+head | minute | 0.0043 | 0.7811 | 0.2599 | 0.0413 | 0.3662 | 0.0651 | 0.2829 | 0.1936 | 0.1800 | 0.3435 |
| joint ASD+LoRA-MoE+PatchTST+head | hour | 0.0028 | 0.2430 | 0.3299 | 0.1404 | 0.2700 | 0.1017 | 0.4024 | 0.1889 | 0.1122 | 0.2965 |

## Conclusion

This result does not support making joint PatchTST fine-tuning the default
three-scale model. Joint training is slightly better on hour, but it weakens
second/minute and the hour gain is smaller than the seed-to-seed variation.

The default recommendation remains the frozen-backbone adapter/head route:

```text
return window
-> per-scale normalization
-> gated pre-ASD + LoRA-MoE
-> frozen raw PatchTST backbone
-> scale-specific heads
```

Joint training can be kept as an hour-oriented exploratory variant.

## Files

- Summary: `outputs/joint_training_true_hour/B_60_45_24_frozen_vs_joint_10stocks_3seed/summary_all.csv`
- Relative by seed: `outputs/joint_training_true_hour/B_60_45_24_frozen_vs_joint_10stocks_3seed/relative_improvement_by_seed.csv`
- Relative aggregate: `outputs/joint_training_true_hour/B_60_45_24_frozen_vs_joint_10stocks_3seed/relative_improvement_aggregate.csv`
- Joint vs frozen: `outputs/joint_training_true_hour/B_60_45_24_frozen_vs_joint_10stocks_3seed/joint_vs_frozen_mse_improvement_delta.csv`
- Diagnostics: `outputs/joint_training_true_hour/B_60_45_24_frozen_vs_joint_10stocks_3seed/diagnostics_compact_by_scale.csv`
