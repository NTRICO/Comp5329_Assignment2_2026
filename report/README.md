# PatchTST Result Index

This directory is the committed evidence index for the PatchTST-focused
assignment direction. Large raw outputs remain in `outputs/`, while this file
keeps the core tables and points to the detailed reports.

## Current Scope

The project now centers on PatchTST optimization for noisy financial time-series
forecasting. FinCast is used as a benchmark or data-generation reference where
helpful, but the main experiment surface is PatchTST and lightweight modules
around it.

No final contribution claim is selected here. This index only preserves the
current evidence and guardrails for later report writing.

## Core Experiment Tables

### FinCast-Paper-Test C Protocol

Source output: `outputs/hf_fincast_paper_test_c_protocol/summary.csv`

| frequency | model | test_mse | direction_accuracy_nonzero | test_corr | evaluated_windows |
| --- | --- | ---: | ---: | ---: | ---: |
| 1m | patchtst | 7.2233e-06 | 0.4696 | 0.1365 | 2048 |
| 1m | fincast_mean | 6.2587e-06 | 0.4972 | 0.3839 | 2048 |
| 1m | zero | 7.3400e-06 | 0.5320 | n/a | 2048 |
| 1h | patchtst | 1.2175e-04 | 0.5057 | -0.0388 | 2048 |
| 1h | fincast_mean | 1.3345e-04 | 0.5181 | 0.0255 | 2048 |
| 1h | zero | 1.2022e-04 | 0.5031 | n/a | 2048 |
| 1d | patchtst | 5.3937e-04 | 0.5020 | 0.0216 | 2048 |
| 1d | fincast_mean | 6.2601e-04 | 0.5109 | -0.0032 | 2048 |
| 1d | zero | 5.3609e-04 | 0.5005 | n/a | 2048 |
| 1wk | patchtst | 2.2820e-03 | 0.5238 | -0.0047 | 2048 |
| 1wk | fincast_mean | 2.9564e-03 | 0.5110 | 0.0253 | 2048 |
| 1wk | zero | 2.1837e-03 | 0.4723 | n/a | 2048 |

### Stock-0 Time-Scale PatchTST Baseline

Source output: `outputs/patchtst_stock0_timescales/summary.csv`

| scale | model | test_mse | direction_accuracy_nonzero | test_corr | test_windows |
| --- | --- | ---: | ---: | ---: | ---: |
| minute | patchtst | 1.0087e-06 | 0.5429 | 0.0986 | 3830 |
| minute | zero | 1.0179e-06 | 0.5008 | n/a | 3830 |
| hour_proxy | patchtst | 8.3774e-05 | 0.5164 | -0.1739 | 122 |
| hour_proxy | zero | 5.2563e-05 | 0.5082 | n/a | 122 |
| day_proxy | patchtst | 4.5055e-04 | 0.6667 | 0.3152 | 18 |
| day_proxy | zero | 5.1076e-04 | 0.5000 | n/a | 18 |

### Raw vs Spectral-Denoised PatchTST

Source output: `outputs/optiver_spectral_denoise_patchtst_scale_summary.csv`

| run | split | raw_mse | denoised_mse | mse_change_pct | raw_dir | denoised_dir |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 30s_second_512t | test | 9.4742e-07 | 9.4723e-07 | -0.0200 | 0.5380 | 0.5372 |
| 30s_second_512t | zero_shot | 8.8033e-07 | 8.7858e-07 | -0.1988 | 0.5083 | 0.5083 |
| minute_512t | test | 1.5872e-06 | 1.5824e-06 | -0.3060 | 0.5189 | 0.5263 |
| minute_512t | zero_shot | 1.5068e-06 | 1.5085e-06 | 0.1089 | 0.5005 | 0.4897 |
| hour_512t | test | 2.9876e-05 | 2.9635e-05 | -0.8050 | 0.6700 | 0.6850 |
| hour_512t | zero_shot | 2.9480e-05 | 2.9003e-05 | -1.6183 | 0.7167 | 0.7271 |

### Scale-Aware ASD Robustness

Source report: `report/asd_utility_verdict.md`

| scale | raw_nmse_mean | asd_nmse_mean | asd_nmse_change | raw_direction | asd_direction |
| --- | ---: | ---: | ---: | ---: | ---: |
| second | 0.988327 | 0.987681 | -0.065% | 0.533133 | 0.530944 |
| minute | 0.994920 | 0.995348 | +0.043% | 0.521516 | 0.529544 |
| hour | 0.565408 | 0.553066 | -2.183% | 0.715000 | 0.713333 |

### ASD + LoRA-MoE Guardrail

Source report: `report/scale_aware_asd_lora_moe_final_decision.md`

| scale | pass | raw_nmse_mean | asd_nmse_mean | combined_nmse_mean | combined_nmse_std |
| --- | ---: | ---: | ---: | ---: | ---: |
| second | 1 | 0.9908 | 0.9879 | 0.9883 | 1.0343e-03 |
| minute | 1 | 1.0008 | 1.0043 | 0.9981 | 4.6980e-03 |
| hour | 0 | 0.5397 | 0.5279 | 0.5399 | 9.1399e-03 |

The combined module is therefore kept as exploratory evidence, not as a selected
final model.

## Supplementary Tables

### Multi-Channel PatchTST

Source report: `report/multichannel_patchtst_asd_experiment.md`

| model | scale | nmse | direction_accuracy_nonzero | corr |
| --- | --- | ---: | ---: | ---: |
| multichannel_asd_frozen_encoder_train_head | second | 1.1051 | 0.5410 | 0.0938 |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 0.9946 | 0.5298 | 0.0954 |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 0.9151 | 0.5450 | 0.4884 |

### Level-ASD PatchTST

Source report: `report/level_asd_lora_moe_patchtst_experiment.md`

| model | scale | nmse | direction_accuracy_nonzero | corr |
| --- | --- | ---: | ---: | ---: |
| level_asd_raw_price_frozen_encoder_train_head | second | 1.0254 | 0.5000 | -0.0400 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | minute | 0.9989 | 0.5054 | 0.0405 |
| lora_moe_frozen_base_train_moe_head | hour | 0.7879 | 0.7050 | 0.5809 |

## Detailed Reports

- `asd_utility_verdict.md`: focused ASD utility and robustness verdict.
- `scale_aware_asd_patchtst_experiment.md`: initial scale-aware ASD run.
- `scale_aware_asd_patchtst_ablation.md`: ASD ablation details.
- `scale_aware_adapter_ablation_patchtst.md`: adapter and robustness sweeps.
- `scale_aware_adapter_targeted_robustness.md`: targeted LoRA-MoE robustness.
- `scale_aware_asd_lora_moe_patchtst_experiment.md`: combined ASD + LoRA-MoE run.
- `scale_aware_asd_lora_moe_final_decision.md`: guardrail report for combined models.
- `multichannel_patchtst_experiment.md`: 15-channel PatchTST run.
- `multichannel_patchtst_asd_experiment.md`: 15-channel ASD/adapters run.
- `level_asd_patchtst_experiment.md`: price-domain ASD comparison.
- `level_asd_lora_moe_patchtst_experiment.md`: price-domain ASD + LoRA-MoE run.
- `tslanet_intraday_baseline.md`: supplementary TSLANet-style baseline.

## Reporting Guardrails

- Compare PatchTST variants against zero-return and last-return baselines.
- Do not overstate second-scale gains when they are near the noise floor.
- Treat hour-scale results with sample-size caution.
- Keep ASD + LoRA-MoE as exploratory unless a later run passes a unified
  robustness gate across all selected scales.
