# Multi-Channel ASD + LoRA-MoE Multi-Seed Confirmation

本报告只展示相对 `multichannel_raw_joint` 的百分比变化；不把原始归一化误差写入主表。

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_32stocks_512t.npz`; seeds: `42, 43, 44`; patch preset: `short_second`; epochs=5; balanced steps/epoch=50; channels=15.

## Test Improvement

| model | scale | n | improvement_pct_mean | direction_pp_mean | corr_delta_mean |
| --- | --- | --- | --- | --- | --- |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 620.0000 | 3.8272 | 1.1290 | 0.0237 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 3,224 | -0.4235 | -0.9031 | 2.6189e-03 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 4,096 | 3.2098 | 1.8184 | -0.0254 |

## Zero-Shot Improvement

| model | scale | n | improvement_pct_mean | direction_pp_mean | corr_delta_mean |
| --- | --- | --- | --- | --- | --- |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 480.0000 | 1.5364 | 0.7639 | 0.0106 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 1,024 | 0.6162 | 0.6849 | -3.7893e-03 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 4,096 | 9.4787 | 0.9611 | -0.0181 |

## Router Diagnostics

| model | scale | router_entropy_mean | router_balance_loss_mean | expert_prob_0_mean | expert_prob_1_mean | expert_prob_2_mean | expert_prob_3_mean | mean_abs_delta_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 0.2770 | 0.1122 | 0.5669 | 0.3277 | 9.2900e-03 | 0.0961 | 0.2475 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 0.0861 | 0.1744 | 0.0155 | 0.0113 | 0.3279 | 0.6454 | 0.5684 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 0.2895 | 0.1213 | 0.5366 | 0.0803 | 0.3554 | 0.0277 | 0.1723 |
| multichannel_raw_joint | hour | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | minute | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | second | nan | nan | nan | nan | nan | nan | nan |

## Channels

`wap1_log_return`, `wap2_log_return`, `mid1_log_return`, `mid2_log_return`, `rel_spread1`, `rel_spread2`, `imbalance1`, `imbalance2`, `log_total_size1`, `log_total_size2`, `total_imbalance`, `updates_in_second`, `is_observed_update`, `seconds_since_update`, `second_frac`
