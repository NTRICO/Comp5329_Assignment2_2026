# LoRA-MoE Targeted Robustness

本轮只确认 4 个核心模型：raw PatchTST、head-only、LoRA-MoE + head、ASD + LoRA-MoE + head。
配置固定为 patch preset `short_second`，rank=8，ASD init gate=-4.0，seeds=[42, 43, 44]。

## 1. Mean/Std NMSE

| patch_preset | model | adapter_rank | scale | nmse_mean | nmse_std | mse_mean | mse_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | hour | 0.7698 | 0.0837 | 3.6071e-05 | 3.9241e-06 | 0.6733 | 0.0275 | 0.5761 | 0.0433 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | minute | 1.0031 | 0.0164 | 1.5854e-06 | 2.5898e-08 | 0.5112 | 3.5245e-03 | 0.0540 | 0.0462 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | second | 1.0215 | 0.0176 | 9.8358e-07 | 1.6973e-08 | 0.4853 | 0.0241 | 9.5448e-03 | 0.0299 |
| short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | hour | 0.7658 | 0.0544 | 3.5881e-05 | 2.5507e-06 | 0.7217 | 0.0161 | 0.5816 | 0.0321 |
| short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | minute | 1.0101 | 0.0145 | 1.5964e-06 | 2.2863e-08 | 0.5217 | 0.0185 | 0.0401 | 0.0492 |
| short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | second | 1.0234 | 7.7384e-03 | 9.8540e-07 | 7.4511e-09 | 0.4967 | 0.0369 | -4.1575e-03 | 0.0345 |
| short_second | raw_frozen_base_train_head | nan | hour | 0.8070 | 0.0614 | 3.7813e-05 | 2.8751e-06 | 0.7067 | 0.0189 | 0.5664 | 0.0296 |
| short_second | raw_frozen_base_train_head | nan | minute | 1.0078 | 5.6560e-03 | 1.5928e-06 | 8.9392e-09 | 0.5181 | 0.0128 | 0.0513 | 0.0443 |
| short_second | raw_frozen_base_train_head | nan | second | 1.0287 | 0.0136 | 9.9045e-07 | 1.3050e-08 | 0.4908 | 0.0251 | -2.8714e-03 | 0.0412 |
| short_second | raw_joint | nan | hour | 0.8625 | 0.0680 | 4.0414e-05 | 3.1882e-06 | 0.6650 | 0.0589 | 0.5285 | 0.0566 |
| short_second | raw_joint | nan | minute | 1.0118 | 0.0130 | 1.5990e-06 | 2.0585e-08 | 0.5155 | 0.0111 | 0.0454 | 0.0605 |
| short_second | raw_joint | nan | second | 1.0313 | 0.0117 | 9.9298e-07 | 1.1301e-08 | 0.4784 | 0.0214 | -3.3740e-04 | 0.0307 |

## 2. Router Expert Usage By Scale

| training_regime | adapter_rank | scale | expert_0 | expert_1 | expert_2 | expert_3 | router_entropy | router_balance_loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | hour | 0.3378 | 0.3288 | 0.0787 | 0.2547 | 0.4368 | 0.0753 |
| asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | minute | 0.3890 | 0.2793 | 0.0964 | 0.2353 | 0.4475 | 0.0758 |
| asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | second | 0.1834 | 0.1607 | 0.2871 | 0.3689 | 0.4853 | 0.0587 |
| lora_moe_frozen_base_train_moe_head | 8.0000 | hour | 0.3766 | 0.2905 | 0.1212 | 0.2117 | 0.4558 | 0.0680 |
| lora_moe_frozen_base_train_moe_head | 8.0000 | minute | 0.4467 | 0.1802 | 0.0257 | 0.3474 | 0.4705 | 0.0547 |
| lora_moe_frozen_base_train_moe_head | 8.0000 | second | 0.4169 | 0.1988 | 0.3446 | 0.0397 | 0.4888 | 0.0535 |

## 3. ASD Gate/Tau By Scale

| training_regime | adapter_rank | scale | gate_mean | gate_std | tau_mean | tau_std | mean_abs_delta_mean | mean_abs_delta_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | hour | 3.1160e-03 | 4.1338e-04 | 0.6799 | 0.3166 | 2.6987e-04 | 1.1849e-04 |
| asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | minute | 9.5952e-03 | 3.6271e-03 | 0.4416 | 0.1811 | 1.0524e-03 | 1.1507e-04 |
| asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | second | 0.0130 | 5.1031e-03 | 2.0247 | 1.1987 | 1.9050e-03 | 1.4567e-03 |

## Files

- summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_adapter_ablation_patchtst\targeted_robustness_short_second_rank8\targeted_robustness_summary.csv`
- aggregate: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_adapter_ablation_patchtst\targeted_robustness_short_second_rank8\targeted_robustness_aggregate.csv`
- router usage: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_adapter_ablation_patchtst\targeted_robustness_short_second_rank8\router_usage_by_scale.csv`
- ASD diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_adapter_ablation_patchtst\targeted_robustness_short_second_rank8\asd_gate_tau_by_scale.csv`
