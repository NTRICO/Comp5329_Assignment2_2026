# ASD + PatchTST + LoRA-MoE 最终决策报告

本报告只确认 `short_second + rank=8 + ASD init gate=-4.0` 的多 seed 稳定性；没有新增 ASB、attention-level LoRA、MoE 层数或 day 数据。

训练仍采用 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，平均主 loss 后加入 `router_balance_weight * router_balance_loss`。

## 1. 已有 Full 结果摘要

已有 full summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_lora_moe_patchtst\round2_full_summary.csv`

| patch_preset | model | init_gate | adapter_rank | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_second | zero | nan | nan | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | second | 263,640 | 9.4926e-07 | 6.1279e-04 | 0.9879 | 0.5402 | 0.1129 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | second | 263,640 | 9.4921e-07 | 6.1296e-04 | 0.9879 | 0.5364 | 0.1109 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | second | 263,640 | 9.4975e-07 | 6.1363e-04 | 0.9884 | 0.5283 | 0.1091 |
| short_second | raw_joint | nan | nan | second | 263,640 | 9.5386e-07 | 6.1692e-04 | 0.9927 | 0.5163 | 0.0958 |
| short_second | zero | nan | nan | minute | 1,040 | 1.5905e-06 | 7.9701e-04 | 1.0036 | 0.4957 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1,040 | 1.5857e-06 | 7.9646e-04 | 1.0006 | 0.5308 | 0.0880 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | minute | 1,040 | 1.5895e-06 | 7.9755e-04 | 1.0030 | 0.5308 | 0.0860 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | minute | 1,040 | 1.5765e-06 | 7.9522e-04 | 0.9948 | 0.5298 | 0.0949 |
| short_second | raw_joint | nan | nan | minute | 1,040 | 1.5807e-06 | 7.9618e-04 | 0.9975 | 0.5192 | 0.0919 |
| short_second | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 2.5056e-05 | 3.1588e-03 | 0.5347 | 0.6800 | 0.6829 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | hour | 200.0000 | 2.4989e-05 | 3.1588e-03 | 0.5333 | 0.7300 | 0.6833 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | hour | 200.0000 | 2.5521e-05 | 3.2598e-03 | 0.5447 | 0.7350 | 0.6797 |
| short_second | raw_joint | nan | nan | hour | 200.0000 | 2.5995e-05 | 3.2755e-03 | 0.5548 | 0.6650 | 0.6836 |

## 2. Compact Robustness

已有 compact robustness: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_lora_moe_patchtst\round3_robustness_aggregate.csv`

| patch_preset | model | init_gate | adapter_rank | scale | mse_mean | mse_std | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 2.5332e-05 | 1.0271e-06 | 0.5406 | 0.0219 | 3.1776e-03 | 4.7227e-05 | 0.7300 | 0.0100 | 0.6816 | 0.0120 |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1.5826e-06 | 7.3722e-10 | 0.9949 | 4.6346e-04 | 8.1454e-04 | 1.0363e-06 | 0.5155 | 0.0116 | 0.0755 | 4.1807e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | second | 9.4993e-07 | 2.2271e-09 | 0.9886 | 2.3178e-03 | 6.1440e-04 | 1.4528e-06 | 0.5308 | 7.3426e-03 | 0.1162 | 3.5118e-03 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | hour | 2.5852e-05 | 1.3177e-06 | 0.5517 | 0.0281 | 3.2500e-03 | 6.4388e-05 | 0.7083 | 0.0437 | 0.6751 | 0.0168 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | minute | 1.5830e-06 | 7.2857e-10 | 0.9951 | 4.5802e-04 | 8.1304e-04 | 1.4348e-06 | 0.5272 | 0.0134 | 0.0752 | 5.6710e-03 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | second | 9.5135e-07 | 1.4531e-09 | 0.9901 | 1.5123e-03 | 6.1517e-04 | 1.8615e-06 | 0.5267 | 0.0113 | 0.1142 | 3.8866e-03 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 2.5512e-05 | 1.2856e-06 | 0.5445 | 0.0274 | 3.2052e-03 | 7.6054e-05 | 0.7083 | 7.6376e-03 | 0.6801 | 0.0129 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 1.5843e-06 | 2.3110e-09 | 0.9960 | 1.4529e-03 | 8.1232e-04 | 2.7267e-07 | 0.5323 | 5.4718e-03 | 0.0789 | 4.1648e-03 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 9.5117e-07 | 2.7733e-09 | 0.9899 | 2.8863e-03 | 6.1476e-04 | 2.8819e-06 | 0.5323 | 7.1715e-03 | 0.1155 | 4.0441e-03 |
| compact | raw_joint | nan | nan | hour | 2.5442e-05 | 9.5872e-07 | 0.5430 | 0.0205 | 3.1655e-03 | 7.1512e-05 | 0.7117 | 7.6376e-03 | 0.6790 | 0.0123 |
| compact | raw_joint | nan | nan | minute | 1.5899e-06 | 8.0458e-09 | 0.9995 | 5.0581e-03 | 8.1430e-04 | 2.3827e-06 | 0.5202 | 0.0123 | 0.0779 | 5.5159e-03 |
| compact | raw_joint | nan | nan | second | 9.5083e-07 | 1.9089e-09 | 0.9896 | 1.9867e-03 | 6.1357e-04 | 9.2016e-07 | 0.5337 | 5.5858e-03 | 0.1091 | 8.4834e-03 |

## 3. Short-Second Robustness

targeted summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_lora_moe_patchtst\round3_robustness_short_second_rank8\round3_short_second_rank8_summary.csv`

targeted aggregate: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_lora_moe_patchtst\round3_robustness_short_second_rank8\round3_short_second_rank8_aggregate.csv`

| patch_preset | model | init_gate | adapter_rank | scale | mse_mean | mse_std | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 2.4738e-05 | 4.9777e-07 | 0.5279 | 0.0106 | 3.1508e-03 | 1.8777e-05 | 0.7033 | 0.0252 | 0.6878 | 7.2121e-03 |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1.5915e-06 | 6.5085e-09 | 1.0043 | 4.1071e-03 | 7.9690e-04 | 1.1325e-06 | 0.5340 | 4.0032e-03 | 0.0700 | 0.0157 |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | second | 9.4925e-07 | 1.1864e-11 | 0.9879 | 1.2347e-05 | 6.1292e-04 | 1.4367e-07 | 0.5408 | 6.3205e-04 | 0.1129 | 6.5523e-04 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | hour | 2.5300e-05 | 4.2826e-07 | 0.5399 | 9.1399e-03 | 3.2046e-03 | 4.0200e-05 | 0.7317 | 0.0176 | 0.6806 | 3.7593e-03 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | minute | 1.5816e-06 | 7.4449e-09 | 0.9981 | 4.6980e-03 | 7.9769e-04 | 2.7120e-06 | 0.5218 | 0.0139 | 0.0866 | 0.0245 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | second | 9.4962e-07 | 9.9381e-10 | 0.9883 | 1.0343e-03 | 6.1362e-04 | 1.4109e-06 | 0.5323 | 0.0117 | 0.1110 | 1.6488e-03 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | hour | 2.5054e-05 | 4.3613e-07 | 0.5347 | 9.3079e-03 | 3.2175e-03 | 3.6614e-05 | 0.7467 | 0.0161 | 0.6879 | 0.0103 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | minute | 1.5804e-06 | 3.9239e-09 | 0.9973 | 2.4761e-03 | 7.9663e-04 | 2.1855e-06 | 0.5179 | 0.0105 | 0.0749 | 0.0230 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | second | 9.4992e-07 | 1.5233e-10 | 0.9886 | 1.5854e-04 | 6.1453e-04 | 9.2555e-07 | 0.5247 | 3.5895e-03 | 0.1101 | 1.5668e-03 |
| short_second | raw_joint | nan | nan | hour | 2.5286e-05 | 6.3704e-07 | 0.5397 | 0.0136 | 3.2080e-03 | 6.7046e-05 | 0.6933 | 0.0247 | 0.6850 | 2.4775e-03 |
| short_second | raw_joint | nan | nan | minute | 1.5860e-06 | 9.0600e-09 | 1.0008 | 5.7172e-03 | 7.9649e-04 | 1.1981e-06 | 0.5247 | 5.7959e-03 | 0.0774 | 0.0154 |
| short_second | raw_joint | nan | nan | second | 9.5201e-07 | 1.6330e-09 | 0.9908 | 1.6995e-03 | 6.1507e-04 | 1.6409e-06 | 0.5246 | 9.1570e-03 | 0.1029 | 6.9759e-03 |

hour test n: `200`，因此 hour 结果需要按低样本量解释。

## 4. 最终模型选择

结论：组合模块没有形成稳定统一主模型；`ASD+LoRA-MoE` 应保留为 exploratory / oracle 候选。

decision reason: ASD+LoRA-MoE does not pass the unified three-scale gate.
seed std warning: hour NMSE 的 seed 波动接近或超过模型间 margin。
fallback unified recommendation: `asd_frozen_encoder_train_head`，按 robustness mean NMSE 在非 combined 模型中选择。

| scale | pass | raw_mse_mean | combined_mse_mean | combined_mse_over_raw | raw_nmse_mean | asd_nmse_mean | combined_nmse_mean | combined_nmse_std | combined_nmse_delta_raw_minus_model | combined_nmse_delta_asd_minus_model | raw_direction_mean | combined_direction_mean | combined_direction_delta | raw_corr_mean | combined_corr_mean | combined_corr_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| second | 1.0000 | 9.5201e-07 | 9.4962e-07 | 0.9975 | 0.9908 | 0.9879 | 0.9883 | 1.0343e-03 | 2.4805e-03 | -3.8837e-04 | 0.5246 | 0.5323 | 7.6405e-03 | 0.1029 | 0.1110 | 8.0760e-03 |
| minute | 1.0000 | 1.5860e-06 | 1.5816e-06 | 0.9972 | 1.0008 | 1.0043 | 0.9981 | 4.6980e-03 | 2.7585e-03 | 6.2451e-03 | 0.5247 | 0.5218 | -2.8846e-03 | 0.0774 | 0.0866 | 9.1998e-03 |
| hour | 0.0000e+00 | 2.5286e-05 | 2.5300e-05 | 1.0005 | 0.5397 | 0.5279 | 0.5399 | 9.1399e-03 | -2.8264e-04 | -0.0120 | 0.6933 | 0.7317 | 0.0383 | 0.6850 | 0.6806 | -4.3752e-03 |

### Robustness Per-Scale Recommendation

| scale | model | init_gate | adapter_rank | mse_mean | mse_std | nmse_mean | nmse_std | direction_accuracy_nonzero_mean | corr_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| second | asd_frozen_encoder_train_head | -4.0000 | nan | 9.4925e-07 | 1.1864e-11 | 0.9879 | 1.2347e-05 | 0.5408 | 0.1129 |
| minute | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | 1.5804e-06 | 3.9239e-09 | 0.9973 | 2.4761e-03 | 0.5179 | 0.0749 |
| hour | asd_frozen_encoder_train_head | -4.0000 | nan | 2.4738e-05 | 4.9777e-07 | 0.5279 | 0.0106 | 0.7033 | 0.6878 |

## 5. Per-Scale Oracle

oracle 允许每个 scale 从 raw / ASD / ASB / LoRA-MoE / ASD+LoRA-MoE 中选 test MSE 最低者，只作为实用上界，不作为单一主模型 claim。

| scale | source | patch_preset | model | init_gate | adapter_rank | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| second | lora_moe_previous_full | compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | 263,640 | 9.4850e-07 | 6.1258e-04 | 0.9871 | 0.5428 | 0.1147 |
| minute | asd_lora_moe_targeted_robustness | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | 1,040 | 1.5746e-06 | 7.9505e-04 | 0.9937 | 0.5288 | 0.1114 |
| hour | asd_lora_moe_targeted_robustness | short_second | asd_frozen_encoder_train_head | -4.0000 | nan | 200.0000 | 2.4164e-05 | 3.1293e-03 | 0.5157 | 0.7000 | 0.6960 |

## 6. Diagnostics 解读

diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_lora_moe_patchtst\round3_robustness_short_second_rank8\short_second_rank8_diagnostics.csv`

| scale | asd_gate_mean | asd_tau_mean | asd_mean_abs_delta | moe_router_entropy | moe_router_balance_loss | moe_expert_prob_0 | moe_expert_prob_1 | moe_expert_prob_2 | moe_expert_prob_3 | moe_mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| hour | 2.8129e-03 | 13.7193 | 1.9942e-03 | 0.4091 | 0.0486 | 0.3376 | 0.1319 | 0.3730 | 0.1576 | 0.3264 |
| minute | 6.4807e-03 | 0.3592 | 8.5659e-04 | 0.3510 | 0.0851 | 0.5842 | 0.0734 | 0.2828 | 0.0596 | 0.5811 |
| second | 0.0345 | 8.6906 | 0.0102 | 0.2604 | 0.1279 | 5.9563e-04 | 0.2813 | 0.4569 | 0.2612 | 1.1572 |

## Guardrails

- 本报告的最终判断基于 seeds `42, 43, 44` 的 mean/std，而不是单 seed full test。
- 若 combined 失败，解释应写成“组合模块不稳定”，不是“ASD 或 LoRA-MoE 一定无效”。
- hour test 样本量小，必须结合 robustness 和 oracle 一起解释。
