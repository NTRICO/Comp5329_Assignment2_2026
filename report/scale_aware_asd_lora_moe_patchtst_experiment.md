# ASD + PatchTST + LoRA-MoE 跨尺度金融预测实验报告

本轮只包含 second / minute / hour，不包含 day。方法定位固定为：ASD 做 scale-aware denoising，shared PatchTST 保留通用时间序列表征，LoRA-style adapter 做金融域低秩适配，MoE router 做 second/minute/hour 的 scale/frequency specialization。

训练采用 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，主 loss 平均后，对 LoRA-MoE 加 `router_balance_weight * router_balance_loss`。

## 1. Small Selection

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`; epochs=3; balanced steps/epoch=12; patch presets=['compact', 'short_second']; ranks=[4, 8]; ASD init gates=[-4.0, -3.0].

完整 small summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_lora_moe_patchtst\round1_small_summary.csv`

### Selection Ranking
| patch_preset | training_regime | init_gate | adapter_rank | quality_pass | strong_pass | selection_score | second_mse_over_raw | second_dir_delta | minute_mse_over_raw | minute_dir_delta | minute_corr_delta | hour_mse_over_asd | hour_nmse_delta_raw_minus_model | hour_nmse_delta_asd_minus_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | 1.0000 | 1.0000 | 0.2769 | 1.0091 | -4.9068e-03 | 0.9889 | 0.0568 | 0.0469 | 0.9449 | 0.1050 | 0.0447 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | 1.0000 | 1.0000 | 0.2381 | 1.0040 | -9.8135e-04 | 1.0051 | -3.9331e-03 | 0.0298 | 0.9636 | 0.1066 | 0.0318 |

### Small Test Comparison
| patch_preset | model | init_gate | adapter_rank | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | nan | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | second | 1,024 | 9.6719e-07 | 6.1808e-04 | 1.0045 | 0.5029 | 0.0466 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | second | 1,024 | 9.9975e-07 | 6.4352e-04 | 1.0383 | 0.4617 | 0.0282 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 1,024 | 9.6515e-07 | 6.1725e-04 | 1.0024 | 0.5324 | 0.0402 |
| compact | raw_joint | nan | nan | second | 1,024 | 9.6628e-07 | 6.2069e-04 | 1.0035 | 0.4764 | 0.0646 |
| compact | zero | nan | nan | minute | 1,024 | 1.6364e-06 | 8.3933e-04 | 1.0004 | 0.4818 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1,024 | 1.6301e-06 | 8.3867e-04 | 0.9966 | 0.5221 | 0.0629 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | minute | 1,024 | 1.6309e-06 | 8.3887e-04 | 0.9971 | 0.5289 | 0.0579 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 1,024 | 1.6329e-06 | 8.3862e-04 | 0.9983 | 0.5289 | 0.0493 |
| compact | raw_joint | nan | nan | minute | 1,024 | 1.6311e-06 | 8.4257e-04 | 0.9972 | 0.4946 | 0.0814 |
| compact | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 3.9266e-05 | 4.1565e-03 | 0.8380 | 0.5950 | 0.5234 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | hour | 200.0000 | 3.6877e-05 | 3.9679e-03 | 0.7870 | 0.6800 | 0.5495 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 200.0000 | 3.6884e-05 | 3.9826e-03 | 0.7872 | 0.6650 | 0.5568 |
| compact | raw_joint | nan | nan | hour | 200.0000 | 4.1760e-05 | 4.3426e-03 | 0.8912 | 0.5600 | 0.4917 |
| short_second | zero | nan | nan | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | second | 1,024 | 9.8925e-07 | 6.2427e-04 | 1.0274 | 0.4686 | -0.0334 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | second | 1,024 | 1.0031e-06 | 6.3814e-04 | 1.0418 | 0.4617 | -0.0222 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | second | 1,024 | 9.9261e-07 | 6.2848e-04 | 1.0309 | 0.5344 | -0.0440 |
| short_second | raw_joint | nan | nan | second | 1,024 | 9.7993e-07 | 6.1876e-04 | 1.0177 | 0.5029 | -0.0334 |
| short_second | zero | nan | nan | minute | 1,024 | 1.5852e-06 | 7.9712e-04 | 1.0030 | 0.4936 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1,024 | 1.5952e-06 | 7.9830e-04 | 1.0094 | 0.5357 | 0.0394 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | minute | 1,024 | 1.5780e-06 | 7.9950e-04 | 0.9985 | 0.5152 | 0.0480 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | minute | 1,024 | 1.5967e-06 | 7.9888e-04 | 1.0103 | 0.5259 | 0.0155 |
| short_second | raw_joint | nan | nan | minute | 1,024 | 1.6069e-06 | 8.0126e-04 | 1.0167 | 0.5064 | 0.0179 |
| short_second | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 4.1026e-05 | 4.2233e-03 | 0.8756 | 0.7000 | 0.5338 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | hour | 200.0000 | 3.9368e-05 | 4.1164e-03 | 0.8402 | 0.6750 | 0.5361 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | hour | 200.0000 | 3.8096e-05 | 4.0485e-03 | 0.8130 | 0.7150 | 0.5508 |
| short_second | raw_joint | nan | nan | hour | 200.0000 | 4.3795e-05 | 4.3949e-03 | 0.9347 | 0.6000 | 0.4649 |

## 2. Full Confirm

完整 full summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_lora_moe_patchtst\round2_full_summary.csv`

| patch_preset | model | init_gate | adapter_rank | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | nan | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | second | 263,640 | 9.5247e-07 | 6.1572e-04 | 0.9913 | 0.5273 | 0.1132 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | second | 263,640 | 9.4987e-07 | 6.1309e-04 | 0.9886 | 0.5378 | 0.1098 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 263,640 | 9.4910e-07 | 6.1274e-04 | 0.9878 | 0.5385 | 0.1136 |
| compact | raw_joint | nan | nan | second | 263,640 | 9.5198e-07 | 6.1363e-04 | 0.9908 | 0.5347 | 0.0995 |
| compact | zero | nan | nan | minute | 3,120 | 1.5911e-06 | 8.1541e-04 | 1.0003 | 0.4878 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 3,120 | 1.5823e-06 | 8.1338e-04 | 0.9947 | 0.5289 | 0.0740 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | minute | 3,120 | 1.5832e-06 | 8.1286e-04 | 0.9953 | 0.5244 | 0.0687 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 3,120 | 1.5823e-06 | 8.1243e-04 | 0.9947 | 0.5353 | 0.0748 |
| compact | raw_joint | nan | nan | minute | 3,120 | 1.5980e-06 | 8.1688e-04 | 1.0046 | 0.5119 | 0.0719 |
| compact | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 2.4648e-05 | 3.1486e-03 | 0.5260 | 0.7400 | 0.6900 |
| compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | hour | 200.0000 | 2.5073e-05 | 3.2283e-03 | 0.5351 | 0.7450 | 0.6869 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 200.0000 | 2.4780e-05 | 3.1590e-03 | 0.5289 | 0.7000 | 0.6883 |
| compact | raw_joint | nan | nan | hour | 200.0000 | 2.4648e-05 | 3.1193e-03 | 0.5260 | 0.7200 | 0.6886 |
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

## 3. Robustness

robustness 配置: `{'patch_preset': 'compact', 'training_regime': 'asd_lora_moe_frozen_base_train_adapters_head', 'init_gate': -4.0, 'adapter_rank': 4, 'seed': 42, 'selection_score': 0.02175708557223594, 'quality_pass': False, 'strong_pass': False, 'second_mse_over_raw': 0.998996775199212, 'second_dir_delta': 0.0011504751540120095, 'minute_mse_over_raw': 0.9988971278121639, 'minute_dir_delta': 0.012442698100851413, 'minute_corr_delta': -0.00786685821387021, 'hour_mse_over_asd': 1.011240390173051, 'hour_nmse_delta_raw_minus_model': 0.013478328202242429, 'hour_nmse_delta_asd_minus_model': -0.0051120136569088315, 'hour_nmse_delta_zero_minus_model': 0.5401643922038857}`

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

## 4. Per-Scale Oracle

oracle 表允许每个 scale 从 raw / ASD / ASB / LoRA-MoE / ASD+LoRA-MoE 中选 test MSE 最低者，只作为实用上界，不作为单一主模型 claim。

| scale | source | patch_preset | model | init_gate | adapter_rank | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| second | lora_moe_previous_full | compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | 263,640 | 9.4850e-07 | 6.1258e-04 | 0.9871 | 0.5428 | 0.1147 |
| minute | asd_lora_moe_current_full | short_second | lora_moe_frozen_base_train_moe_head | nan | 8.0000 | 1,040 | 1.5765e-06 | 7.9522e-04 | 0.9948 | 0.5298 | 0.0949 |
| hour | lora_moe_previous_full | short_second | asd_frozen_encoder_train_head | -4.0000 | nan | 200.0000 | 2.4594e-05 | 3.1000e-03 | 0.5249 | 0.7000 | 0.6899 |

## 5. Diagnostics

| round | patch_preset | training_regime | init_gate | adapter_rank | scale | asd_gate_mean | asd_tau_mean | asd_mean_abs_delta | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | scale_prior_prob_0 | scale_prior_prob_1 | scale_prior_prob_2 | scale_prior_prob_3 | moe_mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | hour | 2.9687e-03 | 0.5980 | 2.3222e-04 | 0.4462 | 0.0733 | 1.3370e-04 | 0.6471 | 0.0000e+00 | 0.3527 | 0.2064 | 0.3549 | 0.1414 | 0.2972 | 0.0610 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | minute | 3.2769e-03 | 0.4014 | 5.1107e-04 | 0.4907 | 0.0576 | 0.5316 | 0.4440 | 1.2464e-03 | 0.0232 | 0.2653 | 0.2501 | 0.1915 | 0.2932 | 0.0319 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | second | 0.0177 | 5.8745 | 5.6529e-03 | 0.4953 | 0.0360 | 1.0787e-03 | 0.3690 | 0.4874 | 0.1425 | 0.2316 | 0.2973 | 0.3182 | 0.1528 | 0.0709 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | hour | 0.0289 | 0.1890 | 7.2686e-04 | 0.4721 | 0.0639 | 0.0000e+00 | 0.0230 | 0.6065 | 0.3705 | 0.2157 | 0.2115 | 0.3177 | 0.2550 | 0.0351 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | minute | 0.0556 | 0.3415 | 7.4117e-03 | 0.4709 | 0.0582 | 0.0270 | 0.3431 | 0.6031 | 0.0268 | 0.1871 | 0.2189 | 0.3021 | 0.2918 | 0.0216 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | second | 8.5841e-03 | 2.6290 | 1.6563e-03 | 0.4902 | 0.0567 | 2.5086e-03 | 0.5575 | 0.4058 | 0.0342 | 0.2498 | 0.3065 | 0.3029 | 0.1409 | 0.0293 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 4.0000 | hour | 7.8792e-03 | 0.5998 | 6.1809e-04 | 0.4455 | 0.0737 | 1.2300e-04 | 0.6500 | 0.0000e+00 | 0.3499 | 0.2058 | 0.3614 | 0.1348 | 0.2980 | 0.0579 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 4.0000 | minute | 8.8195e-03 | 0.4016 | 1.3763e-03 | 0.4907 | 0.0548 | 0.5219 | 0.4421 | 1.4999e-03 | 0.0345 | 0.2597 | 0.2454 | 0.1958 | 0.2991 | 0.0302 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 4.0000 | second | 0.0459 | 5.8493 | 0.0146 | 0.4909 | 0.0467 | 1.2921e-03 | 0.5513 | 0.0959 | 0.3515 | 0.2396 | 0.3173 | 0.2879 | 0.1553 | 0.0327 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 8.0000 | hour | 0.0651 | 0.1815 | 1.5709e-03 | 0.4071 | 0.0846 | 0.0166 | 0.2506 | 0.7249 | 7.9697e-03 | 0.2693 | 0.2161 | 0.3279 | 0.1868 | 0.0362 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 8.0000 | minute | 0.1303 | 0.3360 | 0.0171 | 0.4597 | 0.0659 | 0.0255 | 0.3413 | 0.6295 | 3.6777e-03 | 0.1983 | 0.2258 | 0.3209 | 0.2550 | 0.0224 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 8.0000 | second | 0.0229 | 2.6259 | 4.4113e-03 | 0.4943 | 0.0562 | 2.2769e-03 | 0.5294 | 0.4403 | 0.0281 | 0.2519 | 0.3023 | 0.3060 | 0.1397 | 0.0298 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 4.0000 | hour | 3.5091e-03 | 0.6835 | 3.1242e-04 | 0.4762 | 0.0502 | 6.1554e-03 | 0.5446 | 0.3880 | 0.0612 | 0.2231 | 0.2702 | 0.2746 | 0.2321 | 0.1344 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 4.0000 | minute | 3.3694e-03 | 0.4110 | 5.3768e-04 | 0.4109 | 0.0836 | 0.7282 | 0.2299 | 0.0375 | 4.4297e-03 | 0.2969 | 0.2176 | 0.2356 | 0.2499 | 0.0650 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 4.0000 | second | 0.0216 | 6.3117 | 7.0389e-03 | 0.4809 | 0.0531 | 3.0190e-03 | 0.3320 | 0.0767 | 0.5884 | 0.2482 | 0.2938 | 0.2889 | 0.1691 | 0.0451 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 8.0000 | hour | 0.0320 | 0.2093 | 8.9001e-04 | 0.4746 | 0.0634 | 5.3887e-04 | 0.5453 | 0.4542 | 0.0000e+00 | 0.2239 | 0.3124 | 0.3307 | 0.1330 | 0.1133 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 8.0000 | minute | 0.0584 | 0.3565 | 8.1205e-03 | 0.4441 | 0.0695 | 0.6675 | 0.2829 | 0.0496 | 0.0000e+00 | 0.3096 | 0.2348 | 0.2595 | 0.1961 | 0.0830 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 8.0000 | second | 8.7758e-03 | 2.6633 | 1.7114e-03 | 0.4941 | 0.0400 | 1.0058e-03 | 0.4128 | 0.1091 | 0.4771 | 0.2286 | 0.3025 | 0.2974 | 0.1715 | 0.0331 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 4.0000 | hour | 9.2788e-03 | 0.6817 | 8.2393e-04 | 0.4778 | 0.0519 | 4.9109e-04 | 0.5378 | 0.4066 | 0.0551 | 0.2084 | 0.2749 | 0.2825 | 0.2343 | 0.1323 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 4.0000 | minute | 9.0707e-03 | 0.4121 | 1.4511e-03 | 0.4117 | 0.0846 | 0.7286 | 0.2377 | 0.0295 | 4.2111e-03 | 0.2995 | 0.2195 | 0.2320 | 0.2489 | 0.0617 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 4.0000 | second | 0.0551 | 6.2325 | 0.0179 | 0.4651 | 0.0677 | 1.6979e-03 | 0.3336 | 0.0253 | 0.6393 | 0.2426 | 0.2971 | 0.2799 | 0.1803 | 0.0443 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 8.0000 | hour | 0.0721 | 0.1998 | 1.9152e-03 | 0.4768 | 0.0629 | 1.0758e-03 | 0.5354 | 0.4636 | 0.0000e+00 | 0.2346 | 0.3056 | 0.3292 | 0.1306 | 0.1154 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 8.0000 | minute | 0.1359 | 0.3483 | 0.0185 | 0.4392 | 0.0735 | 0.6803 | 0.2824 | 0.0373 | 0.0000e+00 | 0.3144 | 0.2359 | 0.2536 | 0.1961 | 0.0845 |
| round1_small | compact | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 8.0000 | second | 0.0236 | 2.6684 | 4.6176e-03 | 0.4916 | 0.0423 | 8.2404e-04 | 0.1061 | 0.3782 | 0.5149 | 0.2278 | 0.2919 | 0.3055 | 0.1749 | 0.0381 |
| round1_small | compact | asd_lora_moe_joint | -4.0000 | 4.0000 | hour | 5.1457e-03 | 5.5184 | 2.7401e-03 | 0.4833 | 0.0646 | 7.6502e-04 | 0.4314 | 0.0000e+00 | 0.5679 | 0.2662 | 0.2913 | 0.1123 | 0.3302 | 0.0247 |
| round1_small | compact | asd_lora_moe_joint | -4.0000 | 4.0000 | minute | 3.9078e-03 | 0.8968 | 1.2898e-03 | 0.3138 | 0.1172 | 0.1242 | 0.8379 | 2.7558e-04 | 0.0377 | 0.2152 | 0.3108 | 0.1775 | 0.2966 | 0.0427 |
| round1_small | compact | asd_lora_moe_joint | -4.0000 | 4.0000 | second | 0.0445 | 10.3964 | 0.0148 | 0.3965 | 0.0937 | 5.7287e-04 | 4.6788e-03 | 0.2422 | 0.7525 | 0.2497 | 0.2616 | 0.2874 | 0.2013 | 0.0592 |
| round1_small | compact | asd_lora_moe_joint | -4.0000 | 8.0000 | hour | 0.2148 | 0.3580 | 0.0102 | 0.4285 | 0.0817 | 0.0000e+00 | 0.3039 | 0.6961 | 0.0000e+00 | 0.1989 | 0.2264 | 0.3761 | 0.1987 | 0.0470 |
| round1_small | compact | asd_lora_moe_joint | -4.0000 | 8.0000 | minute | 0.2496 | 0.6760 | 0.0639 | 0.4753 | 0.0645 | 0.0182 | 0.6045 | 0.3772 | 0.0000e+00 | 0.2102 | 0.2632 | 0.3016 | 0.2251 | 0.0509 |
| round1_small | compact | asd_lora_moe_joint | -4.0000 | 8.0000 | second | 6.8959e-03 | 2.4022 | 1.2340e-03 | 0.3339 | 0.1130 | 2.9346e-03 | 2.2937e-03 | 0.1755 | 0.8193 | 0.2492 | 0.2485 | 0.2804 | 0.2219 | 0.0266 |
| round1_small | compact | asd_lora_moe_joint | -3.0000 | 4.0000 | hour | 0.0131 | 5.7689 | 7.1445e-03 | 0.4895 | 0.0627 | 0.0000e+00 | 0.5209 | 0.0000e+00 | 0.4791 | 0.2552 | 0.3145 | 0.1123 | 0.3179 | 0.0246 |
| round1_small | compact | asd_lora_moe_joint | -3.0000 | 4.0000 | minute | 0.0104 | 0.9253 | 3.5354e-03 | 0.2937 | 0.1238 | 0.1143 | 0.8550 | 0.0000e+00 | 0.0307 | 0.2131 | 0.3200 | 0.1765 | 0.2904 | 0.0423 |
| round1_small | compact | asd_lora_moe_joint | -3.0000 | 4.0000 | second | 0.1092 | 10.3330 | 0.0364 | 0.4043 | 0.0915 | 4.6259e-04 | 4.1088e-03 | 0.2521 | 0.7433 | 0.2484 | 0.2622 | 0.2896 | 0.1999 | 0.0596 |
| round1_small | compact | asd_lora_moe_joint | -3.0000 | 8.0000 | hour | 0.6519 | 0.3260 | 0.0281 | 0.4747 | 0.0662 | 0.0000e+00 | 0.4146 | 0.5854 | 0.0000e+00 | 0.1861 | 0.2491 | 0.3545 | 0.2102 | 0.0509 |
| round1_small | compact | asd_lora_moe_joint | -3.0000 | 8.0000 | minute | 0.5808 | 0.6690 | 0.1473 | 0.4429 | 0.0779 | 0.0122 | 0.6859 | 0.3019 | 0.0000e+00 | 0.2021 | 0.2788 | 0.2890 | 0.2301 | 0.0536 |
| round1_small | compact | asd_lora_moe_joint | -3.0000 | 8.0000 | second | 0.0184 | 2.4016 | 3.2967e-03 | 0.3533 | 0.1072 | 2.8687e-03 | 1.9232e-03 | 0.1947 | 0.8005 | 0.2507 | 0.2462 | 0.2842 | 0.2188 | 0.0269 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | hour | 4.7973e-04 | 2.5430 | 1.4266e-04 | 0.4712 | 0.0636 | 0.4534 | 0.5466 | 0.0000e+00 | 0.0000e+00 | 0.1298 | 0.4550 | 0.2742 | 0.1410 | 0.1015 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | minute | 2.3277e-03 | 1.1094 | 6.7760e-04 | 0.4887 | 0.0617 | 0.5137 | 3.5184e-03 | 0.4828 | 0.0000e+00 | 0.1932 | 0.3699 | 0.2922 | 0.1446 | 0.0266 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | second | 0.0108 | 3.9299 | 2.8231e-03 | 0.4912 | 0.0617 | 0.5522 | 6.8819e-04 | 0.4378 | 9.3275e-03 | 0.2280 | 0.2474 | 0.2470 | 0.2776 | 0.0287 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | hour | 3.1127e-03 | 1.0135 | 4.0388e-04 | 0.4686 | 0.0647 | 0.5668 | 0.4332 | 0.0000e+00 | 0.0000e+00 | 0.1394 | 0.4421 | 0.2371 | 0.1815 | 0.0998 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | minute | 9.8831e-03 | 0.4315 | 1.1842e-03 | 0.4213 | 0.0849 | 0.7119 | 0.0000e+00 | 0.2878 | 2.7726e-04 | 0.2317 | 0.3085 | 0.2674 | 0.1923 | 0.0285 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | second | 7.1477e-03 | 2.2654 | 1.2166e-03 | 0.4916 | 0.0631 | 0.4498 | 5.2643e-04 | 0.5477 | 1.9059e-03 | 0.2244 | 0.2364 | 0.2835 | 0.2557 | 0.0116 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 4.0000 | hour | 1.2886e-03 | 2.5589 | 3.8524e-04 | 0.4698 | 0.0640 | 0.4454 | 0.5546 | 0.0000e+00 | 0.0000e+00 | 0.1280 | 0.4541 | 0.2748 | 0.1431 | 0.1055 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 4.0000 | minute | 6.2757e-03 | 1.1178 | 1.8392e-03 | 0.4889 | 0.0617 | 0.5135 | 3.5780e-03 | 0.4829 | 0.0000e+00 | 0.1921 | 0.3707 | 0.2911 | 0.1462 | 0.0268 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 4.0000 | second | 0.0287 | 3.8503 | 7.3591e-03 | 0.4911 | 0.0622 | 0.5531 | 6.8345e-04 | 0.4390 | 7.1838e-03 | 0.2289 | 0.2491 | 0.2480 | 0.2740 | 0.0288 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 8.0000 | hour | 8.1380e-03 | 1.0229 | 1.0651e-03 | 0.4678 | 0.0654 | 0.5756 | 0.4244 | 0.0000e+00 | 0.0000e+00 | 0.1420 | 0.4433 | 0.2321 | 0.1826 | 0.0962 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 8.0000 | minute | 0.0259 | 0.4343 | 3.1242e-03 | 0.4154 | 0.0869 | 0.7209 | 0.0000e+00 | 0.2787 | 3.0800e-04 | 0.2344 | 0.3061 | 0.2656 | 0.1939 | 0.0268 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | -3.0000 | 8.0000 | second | 0.0190 | 2.2557 | 3.2234e-03 | 0.4912 | 0.0631 | 0.4467 | 5.2628e-04 | 0.5504 | 2.3294e-03 | 0.2234 | 0.2368 | 0.2822 | 0.2577 | 0.0117 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 4.0000 | hour | 5.7825e-04 | 2.8068 | 1.8670e-04 | 0.4610 | 0.0631 | 0.4651 | 0.5349 | 0.0000e+00 | 0.0000e+00 | 0.1362 | 0.4475 | 0.2110 | 0.2053 | 0.2772 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 4.0000 | minute | 2.4406e-03 | 1.1324 | 7.2345e-04 | 0.4944 | 0.0505 | 0.0537 | 0.0000e+00 | 0.5007 | 0.4456 | 0.1520 | 0.3016 | 0.2804 | 0.2659 | 0.0803 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 4.0000 | second | 0.0123 | 4.4038 | 3.4249e-03 | 0.4928 | 0.0629 | 0.5382 | 7.2503e-04 | 0.4604 | 6.6847e-04 | 0.2437 | 0.2517 | 0.2738 | 0.2308 | 0.0633 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 8.0000 | hour | 3.6521e-03 | 1.1083 | 5.1545e-04 | 0.4602 | 0.0633 | 0.5404 | 0.4596 | 0.0000e+00 | 0.0000e+00 | 0.1290 | 0.4321 | 0.2647 | 0.1742 | 0.2399 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 8.0000 | minute | 0.0104 | 0.4489 | 1.2982e-03 | 0.4805 | 0.0625 | 0.4589 | 0.0000e+00 | 0.5381 | 2.9871e-03 | 0.1802 | 0.2781 | 0.3089 | 0.2328 | 0.1348 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -4.0000 | 8.0000 | second | 7.4771e-03 | 2.4344 | 1.3532e-03 | 0.4921 | 0.0634 | 0.4477 | 6.4766e-04 | 0.5507 | 9.4414e-04 | 0.2275 | 0.2356 | 0.2916 | 0.2454 | 0.0474 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 4.0000 | hour | 1.5617e-03 | 2.7926 | 5.0213e-04 | 0.4603 | 0.0633 | 0.4591 | 0.5409 | 0.0000e+00 | 0.0000e+00 | 0.1318 | 0.4362 | 0.2655 | 0.1665 | 0.2777 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 4.0000 | minute | 6.6189e-03 | 1.1328 | 1.9627e-03 | 0.4268 | 0.0753 | 0.1883 | 7.5786e-05 | 0.7110 | 0.1007 | 0.1488 | 0.2900 | 0.3229 | 0.2383 | 0.0993 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 4.0000 | second | 0.0319 | 4.3452 | 8.8334e-03 | 0.4917 | 0.0634 | 0.5482 | 7.5987e-04 | 0.4505 | 5.8392e-04 | 0.2444 | 0.2524 | 0.2718 | 0.2313 | 0.0629 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 8.0000 | hour | 9.5281e-03 | 1.1366 | 1.3769e-03 | 0.4628 | 0.0625 | 0.5091 | 0.4909 | 0.0000e+00 | 0.0000e+00 | 0.1293 | 0.4557 | 0.2208 | 0.1941 | 0.2393 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 8.0000 | minute | 0.0273 | 0.4514 | 3.4230e-03 | 0.4895 | 0.0505 | 0.5288 | 0.0000e+00 | 0.0604 | 0.4108 | 0.1774 | 0.2904 | 0.2514 | 0.2808 | 0.1457 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_only | -3.0000 | 8.0000 | second | 0.0199 | 2.4285 | 3.6009e-03 | 0.4877 | 0.0649 | 0.4248 | 6.2345e-04 | 0.5737 | 9.1570e-04 | 0.2249 | 0.2346 | 0.2986 | 0.2418 | 0.0483 |
| round1_small | short_second | asd_lora_moe_joint | -4.0000 | 4.0000 | hour | 1.5918e-03 | 10.7453 | 1.0948e-03 | 0.2132 | 0.1464 | 0.0904 | 0.9095 | 0.0000e+00 | 5.0475e-05 | 0.0936 | 0.5823 | 0.1395 | 0.1846 | 0.0674 |
| round1_small | short_second | asd_lora_moe_joint | -4.0000 | 4.0000 | minute | 3.5117e-03 | 2.2463 | 1.7355e-03 | 0.4909 | 0.0613 | 0.4219 | 0.0126 | 2.5986e-03 | 0.5630 | 0.1842 | 0.3492 | 0.1992 | 0.2675 | 0.0230 |
| round1_small | short_second | asd_lora_moe_joint | -4.0000 | 4.0000 | second | 0.0179 | 6.3140 | 5.8424e-03 | 0.4931 | 0.0620 | 0.5106 | 3.8761e-04 | 0.4872 | 1.7432e-03 | 0.2442 | 0.2433 | 0.2762 | 0.2362 | 0.0258 |
| round1_small | short_second | asd_lora_moe_joint | -4.0000 | 8.0000 | hour | 8.8661e-03 | 13.8161 | 6.2952e-03 | 0.4694 | 0.0662 | 0.4145 | 0.5855 | 0.0000e+00 | 0.0000e+00 | 0.1882 | 0.5549 | 0.1386 | 0.1182 | 0.1334 |
| round1_small | short_second | asd_lora_moe_joint | -4.0000 | 8.0000 | minute | 0.0116 | 1.6831 | 4.7571e-03 | 0.1621 | 0.1586 | 0.9384 | 0.0612 | 2.4174e-04 | 9.0670e-05 | 0.3370 | 0.3482 | 0.1724 | 0.1424 | 0.0125 |
| round1_small | short_second | asd_lora_moe_joint | -4.0000 | 8.0000 | second | 8.3913e-03 | 2.9318 | 1.7672e-03 | 0.4875 | 0.0558 | 0.3911 | 5.9994e-04 | 0.5641 | 0.0442 | 0.2038 | 0.2429 | 0.2605 | 0.2927 | 0.0116 |
| round1_small | short_second | asd_lora_moe_joint | -3.0000 | 4.0000 | hour | 4.0698e-03 | 10.7976 | 2.8018e-03 | 0.1920 | 0.1517 | 0.0774 | 0.9224 | 0.0000e+00 | 2.4371e-04 | 0.0892 | 0.5873 | 0.1354 | 0.1880 | 0.0657 |
| round1_small | short_second | asd_lora_moe_joint | -3.0000 | 4.0000 | minute | 9.2464e-03 | 2.2564 | 4.5808e-03 | 0.4852 | 0.0611 | 0.3819 | 0.0265 | 1.6745e-03 | 0.5899 | 0.1775 | 0.3652 | 0.1910 | 0.2662 | 0.0223 |
| round1_small | short_second | asd_lora_moe_joint | -3.0000 | 4.0000 | second | 0.0467 | 6.2510 | 0.0152 | 0.4934 | 0.0621 | 0.4944 | 3.2842e-04 | 0.5040 | 1.3513e-03 | 0.2445 | 0.2416 | 0.2810 | 0.2328 | 0.0258 |
| round1_small | short_second | asd_lora_moe_joint | -3.0000 | 8.0000 | hour | 0.0211 | 13.9893 | 0.0150 | 0.4693 | 0.0664 | 0.4114 | 0.5886 | 0.0000e+00 | 0.0000e+00 | 0.1877 | 0.5585 | 0.1390 | 0.1148 | 0.1295 |
| round1_small | short_second | asd_lora_moe_joint | -3.0000 | 8.0000 | minute | 0.0295 | 1.7026 | 0.0122 | 0.1579 | 0.1595 | 0.9406 | 0.0591 | 2.5067e-04 | 4.1918e-05 | 0.3401 | 0.3487 | 0.1724 | 0.1388 | 0.0123 |
| round1_small | short_second | asd_lora_moe_joint | -3.0000 | 8.0000 | second | 0.0226 | 2.9340 | 4.7613e-03 | 0.4861 | 0.0567 | 0.3860 | 5.5867e-04 | 0.5711 | 0.0423 | 0.2038 | 0.2421 | 0.2619 | 0.2922 | 0.0116 |
| round2_full_confirm | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | hour | 5.5550e-03 | 10.4742 | 3.8002e-03 | 0.3970 | 0.0575 | 0.5458 | 0.4259 | 0.0282 | 2.4040e-05 | 0.3270 | 0.3278 | 0.1933 | 0.1519 | 0.1188 |
| round2_full_confirm | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | minute | 0.0198 | 1.8321 | 0.0104 | 0.4305 | 0.0715 | 0.3605 | 0.6372 | 1.3145e-04 | 2.0874e-03 | 0.1787 | 0.4238 | 0.1523 | 0.2452 | 0.1866 |
| round2_full_confirm | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | second | 5.3263e-03 | 6.9604 | 1.6920e-03 | 0.4259 | 0.0852 | 0.0000e+00 | 0.2854 | 7.9194e-04 | 0.7138 | 0.2021 | 0.3205 | 0.2644 | 0.2130 | 0.1471 |
| round2_full_confirm | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | hour | 5.9819e-03 | 13.2473 | 4.2398e-03 | 0.4728 | 0.0551 | 0.4367 | 0.0194 | 0.5284 | 0.0155 | 0.2516 | 0.1858 | 0.4546 | 0.1079 | 0.1145 |
| round2_full_confirm | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | minute | 0.0144 | 0.5591 | 2.2181e-03 | 0.4226 | 0.0670 | 0.6800 | 0.0000e+00 | 0.2064 | 0.1136 | 0.3257 | 0.1912 | 0.2842 | 0.1989 | 0.1095 |
| round2_full_confirm | short_second | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 8.0000 | second | 8.6145e-03 | 8.0752 | 2.7605e-03 | 0.3349 | 0.1105 | 7.2211e-06 | 0.8098 | 0.1902 | 1.9184e-05 | 0.1240 | 0.4314 | 0.3689 | 0.0757 | 0.5360 |
| round3_robustness | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | hour | 2.4814e-03 | 8.9096 | 1.6576e-03 | 0.3470 | 0.0680 | 0.4431 | 0.3195 | 0.1907 | 0.0467 | 0.3162 | 0.3389 | 0.1509 | 0.1939 | 0.2686 |
| round3_robustness | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | minute | 0.0756 | 2.2231 | 0.0385 | 0.4607 | 0.0646 | 0.2951 | 0.5246 | 0.1760 | 4.2787e-03 | 0.2120 | 0.3685 | 0.2106 | 0.2089 | 0.2894 |
| round3_robustness | compact | asd_lora_moe_frozen_base_train_adapters_head | -4.0000 | 4.0000 | second | 2.8942e-03 | 7.7268 | 9.1571e-04 | 0.3447 | 0.1024 | 0.1755 | 0.0952 | 0.1637 | 0.5657 | 0.2432 | 0.2047 | 0.2559 | 0.2961 | 0.5794 |

## Interpretation Guardrails

- `asd_lora_moe_joint` 训练 ASD + PatchTST + LoRA-MoE 全部参数。
- `asd_lora_moe_frozen_base_train_adapters_only` 从 raw checkpoint 加载，只训练 ASD + LoRA-MoE。
- `asd_lora_moe_frozen_base_train_adapters_head` 从 raw checkpoint 加载，只训练 ASD + LoRA-MoE + scale heads，是本轮主候选。
- hour test 的 `n` 很小，约 200 个窗口；即使 full confirm 有提升，也需要 robustness 和 oracle 一起解释。
- 若 combined 没有通过三尺度 gate，结论应写为组合模块未形成稳定统一主模型，而不是模块完全无效。
