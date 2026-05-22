# LoRA-MoE PatchTST 跨尺度金融预测实验报告

本轮只包含 second / minute / hour，不包含 day。方法定位是：shared PatchTST 保留通用时间序列预测能力；LoRA-style low-rank adapter 负责把模型适配到金融 intraday return 分布；MoE router 负责不同 temporal scale 之间的专家选择。ASD/ASB 只作为对照和 oracle 候选。

训练仍采用 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，主 loss 平均后，对 LoRA-MoE 加 `router_balance_weight * router_balance_loss`。

实现说明：下面的数值结果来自第一版 LoRA-MoE router。根据 FinCast 的做法，代码已在本报告生成后进一步对齐为 `log(delta_seconds) + discrete scale id embedding`，并在 MoE router 中加入 scale-conditioned expert prior；下一次 rerun 会刷新这份报告中的结果。

## 1. Small Selection

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`; epochs=3; balanced steps/epoch=12; patch presets=['compact', 'short_second']; ranks=[4, 8].

完整 small summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_lora_moe_patchtst\round1_small_summary.csv`

### Selection Ranking
| patch_preset | training_regime | adapter_rank | quality_pass | strong_pass | selection_score | second_mse_over_raw | second_dir_delta | minute_mse_over_raw | minute_dir_delta | minute_corr_delta | hour_nmse_delta_raw_minus_model | hour_nmse_delta_asd_minus_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | lora_moe_frozen_base_train_moe_head | 4.0000 | 1.0000 | 1.0000 | 0.3727 | 0.9897 | 4.9068e-03 | 0.9955 | 0.0646 | -0.0129 | 0.1490 | 0.0608 |
| short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | 1.0000 | 1.0000 | 0.3671 | 0.9980 | 4.9068e-03 | 0.9937 | 0.0482 | -0.0340 | 0.1453 | 0.0813 |

### Small Test Comparison
| patch_preset | model | init_gate | adapter_rank | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | nan | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | second | 1,024 | 1.0150e-06 | 6.3413e-04 | 1.0542 | 0.4627 | -0.0697 |
| compact | raw_joint | nan | nan | second | 1,024 | 1.0127e-06 | 6.3993e-04 | 1.0518 | 0.5275 | -0.1051 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 1,024 | 1.0056e-06 | 6.2596e-04 | 1.0444 | 0.5079 | -0.0789 |
| compact | zero | nan | nan | minute | 1,024 | 1.6364e-06 | 8.3933e-04 | 1.0004 | 0.4818 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1,024 | 1.6362e-06 | 8.3900e-04 | 1.0003 | 0.5407 | 0.0368 |
| compact | raw_joint | nan | nan | minute | 1,024 | 1.6441e-06 | 8.4578e-04 | 1.0052 | 0.4966 | 0.0312 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 1,024 | 1.6413e-06 | 8.4070e-04 | 1.0035 | 0.5093 | 0.0275 |
| compact | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 3.7533e-05 | 4.0700e-03 | 0.8010 | 0.6150 | 0.5484 |
| compact | raw_joint | nan | nan | hour | 200.0000 | 4.0935e-05 | 4.2538e-03 | 0.8736 | 0.6300 | 0.5129 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 200.0000 | 3.4432e-05 | 3.8413e-03 | 0.7349 | 0.7200 | 0.5617 |
| short_second | zero | nan | nan | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | second | 1,024 | 9.8243e-07 | 6.2548e-04 | 1.0203 | 0.5295 | 0.0269 |
| short_second | raw_joint | nan | nan | second | 1,024 | 9.8821e-07 | 6.3196e-04 | 1.0263 | 0.4715 | 0.0205 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 1,024 | 9.8554e-07 | 6.2554e-04 | 1.0235 | 0.5334 | 0.0146 |
| short_second | zero | nan | nan | minute | 1,024 | 1.5852e-06 | 7.9712e-04 | 1.0030 | 0.4936 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1,024 | 1.5958e-06 | 8.0162e-04 | 1.0097 | 0.5044 | -0.0161 |
| short_second | raw_joint | nan | nan | minute | 1,024 | 1.5975e-06 | 8.0792e-04 | 1.0108 | 0.4946 | -0.0338 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 1,024 | 1.6019e-06 | 7.9913e-04 | 1.0136 | 0.5308 | -0.0199 |
| short_second | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 3.9706e-05 | 4.1507e-03 | 0.8474 | 0.6450 | 0.5062 |
| short_second | raw_joint | nan | nan | hour | 200.0000 | 4.2176e-05 | 4.3140e-03 | 0.9001 | 0.5800 | 0.4745 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 200.0000 | 3.6205e-05 | 3.9404e-03 | 0.7727 | 0.7050 | 0.5459 |

## 2. Full Confirm

完整 full summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_lora_moe_patchtst\round2_full_summary.csv`

| patch_preset | model | init_gate | adapter_rank | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | nan | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | second | 263,640 | 9.4973e-07 | 6.1516e-04 | 0.9884 | 0.5226 | 0.1143 |
| compact | raw_joint | nan | nan | second | 263,640 | 9.4859e-07 | 6.1349e-04 | 0.9872 | 0.5301 | 0.1143 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 263,640 | 9.4850e-07 | 6.1258e-04 | 0.9871 | 0.5428 | 0.1147 |
| compact | zero | nan | nan | minute | 3,120 | 1.5911e-06 | 8.1541e-04 | 1.0003 | 0.4878 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 3,120 | 1.5812e-06 | 8.1385e-04 | 0.9940 | 0.5170 | 0.0784 |
| compact | raw_joint | nan | nan | minute | 3,120 | 1.5836e-06 | 8.1870e-04 | 0.9955 | 0.4990 | 0.0828 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 3,120 | 1.5877e-06 | 8.1289e-04 | 0.9982 | 0.5196 | 0.0738 |
| compact | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 2.6405e-05 | 3.2886e-03 | 0.5635 | 0.7400 | 0.6709 |
| compact | raw_joint | nan | nan | hour | 200.0000 | 2.6938e-05 | 3.3242e-03 | 0.5749 | 0.7300 | 0.6633 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 200.0000 | 2.7440e-05 | 3.3605e-03 | 0.5856 | 0.7300 | 0.6592 |
| short_second | zero | nan | nan | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | second | 263,640 | 9.4978e-07 | 6.1286e-04 | 0.9885 | 0.5399 | 0.1085 |
| short_second | raw_joint | nan | nan | second | 263,640 | 9.5155e-07 | 6.1482e-04 | 0.9903 | 0.5218 | 0.1017 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 263,640 | 9.5303e-07 | 6.1694e-04 | 0.9918 | 0.5156 | 0.1005 |
| short_second | zero | nan | nan | minute | 1,040 | 1.5905e-06 | 7.9701e-04 | 1.0036 | 0.4957 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1,040 | 1.5889e-06 | 7.9664e-04 | 1.0026 | 0.5327 | 0.0815 |
| short_second | raw_joint | nan | nan | minute | 1,040 | 1.5833e-06 | 7.9763e-04 | 0.9991 | 0.5106 | 0.0702 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 1,040 | 1.5931e-06 | 8.0125e-04 | 1.0053 | 0.5298 | 0.0704 |
| short_second | zero | nan | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| short_second | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 200.0000 | 2.4594e-05 | 3.1000e-03 | 0.5249 | 0.7000 | 0.6899 |
| short_second | raw_joint | nan | nan | hour | 200.0000 | 2.4762e-05 | 3.1099e-03 | 0.5285 | 0.7150 | 0.6872 |
| short_second | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 200.0000 | 2.4831e-05 | 3.1710e-03 | 0.5299 | 0.7400 | 0.6886 |

## 3. Robustness

robustness 配置: `{'patch_preset': 'compact', 'training_regime': 'lora_moe_frozen_base_train_moe_head', 'init_gate': None, 'adapter_rank': 4, 'seed': 42, 'selection_score': 0.010233042640365873, 'quality_pass': False, 'strong_pass': False, 'second_mse_over_raw': 0.9998340655107301, 'second_dir_delta': 0.008169928289639916, 'minute_mse_over_raw': 0.9996393565049887, 'minute_dir_delta': 0.019973804846103504, 'minute_corr_delta': -0.004172700402321857, 'hour_nmse_delta_raw_minus_model': -0.001162740648934124, 'hour_nmse_delta_asd_minus_model': 0.004819728821017111, 'hour_nmse_delta_zero_minus_model': 0.5016946208661761}`

| patch_preset | model | init_gate | adapter_rank | scale | mse_mean | mse_std | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | hour | 2.5914e-05 | 7.6408e-07 | 0.5531 | 0.0163 | 3.2534e-03 | 8.6451e-05 | 0.7133 | 0.0306 | 0.6743 | 7.9928e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | minute | 1.5833e-06 | 1.9222e-09 | 0.9953 | 1.2084e-03 | 8.1291e-04 | 8.5329e-07 | 0.5295 | 0.0112 | 0.0714 | 6.1079e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | nan | second | 9.4903e-07 | 8.3561e-10 | 0.9877 | 8.6964e-04 | 6.1400e-04 | 1.3369e-06 | 0.5309 | 0.0107 | 0.1150 | 3.5134e-03 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | hour | 2.6810e-05 | 5.4596e-07 | 0.5722 | 0.0117 | 3.3536e-03 | 2.5635e-05 | 0.7133 | 0.0176 | 0.6689 | 0.0125 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | minute | 1.5853e-06 | 3.0486e-09 | 0.9966 | 1.9165e-03 | 8.1268e-04 | 2.2423e-07 | 0.5284 | 7.9270e-03 | 0.0743 | 2.5985e-03 |
| compact | lora_moe_frozen_base_train_moe_head | nan | 4.0000 | second | 9.5043e-07 | 2.5847e-09 | 0.9891 | 2.6900e-03 | 6.1481e-04 | 2.0323e-06 | 0.5303 | 0.0110 | 0.1139 | 9.0464e-04 |
| compact | raw_joint | nan | nan | hour | 2.6493e-05 | 6.1119e-07 | 0.5654 | 0.0130 | 3.2824e-03 | 8.7012e-05 | 0.7150 | 0.0218 | 0.6670 | 7.2972e-03 |
| compact | raw_joint | nan | nan | minute | 1.5826e-06 | 9.2967e-10 | 0.9949 | 5.8444e-04 | 8.1441e-04 | 3.7150e-06 | 0.5215 | 0.0195 | 0.0794 | 4.2736e-03 |
| compact | raw_joint | nan | nan | second | 9.4965e-07 | 1.9461e-09 | 0.9883 | 2.0254e-03 | 6.1362e-04 | 1.0832e-06 | 0.5331 | 6.1481e-03 | 0.1123 | 4.6480e-03 |

## 4. Per-Scale Oracle

oracle 表允许每个 scale 从 raw / ASD / ASB / LoRA-MoE 中选 test MSE 最低者，只作为实用上界，不作为单一主模型 claim。

| scale | source | patch_preset | model | adapter_rank | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| second | lora_moe_current_full | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | 263,640 | 9.4850e-07 | 6.1258e-04 | 0.9871 | 0.5428 | 0.1147 |
| minute | asb_previous_full | compact | asb_encoder_frozen_base_train_asb_head | nan | 3,120 | 1.5810e-06 | 8.1359e-04 | 0.9939 | 0.5177 | 0.0787 |
| hour | lora_moe_current_full | short_second | asd_frozen_encoder_train_head | nan | 200.0000 | 2.4594e-05 | 3.1000e-03 | 0.5249 | 0.7000 | 0.6899 |

## 5. Router Diagnostics

| round | patch_preset | training_regime | adapter_rank | scale | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round1_small | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.4608 | 0.0619 | 0.5122 | 1.9735e-03 | 9.5616e-04 | 0.4849 | 0.1334 |
| round1_small | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | minute | 0.4873 | 0.0627 | 0.5875 | 0.0136 | 4.6844e-03 | 0.3942 | 0.0576 |
| round1_small | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | second | 0.4975 | 0.0208 | 0.2877 | 0.4449 | 0.2253 | 0.0421 | 0.0362 |
| round1_small | compact | lora_moe_frozen_base_train_moe_head | 8.0000 | hour | 0.4499 | 0.0661 | 0.3706 | 0.6140 | 0.0154 | 0.0000e+00 | 0.1113 |
| round1_small | compact | lora_moe_frozen_base_train_moe_head | 8.0000 | minute | 0.4945 | 0.0620 | 0.4340 | 0.5566 | 9.4658e-03 | 0.0000e+00 | 0.0429 |
| round1_small | compact | lora_moe_frozen_base_train_moe_head | 8.0000 | second | 0.4969 | 0.0249 | 0.1299 | 0.0595 | 0.4240 | 0.3867 | 0.0127 |
| round1_small | compact | lora_moe_frozen_base_train_moe_only | 4.0000 | hour | 0.4596 | 0.0626 | 9.1969e-05 | 2.6305e-04 | 0.5203 | 0.4794 | 0.1431 |
| round1_small | compact | lora_moe_frozen_base_train_moe_only | 4.0000 | minute | 0.4949 | 0.0586 | 0.0000e+00 | 0.0172 | 0.5150 | 0.4678 | 0.0788 |
| round1_small | compact | lora_moe_frozen_base_train_moe_only | 4.0000 | second | 0.4969 | 0.0451 | 0.0189 | 0.4674 | 0.0572 | 0.4565 | 0.0677 |
| round1_small | compact | lora_moe_frozen_base_train_moe_only | 8.0000 | hour | 0.4430 | 0.0652 | 1.8217e-04 | 0.0270 | 0.3503 | 0.6225 | 0.1222 |
| round1_small | compact | lora_moe_frozen_base_train_moe_only | 8.0000 | minute | 0.4967 | 0.0389 | 0.0000e+00 | 0.3885 | 0.1220 | 0.4895 | 0.0824 |
| round1_small | compact | lora_moe_frozen_base_train_moe_only | 8.0000 | second | 0.4967 | 0.0439 | 0.0271 | 0.4561 | 0.0545 | 0.4623 | 0.0549 |
| round1_small | compact | lora_moe_joint | 4.0000 | hour | 0.4578 | 0.0620 | 0.5196 | 0.4776 | 3.3874e-04 | 2.4722e-03 | 0.1640 |
| round1_small | compact | lora_moe_joint | 4.0000 | minute | 0.4747 | 0.0694 | 0.3770 | 0.6210 | 0.0000e+00 | 2.0372e-03 | 0.0849 |
| round1_small | compact | lora_moe_joint | 4.0000 | second | 0.4965 | 0.0248 | 0.1243 | 0.0638 | 0.4008 | 0.4111 | 0.0177 |
| round1_small | compact | lora_moe_joint | 8.0000 | hour | 0.4594 | 0.0629 | 0.0000e+00 | 5.6018e-04 | 0.4662 | 0.5333 | 0.0735 |
| round1_small | compact | lora_moe_joint | 8.0000 | minute | 0.4909 | 0.0631 | 8.4491e-05 | 7.9182e-04 | 0.5403 | 0.4588 | 0.0533 |
| round1_small | compact | lora_moe_joint | 8.0000 | second | 0.4935 | 0.0286 | 0.1701 | 0.0123 | 0.4325 | 0.3851 | 0.0133 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.4514 | 0.0607 | 0.6011 | 0.0379 | 0.3605 | 4.2683e-04 | 0.1596 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | minute | 0.4930 | 0.0210 | 0.2714 | 0.3825 | 0.3373 | 8.7803e-03 | 0.0774 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | second | 0.4942 | 0.0533 | 0.0170 | 0.4925 | 0.4687 | 0.0218 | 0.0127 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | hour | 0.4687 | 0.0622 | 0.5215 | 0.0000e+00 | 0.4763 | 2.1721e-03 | 0.1544 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | minute | 0.4753 | 0.0627 | 0.3506 | 9.2006e-03 | 0.6140 | 0.0262 | 0.0566 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | second | 0.4920 | 0.0546 | 0.0242 | 0.5184 | 0.4458 | 0.0116 | 0.0141 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_only | 4.0000 | hour | 0.4649 | 0.0628 | 0.5228 | 1.6774e-05 | 0.4772 | 0.0000e+00 | 0.2275 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_only | 4.0000 | minute | 0.4867 | 0.0509 | 0.3343 | 0.0828 | 0.5787 | 4.2520e-03 | 0.1025 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_only | 4.0000 | second | 0.4893 | 0.0562 | 0.0187 | 0.5425 | 0.4244 | 0.0144 | 0.0381 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_only | 8.0000 | hour | 0.4668 | 0.0623 | 0.5102 | 1.4981e-05 | 0.4888 | 1.0625e-03 | 0.1826 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_only | 8.0000 | minute | 0.4808 | 0.0644 | 0.3793 | 2.5179e-03 | 0.6030 | 0.0152 | 0.0760 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_only | 8.0000 | second | 0.4938 | 0.0547 | 0.0211 | 0.4988 | 0.4686 | 0.0115 | 0.0407 |
| round1_small | short_second | lora_moe_joint | 4.0000 | hour | 0.4923 | 0.0625 | 0.4972 | 0.5028 | 0.0000e+00 | 0.0000e+00 | 0.0208 |
| round1_small | short_second | lora_moe_joint | 4.0000 | minute | 0.4791 | 0.0662 | 0.3981 | 0.5968 | 1.2532e-03 | 3.8796e-03 | 0.0125 |
| round1_small | short_second | lora_moe_joint | 4.0000 | second | 0.4905 | 0.0508 | 0.0463 | 0.5221 | 0.4223 | 9.3774e-03 | 0.0155 |
| round1_small | short_second | lora_moe_joint | 8.0000 | hour | 0.4826 | 0.0622 | 0.4804 | 0.0000e+00 | 0.5178 | 1.7884e-03 | 0.0424 |
| round1_small | short_second | lora_moe_joint | 8.0000 | minute | 0.4901 | 0.0584 | 0.4356 | 2.1995e-03 | 0.5417 | 0.0205 | 0.0286 |
| round1_small | short_second | lora_moe_joint | 8.0000 | second | 0.4926 | 0.0186 | 0.4069 | 0.2061 | 0.3376 | 0.0495 | 0.0176 |
| round2_full_confirm | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.3591 | 0.0453 | 0.5373 | 0.3707 | 0.0271 | 0.0649 | 0.3219 |
| round2_full_confirm | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | minute | 0.4835 | 0.0507 | 0.5582 | 0.3683 | 0.0000e+00 | 0.0735 | 0.3840 |
| round2_full_confirm | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | second | 0.4826 | 0.0559 | 0.5291 | 0.4392 | 0.0302 | 1.4680e-03 | 0.4544 |
| round2_full_confirm | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.3660 | 0.0694 | 0.0000e+00 | 0.6272 | 0.3663 | 6.4744e-03 | 0.4167 |
| round2_full_confirm | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | minute | 0.4592 | 0.0589 | 6.9667e-05 | 0.5511 | 0.4258 | 0.0231 | 0.3981 |
| round2_full_confirm | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | second | 0.2906 | 0.1269 | 0.8595 | 0.1379 | 2.3690e-03 | 1.8929e-04 | 1.2319 |
| round3_robustness | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.4242 | 0.0434 | 0.3654 | 0.2574 | 0.3107 | 0.0665 | 0.1966 |
| round3_robustness | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | minute | 0.4855 | 0.0587 | 0.3660 | 0.2620 | 0.3422 | 0.0298 | 0.2258 |
| round3_robustness | compact | lora_moe_frozen_base_train_moe_head | 4.0000 | second | 0.4731 | 0.0653 | 0.1774 | 0.4873 | 0.3332 | 2.0786e-03 | 0.2628 |

## Interpretation Guardrails

- `lora_moe_joint` 训练 PatchTST + LoRA-MoE 全部参数。
- `lora_moe_frozen_base_train_moe_only` 从 raw checkpoint 加载 shared backbone 和 heads，只训练 LoRA-MoE。
- `lora_moe_frozen_base_train_moe_head` 从 raw checkpoint 加载并冻结 encoder/patch/scale embedding，只训练 LoRA-MoE + scale heads，是本轮推荐主候选。
- 如果没有统一 LoRA-MoE 通过三尺度 gate，结论应写为单一跨尺度最优模型未形成，并使用 oracle 表说明 per-scale 上界。
