# ASB-Style Encoder PatchTST 实验报告

本轮只包含 second / minute / hour，不包含 day、MoE 或 LoRA。ASB 放在 PatchTST encoder 最后一层后，对 patch-token 序列做 learnable spectral filtering。

训练方式保持 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，三个 scale loss 平均后更新一次。

## 1. Round 1 Small Selection

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`；epochs=3；balanced steps/epoch=12；patch presets=['compact', 'short_second']；ASB init gates=[-4.0, -3.0]。

完整 small summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asb_encoder_patchtst\round1_small_summary.csv`

### Selection Ranking
| patch_preset | training_regime | init_gate | quality_pass | selection_score | second_mse_over_raw | second_dir_delta | minute_mse_over_raw | minute_dir_delta | minute_corr_delta | hour_nmse_delta_raw_minus_model | hour_nmse_delta_asd_minus_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | 1.0000 | 0.2190 | 0.9859 | 0.0255 | 0.9961 | 0.0578 | -0.0151 | 0.0946 | 6.3555e-03 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | 1.0000 | 0.2117 | 0.9859 | 0.0255 | 0.9960 | 0.0578 | -0.0149 | 0.0919 | 3.6614e-03 |

### Small Test Comparison
| patch_preset | model | init_gate | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | second | 1,024 | 1.0150e-06 | 6.3413e-04 | 1.0542 | 0.4627 | -0.0697 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | second | 1,024 | 1.0151e-06 | 6.3414e-04 | 1.0543 | 0.4617 | -0.0694 |
| compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | second | 1,024 | 1.0151e-06 | 6.3413e-04 | 1.0543 | 0.4617 | -0.0693 |
| compact | raw_joint | nan | second | 1,024 | 1.0127e-06 | 6.3993e-04 | 1.0518 | 0.5275 | -0.1051 |
| compact | zero | nan | minute | 1,024 | 1.6364e-06 | 8.3933e-04 | 1.0004 | 0.4818 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | minute | 1,024 | 1.6362e-06 | 8.3900e-04 | 1.0003 | 0.5407 | 0.0368 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | minute | 1,024 | 1.6361e-06 | 8.3900e-04 | 1.0003 | 0.5388 | 0.0369 |
| compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | minute | 1,024 | 1.6361e-06 | 8.3901e-04 | 1.0003 | 0.5368 | 0.0370 |
| compact | raw_joint | nan | minute | 1,024 | 1.6441e-06 | 8.4578e-04 | 1.0052 | 0.4966 | 0.0312 |
| compact | zero | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | hour | 200.0000 | 3.7533e-05 | 4.0700e-03 | 0.8010 | 0.6150 | 0.5484 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | hour | 200.0000 | 3.7394e-05 | 4.0614e-03 | 0.7981 | 0.6150 | 0.5491 |
| compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | hour | 200.0000 | 3.7289e-05 | 4.0549e-03 | 0.7958 | 0.6150 | 0.5496 |
| compact | raw_joint | nan | hour | 200.0000 | 4.0935e-05 | 4.2538e-03 | 0.8736 | 0.6300 | 0.5129 |

## 2. Round 2 Full Confirm

完整 full summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asb_encoder_patchtst\round2_full_summary.csv`

| patch_preset | model | init_gate | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | second | 263,640 | 9.4973e-07 | 6.1516e-04 | 0.9884 | 0.5226 | 0.1143 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | second | 263,640 | 9.4977e-07 | 6.1514e-04 | 0.9884 | 0.5231 | 0.1139 |
| compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | second | 263,640 | 9.4970e-07 | 6.1509e-04 | 0.9884 | 0.5231 | 0.1141 |
| compact | raw_joint | nan | second | 263,640 | 9.4859e-07 | 6.1349e-04 | 0.9872 | 0.5301 | 0.1143 |
| compact | zero | nan | minute | 3,120 | 1.5911e-06 | 8.1541e-04 | 1.0003 | 0.4878 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | minute | 3,120 | 1.5812e-06 | 8.1385e-04 | 0.9940 | 0.5170 | 0.0784 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | minute | 3,120 | 1.5811e-06 | 8.1367e-04 | 0.9940 | 0.5186 | 0.0786 |
| compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | minute | 3,120 | 1.5810e-06 | 8.1359e-04 | 0.9939 | 0.5177 | 0.0787 |
| compact | raw_joint | nan | minute | 3,120 | 1.5836e-06 | 8.1870e-04 | 0.9955 | 0.4990 | 0.0828 |
| compact | zero | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | hour | 200.0000 | 2.6405e-05 | 3.2886e-03 | 0.5635 | 0.7400 | 0.6709 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | hour | 200.0000 | 2.6704e-05 | 3.3104e-03 | 0.5699 | 0.7450 | 0.6675 |
| compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | hour | 200.0000 | 2.6689e-05 | 3.3092e-03 | 0.5696 | 0.7450 | 0.6677 |
| compact | raw_joint | nan | hour | 200.0000 | 2.6938e-05 | 3.3242e-03 | 0.5749 | 0.7300 | 0.6633 |

## 3. Round 3 Robustness

robustness 配置（按 full validation ASB selection score 选择）: `{'patch_preset': 'compact', 'training_regime': 'asb_encoder_frozen_base_train_asb_head', 'init_gate': -4.0, 'seed': 42, 'selection_score': 0.0038371141111922264, 'quality_pass': False, 'second_mse_over_raw': 0.9993041040505958, 'second_dir_delta': -0.006479196222088368, 'minute_mse_over_raw': 0.9981313192064155, 'minute_dir_delta': 0.01669941060903729, 'minute_corr_delta': 0.00223454450120398, 'hour_nmse_delta_raw_minus_model': -0.0034756652440330327, 'hour_nmse_delta_asd_minus_model': 0.002506804225918202, 'hour_nmse_delta_zero_minus_model': 0.4993816962710772}`

| patch_preset | model | init_gate | scale | mse_mean | mse_std | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | hour | 2.6170e-05 | 9.8668e-07 | 0.5585 | 0.0211 | 3.2734e-03 | 1.0347e-04 | 0.7200 | 0.0278 | 0.6715 | 0.0107 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | minute | 1.5828e-06 | 1.5009e-09 | 0.9950 | 9.4355e-04 | 8.1283e-04 | 7.7988e-07 | 0.5298 | 0.0105 | 0.0722 | 5.7843e-03 |
| compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | second | 9.4904e-07 | 7.8875e-10 | 0.9877 | 8.2087e-04 | 6.1391e-04 | 1.3337e-06 | 0.5317 | 0.0111 | 0.1145 | 2.8507e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | hour | 2.5914e-05 | 7.6408e-07 | 0.5531 | 0.0163 | 3.2534e-03 | 8.6451e-05 | 0.7133 | 0.0306 | 0.6743 | 7.9928e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | minute | 1.5833e-06 | 1.9222e-09 | 0.9953 | 1.2084e-03 | 8.1291e-04 | 8.5329e-07 | 0.5295 | 0.0112 | 0.0714 | 6.1079e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | second | 9.4903e-07 | 8.3561e-10 | 0.9877 | 8.6964e-04 | 6.1400e-04 | 1.3369e-06 | 0.5309 | 0.0107 | 0.1150 | 3.5134e-03 |
| compact | raw_joint | nan | hour | 2.6493e-05 | 6.1119e-07 | 0.5654 | 0.0130 | 3.2824e-03 | 8.7012e-05 | 0.7150 | 0.0218 | 0.6670 | 7.2972e-03 |
| compact | raw_joint | nan | minute | 1.5826e-06 | 9.2967e-10 | 0.9949 | 5.8444e-04 | 8.1441e-04 | 3.7150e-06 | 0.5215 | 0.0195 | 0.0794 | 4.2736e-03 |
| compact | raw_joint | nan | second | 9.4965e-07 | 1.9461e-09 | 0.9883 | 2.0254e-03 | 6.1362e-04 | 1.0832e-06 | 0.5331 | 6.1481e-03 | 0.1123 | 4.6480e-03 |

## 4. ASB Diagnostics

| round | patch_preset | training_regime | init_gate | scale | gate_mean | tau_mean | local_mask_mean | mean_abs_delta | global_filter_norm | local_filter_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round1_small | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | hour | 0.2955 | 0.1688 | 0.9710 | 7.7528e-03 | 0.0167 | 0.0165 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | minute | 0.0640 | 0.4603 | 0.7341 | 1.0289e-03 | 0.0120 | 0.0119 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | second | 0.0169 | 1.4235 | 0.3527 | 2.4911e-04 | 0.0167 | 0.0165 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | hour | 0.5340 | 0.1750 | 0.9700 | 0.0138 | 0.0165 | 0.0163 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | minute | 0.1568 | 0.4691 | 0.7294 | 2.4831e-03 | 0.0118 | 0.0117 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | second | 0.0447 | 1.4272 | 0.3517 | 6.4917e-04 | 0.0165 | 0.0163 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_only | -4.0000 | hour | 0.4772 | 0.1307 | 0.9773 | 0.0625 | 0.0263 | 0.0257 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_only | -4.0000 | minute | 0.0924 | 0.3971 | 0.7687 | 7.8559e-03 | 0.0274 | 0.0271 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_only | -4.0000 | second | 0.0178 | 1.3696 | 0.3661 | 9.1983e-04 | 0.0263 | 0.0257 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_only | -3.0000 | hour | 0.7086 | 0.1376 | 0.9763 | 0.0917 | 0.0259 | 0.0254 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_only | -3.0000 | minute | 0.2154 | 0.4061 | 0.7636 | 0.0182 | 0.0271 | 0.0269 |
| round1_small | compact | asb_encoder_frozen_base_train_asb_only | -3.0000 | second | 0.0470 | 1.3698 | 0.3660 | 2.4639e-03 | 0.0259 | 0.0254 |
| round1_small | compact | asb_encoder_joint | -4.0000 | hour | 0.1987 | 0.1449 | 0.9724 | 6.5680e-03 | 0.0180 | 0.0179 |
| round1_small | compact | asb_encoder_joint | -4.0000 | minute | 0.0527 | 0.4072 | 0.7453 | 1.0842e-03 | 0.0124 | 0.0121 |
| round1_small | compact | asb_encoder_joint | -4.0000 | second | 0.0173 | 1.3870 | 0.3658 | 3.1624e-04 | 0.0180 | 0.0179 |
| round1_small | compact | asb_encoder_joint | -3.0000 | hour | 0.4137 | 0.1424 | 0.9728 | 0.0137 | 0.0180 | 0.0178 |
| round1_small | compact | asb_encoder_joint | -3.0000 | minute | 0.1337 | 0.4015 | 0.7486 | 2.7489e-03 | 0.0124 | 0.0121 |
| round1_small | compact | asb_encoder_joint | -3.0000 | second | 0.0456 | 1.3791 | 0.3677 | 8.3488e-04 | 0.0180 | 0.0178 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_head | -4.0000 | hour | 0.0816 | 0.5301 | 0.8524 | 1.9298e-03 | 0.0128 | 0.0109 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_head | -4.0000 | minute | 0.0350 | 0.9195 | 0.5397 | 5.3889e-04 | 9.7106e-03 | 8.9625e-03 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_head | -4.0000 | second | 0.0162 | 2.1315 | 0.2637 | 2.2697e-04 | 0.0113 | 0.0108 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_head | -3.0000 | hour | 0.1809 | 2.6015 | 0.2740 | 3.8030e-03 | 0.0127 | 9.4426e-03 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_head | -3.0000 | minute | 0.0866 | 1.7900 | 0.3648 | 1.2257e-03 | 9.6594e-03 | 8.0857e-03 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_head | -3.0000 | second | 0.0429 | 2.1518 | 0.2604 | 5.9103e-04 | 0.0113 | 0.0101 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_only | -4.0000 | hour | 0.4060 | 0.2117 | 0.9575 | 0.0521 | 0.0235 | 0.0213 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_only | -4.0000 | minute | 0.0866 | 0.5895 | 0.6761 | 7.8646e-03 | 0.0256 | 0.0243 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_only | -4.0000 | second | 0.0168 | 1.9909 | 0.2886 | 8.8378e-04 | 0.0171 | 0.0164 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_only | -3.0000 | hour | 0.6396 | 0.2180 | 0.9561 | 0.0806 | 0.0232 | 0.0210 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_only | -3.0000 | minute | 0.2023 | 0.5970 | 0.6723 | 0.0180 | 0.0251 | 0.0238 |
| round1_small | short_second | asb_encoder_frozen_base_train_asb_only | -3.0000 | second | 0.0445 | 1.9904 | 0.2887 | 2.2795e-03 | 0.0169 | 0.0163 |
| round1_small | short_second | asb_encoder_joint | -4.0000 | hour | 0.1726 | 0.3530 | 0.9200 | 7.0230e-03 | 0.0171 | 0.0157 |
| round1_small | short_second | asb_encoder_joint | -4.0000 | minute | 0.0538 | 0.6923 | 0.6019 | 1.5119e-03 | 0.0134 | 0.0132 |
| round1_small | short_second | asb_encoder_joint | -4.0000 | second | 0.0170 | 2.0409 | 0.2749 | 3.9074e-04 | 0.0159 | 0.0151 |
| round1_small | short_second | asb_encoder_joint | -3.0000 | hour | 0.3735 | 0.3517 | 0.9204 | 0.0152 | 0.0171 | 0.0158 |
| round1_small | short_second | asb_encoder_joint | -3.0000 | minute | 0.1367 | 0.6895 | 0.6027 | 3.8445e-03 | 0.0133 | 0.0132 |
| round1_small | short_second | asb_encoder_joint | -3.0000 | second | 0.0450 | 2.0355 | 0.2758 | 1.0336e-03 | 0.0160 | 0.0152 |
| round2_full_confirm | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | hour | 0.9153 | 2.0871 | 0.2912 | 0.1462 | 0.0730 | 0.0959 |
| round2_full_confirm | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | minute | 0.3231 | 1.6427 | 0.2388 | 0.0600 | 0.0661 | 0.0819 |
| round2_full_confirm | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | second | 0.0322 | 1.7150 | 0.2554 | 5.2325e-03 | 0.0730 | 0.0959 |
| round2_full_confirm | compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | hour | 0.9374 | 2.1051 | 0.2877 | 0.1415 | 0.0676 | 0.0892 |
| round2_full_confirm | compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | minute | 0.5312 | 1.6334 | 0.2404 | 0.0946 | 0.0618 | 0.0766 |
| round2_full_confirm | compact | asb_encoder_frozen_base_train_asb_head | -3.0000 | second | 0.1133 | 1.7226 | 0.2540 | 0.0203 | 0.0676 | 0.0892 |
| round3_robustness | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | hour | 0.8951 | 2.0634 | 0.2808 | 0.1451 | 0.0786 | 0.1113 |
| round3_robustness | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | minute | 0.5234 | 1.4845 | 0.2689 | 0.0954 | 0.0697 | 0.0938 |
| round3_robustness | compact | asb_encoder_frozen_base_train_asb_head | -4.0000 | second | 0.3085 | 1.7724 | 0.2602 | 0.0982 | 0.0786 | 0.1113 |

## Interpretation Guardrails

- raw PatchTST 是全部训练的 baseline；当前输入端 ASD 对照为 `asd_frozen_encoder_train_head`。
- `asb_encoder_joint` 训练 ASB + PatchTST；`asb_encoder_frozen_base_train_asb_only` 只训练 ASB；`asb_encoder_frozen_base_train_asb_head` 只训练 ASB + scale heads。
- second 要求 MSE 不差于 raw 超过 1%，direction 不下降超过 1 个百分点；minute 要求 MSE 不差于 raw 超过 2%，且 direction/corr 至少一个不低于 raw；hour 要求 NMSE 优于 raw。
- 若 ASB 只改善 hour 但损伤 second，则只作为低频 scale 特化模块，不作为跨尺度主模型。
