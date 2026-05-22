# 后续实验：训练机制、Patch Scaling 与 ASD 归因

本轮只包含 second / minute / hour，不包含 day，也不加入 MoE 或 LoRA。没有新增 smoke test；small experiment 本身是第一道运行验证。

训练方式固定为 balanced multi-scale step：每个 optimizer step 同时取 second、minute、hour 各一个 batch，三个 scale loss 平均后更新一次。

## 1. Round 1 Small Selection

数据 cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`；epochs=3；balanced steps/epoch=12；训练股票 `0,1,2,3,4,5,6,7,8,9`，zero-shot stock `10`。

完整 small summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_patchtst_ablation\round1_small_summary.csv`

### Selection Ranking
| patch_preset | training_regime | init_gate | quality_pass | selection_score | second_mse_over_raw | second_dir_delta | minute_mse_over_raw | minute_dir_delta | minute_corr_delta | hour_nmse_delta_raw_minus_model |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | asd_frozen_encoder_train_head | -2.0000 | 1.0000 | 0.2164 | 0.9859 | 0.0255 | 0.9961 | 0.0597 | -0.0153 | 0.0877 |
| compact | asd_frozen_encoder_train_head | -4.0000 | 1.0000 | 0.2161 | 0.9859 | 0.0275 | 0.9960 | 0.0578 | -0.0148 | 0.0882 |

### Top Config Test Metrics
| patch_preset | model | init_gate | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | second | 1,024 | 1.0150e-06 | 6.3413e-04 | 1.0542 | 0.4627 | -0.0697 |
| compact | asd_frozen_encoder_train_head | -2.0000 | second | 1,024 | 1.0145e-06 | 6.3420e-04 | 1.0537 | 0.4627 | -0.0709 |
| compact | raw_joint | nan | second | 1,024 | 1.0127e-06 | 6.3993e-04 | 1.0518 | 0.5275 | -0.1051 |
| compact | zero | nan | minute | 1,024 | 1.6364e-06 | 8.3933e-04 | 1.0004 | 0.4818 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | minute | 1,024 | 1.6362e-06 | 8.3900e-04 | 1.0003 | 0.5407 | 0.0368 |
| compact | asd_frozen_encoder_train_head | -2.0000 | minute | 1,024 | 1.6363e-06 | 8.3904e-04 | 1.0004 | 0.5417 | 0.0364 |
| compact | raw_joint | nan | minute | 1,024 | 1.6441e-06 | 8.4578e-04 | 1.0052 | 0.4966 | 0.0312 |
| compact | zero | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | hour | 200.0000 | 3.7533e-05 | 4.0700e-03 | 0.8010 | 0.6150 | 0.5484 |
| compact | asd_frozen_encoder_train_head | -2.0000 | hour | 200.0000 | 3.7540e-05 | 4.0708e-03 | 0.8012 | 0.6150 | 0.5483 |
| compact | raw_joint | nan | hour | 200.0000 | 4.0935e-05 | 4.2538e-03 | 0.8736 | 0.6300 | 0.5129 |

## 2. Round 2 Full Confirm

完整 full summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_asd_patchtst_ablation\round2_full_summary.csv`

| patch_preset | model | init_gate | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | nan | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | second | 263,640 | 9.4973e-07 | 6.1516e-04 | 0.9884 | 0.5226 | 0.1143 |
| compact | asd_frozen_encoder_train_head | -2.0000 | second | 263,640 | 9.4961e-07 | 6.1523e-04 | 0.9883 | 0.5218 | 0.1155 |
| compact | raw_joint | nan | second | 263,640 | 9.4859e-07 | 6.1349e-04 | 0.9872 | 0.5301 | 0.1143 |
| compact | zero | nan | minute | 3,120 | 1.5911e-06 | 8.1541e-04 | 1.0003 | 0.4878 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | minute | 3,120 | 1.5812e-06 | 8.1385e-04 | 0.9940 | 0.5170 | 0.0784 |
| compact | asd_frozen_encoder_train_head | -2.0000 | minute | 3,120 | 1.5812e-06 | 8.1390e-04 | 0.9940 | 0.5167 | 0.0783 |
| compact | raw_joint | nan | minute | 3,120 | 1.5836e-06 | 8.1870e-04 | 0.9955 | 0.4990 | 0.0828 |
| compact | zero | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | -4.0000 | hour | 200.0000 | 2.6405e-05 | 3.2886e-03 | 0.5635 | 0.7400 | 0.6709 |
| compact | asd_frozen_encoder_train_head | -2.0000 | hour | 200.0000 | 2.6395e-05 | 3.2885e-03 | 0.5633 | 0.7400 | 0.6711 |
| compact | raw_joint | nan | hour | 200.0000 | 2.6938e-05 | 3.3242e-03 | 0.5749 | 0.7300 | 0.6633 |

## 3. Round 3 Robustness

robustness 配置（按 full validation selection score 选择）: `{'patch_preset': 'compact', 'training_regime': 'asd_frozen_encoder_train_head', 'init_gate': -4.0, 'seed': 42, 'selection_score': 0.0004694020704991014, 'quality_pass': False, 'second_mse_over_raw': 0.9992270185293363, 'second_dir_delta': -0.006817342635598633, 'minute_mse_over_raw': 0.9978127139314904, 'minute_dir_delta': 0.015062213490504295, 'minute_corr_delta': 0.0045507235700934, 'hour_nmse_delta_raw_minus_model': -0.005982469469951235, 'hour_nmse_delta_zero_minus_model': 0.496874892045159}`

| patch_preset | model | init_gate | scale | mse_mean | mse_std | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | asd_frozen_encoder_train_head | -4.0000 | hour | 2.5914e-05 | 7.6408e-07 | 0.5531 | 0.0163 | 3.2534e-03 | 8.6451e-05 | 0.7133 | 0.0306 | 0.6743 | 7.9928e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | minute | 1.5833e-06 | 1.9222e-09 | 0.9953 | 1.2084e-03 | 8.1291e-04 | 8.5329e-07 | 0.5295 | 0.0112 | 0.0714 | 6.1079e-03 |
| compact | asd_frozen_encoder_train_head | -4.0000 | second | 9.4903e-07 | 8.3561e-10 | 0.9877 | 8.6964e-04 | 6.1400e-04 | 1.3369e-06 | 0.5309 | 0.0107 | 0.1150 | 3.5134e-03 |
| compact | raw_joint | nan | hour | 2.6493e-05 | 6.1119e-07 | 0.5654 | 0.0130 | 3.2824e-03 | 8.7012e-05 | 0.7150 | 0.0218 | 0.6670 | 7.2972e-03 |
| compact | raw_joint | nan | minute | 1.5826e-06 | 9.2967e-10 | 0.9949 | 5.8444e-04 | 8.1441e-04 | 3.7150e-06 | 0.5215 | 0.0195 | 0.0794 | 4.2736e-03 |
| compact | raw_joint | nan | second | 9.4965e-07 | 1.9461e-09 | 0.9883 | 2.0254e-03 | 6.1362e-04 | 1.0832e-06 | 0.5331 | 6.1481e-03 | 0.1123 | 4.6480e-03 |

## 4. ASD Diagnostics

| round | patch_preset | training_regime | init_gate | scale | gate_mean | tau_mean | mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| round1_small | compact | asd_frozen_encoder_train_head | -4.0000 | hour | 6.9137e-03 | 0.1594 | 1.4653e-04 |
| round1_small | compact | asd_frozen_encoder_train_head | -4.0000 | minute | 0.0112 | 0.2592 | 1.1426e-03 |
| round1_small | compact | asd_frozen_encoder_train_head | -4.0000 | second | 0.0166 | 3.6196 | 4.0787e-03 |
| round1_small | compact | asd_frozen_encoder_train_head | -3.0000 | hour | 0.0179 | 0.1575 | 3.7524e-04 |
| round1_small | compact | asd_frozen_encoder_train_head | -3.0000 | minute | 0.0294 | 0.2574 | 2.9685e-03 |
| round1_small | compact | asd_frozen_encoder_train_head | -3.0000 | second | 0.0436 | 3.6130 | 0.0107 |
| round1_small | compact | asd_frozen_encoder_train_head | -2.0000 | hour | 0.0435 | 0.1530 | 8.8584e-04 |
| round1_small | compact | asd_frozen_encoder_train_head | -2.0000 | minute | 0.0726 | 0.2531 | 7.2195e-03 |
| round1_small | compact | asd_frozen_encoder_train_head | -2.0000 | second | 0.1090 | 3.5942 | 0.0267 |
| round1_small | compact | asd_joint | -4.0000 | hour | 0.0828 | 0.3216 | 3.5242e-03 |
| round1_small | compact | asd_joint | -4.0000 | minute | 0.0365 | 0.3476 | 4.9564e-03 |
| round1_small | compact | asd_joint | -4.0000 | second | 0.0173 | 3.5936 | 4.2436e-03 |
| round1_small | compact | asd_joint | -3.0000 | hour | 0.2290 | 0.2947 | 8.9469e-03 |
| round1_small | compact | asd_joint | -3.0000 | minute | 0.1014 | 0.3355 | 0.0133 |
| round1_small | compact | asd_joint | -3.0000 | second | 0.0459 | 3.5985 | 0.0113 |
| round1_small | compact | asd_joint | -2.0000 | hour | 0.5666 | 0.2752 | 0.0207 |
| round1_small | compact | asd_joint | -2.0000 | minute | 0.2771 | 0.3265 | 0.0354 |
| round1_small | compact | asd_joint | -2.0000 | second | 0.1160 | 3.6080 | 0.0285 |
| round1_small | compact | asd_only_frozen_backbone | -4.0000 | hour | 7.5455e-03 | 0.1734 | 1.7400e-04 |
| round1_small | compact | asd_only_frozen_backbone | -4.0000 | minute | 0.0116 | 0.2663 | 1.2108e-03 |
| round1_small | compact | asd_only_frozen_backbone | -4.0000 | second | 0.0164 | 3.5900 | 4.0170e-03 |
| round1_small | compact | asd_only_frozen_backbone | -3.0000 | hour | 0.0196 | 0.1713 | 4.4540e-04 |
| round1_small | compact | asd_only_frozen_backbone | -3.0000 | minute | 0.0303 | 0.2643 | 3.1415e-03 |
| round1_small | compact | asd_only_frozen_backbone | -3.0000 | second | 0.0432 | 3.5813 | 0.0105 |
| round1_small | compact | asd_only_frozen_backbone | -2.0000 | hour | 0.0476 | 0.1661 | 1.0505e-03 |
| round1_small | compact | asd_only_frozen_backbone | -2.0000 | minute | 0.0748 | 0.2592 | 7.6125e-03 |
| round1_small | compact | asd_only_frozen_backbone | -2.0000 | second | 0.1078 | 3.5556 | 0.0262 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -4.0000 | hour | 5.6077e-04 | 0.2592 | 1.9284e-05 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -4.0000 | minute | 3.3163e-03 | 0.4315 | 3.9738e-04 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -4.0000 | second | 0.0188 | 4.9711 | 4.6136e-03 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -3.0000 | hour | 1.5056e-03 | 0.2573 | 5.1410e-05 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -3.0000 | minute | 8.8424e-03 | 0.4265 | 1.0476e-03 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -3.0000 | second | 0.0490 | 4.9225 | 0.0119 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -2.0000 | hour | 4.0058e-03 | 0.2543 | 1.3517e-04 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -2.0000 | minute | 0.0231 | 0.4184 | 2.6839e-03 |
| round1_small | fincast_adapted | asd_frozen_encoder_train_head | -2.0000 | second | 0.1202 | 4.8431 | 0.0289 |
| round1_small | fincast_adapted | asd_joint | -4.0000 | hour | 3.9753e-03 | 4.4178 | 1.8278e-03 |
| round1_small | fincast_adapted | asd_joint | -4.0000 | minute | 8.9059e-03 | 1.5779 | 3.4704e-03 |
| round1_small | fincast_adapted | asd_joint | -4.0000 | second | 0.0204 | 5.3266 | 5.2562e-03 |
| round1_small | fincast_adapted | asd_joint | -3.0000 | hour | 0.0107 | 4.5337 | 4.9991e-03 |
| round1_small | fincast_adapted | asd_joint | -3.0000 | minute | 0.0238 | 1.5976 | 9.3543e-03 |
| round1_small | fincast_adapted | asd_joint | -3.0000 | second | 0.0536 | 5.3256 | 0.0138 |
| round1_small | fincast_adapted | asd_joint | -2.0000 | hour | 0.0282 | 4.7587 | 0.0136 |
| round1_small | fincast_adapted | asd_joint | -2.0000 | minute | 0.0617 | 1.6346 | 0.0247 |
| round1_small | fincast_adapted | asd_joint | -2.0000 | second | 0.1331 | 5.3203 | 0.0342 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -4.0000 | hour | 1.3753e-03 | 0.6485 | 1.1636e-04 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -4.0000 | minute | 5.1272e-03 | 0.6419 | 9.0049e-04 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -4.0000 | second | 0.0194 | 5.0222 | 4.7915e-03 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -3.0000 | hour | 3.7244e-03 | 0.6486 | 3.1517e-04 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -3.0000 | minute | 0.0138 | 0.6408 | 2.4160e-03 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -3.0000 | second | 0.0509 | 5.0139 | 0.0126 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -2.0000 | hour | 0.0101 | 0.6490 | 8.5428e-04 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -2.0000 | minute | 0.0365 | 0.6385 | 6.3714e-03 |
| round1_small | fincast_adapted | asd_only_frozen_backbone | -2.0000 | second | 0.1265 | 4.9934 | 0.0311 |
| round1_small | short_second | asd_frozen_encoder_train_head | -4.0000 | hour | 2.6997e-04 | 7.4535 | 1.6735e-04 |
| round1_small | short_second | asd_frozen_encoder_train_head | -4.0000 | minute | 3.0615e-03 | 1.9508 | 1.3893e-03 |
| round1_small | short_second | asd_frozen_encoder_train_head | -4.0000 | second | 0.0186 | 3.2440 | 4.2288e-03 |
| round1_small | short_second | asd_frozen_encoder_train_head | -3.0000 | hour | 7.2707e-04 | 7.5308 | 4.5252e-04 |
| round1_small | short_second | asd_frozen_encoder_train_head | -3.0000 | minute | 8.2356e-03 | 1.9379 | 3.7210e-03 |
| round1_small | short_second | asd_frozen_encoder_train_head | -3.0000 | second | 0.0489 | 3.1766 | 0.0109 |
| round1_small | short_second | asd_frozen_encoder_train_head | -2.0000 | hour | 1.9507e-03 | 6.2761 | 1.1157e-03 |
| round1_small | short_second | asd_frozen_encoder_train_head | -2.0000 | minute | 0.0219 | 1.7143 | 9.0689e-03 |
| round1_small | short_second | asd_frozen_encoder_train_head | -2.0000 | second | 0.1221 | 3.0894 | 0.0268 |
| round1_small | short_second | asd_joint | -4.0000 | hour | 1.0592e-03 | 12.1391 | 7.4417e-04 |
| round1_small | short_second | asd_joint | -4.0000 | minute | 6.2151e-03 | 2.7775 | 3.3983e-03 |
| round1_small | short_second | asd_joint | -4.0000 | second | 0.0204 | 3.5799 | 4.9864e-03 |
| round1_small | short_second | asd_joint | -3.0000 | hour | 2.7532e-03 | 12.1282 | 1.9341e-03 |
| round1_small | short_second | asd_joint | -3.0000 | minute | 0.0164 | 2.7712 | 8.9389e-03 |
| round1_small | short_second | asd_joint | -3.0000 | second | 0.0534 | 3.5691 | 0.0130 |
| round1_small | short_second | asd_joint | -2.0000 | hour | 6.7557e-03 | 12.1963 | 4.7498e-03 |
| round1_small | short_second | asd_joint | -2.0000 | minute | 0.0413 | 2.7762 | 0.0226 |
| round1_small | short_second | asd_joint | -2.0000 | second | 0.1325 | 3.5557 | 0.0322 |
| round1_small | short_second | asd_only_frozen_backbone | -4.0000 | hour | 3.4743e-04 | 8.5142 | 2.2578e-04 |
| round1_small | short_second | asd_only_frozen_backbone | -4.0000 | minute | 3.4367e-03 | 2.0949 | 1.6313e-03 |
| round1_small | short_second | asd_only_frozen_backbone | -4.0000 | second | 0.0186 | 3.2087 | 4.1945e-03 |
| round1_small | short_second | asd_only_frozen_backbone | -3.0000 | hour | 9.3574e-04 | 8.7050 | 6.1207e-04 |
| round1_small | short_second | asd_only_frozen_backbone | -3.0000 | minute | 9.2361e-03 | 2.1098 | 4.4031e-03 |
| round1_small | short_second | asd_only_frozen_backbone | -3.0000 | second | 0.0489 | 3.1671 | 0.0109 |
| round1_small | short_second | asd_only_frozen_backbone | -2.0000 | hour | 2.4985e-03 | 7.5118 | 1.5535e-03 |
| round1_small | short_second | asd_only_frozen_backbone | -2.0000 | minute | 0.0244 | 1.8972 | 0.0109 |
| round1_small | short_second | asd_only_frozen_backbone | -2.0000 | second | 0.1221 | 3.0805 | 0.0267 |
| round2_full_confirm | compact | asd_frozen_encoder_train_head | -4.0000 | hour | 0.0109 | 9.2281 | 7.2128e-03 |
| round2_full_confirm | compact | asd_frozen_encoder_train_head | -4.0000 | minute | 0.0216 | 1.9760 | 0.0117 |
| round2_full_confirm | compact | asd_frozen_encoder_train_head | -4.0000 | second | 0.0411 | 7.9698 | 0.0132 |
| round2_full_confirm | compact | asd_frozen_encoder_train_head | -2.0000 | hour | 0.0245 | 7.8467 | 0.0155 |
| round2_full_confirm | compact | asd_frozen_encoder_train_head | -2.0000 | minute | 0.0752 | 1.7956 | 0.0392 |
| round2_full_confirm | compact | asd_frozen_encoder_train_head | -2.0000 | second | 0.1841 | 7.5012 | 0.0589 |
| round3_robustness | compact | asd_frozen_encoder_train_head | -4.0000 | hour | 0.0103 | 8.8794 | 6.1090e-03 |
| round3_robustness | compact | asd_frozen_encoder_train_head | -4.0000 | minute | 0.0259 | 1.9309 | 0.0135 |
| round3_robustness | compact | asd_frozen_encoder_train_head | -4.0000 | second | 0.0850 | 9.1335 | 0.0256 |

## Interpretation Guardrails

- `asd_only_frozen_backbone` 用 raw checkpoint 初始化并冻结 PatchTST backbone、patch projection、scale embedding 和 heads，只训练 ASD gate/threshold。
- `asd_frozen_encoder_train_head` 用 raw checkpoint 初始化，冻结 shared encoder 与 patch/scale embedding，只训练 ASD 和 scale-specific heads。
- second 判断为 MSE 不劣于 raw 且 direction 不明显下降；minute 允许 MSE 小幅不占优，但 direction/corr 要稳定；hour 要求 MSE/NMSE 明确优于 raw 和 zero。
- hour test 的 n 较小，报告中保留 `n`，避免过度解释。
