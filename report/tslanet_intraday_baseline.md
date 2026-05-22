# TSLANet-Style Intraday Baseline

本报告加入一个轻量 TSLANet-style baseline，用 ASB + ICB blocks 替换 PatchTST Transformer encoder。它复用当前 second/minute/hour 数据协议、scale-specific patch projection 和 scale-specific heads。

说明：这是本地轻量复现，用于和当前 PatchTST/ASD 线做公平工程对比；不是官方 TSLANet 代码的逐行移植。

small setting: epochs=3, steps/epoch=12; full setting: epochs=5, steps/epoch=250; patch presets=['compact', 'short_second'].

## 1. Small Result

summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\tslanet_intraday_baseline\round1_small_summary.csv`

| patch_preset | model | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| compact | asd_frozen_encoder_train_head | second | 1,024 | 9.6719e-07 | 6.1808e-04 | 1.0045 | 0.5029 | 0.0466 |
| compact | raw_joint | second | 1,024 | 9.6628e-07 | 6.2069e-04 | 1.0035 | 0.4764 | 0.0646 |
| compact | tslanet_joint | second | 1,024 | 9.9434e-07 | 6.2450e-04 | 1.0327 | 0.5138 | -0.0553 |
| compact | zero | minute | 1,024 | 1.6364e-06 | 8.3933e-04 | 1.0004 | 0.4818 | nan |
| compact | asd_frozen_encoder_train_head | minute | 1,024 | 1.6301e-06 | 8.3867e-04 | 0.9966 | 0.5221 | 0.0629 |
| compact | raw_joint | minute | 1,024 | 1.6311e-06 | 8.4257e-04 | 0.9972 | 0.4946 | 0.0814 |
| compact | tslanet_joint | minute | 1,024 | 1.6397e-06 | 8.4454e-04 | 1.0025 | 0.5093 | 0.0464 |
| compact | zero | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | hour | 200.0000 | 3.9266e-05 | 4.1565e-03 | 0.8380 | 0.5950 | 0.5234 |
| compact | raw_joint | hour | 200.0000 | 4.1760e-05 | 4.3426e-03 | 0.8912 | 0.5600 | 0.4917 |
| compact | tslanet_joint | hour | 200.0000 | 3.7776e-05 | 4.0203e-03 | 0.8062 | 0.7100 | 0.5624 |
| short_second | zero | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| short_second | asd_frozen_encoder_train_head | second | 1,024 | 9.8925e-07 | 6.2427e-04 | 1.0274 | 0.4686 | -0.0334 |
| short_second | raw_joint | second | 1,024 | 9.7993e-07 | 6.1876e-04 | 1.0177 | 0.5029 | -0.0334 |
| short_second | tslanet_joint | second | 1,024 | 9.8775e-07 | 6.2459e-04 | 1.0258 | 0.5147 | -0.0371 |
| short_second | zero | minute | 1,024 | 1.5852e-06 | 7.9712e-04 | 1.0030 | 0.4936 | nan |
| short_second | asd_frozen_encoder_train_head | minute | 1,024 | 1.5952e-06 | 7.9830e-04 | 1.0094 | 0.5357 | 0.0394 |
| short_second | raw_joint | minute | 1,024 | 1.6069e-06 | 8.0126e-04 | 1.0167 | 0.5064 | 0.0179 |
| short_second | tslanet_joint | minute | 1,024 | 1.5885e-06 | 7.9786e-04 | 1.0051 | 0.5455 | 0.0764 |
| short_second | zero | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| short_second | asd_frozen_encoder_train_head | hour | 200.0000 | 4.1026e-05 | 4.2233e-03 | 0.8756 | 0.7000 | 0.5338 |
| short_second | raw_joint | hour | 200.0000 | 4.3795e-05 | 4.3949e-03 | 0.9347 | 0.6000 | 0.4649 |
| short_second | tslanet_joint | hour | 200.0000 | 4.2164e-05 | 4.2917e-03 | 0.8999 | 0.6450 | 0.4463 |

## 2. Full Result

summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\tslanet_intraday_baseline\round2_full_summary.csv`

| patch_preset | model | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| compact | zero | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| compact | asd_frozen_encoder_train_head | second | 263,640 | 9.5247e-07 | 6.1572e-04 | 0.9913 | 0.5273 | 0.1132 |
| compact | raw_joint | second | 263,640 | 9.5198e-07 | 6.1363e-04 | 0.9908 | 0.5347 | 0.0995 |
| compact | tslanet_joint | second | 263,640 | 9.5118e-07 | 6.1333e-04 | 0.9899 | 0.5347 | 0.1043 |
| compact | zero | minute | 3,120 | 1.5911e-06 | 8.1541e-04 | 1.0003 | 0.4878 | nan |
| compact | asd_frozen_encoder_train_head | minute | 3,120 | 1.5823e-06 | 8.1338e-04 | 0.9947 | 0.5289 | 0.0740 |
| compact | raw_joint | minute | 3,120 | 1.5980e-06 | 8.1688e-04 | 1.0046 | 0.5119 | 0.0719 |
| compact | tslanet_joint | minute | 3,120 | 1.5880e-06 | 8.1278e-04 | 0.9983 | 0.5250 | 0.0693 |
| compact | zero | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| compact | asd_frozen_encoder_train_head | hour | 200.0000 | 2.4648e-05 | 3.1486e-03 | 0.5260 | 0.7400 | 0.6900 |
| compact | raw_joint | hour | 200.0000 | 2.4648e-05 | 3.1193e-03 | 0.5260 | 0.7200 | 0.6886 |
| compact | tslanet_joint | hour | 200.0000 | 2.5649e-05 | 3.2367e-03 | 0.5474 | 0.7000 | 0.6787 |
| short_second | zero | second | 263,640 | 9.6087e-07 | 6.1651e-04 | 1.0000 | 0.4941 | nan |
| short_second | asd_frozen_encoder_train_head | second | 263,640 | 9.4926e-07 | 6.1279e-04 | 0.9879 | 0.5402 | 0.1129 |
| short_second | raw_joint | second | 263,640 | 9.5386e-07 | 6.1692e-04 | 0.9927 | 0.5163 | 0.0958 |
| short_second | tslanet_joint | second | 263,640 | 9.5984e-07 | 6.2080e-04 | 0.9989 | 0.5153 | 0.1093 |
| short_second | zero | minute | 1,040 | 1.5905e-06 | 7.9701e-04 | 1.0036 | 0.4957 | nan |
| short_second | asd_frozen_encoder_train_head | minute | 1,040 | 1.5857e-06 | 7.9646e-04 | 1.0006 | 0.5308 | 0.0880 |
| short_second | raw_joint | minute | 1,040 | 1.5807e-06 | 7.9618e-04 | 0.9975 | 0.5192 | 0.0919 |
| short_second | tslanet_joint | minute | 1,040 | 1.5521e-06 | 7.9549e-04 | 0.9794 | 0.5308 | 0.1696 |
| short_second | zero | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| short_second | asd_frozen_encoder_train_head | hour | 200.0000 | 2.5056e-05 | 3.1588e-03 | 0.5347 | 0.6800 | 0.6829 |
| short_second | raw_joint | hour | 200.0000 | 2.5995e-05 | 3.2755e-03 | 0.5548 | 0.6650 | 0.6836 |
| short_second | tslanet_joint | hour | 200.0000 | 2.5875e-05 | 3.2860e-03 | 0.5522 | 0.7200 | 0.6783 |

## 3. Diagnostics

| round | patch_preset | scale | tslanet_gate_mean | tslanet_tau_mean | tslanet_local_mask_mean | tslanet_mean_abs_delta | tslanet_global_filter_norm | tslanet_local_filter_norm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round1_small | compact | hour | 0.0604 | 2.8203 | 0.1568 | 5.9046e-04 | 0.0150 | 0.0134 |
| round1_small | compact | minute | 0.0228 | 1.5788 | 0.3241 | 2.3723e-04 | 0.0117 | 0.0110 |
| round1_small | compact | second | 0.0640 | 0.4918 | 0.6676 | 8.5133e-04 | 0.0150 | 0.0134 |
| round1_small | short_second | hour | 0.2251 | 0.5689 | 0.4947 | 2.9834e-03 | 0.0171 | 0.0119 |
| round1_small | short_second | minute | 0.0212 | 1.0041 | 0.4279 | 2.7402e-04 | 0.0139 | 9.0765e-03 |
| round1_small | short_second | second | 0.0261 | 0.6939 | 0.6055 | 3.3011e-04 | 0.0154 | 0.0127 |
| round2_full | compact | hour | 0.8572 | 1.5885 | 0.1711 | 0.0281 | 0.0420 | 0.0351 |
| round2_full | compact | minute | 0.0360 | 1.2123 | 0.2791 | 1.2252e-03 | 0.0355 | 0.0306 |
| round2_full | compact | second | 0.9647 | 0.2277 | 0.8194 | 0.0803 | 0.0420 | 0.0351 |
| round2_full | short_second | hour | 0.8981 | 0.1148 | 0.9449 | 0.0551 | 0.0511 | 0.0463 |
| round2_full | short_second | minute | 0.1741 | 0.4125 | 0.5695 | 8.7704e-03 | 0.0398 | 0.0369 |
| round2_full | short_second | second | 0.0484 | 0.2702 | 0.7134 | 1.7426e-03 | 0.0369 | 0.0344 |

## Interpretation Guardrails

- `tslanet_joint` 是 ASB + ICB 的轻量时间序列 baseline，不包含 LoRA/MoE。
- 对比重点是 raw PatchTST、ASD+PatchTST 和 TSLANet-style baseline 在同一数据协议下的 test metrics。
- hour test 的 `n` 约 200，低频结果需要谨慎解释。
