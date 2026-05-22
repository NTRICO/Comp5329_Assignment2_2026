# Scale-Aware ASD PatchTST 小数据优先实验报告

本报告只包含 second / minute / hour 三个 intraday scale，不包含 day 数据。

## 1. 小数据真实结果

small cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`；epochs=3；balanced steps/epoch=12；patch preset=`fincast_adapted`。

| scale | model | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| second | zero | 1,024 | 9.4538e-07 | 6.1308e-04 | 1.0003 | 0.5231 | nan |
| second | raw_patchtst | 1,024 | 9.6549e-07 | 6.2018e-04 | 1.0216 | 0.5044 | 5.8451e-03 |
| second | static_asd_patchtst | 1,024 | 9.6306e-07 | 6.1901e-04 | 1.0190 | 0.5172 | 5.0910e-03 |
| second | scale_aware_asd_patchtst | 1,024 | 9.6743e-07 | 6.2191e-04 | 1.0237 | 0.5005 | 0.0141 |
| minute | zero | 1,024 | 1.5852e-06 | 7.9712e-04 | 1.0030 | 0.4936 | nan |
| minute | raw_patchtst | 1,024 | 1.5905e-06 | 8.0995e-04 | 1.0063 | 0.4897 | 4.7215e-03 |
| minute | static_asd_patchtst | 1,024 | 1.5875e-06 | 8.0964e-04 | 1.0045 | 0.4917 | 0.0130 |
| minute | scale_aware_asd_patchtst | 1,024 | 1.6097e-06 | 8.0302e-04 | 1.0185 | 0.5112 | 0.0367 |
| hour | zero | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| hour | raw_patchtst | 200.0000 | 4.2841e-05 | 4.3088e-03 | 0.9143 | 0.7200 | 0.4990 |
| hour | static_asd_patchtst | 200.0000 | 4.3481e-05 | 4.3480e-03 | 0.9280 | 0.7300 | 0.4900 |
| hour | scale_aware_asd_patchtst | 200.0000 | 4.2224e-05 | 4.2905e-03 | 0.9011 | 0.6350 | 0.5218 |

### Quality Gate
- gate: **PASS**
- diagnostics finite; required splits and scales are present.

## 2. 全量数据预估水平

以下是 full actual 运行前预估，不作为最终实验结论。性能预估使用 `existing_full_raw_metric * small(scale-aware/raw)`。

| scale | source | full_sample_count | full_reference_raw_mse | estimated_scale_aware_mse | estimated_scale_aware_mae | estimated_direction_accuracy_nonzero |
| --- | --- | --- | --- | --- | --- | --- |
| second | existing_full_raw_reference_context_proxy | 263,640 | 9.4742e-07 | 9.4933e-07 | 6.1492e-04 | 0.5341 |
| minute | existing_full_raw_reference_context_proxy | 3,120 | 1.5872e-06 | 1.6064e-06 | 8.0689e-04 | 0.5405 |
| hour | existing_full_raw_reference_context_proxy | 200.0000 | 2.9876e-05 | 2.9445e-05 | 3.4062e-03 | 0.5850 |

预计 full scale-aware 训练时间约 53.7 秒 （small_scale_aware_elapsed * full_train_steps / small_train_steps）。

## 3. 全量实际结果

full cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`；epochs=5；balanced steps/epoch=250；patch preset=`fincast_adapted`。

| scale | model | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| second | zero | 230,360 | 9.2105e-07 | 6.0623e-04 | 1.0000 | 0.4934 | nan |
| second | raw_patchtst | 230,360 | 9.1157e-07 | 6.0340e-04 | 0.9897 | 0.5342 | 0.1051 |
| second | static_asd_patchtst | 230,360 | 9.1085e-07 | 6.0320e-04 | 0.9889 | 0.5352 | 0.1090 |
| second | scale_aware_asd_patchtst | 230,360 | 9.1244e-07 | 6.0526e-04 | 0.9906 | 0.5196 | 0.1015 |
| minute | zero | 1,040 | 1.5905e-06 | 7.9701e-04 | 1.0036 | 0.4957 | nan |
| minute | raw_patchtst | 1,040 | 1.5754e-06 | 7.9672e-04 | 0.9941 | 0.4990 | 0.0888 |
| minute | static_asd_patchtst | 1,040 | 1.5765e-06 | 7.9715e-04 | 0.9948 | 0.5038 | 0.0837 |
| minute | scale_aware_asd_patchtst | 1,040 | 1.5821e-06 | 7.9634e-04 | 0.9984 | 0.5327 | 0.0943 |
| hour | zero | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| hour | raw_patchtst | 200.0000 | 2.5449e-05 | 3.1699e-03 | 0.5431 | 0.7050 | 0.6826 |
| hour | static_asd_patchtst | 200.0000 | 2.5300e-05 | 3.1505e-03 | 0.5400 | 0.6800 | 0.6885 |
| hour | scale_aware_asd_patchtst | 200.0000 | 2.5073e-05 | 3.1083e-03 | 0.5351 | 0.7150 | 0.6839 |

### 预估 vs 实际
| scale | estimated_mse | actual_mse | actual_minus_estimated_mse | estimated_mae | actual_mae |
| --- | --- | --- | --- | --- | --- |
| second | 9.4933e-07 | 9.1244e-07 | -3.6892e-08 | 6.1492e-04 | 6.0526e-04 |
| minute | 1.6064e-06 | 1.5821e-06 | -2.4278e-08 | 8.0689e-04 | 7.9634e-04 |
| hour | 2.9445e-05 | 2.5073e-05 | -4.3723e-06 | 3.4062e-03 | 3.1083e-03 |

## Notes

- hour scale 沿用当前项目的 intraday hour/proxy 构造，不写成严格自然小时。
- patch/context 不固定为 32；默认 preset 参考 FinCast 的 frequency-aware context 思路，并按 Optiver 可用窗口长度调整。
- full preset 使用 balanced training；不会强制每个 epoch 穷尽 second 全部 windows。
- 第一版不包含 Adapter-MoE 或 LoRA。
