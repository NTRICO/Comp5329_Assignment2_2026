# Padded Frequency-Router PatchTST Small Run

本实验测试一个新的小数据架构：

`normalized input -> padding + mask -> FFT frequency embedding -> MoE router -> identity/scale-specific ASD experts -> crop -> PatchTST -> scale head`

训练仍采用 mixed-frequency balanced step：每个 optimizer step 同时取 `second/minute/hour` 各一个 batch，并平均 loss。

cache: `data/cache/position_optiver_hf_second_feature_cache_32stocks_512t.npz`; patch preset: `short_second`; seed=42; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096; identity expert=on.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 4,096 | 1.0000 | 8.0250e-07 | 5.5585e-04 | 0.4869 | nan |
| shared_raw_patchtst | second | 4,096 | 0.9855 | 7.9084e-07 | 5.5585e-04 | 0.5158 | 0.1228 |
| scale_specific_raw_patchtst | second | 4,096 | 0.9872 | 7.9224e-07 | 5.5888e-04 | 0.5080 | 0.1255 |
| zero | minute | 3,224 | 1.0009 | 1.2435e-06 | 7.0896e-04 | 0.5067 | nan |
| shared_raw_patchtst | minute | 3,224 | 0.9937 | 1.2345e-06 | 7.0643e-04 | 0.5263 | 0.0913 |
| scale_specific_raw_patchtst | minute | 3,224 | 1.0043 | 1.2477e-06 | 7.1305e-04 | 0.5048 | 0.0772 |
| zero | hour | 620.0000 | 1.0000 | 3.5023e-05 | 4.1690e-03 | 0.5048 | nan |
| shared_raw_patchtst | hour | 620.0000 | 0.5239 | 1.8350e-05 | 2.7924e-03 | 0.7452 | 0.6918 |
| scale_specific_raw_patchtst | hour | 620.0000 | 0.5286 | 1.8512e-05 | 2.8195e-03 | 0.7387 | 0.6889 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| shared_raw_patchtst | hour | 620.0000 | 0.5239 | 1.8350e-05 | 2.7924e-03 | 0.7452 | 0.6918 |
| scale_specific_raw_patchtst | hour | 620.0000 | 0.5286 | 1.8512e-05 | 2.8195e-03 | 0.7387 | 0.6889 |
| shared_raw_patchtst | minute | 3,224 | 0.9937 | 1.2345e-06 | 7.0643e-04 | 0.5263 | 0.0913 |
| scale_specific_raw_patchtst | minute | 3,224 | 1.0043 | 1.2477e-06 | 7.1305e-04 | 0.5048 | 0.0772 |
| shared_raw_patchtst | second | 4,096 | 0.9855 | 7.9084e-07 | 5.5585e-04 | 0.5158 | 0.1228 |
| scale_specific_raw_patchtst | second | 4,096 | 0.9872 | 7.9224e-07 | 5.5888e-04 | 0.5080 | 0.1255 |

## Diagnostics

| model | scale |
| --- | --- |
| shared_raw_patchtst | second |
| shared_raw_patchtst | minute |
| shared_raw_patchtst | hour |
| scale_specific_raw_patchtst | second |
| scale_specific_raw_patchtst | minute |
| scale_specific_raw_patchtst | hour |

## Files

- summary: `outputs\frequency_router_scale_specific_patchtst_32stock_raw_compare\summary.csv`
- diagnostics: `outputs\frequency_router_scale_specific_patchtst_32stock_raw_compare\diagnostics.csv`
- aggregate: `outputs\frequency_router_scale_specific_patchtst_32stock_raw_compare\aggregate.csv`
