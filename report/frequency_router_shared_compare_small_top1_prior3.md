# Padded Frequency-Router PatchTST Small Run

本实验测试一个新的小数据架构：

`normalized input -> padding + mask -> FFT frequency embedding -> MoE router -> identity/scale-specific ASD experts -> crop -> PatchTST -> scale head`

训练仍采用 mixed-frequency balanced step：每个 optimizer step 同时取 `second/minute/hour` 各一个 batch，并平均 loss。

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`; patch preset: `short_second`; seed=42; epochs=3; steps/epoch=12; train cap=4096; eval cap=1024; identity expert=on.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 1,024 | 1.0023 | 9.6510e-07 | 6.1415e-04 | 0.4568 | nan |
| shared_raw_patchtst | second | 1,024 | 1.0177 | 9.7993e-07 | 6.1876e-04 | 0.5029 | -0.0334 |
| scale_specific_raw_patchtst | second | 1,024 | 1.0221 | 9.8417e-07 | 6.1911e-04 | 0.5334 | -0.0257 |
| padded_freq_router_scale_specific_asd_patchtst | second | 1,024 | 1.0067 | 9.6931e-07 | 6.1976e-04 | 0.5147 | 2.5724e-04 |
| padded_freq_router_shared_asd_patchtst | second | 1,024 | 1.0087 | 9.7125e-07 | 6.2249e-04 | 0.5255 | 0.0106 |
| zero | minute | 1,024 | 1.0030 | 1.5852e-06 | 7.9712e-04 | 0.4936 | nan |
| shared_raw_patchtst | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| scale_specific_raw_patchtst | minute | 1,024 | 1.0335 | 1.6333e-06 | 8.1434e-04 | 0.5044 | 0.0218 |
| padded_freq_router_scale_specific_asd_patchtst | minute | 1,024 | 0.9957 | 1.5736e-06 | 7.9921e-04 | 0.5210 | 0.0672 |
| padded_freq_router_shared_asd_patchtst | minute | 1,024 | 1.0272 | 1.6235e-06 | 8.1047e-04 | 0.5054 | 0.0262 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| shared_raw_patchtst | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| scale_specific_raw_patchtst | hour | 200.0000 | 0.7624 | 3.5724e-05 | 3.8835e-03 | 0.6800 | 0.5728 |
| padded_freq_router_scale_specific_asd_patchtst | hour | 200.0000 | 0.8998 | 4.2162e-05 | 4.3796e-03 | 0.5350 | 0.5335 |
| padded_freq_router_shared_asd_patchtst | hour | 200.0000 | 0.9257 | 4.3376e-05 | 4.3776e-03 | 0.5600 | 0.4413 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scale_specific_raw_patchtst | hour | 200.0000 | 0.7624 | 3.5724e-05 | 3.8835e-03 | 0.6800 | 0.5728 |
| padded_freq_router_scale_specific_asd_patchtst | hour | 200.0000 | 0.8998 | 4.2162e-05 | 4.3796e-03 | 0.5350 | 0.5335 |
| padded_freq_router_shared_asd_patchtst | hour | 200.0000 | 0.9257 | 4.3376e-05 | 4.3776e-03 | 0.5600 | 0.4413 |
| shared_raw_patchtst | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| padded_freq_router_scale_specific_asd_patchtst | minute | 1,024 | 0.9957 | 1.5736e-06 | 7.9921e-04 | 0.5210 | 0.0672 |
| shared_raw_patchtst | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| padded_freq_router_shared_asd_patchtst | minute | 1,024 | 1.0272 | 1.6235e-06 | 8.1047e-04 | 0.5054 | 0.0262 |
| scale_specific_raw_patchtst | minute | 1,024 | 1.0335 | 1.6333e-06 | 8.1434e-04 | 0.5044 | 0.0218 |
| padded_freq_router_scale_specific_asd_patchtst | second | 1,024 | 1.0067 | 9.6931e-07 | 6.1976e-04 | 0.5147 | 2.5724e-04 |
| padded_freq_router_shared_asd_patchtst | second | 1,024 | 1.0087 | 9.7125e-07 | 6.2249e-04 | 0.5255 | 0.0106 |
| shared_raw_patchtst | second | 1,024 | 1.0177 | 9.7993e-07 | 6.1876e-04 | 0.5029 | -0.0334 |
| scale_specific_raw_patchtst | second | 1,024 | 1.0221 | 9.8417e-07 | 6.1911e-04 | 0.5334 | -0.0257 |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | frequency_embedding_norm | mean_abs_delta | asd_gate_mean | asd_tau_mean | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | expert_identity_prob | expert_second_prob | expert_minute_prob | expert_hour_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared_raw_patchtst | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| shared_raw_patchtst | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| shared_raw_patchtst | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| scale_specific_raw_patchtst | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| scale_specific_raw_patchtst | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| scale_specific_raw_patchtst | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| padded_freq_router_scale_specific_asd_patchtst | second | -0.0000e+00 | 0.2222 | 1.6043 | 1.2679e-03 | 0.0115 | 1.3998 | 1.0000 | 0.0000e+00 | 0.0000e+00 | nan | nan | 1.0000 | 0.0000e+00 | 0.0000e+00 |
| padded_freq_router_scale_specific_asd_patchtst | minute | -0.0000e+00 | 0.2222 | 1.5553 | 1.2098e-03 | 6.0839e-03 | 0.7320 | 0.0000e+00 | 1.0000 | 0.0000e+00 | nan | nan | 0.0000e+00 | 1.0000 | 0.0000e+00 |
| padded_freq_router_scale_specific_asd_patchtst | hour | -0.0000e+00 | 0.2222 | 1.4756 | 1.3722e-05 | 4.4673e-05 | 2.6427 | 0.0000e+00 | 0.0000e+00 | 1.0000 | nan | nan | 0.0000e+00 | 0.0000e+00 | 1.0000 |
| padded_freq_router_shared_asd_patchtst | second | -0.0000e+00 | 0.1875 | 1.7129 | 0.0249 | 0.0739 | 8.0569 | 0.0000e+00 | 1.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000 | 0.0000e+00 | 0.0000e+00 |
| padded_freq_router_shared_asd_patchtst | minute | -0.0000e+00 | 0.1875 | 1.7384 | 0.3295 | 0.5376 | 6.1138 | 0.0000e+00 | 0.0000e+00 | 1.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000 | 0.0000e+00 |
| padded_freq_router_shared_asd_patchtst | hour | -0.0000e+00 | 0.1875 | 1.9664 | 1.4389e-05 | 3.7721e-04 | 0.3075 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000 |

## Files

- summary: `outputs\frequency_router_shared_compare_small_top1_prior3\summary.csv`
- diagnostics: `outputs\frequency_router_shared_compare_small_top1_prior3\diagnostics.csv`
- aggregate: `outputs\frequency_router_shared_compare_small_top1_prior3\aggregate.csv`
