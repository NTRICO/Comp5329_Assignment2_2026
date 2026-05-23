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
| padded_freq_router_scale_specific_asd_patchtst | second | 1,024 | 1.0066 | 9.6926e-07 | 6.1975e-04 | 0.5157 | 4.7443e-04 |
| padded_freq_router_shared_asd_patchtst | second | 1,024 | 1.0089 | 9.7143e-07 | 6.2255e-04 | 0.5246 | 0.0120 |
| zero | minute | 1,024 | 1.0030 | 1.5852e-06 | 7.9712e-04 | 0.4936 | nan |
| shared_raw_patchtst | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| scale_specific_raw_patchtst | minute | 1,024 | 1.0335 | 1.6333e-06 | 8.1434e-04 | 0.5044 | 0.0218 |
| padded_freq_router_scale_specific_asd_patchtst | minute | 1,024 | 0.9957 | 1.5736e-06 | 7.9922e-04 | 0.5210 | 0.0671 |
| padded_freq_router_shared_asd_patchtst | minute | 1,024 | 1.0276 | 1.6241e-06 | 8.1100e-04 | 0.5005 | 0.0332 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| shared_raw_patchtst | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| scale_specific_raw_patchtst | hour | 200.0000 | 0.7624 | 3.5724e-05 | 3.8835e-03 | 0.6800 | 0.5728 |
| padded_freq_router_scale_specific_asd_patchtst | hour | 200.0000 | 0.8998 | 4.2162e-05 | 4.3796e-03 | 0.5350 | 0.5335 |
| padded_freq_router_shared_asd_patchtst | hour | 200.0000 | 0.9262 | 4.3398e-05 | 4.3788e-03 | 0.5600 | 0.4406 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| scale_specific_raw_patchtst | hour | 200.0000 | 0.7624 | 3.5724e-05 | 3.8835e-03 | 0.6800 | 0.5728 |
| padded_freq_router_scale_specific_asd_patchtst | hour | 200.0000 | 0.8998 | 4.2162e-05 | 4.3796e-03 | 0.5350 | 0.5335 |
| padded_freq_router_shared_asd_patchtst | hour | 200.0000 | 0.9262 | 4.3398e-05 | 4.3788e-03 | 0.5600 | 0.4406 |
| shared_raw_patchtst | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| padded_freq_router_scale_specific_asd_patchtst | minute | 1,024 | 0.9957 | 1.5736e-06 | 7.9922e-04 | 0.5210 | 0.0671 |
| shared_raw_patchtst | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| padded_freq_router_shared_asd_patchtst | minute | 1,024 | 1.0276 | 1.6241e-06 | 8.1100e-04 | 0.5005 | 0.0332 |
| scale_specific_raw_patchtst | minute | 1,024 | 1.0335 | 1.6333e-06 | 8.1434e-04 | 0.5044 | 0.0218 |
| padded_freq_router_scale_specific_asd_patchtst | second | 1,024 | 1.0066 | 9.6926e-07 | 6.1975e-04 | 0.5157 | 4.7443e-04 |
| padded_freq_router_shared_asd_patchtst | second | 1,024 | 1.0089 | 9.7143e-07 | 6.2255e-04 | 0.5246 | 0.0120 |
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
| padded_freq_router_scale_specific_asd_patchtst | second | 0.6909 | 0.0556 | 1.5844 | 2.4550e-03 | 0.0175 | 1.7537 | 0.5054 | 0.4946 | 0.0000e+00 | nan | nan | 0.5054 | 0.4946 | 0.0000e+00 |
| padded_freq_router_scale_specific_asd_patchtst | minute | 0.6926 | 0.0556 | 1.5283 | 1.4675e-03 | 5.7533e-03 | 0.9752 | 0.4934 | 0.5066 | 0.0000e+00 | nan | nan | 0.4934 | 0.5066 | 0.0000e+00 |
| padded_freq_router_scale_specific_asd_patchtst | hour | 0.6925 | 0.0558 | 1.3720 | 1.8764e-05 | 2.0732e-04 | 1.4219 | 0.5184 | 0.0000e+00 | 0.4816 | nan | nan | 0.5184 | 0.0000e+00 | 0.4816 |
| padded_freq_router_shared_asd_patchtst | second | 0.6895 | 0.0630 | 1.7363 | 8.0234e-03 | 0.0240 | 7.2696 | 0.0000e+00 | 0.4680 | 0.0000e+00 | 0.5320 | 0.0000e+00 | 0.4680 | 0.0000e+00 | 0.5320 |
| padded_freq_router_shared_asd_patchtst | minute | 0.6406 | 0.0716 | 1.7747 | 0.1941 | 0.3165 | 3.5062 | 0.3164 | 0.0236 | 0.6600 | 0.0000e+00 | 0.3164 | 0.0236 | 0.6600 | 0.0000e+00 |
| padded_freq_router_shared_asd_patchtst | hour | 0.6921 | 0.0628 | 2.0182 | 6.6443e-05 | 2.2594e-04 | 1.2502 | 0.5229 | 0.0000e+00 | 0.0000e+00 | 0.4771 | 0.5229 | 0.0000e+00 | 0.0000e+00 | 0.4771 |

## Files

- summary: `outputs\frequency_router_shared_compare_small\summary.csv`
- diagnostics: `outputs\frequency_router_shared_compare_small\diagnostics.csv`
- aggregate: `outputs\frequency_router_shared_compare_small\aggregate.csv`
