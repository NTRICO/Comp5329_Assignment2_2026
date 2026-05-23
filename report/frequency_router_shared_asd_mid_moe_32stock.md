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
| padded_freq_router_shared_asd_patchtst | second | 4,096 | 0.9844 | 7.8999e-07 | 5.5441e-04 | 0.5271 | 0.1263 |
| padded_freq_router_shared_asd_mid_moe_patchtst | second | 4,096 | 0.9851 | 7.9050e-07 | 5.5430e-04 | 0.5116 | 0.1240 |
| zero | minute | 3,224 | 1.0009 | 1.2435e-06 | 7.0896e-04 | 0.5067 | nan |
| shared_raw_patchtst | minute | 3,224 | 0.9937 | 1.2345e-06 | 7.0643e-04 | 0.5263 | 0.0913 |
| padded_freq_router_shared_asd_patchtst | minute | 3,224 | 0.9910 | 1.2311e-06 | 7.0651e-04 | 0.5232 | 0.1017 |
| padded_freq_router_shared_asd_mid_moe_patchtst | minute | 3,224 | 0.9916 | 1.2320e-06 | 7.0660e-04 | 0.5263 | 0.1122 |
| zero | hour | 620.0000 | 1.0000 | 3.5023e-05 | 4.1690e-03 | 0.5048 | nan |
| shared_raw_patchtst | hour | 620.0000 | 0.5239 | 1.8350e-05 | 2.7924e-03 | 0.7452 | 0.6918 |
| padded_freq_router_shared_asd_patchtst | hour | 620.0000 | 0.5227 | 1.8308e-05 | 2.8338e-03 | 0.7435 | 0.6985 |
| padded_freq_router_shared_asd_mid_moe_patchtst | hour | 620.0000 | 0.5282 | 1.8500e-05 | 2.8433e-03 | 0.7323 | 0.6937 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| padded_freq_router_shared_asd_patchtst | hour | 620.0000 | 0.5227 | 1.8308e-05 | 2.8338e-03 | 0.7435 | 0.6985 |
| shared_raw_patchtst | hour | 620.0000 | 0.5239 | 1.8350e-05 | 2.7924e-03 | 0.7452 | 0.6918 |
| padded_freq_router_shared_asd_mid_moe_patchtst | hour | 620.0000 | 0.5282 | 1.8500e-05 | 2.8433e-03 | 0.7323 | 0.6937 |
| padded_freq_router_shared_asd_patchtst | minute | 3,224 | 0.9910 | 1.2311e-06 | 7.0651e-04 | 0.5232 | 0.1017 |
| padded_freq_router_shared_asd_mid_moe_patchtst | minute | 3,224 | 0.9916 | 1.2320e-06 | 7.0660e-04 | 0.5263 | 0.1122 |
| shared_raw_patchtst | minute | 3,224 | 0.9937 | 1.2345e-06 | 7.0643e-04 | 0.5263 | 0.0913 |
| padded_freq_router_shared_asd_patchtst | second | 4,096 | 0.9844 | 7.8999e-07 | 5.5441e-04 | 0.5271 | 0.1263 |
| padded_freq_router_shared_asd_mid_moe_patchtst | second | 4,096 | 0.9851 | 7.9050e-07 | 5.5430e-04 | 0.5116 | 0.1240 |
| shared_raw_patchtst | second | 4,096 | 0.9855 | 7.9084e-07 | 5.5585e-04 | 0.5158 | 0.1228 |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | asd_router_entropy | asd_router_balance_loss | mid_moe_router_entropy | mid_moe_router_balance_loss | backbone_mid_moe_router_entropy | backbone_mid_moe_router_balance_loss | frequency_embedding_norm | mean_abs_delta | asd_gate_mean | asd_tau_mean | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | expert_identity_prob | expert_second_prob | expert_minute_prob | expert_hour_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| shared_raw_patchtst | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| shared_raw_patchtst | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| shared_raw_patchtst | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| padded_freq_router_shared_asd_patchtst | second | 0.6700 | 0.0199 | 0.6700 | 0.0199 | nan | nan | nan | nan | 2.2707 | 4.5351e-03 | 0.0122 | 5.8154 | 0.3321 | 0.3083 | 0.3524 | 7.2108e-03 | 0.3321 | 0.3083 | 0.3524 | 7.2108e-03 |
| padded_freq_router_shared_asd_patchtst | minute | 0.6915 | 0.0625 | 0.6915 | 0.0625 | nan | nan | nan | nan | 2.7115 | 9.9371e-05 | 2.1941e-03 | 0.3142 | 0.0000e+00 | 0.5031 | 0.4969 | 0.0000e+00 | 0.0000e+00 | 0.5031 | 0.4969 | 0.0000e+00 |
| padded_freq_router_shared_asd_patchtst | hour | 0.6929 | 0.0625 | 0.6929 | 0.0625 | nan | nan | nan | nan | 3.9506 | 7.3698e-07 | 8.4011e-04 | 0.1413 | 0.0000e+00 | 0.5080 | 0.4920 | 0.0000e+00 | 0.0000e+00 | 0.5080 | 0.4920 | 0.0000e+00 |
| padded_freq_router_shared_asd_mid_moe_patchtst | second | 0.1470 | 0.1596 | 1.4446e-03 | 0.1874 | 0.1470 | 0.1596 | 0.1470 | 0.1596 | 2.9713 | 0.9427 | 0.0238 | 5.1163 | 3.2330e-03 | 1.3071e-04 | 0.0558 | 0.9408 | 1.5178e-04 | 0.0000e+00 | 0.0000e+00 | 0.9998 |
| padded_freq_router_shared_asd_mid_moe_patchtst | minute | 0.4439 | 0.0759 | 0.0185 | 0.1861 | 0.4439 | 0.0759 | 0.4439 | 0.0759 | 3.4308 | 0.2269 | 5.1530e-04 | 2.3276 | 5.7497e-04 | 0.6642 | 0.3353 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 2.8035e-03 | 0.9972 |
| padded_freq_router_shared_asd_mid_moe_patchtst | hour | 0.3761 | 0.0906 | 0.0217 | 0.1859 | 0.3761 | 0.0906 | 0.3761 | 0.0906 | 4.1770 | 0.1454 | 3.1694e-05 | 6.9899 | 8.0805e-03 | 0.7412 | 0.2508 | 0.0000e+00 | 3.2536e-03 | 0.0000e+00 | 0.0000e+00 | 0.9967 |

## Files

- summary: `outputs\frequency_router_shared_asd_mid_moe_32stock\summary.csv`
- diagnostics: `outputs\frequency_router_shared_asd_mid_moe_32stock\diagnostics.csv`
- aggregate: `outputs\frequency_router_shared_asd_mid_moe_32stock\aggregate.csv`
