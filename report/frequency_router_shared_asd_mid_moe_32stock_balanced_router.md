# Padded Frequency-Router PatchTST Small Run

本实验测试一个新的小数据架构：

`normalized input -> padding + mask -> FFT frequency embedding -> MoE router -> identity/scale-specific ASD experts -> crop -> PatchTST -> scale head`

训练仍采用 mixed-frequency balanced step：每个 optimizer step 同时取 `second/minute/hour` 各一个 batch，并平均 loss。

cache: `data/cache/position_optiver_hf_second_feature_cache_32stocks_512t.npz`; patch preset: `short_second`; seed=42; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096; identity expert=on.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 4,096 | 1.0000 | 8.0250e-07 | 5.5585e-04 | 0.4869 | nan |
| padded_freq_router_shared_asd_mid_moe_patchtst | second | 4,096 | 0.9836 | 7.8935e-07 | 5.5435e-04 | 0.5212 | 0.1292 |
| zero | minute | 3,224 | 1.0009 | 1.2435e-06 | 7.0896e-04 | 0.5067 | nan |
| padded_freq_router_shared_asd_mid_moe_patchtst | minute | 3,224 | 0.9916 | 1.2319e-06 | 7.0649e-04 | 0.5192 | 0.0997 |
| zero | hour | 620.0000 | 1.0000 | 3.5023e-05 | 4.1690e-03 | 0.5048 | nan |
| padded_freq_router_shared_asd_mid_moe_patchtst | hour | 620.0000 | 0.5190 | 1.8176e-05 | 2.7911e-03 | 0.7403 | 0.7021 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| padded_freq_router_shared_asd_mid_moe_patchtst | hour | 620.0000 | 0.5190 | 1.8176e-05 | 2.7911e-03 | 0.7403 | 0.7021 |
| padded_freq_router_shared_asd_mid_moe_patchtst | minute | 3,224 | 0.9916 | 1.2319e-06 | 7.0649e-04 | 0.5192 | 0.0997 |
| padded_freq_router_shared_asd_mid_moe_patchtst | second | 4,096 | 0.9836 | 7.8935e-07 | 5.5435e-04 | 0.5212 | 0.1292 |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | asd_router_entropy | asd_router_balance_loss | mid_moe_router_entropy | mid_moe_router_balance_loss | backbone_mid_moe_router_entropy | backbone_mid_moe_router_balance_loss | frequency_embedding_norm | mean_abs_delta | asd_gate_mean | asd_tau_mean | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | expert_identity_prob | expert_second_prob | expert_minute_prob | expert_hour_prob |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| padded_freq_router_shared_asd_mid_moe_patchtst | second | 0.3650 | 0.1466 | 0.6520 | 0.0448 | 0.3650 | 0.1018 | 0.3650 | 0.1018 | 2.1306 | 0.3363 | 0.0297 | 7.8613 | 1.0270e-03 | 0.2165 | 0.7812 | 1.3028e-03 | 0.5248 | 0.3804 | 1.1339e-04 | 0.0947 |
| padded_freq_router_shared_asd_mid_moe_patchtst | minute | 0.4654 | 0.0905 | 0.6812 | 0.0220 | 0.4654 | 0.0685 | 0.4654 | 0.0685 | 1.8671 | 0.1757 | 1.4674e-03 | 1.3745 | 4.1706e-03 | 5.6553e-03 | 0.6250 | 0.3652 | 0.3780 | 0.0000e+00 | 0.3396 | 0.2824 |
| padded_freq_router_shared_asd_mid_moe_patchtst | hour | 0.1335 | 0.1801 | 0.6897 | 0.0212 | 0.1335 | 0.1589 | 0.1335 | 0.1589 | 1.5651 | 0.1221 | 2.4271e-05 | 3.2663 | 0.0601 | 4.9614e-04 | 2.8816e-04 | 0.9391 | 0.3315 | 0.0000e+00 | 0.3060 | 0.3626 |

## Files

- summary: `outputs\frequency_router_shared_asd_mid_moe_32stock_balanced_router\summary.csv`
- diagnostics: `outputs\frequency_router_shared_asd_mid_moe_32stock_balanced_router\diagnostics.csv`
- aggregate: `outputs\frequency_router_shared_asd_mid_moe_32stock_balanced_router\aggregate.csv`
