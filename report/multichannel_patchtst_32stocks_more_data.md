# Multi-Channel PatchTST Experiment

This run uses a 15-channel intraday input: WAP/MID returns plus spread, imbalance, size, update, and time features. PatchTST still shares the encoder across channels, but the target head flattens all channel tokens to predict WAP1 future return.

patch preset: `short_second`; epochs=5; balanced steps/epoch=50; channels=15.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 4,096 | 1.0000 | 8.3036e-07 | 5.5799e-04 | 0.5054 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 4,096 | 1.0318 | 8.5672e-07 | 5.8105e-04 | 0.5295 | 0.0626 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 4,096 | 1.0050 | 8.3449e-07 | 5.6344e-04 | 0.5393 | 0.0637 |
| multichannel_raw_joint | second | 4,096 | 1.0048 | 8.3430e-07 | 5.7420e-04 | 0.5292 | 0.1035 |
| multichannel_asd_frozen_encoder_train_head | second | 4,096 | 1.0006 | 8.3081e-07 | 5.7161e-04 | 0.5317 | 0.1142 |
| zero | minute | 3,224 | 1.0009 | 1.2435e-06 | 7.0896e-04 | 0.5067 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 3,224 | 0.9854 | 1.2242e-06 | 7.1176e-04 | 0.5382 | 0.1240 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 3,224 | 0.9760 | 1.2125e-06 | 7.0796e-04 | 0.5332 | 0.1584 |
| multichannel_raw_joint | minute | 3,224 | 0.9885 | 1.2281e-06 | 7.1047e-04 | 0.5428 | 0.1099 |
| multichannel_asd_frozen_encoder_train_head | minute | 3,224 | 0.9956 | 1.2370e-06 | 7.2266e-04 | 0.5251 | 0.1112 |
| zero | hour | 620.0000 | 1.0000 | 3.5023e-05 | 4.1690e-03 | 0.5048 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 620.0000 | 0.5788 | 2.0272e-05 | 2.9611e-03 | 0.7339 | 0.6492 |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 620.0000 | 0.5678 | 1.9885e-05 | 2.9977e-03 | 0.7065 | 0.6726 |
| multichannel_raw_joint | hour | 620.0000 | 0.6003 | 2.1023e-05 | 3.0787e-03 | 0.7258 | 0.6343 |
| multichannel_asd_frozen_encoder_train_head | hour | 620.0000 | 0.5673 | 1.9867e-05 | 2.9131e-03 | 0.7048 | 0.6613 |

## Test NMSE Relative To Multi-Channel Raw

| model | scale | nmse | nmse_vs_multichannel_raw_pct |
| --- | --- | --- | --- |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 1.0318 | 2.6874 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 1.0050 | 0.0234 |
| multichannel_asd_frozen_encoder_train_head | second | 1.0006 | -0.4177 |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 0.9854 | -0.3188 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 0.9760 | -1.2678 |
| multichannel_asd_frozen_encoder_train_head | minute | 0.9956 | 0.7215 |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 0.5788 | -3.5698 |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 0.5678 | -5.4105 |
| multichannel_asd_frozen_encoder_train_head | hour | 0.5673 | -5.4978 |

## Router Diagnostics

| model | scale | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multichannel_raw_joint | second | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | minute | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | hour | nan | nan | nan | nan | nan | nan | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 0.2030 | 0.1485 | 0.9163 | 0.0632 | 9.0227e-03 | 0.0115 | 0.1423 |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 0.1229 | 0.1674 | 2.4466e-03 | 0.0372 | 2.0468e-03 | 0.9583 | 0.5636 |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 0.3320 | 0.1078 | 0.1951 | 0.8020 | 1.5180e-03 | 1.4401e-03 | 0.2609 |
| multichannel_asd_frozen_encoder_train_head | second | nan | nan | nan | nan | nan | nan | 0.1709 |
| multichannel_asd_frozen_encoder_train_head | minute | nan | nan | nan | nan | nan | nan | 3.2266e-03 |
| multichannel_asd_frozen_encoder_train_head | hour | nan | nan | nan | nan | nan | nan | 1.2318e-03 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 0.0405 | 0.1824 | 2.4065e-05 | 3.0133e-03 | 7.2276e-03 | 0.9897 | 0.7559 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 0.0235 | 0.1849 | 0.9947 | 4.5294e-05 | 5.2170e-03 | 8.6370e-07 | 0.8053 |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 0.3855 | 0.0851 | 0.2874 | 0.0000e+00 | 0.7126 | 3.1585e-06 | 0.3903 |

## Channels

`wap1_log_return`, `wap2_log_return`, `mid1_log_return`, `mid2_log_return`, `rel_spread1`, `rel_spread2`, `imbalance1`, `imbalance2`, `log_total_size1`, `log_total_size2`, `total_imbalance`, `updates_in_second`, `is_observed_update`, `seconds_since_update`, `second_frac`
