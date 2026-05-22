# Multi-Channel PatchTST Experiment

This run uses a 15-channel intraday input: WAP/MID returns plus spread, imbalance, size, update, and time features. PatchTST still shares the encoder across channels, but the target head flattens all channel tokens to predict WAP1 future return.

patch preset: `short_second`; epochs=3; balanced steps/epoch=12; channels=15.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 1,024 | 1.0000 | 9.2808e-07 | 6.0166e-04 | 0.4748 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 1,024 | 1.1562 | 1.0731e-06 | 7.0318e-04 | 0.5074 | 0.0190 |
| multichannel_raw_joint | second | 1,024 | 1.6882 | 1.5668e-06 | 9.8505e-04 | 0.5223 | -0.0526 |
| zero | minute | 1,024 | 1.0036 | 1.5548e-06 | 7.8796e-04 | 0.4956 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 1,024 | 0.9946 | 1.5409e-06 | 7.9389e-04 | 0.5298 | 0.0954 |
| multichannel_raw_joint | minute | 1,024 | 1.2044 | 1.8659e-06 | 1.0045e-03 | 0.4956 | 0.0609 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 200.0000 | 0.9151 | 4.2880e-05 | 4.5667e-03 | 0.5450 | 0.4884 |
| multichannel_raw_joint | hour | 200.0000 | 1.1435 | 5.3582e-05 | 5.3010e-03 | 0.4750 | 0.3236 |

## Test NMSE Relative To Multi-Channel Raw

| model | scale | nmse | nmse_vs_multichannel_raw_pct |
| --- | --- | --- | --- |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 1.1562 | -31.5122 |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 0.9946 | -17.4190 |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 0.9151 | -19.9738 |

## Router Diagnostics

| model | scale | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multichannel_raw_joint | second | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | minute | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | hour | nan | nan | nan | nan | nan | nan | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 0.4609 | 0.0703 | 0.6422 | 0.3443 | 9.6701e-03 | 3.8413e-03 | 0.0777 |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 0.4163 | 0.0859 | 2.3208e-04 | 0.2823 | 0.7168 | 7.2245e-04 | 0.0732 |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 0.0604 | 0.1794 | 2.2735e-03 | 0.0134 | 0.9835 | 8.1543e-04 | 0.2398 |

## Channels

`wap1_log_return`, `wap2_log_return`, `mid1_log_return`, `mid2_log_return`, `rel_spread1`, `rel_spread2`, `imbalance1`, `imbalance2`, `log_total_size1`, `log_total_size2`, `total_imbalance`, `updates_in_second`, `is_observed_update`, `seconds_since_update`, `second_frac`
