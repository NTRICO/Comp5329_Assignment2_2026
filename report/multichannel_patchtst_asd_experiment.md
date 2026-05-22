# Multi-Channel PatchTST Experiment

This run uses a 15-channel intraday input: WAP/MID returns plus spread, imbalance, size, update, and time features. PatchTST still shares the encoder across channels, but the target head flattens all channel tokens to predict WAP1 future return.

patch preset: `short_second`; epochs=3; balanced steps/epoch=12; channels=15.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 1,024 | 1.0000 | 9.2808e-07 | 6.0166e-04 | 0.4748 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 1,024 | 1.1562 | 1.0731e-06 | 7.0318e-04 | 0.5074 | 0.0190 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 1,024 | 1.1759 | 1.0914e-06 | 7.0170e-04 | 0.5134 | 0.0194 |
| multichannel_raw_joint | second | 1,024 | 1.6882 | 1.5668e-06 | 9.8505e-04 | 0.5223 | -0.0526 |
| multichannel_asd_frozen_encoder_train_head | second | 1,024 | 1.1051 | 1.0256e-06 | 6.7604e-04 | 0.5410 | 0.0938 |
| zero | minute | 1,024 | 1.0036 | 1.5548e-06 | 7.8796e-04 | 0.4956 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 1,024 | 0.9946 | 1.5409e-06 | 7.9389e-04 | 0.5298 | 0.0954 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 1,024 | 1.0234 | 1.5855e-06 | 8.0987e-04 | 0.5083 | 0.0805 |
| multichannel_raw_joint | minute | 1,024 | 1.2044 | 1.8659e-06 | 1.0045e-03 | 0.4956 | 0.0609 |
| multichannel_asd_frozen_encoder_train_head | minute | 1,024 | 0.9993 | 1.5481e-06 | 8.1705e-04 | 0.5122 | 0.1071 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 200.0000 | 0.9151 | 4.2880e-05 | 4.5667e-03 | 0.5450 | 0.4884 |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 200.0000 | 0.9584 | 4.4906e-05 | 4.8772e-03 | 0.6100 | 0.4828 |
| multichannel_raw_joint | hour | 200.0000 | 1.1435 | 5.3582e-05 | 5.3010e-03 | 0.4750 | 0.3236 |
| multichannel_asd_frozen_encoder_train_head | hour | 200.0000 | 0.9900 | 4.6388e-05 | 5.0399e-03 | 0.5850 | 0.4694 |

## Test NMSE Relative To Multi-Channel Raw

| model | scale | nmse | nmse_vs_multichannel_raw_pct |
| --- | --- | --- | --- |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 1.1562 | -31.5122 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 1.1759 | -30.3448 |
| multichannel_asd_frozen_encoder_train_head | second | 1.1051 | -34.5419 |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 0.9946 | -17.4190 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 1.0234 | -15.0252 |
| multichannel_asd_frozen_encoder_train_head | minute | 0.9993 | -17.0320 |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 0.9151 | -19.9738 |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 0.9584 | -16.1916 |
| multichannel_asd_frozen_encoder_train_head | hour | 0.9900 | -13.4265 |

## Router Diagnostics

| model | scale | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | mean_abs_delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| multichannel_raw_joint | second | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | minute | nan | nan | nan | nan | nan | nan | nan |
| multichannel_raw_joint | hour | nan | nan | nan | nan | nan | nan | nan |
| multichannel_lora_moe_frozen_base_train_moe_head | second | 0.4609 | 0.0703 | 0.6422 | 0.3443 | 9.6701e-03 | 3.8413e-03 | 0.0777 |
| multichannel_lora_moe_frozen_base_train_moe_head | minute | 0.4163 | 0.0859 | 2.3208e-04 | 0.2823 | 0.7168 | 7.2245e-04 | 0.0732 |
| multichannel_lora_moe_frozen_base_train_moe_head | hour | 0.0604 | 0.1794 | 2.2735e-03 | 0.0134 | 0.9835 | 8.1543e-04 | 0.2398 |
| multichannel_asd_frozen_encoder_train_head | second | nan | nan | nan | nan | nan | nan | 2.5308e-03 |
| multichannel_asd_frozen_encoder_train_head | minute | nan | nan | nan | nan | nan | nan | 2.3323e-03 |
| multichannel_asd_frozen_encoder_train_head | hour | nan | nan | nan | nan | nan | nan | 8.5138e-04 |
| multichannel_asd_lora_moe_frozen_adapters_head | second | 0.3699 | 0.0960 | 2.5883e-03 | 0.1821 | 0.7741 | 0.0412 | 0.0961 |
| multichannel_asd_lora_moe_frozen_adapters_head | minute | 0.4873 | 0.0637 | 0.4188 | 0.5740 | 6.9258e-03 | 2.7571e-04 | 0.0657 |
| multichannel_asd_lora_moe_frozen_adapters_head | hour | 0.0461 | 0.1816 | 7.7729e-03 | 0.9880 | 1.9425e-03 | 2.2531e-03 | 0.2705 |

## Channels

`wap1_log_return`, `wap2_log_return`, `mid1_log_return`, `mid2_log_return`, `rel_spread1`, `rel_spread2`, `imbalance1`, `imbalance2`, `log_total_size1`, `log_total_size2`, `total_imbalance`, `updates_in_second`, `is_observed_update`, `seconds_since_update`, `second_frac`
