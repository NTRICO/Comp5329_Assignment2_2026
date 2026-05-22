# Level-ASD PatchTST Experiment

This small experiment compares return-domain ASD with price-domain ASD. For price-domain ASD, the module first cleans WAP1 price/log-price, then the model converts the cleaned path to returns.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`; patch preset: `short_second`; epochs=3; balanced steps/epoch=12; init_gate=-4.0.

## Test NMSE

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 1,024 | 1.0023 | 9.6510e-07 | 6.1415e-04 | 0.4568 | nan |
| last_return | second | 1,024 | 49.8953 | 4.8042e-05 | 3.1047e-03 | 0.4882 | -0.0112 |
| raw_joint | second | 1,024 | 1.0177 | 9.7993e-07 | 6.1876e-04 | 0.5029 | -0.0334 |
| return_asd_frozen_encoder_train_head | second | 1,024 | 1.0384 | 9.9980e-07 | 6.3007e-04 | 0.4745 | -0.0476 |
| level_asd_log_price_frozen_encoder_train_head | second | 1,024 | 1.0302 | 9.9193e-07 | 6.2379e-04 | 0.5393 | -0.0572 |
| level_asd_raw_price_frozen_encoder_train_head | second | 1,024 | 1.0226 | 9.8459e-07 | 6.2255e-04 | 0.5413 | -0.0277 |
| zero | minute | 1,024 | 1.0030 | 1.5852e-06 | 7.9712e-04 | 0.4936 | nan |
| last_return | minute | 1,024 | 2.0259 | 3.2019e-06 | 1.1921e-03 | 0.4673 | -0.0169 |
| raw_joint | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| return_asd_frozen_encoder_train_head | minute | 1,024 | 1.0038 | 1.5865e-06 | 7.9879e-04 | 0.5024 | 0.0259 |
| level_asd_log_price_frozen_encoder_train_head | minute | 1,024 | 1.0016 | 1.5830e-06 | 7.9628e-04 | 0.5112 | 0.0426 |
| level_asd_raw_price_frozen_encoder_train_head | minute | 1,024 | 0.9980 | 1.5773e-06 | 7.9754e-04 | 0.5034 | 0.0503 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| last_return | hour | 200.0000 | 3.0770 | 1.4418e-04 | 8.1385e-03 | 0.3150 | -0.5431 |
| raw_joint | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| return_asd_frozen_encoder_train_head | hour | 200.0000 | 0.8696 | 4.0745e-05 | 4.2084e-03 | 0.7000 | 0.5328 |
| level_asd_log_price_frozen_encoder_train_head | hour | 200.0000 | 0.8826 | 4.1357e-05 | 4.2815e-03 | 0.5700 | 0.5409 |
| level_asd_raw_price_frozen_encoder_train_head | hour | 200.0000 | 0.8692 | 4.0729e-05 | 4.2026e-03 | 0.6850 | 0.5284 |

## ASD Diagnostics

| model | scale | gate_mean | tau_mean | mean_abs_delta | level_asd_gate_mean | level_asd_tau_mean | level_asd_mean_abs_delta | clean_return_abs_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan |
| return_asd_frozen_encoder_train_head | second | 0.0103 | 3.0163 | 2.2165e-03 | nan | nan | nan | nan |
| return_asd_frozen_encoder_train_head | minute | 8.5216e-03 | 0.4946 | 1.1657e-03 | nan | nan | nan | nan |
| return_asd_frozen_encoder_train_head | hour | 1.1422e-03 | 0.2558 | 3.8772e-05 | nan | nan | nan | nan |
| level_asd_log_price_frozen_encoder_train_head | second | 0.0199 | 4.9720e-03 | 4.1287e-06 | 0.0199 | 4.9720e-03 | 4.1287e-06 | 0.3209 |
| level_asd_log_price_frozen_encoder_train_head | minute | 8.2845e-03 | 1.9092e-03 | 3.9834e-06 | 8.2845e-03 | 1.9092e-03 | 3.9834e-06 | 0.5672 |
| level_asd_log_price_frozen_encoder_train_head | hour | 3.1881e-03 | 4.0455e-03 | 1.5596e-06 | 3.1881e-03 | 4.0455e-03 | 1.5596e-06 | 0.6923 |
| level_asd_raw_price_frozen_encoder_train_head | second | 6.7583e-03 | 1.4435 | 1.5243e-04 | 6.7583e-03 | 1.4435 | 1.5243e-04 | 0.3241 |
| level_asd_raw_price_frozen_encoder_train_head | minute | 8.8290e-03 | 1.0124 | 1.1173e-03 | 8.8290e-03 | 1.0124 | 1.1173e-03 | 0.5652 |
| level_asd_raw_price_frozen_encoder_train_head | hour | 1.2556e-03 | 4.8439 | 1.9007e-04 | 1.2556e-03 | 4.8439 | 1.9007e-04 | 0.6918 |

## Files

- summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\level_asd_patchtst\summary.csv`
- diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\level_asd_patchtst\diagnostics.csv`
