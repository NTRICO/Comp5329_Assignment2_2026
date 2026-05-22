# Level-ASD PatchTST Experiment

This small experiment compares return-domain ASD with price-domain ASD. For price-domain ASD, the module first cleans WAP1 price/log-price, then the model converts the cleaned path to returns.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`; patch preset: `short_second`; epochs=3; balanced steps/epoch=12; init_gate=-4.0.

## Test NMSE

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 1,024 | 1.0023 | 9.6510e-07 | 6.1415e-04 | 0.4568 | nan |
| last_return | second | 1,024 | 49.8953 | 4.8042e-05 | 3.1047e-03 | 0.4882 | -0.0112 |
| lora_moe_frozen_base_train_moe_head | second | 1,024 | 1.0288 | 9.9055e-07 | 6.2279e-04 | 0.5442 | -0.0513 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | second | 1,024 | 1.0329 | 9.9454e-07 | 6.2778e-04 | 0.4794 | -0.0366 |
| raw_joint | second | 1,024 | 1.0177 | 9.7993e-07 | 6.1876e-04 | 0.5029 | -0.0334 |
| return_asd_frozen_encoder_train_head | second | 1,024 | 1.0384 | 9.9980e-07 | 6.3007e-04 | 0.4745 | -0.0476 |
| level_asd_log_price_frozen_encoder_train_head | second | 1,024 | 1.0320 | 9.9369e-07 | 6.2817e-04 | 0.4745 | -0.0298 |
| level_asd_raw_price_frozen_encoder_train_head | second | 1,024 | 1.0254 | 9.8729e-07 | 6.2189e-04 | 0.5000 | -0.0400 |
| zero | minute | 1,024 | 1.0030 | 1.5852e-06 | 7.9712e-04 | 0.4936 | nan |
| last_return | minute | 1,024 | 2.0259 | 3.2019e-06 | 1.1921e-03 | 0.4673 | -0.0169 |
| lora_moe_frozen_base_train_moe_head | minute | 1,024 | 1.0175 | 1.6082e-06 | 8.0255e-04 | 0.5181 | 0.0353 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | minute | 1,024 | 0.9989 | 1.5787e-06 | 7.9953e-04 | 0.5054 | 0.0405 |
| raw_joint | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| return_asd_frozen_encoder_train_head | minute | 1,024 | 1.0038 | 1.5865e-06 | 7.9879e-04 | 0.5024 | 0.0259 |
| level_asd_log_price_frozen_encoder_train_head | minute | 1,024 | 1.0015 | 1.5828e-06 | 7.9611e-04 | 0.5073 | 0.0394 |
| level_asd_raw_price_frozen_encoder_train_head | minute | 1,024 | 1.0038 | 1.5864e-06 | 7.9670e-04 | 0.5249 | 0.0263 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| last_return | hour | 200.0000 | 3.0770 | 1.4418e-04 | 8.1385e-03 | 0.3150 | -0.5431 |
| lora_moe_frozen_base_train_moe_head | hour | 200.0000 | 0.7879 | 3.6916e-05 | 3.9822e-03 | 0.7050 | 0.5809 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | hour | 200.0000 | 0.8436 | 3.9528e-05 | 4.1293e-03 | 0.6950 | 0.5404 |
| raw_joint | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| return_asd_frozen_encoder_train_head | hour | 200.0000 | 0.8696 | 4.0745e-05 | 4.2084e-03 | 0.7000 | 0.5328 |
| level_asd_log_price_frozen_encoder_train_head | hour | 200.0000 | 0.8851 | 4.1471e-05 | 4.2986e-03 | 0.5650 | 0.5387 |
| level_asd_raw_price_frozen_encoder_train_head | hour | 200.0000 | 0.8763 | 4.1059e-05 | 4.2410e-03 | 0.6050 | 0.5394 |

## Test NMSE Relative To Raw

| model | scale | nmse | nmse_vs_raw_pct |
| --- | --- | --- | --- |
| lora_moe_frozen_base_train_moe_head | second | 1.0288 | 1.0836 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | second | 1.0329 | 1.4902 |
| return_asd_frozen_encoder_train_head | second | 1.0384 | 2.0275 |
| level_asd_log_price_frozen_encoder_train_head | second | 1.0320 | 1.4041 |
| level_asd_raw_price_frozen_encoder_train_head | second | 1.0254 | 0.7501 |
| lora_moe_frozen_base_train_moe_head | minute | 1.0175 | 0.0814 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | minute | 0.9989 | -1.7522 |
| return_asd_frozen_encoder_train_head | minute | 1.0038 | -1.2692 |
| level_asd_log_price_frozen_encoder_train_head | minute | 1.0015 | -1.4990 |
| level_asd_raw_price_frozen_encoder_train_head | minute | 1.0038 | -1.2735 |
| lora_moe_frozen_base_train_moe_head | hour | 0.7879 | -15.7065 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | hour | 0.8436 | -9.7417 |
| return_asd_frozen_encoder_train_head | hour | 0.8696 | -6.9635 |
| level_asd_log_price_frozen_encoder_train_head | hour | 0.8851 | -5.3058 |
| level_asd_raw_price_frozen_encoder_train_head | hour | 0.8763 | -6.2456 |

## ASD Diagnostics

| model | scale | gate_mean | tau_mean | mean_abs_delta | level_asd_gate_mean | level_asd_tau_mean | level_asd_mean_abs_delta | clean_return_abs_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan |
| return_asd_frozen_encoder_train_head | second | 0.0103 | 3.0163 | 2.2165e-03 | nan | nan | nan | nan |
| return_asd_frozen_encoder_train_head | minute | 8.5216e-03 | 0.4946 | 1.1657e-03 | nan | nan | nan | nan |
| return_asd_frozen_encoder_train_head | hour | 1.1422e-03 | 0.2558 | 3.8772e-05 | nan | nan | nan | nan |
| lora_moe_frozen_base_train_moe_head | second | nan | nan | 0.0216 | nan | nan | nan | nan |
| lora_moe_frozen_base_train_moe_head | minute | nan | nan | 0.1030 | nan | nan | nan | nan |
| lora_moe_frozen_base_train_moe_head | hour | nan | nan | 0.1231 | nan | nan | nan | nan |
| level_asd_log_price_frozen_encoder_train_head | second | 0.0147 | 5.0311e-03 | 3.0669e-06 | 0.0147 | 5.0311e-03 | 3.0669e-06 | 0.3223 |
| level_asd_log_price_frozen_encoder_train_head | minute | 0.0240 | 8.6454e-04 | 5.7154e-06 | 0.0240 | 8.6454e-04 | 5.7154e-06 | 0.5662 |
| level_asd_log_price_frozen_encoder_train_head | hour | 5.9873e-03 | 3.3246e-03 | 2.4124e-06 | 5.9873e-03 | 3.3246e-03 | 2.4124e-06 | 0.6922 |
| level_asd_raw_price_frozen_encoder_train_head | second | 0.0181 | 1.3147 | 3.7180e-04 | 0.0181 | 1.3147 | 3.7180e-04 | 0.3204 |
| level_asd_raw_price_frozen_encoder_train_head | minute | 7.3686e-03 | 2.6094 | 2.4035e-03 | 7.3686e-03 | 2.6094 | 2.4035e-03 | 0.5668 |
| level_asd_raw_price_frozen_encoder_train_head | hour | 5.1223e-03 | 8.0326 | 1.2858e-03 | 5.1223e-03 | 8.0326 | 1.2858e-03 | 0.6899 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | second | 0.0115 | 1.6058 | 0.0167 | 0.0115 | 1.6058 | 2.8929e-04 | 0.3226 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | minute | 0.0142 | 0.0756 | 0.0132 | 0.0142 | 0.0756 | 1.3448e-04 | 0.5616 |
| level_asd_raw_price_lora_moe_frozen_adapters_head | hour | 4.9943e-03 | 1.9369e-03 | 0.0259 | 4.9943e-03 | 1.9369e-03 | 1.1934e-06 | 0.6924 |

## Files

- summary: `outputs\level_asd_lora_moe_patchtst\summary.csv`
- diagnostics: `outputs\level_asd_lora_moe_patchtst\diagnostics.csv`
