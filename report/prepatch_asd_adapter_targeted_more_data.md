# Pre-PatchTST Targeted More-Data Run

This run compares five targeted architectures with larger sample caps and more balanced training steps.

patch preset: `short_second`; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096; seed=42.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 4,096 | 1.0000 | 9.4161e-07 | 6.1005e-04 | 0.4877 | nan |
| zero | minute | 1,040 | 1.0036 | 1.5905e-06 | 7.9701e-04 | 0.4957 | nan |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| raw_joint | second | 4,096 | 1.0044 | 9.4573e-07 | 6.1416e-04 | 0.5221 | 0.0573 |
| post_return_lora_moe_head | second | 4,096 | 0.9987 | 9.4041e-07 | 6.0956e-04 | 0.5424 | 0.0630 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 1.0042 | 9.4556e-07 | 6.1452e-04 | 0.5267 | 0.0689 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 1.0004 | 9.4202e-07 | 6.1034e-04 | 0.5366 | 0.0557 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 1.0000 | 9.4164e-07 | 6.1183e-04 | 0.5147 | 0.0626 |
| raw_joint | minute | 1,040 | 1.0066 | 1.5952e-06 | 7.9833e-04 | 0.5269 | 0.0859 |
| post_return_lora_moe_head | minute | 1,040 | 0.9946 | 1.5761e-06 | 7.9758e-04 | 0.5260 | 0.0793 |
| pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0003 | 1.5851e-06 | 7.9653e-04 | 0.5202 | 0.0713 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0014 | 1.5870e-06 | 7.9640e-04 | 0.5298 | 0.0870 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 1,040 | 1.0051 | 1.5929e-06 | 7.9776e-04 | 0.5356 | 0.0809 |
| raw_joint | hour | 200.0000 | 0.5494 | 2.5743e-05 | 3.1412e-03 | 0.6950 | 0.6728 |
| post_return_lora_moe_head | hour | 200.0000 | 0.5242 | 2.4564e-05 | 3.1069e-03 | 0.7250 | 0.6898 |
| pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5394 | 2.5274e-05 | 3.1347e-03 | 0.7200 | 0.6796 |
| gated_pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5552 | 2.6015e-05 | 3.2424e-03 | 0.7300 | 0.6744 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 200.0000 | 0.5403 | 2.5315e-05 | 3.1576e-03 | 0.7100 | 0.6815 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| post_return_lora_moe_head | hour | 200.0000 | 0.5242 | 2.4564e-05 | 3.1069e-03 | 0.7250 | 0.6898 |
| pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5394 | 2.5274e-05 | 3.1347e-03 | 0.7200 | 0.6796 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 200.0000 | 0.5403 | 2.5315e-05 | 3.1576e-03 | 0.7100 | 0.6815 |
| raw_joint | hour | 200.0000 | 0.5494 | 2.5743e-05 | 3.1412e-03 | 0.6950 | 0.6728 |
| gated_pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5552 | 2.6015e-05 | 3.2424e-03 | 0.7300 | 0.6744 |
| post_return_lora_moe_head | minute | 1,040 | 0.9946 | 1.5761e-06 | 7.9758e-04 | 0.5260 | 0.0793 |
| pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0003 | 1.5851e-06 | 7.9653e-04 | 0.5202 | 0.0713 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0014 | 1.5870e-06 | 7.9640e-04 | 0.5298 | 0.0870 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 1,040 | 1.0051 | 1.5929e-06 | 7.9776e-04 | 0.5356 | 0.0809 |
| raw_joint | minute | 1,040 | 1.0066 | 1.5952e-06 | 7.9833e-04 | 0.5269 | 0.0859 |
| post_return_lora_moe_head | second | 4,096 | 0.9987 | 9.4041e-07 | 6.0956e-04 | 0.5424 | 0.0630 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 1.0000 | 9.4164e-07 | 6.1183e-04 | 0.5147 | 0.0626 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 1.0004 | 9.4202e-07 | 6.1034e-04 | 0.5366 | 0.0557 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 1.0042 | 9.4556e-07 | 6.1452e-04 | 0.5267 | 0.0689 |
| raw_joint | second | 4,096 | 1.0044 | 9.4573e-07 | 6.1416e-04 | 0.5221 | 0.0573 |

## Diagnostics

| model | scale | asd_gate_mean | asd_tau_mean | asd_mean_abs_delta | pre_adapter_gate_mean | pre_adapter_mean_abs_delta | final_gate_mean | final_mean_abs_delta | router_entropy | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | adapter_kind_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| post_return_lora_moe_head | second | nan | nan | nan | nan | nan | nan | nan | 0.2430 | 1.8178e-03 | 4.3918e-04 | 0.8823 | 0.1155 | nan |
| post_return_lora_moe_head | minute | nan | nan | nan | nan | nan | nan | nan | 0.4357 | 0.6438 | 9.9301e-04 | 0.0000e+00 | 0.3552 | nan |
| post_return_lora_moe_head | hour | nan | nan | nan | nan | nan | nan | nan | 0.3361 | 0.7342 | 5.4263e-06 | 0.0983 | 0.1675 | nan |
| pre_return_asd_lora_moe_patchtst | second | 0.0185 | 6.6804 | 5.8797e-03 | 0.0111 | 0.0460 | 0.1192 | 5.1342e-03 | 0.4743 | 0.4245 | 0.4708 | 0.1046 | 1.9578e-08 | nan |
| pre_return_asd_lora_moe_patchtst | minute | 7.1905e-03 | 0.6501 | 1.2773e-03 | 0.0118 | 0.0545 | 0.1192 | 3.2552e-03 | 0.4505 | 0.3056 | 0.0000e+00 | 0.3111 | 0.3833 | nan |
| pre_return_asd_lora_moe_patchtst | hour | 2.0175e-03 | 0.4373 | 1.1630e-04 | 0.0208 | 0.0516 | 0.1192 | 5.1685e-03 | 0.4641 | 0.4678 | 0.4195 | 0.1127 | 0.0000e+00 | nan |
| gated_pre_return_asd_lora_moe_patchtst | second | 0.0138 | 2.7198 | 2.5721e-03 | 7.7306e-03 | 2.4365e-03 | 0.0933 | 2.6303e-04 | 0.4947 | 0.4673 | 0.0474 | 0.4384 | 0.0469 | nan |
| gated_pre_return_asd_lora_moe_patchtst | minute | 0.0129 | 0.7441 | 2.6068e-03 | 0.0167 | 0.0204 | 0.1108 | 2.4276e-04 | 0.4848 | 0.3316 | 0.3299 | 7.2119e-05 | 0.3384 | nan |
| gated_pre_return_asd_lora_moe_patchtst | hour | 0.0135 | 6.7363 | 7.9788e-03 | 9.2386e-03 | 2.2773e-03 | 0.0957 | 7.6459e-04 | 0.4954 | 0.4985 | 8.9286e-03 | 0.4827 | 9.9289e-03 | nan |
| scale_specific_gated_pre_asd_moe_patchtst | second | 0.0231 | 4.1987 | 5.9481e-03 | 6.4098e-03 | 9.8364e-03 | 2.3559e-03 | 1.3954e-05 | 0.4834 | 0.0263 | 0.1495 | 0.4342 | 0.3900 | 0.0000e+00 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 0.0119 | 1.5107 | 4.4767e-03 | 0.0242 | 7.4729e-03 | 0.0611 | 3.0374e-04 | 0.4769 | 0.3477 | 0.3351 | 0.3172 | 0.0000e+00 | 1.0000 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 0.0171 | 10.5960 | 0.0118 | 0.0205 | 0.0131 | 0.1424 | 1.8028e-03 | 0.4573 | 0.0000e+00 | 0.4202 | 0.2925 | 0.2873 | 0.0000e+00 |

## Files

- summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\targeted_more_data\summary.csv`
- diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\targeted_more_data\diagnostics.csv`
