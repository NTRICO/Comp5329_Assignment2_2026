# Pre-PatchTST Targeted More-Data Run

This run compares targeted architectures with larger sample caps and more balanced training steps. `mid_token_asd_lora_moe_patchtst` places token-level spectral denoising between PatchTST and LoRA-MoE.

patch preset: `short_second`; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096; seed=42.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 4,096 | 1.0000 | 9.4161e-07 | 6.1005e-04 | 0.4877 | nan |
| zero | minute | 1,040 | 1.0036 | 1.5905e-06 | 7.9701e-04 | 0.4957 | nan |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| raw_joint | second | 4,096 | 1.0044 | 9.4573e-07 | 6.1416e-04 | 0.5221 | 0.0573 |
| post_return_lora_moe_head | second | 4,096 | 0.9987 | 9.4041e-07 | 6.0956e-04 | 0.5424 | 0.0630 |
| mid_token_asd_lora_moe_patchtst | second | 4,096 | 0.9978 | 9.3951e-07 | 6.1028e-04 | 0.5316 | 0.0764 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9984 | 9.4011e-07 | 6.1010e-04 | 0.5329 | 0.0683 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9951 | 9.3704e-07 | 6.0923e-04 | 0.5429 | 0.0795 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 1.0009 | 9.4249e-07 | 6.1120e-04 | 0.5312 | 0.0655 |
| raw_joint | minute | 1,040 | 1.0066 | 1.5952e-06 | 7.9833e-04 | 0.5269 | 0.0859 |
| post_return_lora_moe_head | minute | 1,040 | 0.9946 | 1.5761e-06 | 7.9758e-04 | 0.5260 | 0.0793 |
| mid_token_asd_lora_moe_patchtst | minute | 1,040 | 1.0108 | 1.6018e-06 | 8.0165e-04 | 0.5096 | 0.0927 |
| pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0103 | 1.6011e-06 | 8.0020e-04 | 0.5115 | 0.0786 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0007 | 1.5858e-06 | 7.9651e-04 | 0.5308 | 0.0846 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 1,040 | 1.0038 | 1.5908e-06 | 7.9727e-04 | 0.5269 | 0.0784 |
| raw_joint | hour | 200.0000 | 0.5494 | 2.5743e-05 | 3.1412e-03 | 0.6950 | 0.6728 |
| post_return_lora_moe_head | hour | 200.0000 | 0.5242 | 2.4564e-05 | 3.1069e-03 | 0.7250 | 0.6898 |
| mid_token_asd_lora_moe_patchtst | hour | 200.0000 | 0.5340 | 2.5021e-05 | 3.1688e-03 | 0.7550 | 0.6859 |
| pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5438 | 2.5482e-05 | 3.1358e-03 | 0.7250 | 0.6767 |
| gated_pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5473 | 2.5643e-05 | 3.1604e-03 | 0.7350 | 0.6747 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 200.0000 | 0.5444 | 2.5508e-05 | 3.1651e-03 | 0.7500 | 0.6801 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| post_return_lora_moe_head | hour | 200.0000 | 0.5242 | 2.4564e-05 | 3.1069e-03 | 0.7250 | 0.6898 |
| mid_token_asd_lora_moe_patchtst | hour | 200.0000 | 0.5340 | 2.5021e-05 | 3.1688e-03 | 0.7550 | 0.6859 |
| pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5438 | 2.5482e-05 | 3.1358e-03 | 0.7250 | 0.6767 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 200.0000 | 0.5444 | 2.5508e-05 | 3.1651e-03 | 0.7500 | 0.6801 |
| gated_pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5473 | 2.5643e-05 | 3.1604e-03 | 0.7350 | 0.6747 |
| post_return_lora_moe_head | minute | 1,040 | 0.9946 | 1.5761e-06 | 7.9758e-04 | 0.5260 | 0.0793 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0007 | 1.5858e-06 | 7.9651e-04 | 0.5308 | 0.0846 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 1,040 | 1.0038 | 1.5908e-06 | 7.9727e-04 | 0.5269 | 0.0784 |
| raw_joint | minute | 1,040 | 1.0066 | 1.5952e-06 | 7.9833e-04 | 0.5269 | 0.0859 |
| pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0103 | 1.6011e-06 | 8.0020e-04 | 0.5115 | 0.0786 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9951 | 9.3704e-07 | 6.0923e-04 | 0.5429 | 0.0795 |
| mid_token_asd_lora_moe_patchtst | second | 4,096 | 0.9978 | 9.3951e-07 | 6.1028e-04 | 0.5316 | 0.0764 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9984 | 9.4011e-07 | 6.1010e-04 | 0.5329 | 0.0683 |
| post_return_lora_moe_head | second | 4,096 | 0.9987 | 9.4041e-07 | 6.0956e-04 | 0.5424 | 0.0630 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 1.0009 | 9.4249e-07 | 6.1120e-04 | 0.5312 | 0.0655 |

## Diagnostics

| model | scale | asd_gate_mean | asd_tau_mean | asd_mean_abs_delta | gate_mean | tau_mean | local_mask_mean | pre_adapter_gate_mean | pre_adapter_mean_abs_delta | final_gate_mean | final_mean_abs_delta | router_entropy | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | adapter_kind_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| post_return_lora_moe_head | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.2430 | 1.8178e-03 | 4.3918e-04 | 0.8823 | 0.1155 | nan |
| post_return_lora_moe_head | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4357 | 0.6438 | 9.9301e-04 | 0.0000e+00 | 0.3552 | nan |
| post_return_lora_moe_head | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.3361 | 0.7342 | 5.4263e-06 | 0.0983 | 0.1675 | nan |
| mid_token_asd_lora_moe_patchtst | second | nan | nan | nan | 0.9791 | 3.5247 | 0.1295 | nan | nan | nan | nan | 0.4235 | 3.5689e-03 | 0.2842 | 4.4076e-03 | 0.7078 | nan |
| mid_token_asd_lora_moe_patchtst | minute | nan | nan | nan | 7.1049e-03 | 1.6324 | 0.3734 | nan | nan | nan | nan | 0.4438 | 0.6679 | 0.3131 | 0.0187 | 3.1225e-04 | nan |
| mid_token_asd_lora_moe_patchtst | hour | nan | nan | nan | 0.9298 | 1.6589 | 0.4112 | nan | nan | nan | nan | 0.4409 | 0.2269 | 0.5349 | 0.2382 | 0.0000e+00 | nan |
| pre_return_asd_lora_moe_patchtst | second | 9.7538e-03 | 3.4750 | 2.2070e-03 | 9.7538e-03 | 3.4750 | nan | 7.8007e-03 | 7.8181e-03 | 0.1192 | 2.3542e-03 | 0.4798 | 3.4754e-06 | 0.4345 | 0.4726 | 0.0929 | nan |
| pre_return_asd_lora_moe_patchtst | minute | 0.0403 | 1.0231 | 0.0109 | 0.0403 | 1.0231 | nan | 8.3935e-03 | 9.1979e-03 | 0.1192 | 0.0110 | 0.4686 | 0.0684 | 0.3511 | 0.2559 | 0.3246 | nan |
| pre_return_asd_lora_moe_patchtst | hour | 7.6124e-03 | 6.1981 | 4.3259e-03 | 7.6124e-03 | 6.1981 | nan | 8.1755e-03 | 7.7326e-03 | 0.1192 | 4.3787e-03 | 0.4733 | 0.3243 | 0.1409 | 0.3651 | 0.1698 | nan |
| gated_pre_return_asd_lora_moe_patchtst | second | 0.0382 | 11.4128 | 0.0122 | 0.0382 | 11.4128 | nan | 0.0107 | 0.0688 | 0.0987 | 1.3852e-03 | 0.4862 | 0.4209 | 0.0117 | 0.0449 | 0.5226 | nan |
| gated_pre_return_asd_lora_moe_patchtst | minute | 0.0128 | 0.9611 | 3.2769e-03 | 0.0128 | 0.9611 | nan | 0.0229 | 0.0499 | 0.1212 | 6.1732e-04 | 0.4852 | 0.1023 | 0.3369 | 0.2840 | 0.2768 | nan |
| gated_pre_return_asd_lora_moe_patchtst | hour | 0.0425 | 15.9942 | 0.0303 | 0.0425 | 15.9942 | nan | 0.0240 | 0.0715 | 0.2062 | 6.7131e-03 | 0.4771 | 0.3446 | 0.0614 | 0.0854 | 0.5086 | nan |
| scale_specific_gated_pre_asd_moe_patchtst | second | 0.0125 | 7.4789 | 4.0680e-03 | 0.0125 | 7.4789 | nan | 0.0284 | 0.1873 | 4.0267e-03 | 1.0097e-04 | 0.4639 | 0.4541 | 0.0928 | 0.4454 | 7.6263e-03 | 0.0000e+00 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 0.0232 | 0.8952 | 5.5691e-03 | 0.0232 | 0.8952 | nan | 9.8220e-03 | 0.0157 | 0.0526 | 2.7028e-04 | 0.4954 | 0.2520 | 0.2002 | 0.2722 | 0.2756 | 1.0000 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 0.0918 | 18.5653 | 0.0655 | 0.0918 | 18.5653 | nan | 8.9190e-03 | 0.0999 | 0.5575 | 0.0343 | 0.4693 | 0.4357 | 0.0000e+00 | 0.0640 | 0.5003 | 0.0000e+00 |

## Files

- summary: `outputs\prepatch_asd_adapter_patchtst\targeted_mid_token_more_data\summary.csv`
- diagnostics: `outputs\prepatch_asd_adapter_patchtst\targeted_mid_token_more_data\diagnostics.csv`
