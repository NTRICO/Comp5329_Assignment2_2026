# Pre-PatchTST Targeted More-Data Run

This run compares targeted architectures with larger sample caps and more balanced training steps. `mid_token_asd_lora_moe_patchtst` places token-level spectral denoising between PatchTST and LoRA-MoE. `internal_after1_asd_lora_moe_patchtst` places it after the first encoder layer, before later encoder layers.

patch preset: `short_second`; layers=2; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096; seed=42.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 4,096 | 1.0000 | 8.0250e-07 | 5.5585e-04 | 0.4869 | nan |
| zero | minute | 3,224 | 1.0009 | 1.2435e-06 | 7.0896e-04 | 0.5067 | nan |
| zero | hour | 620.0000 | 1.0000 | 3.5023e-05 | 4.1690e-03 | 0.5048 | nan |
| raw_joint | second | 4,096 | 0.9855 | 7.9084e-07 | 5.5585e-04 | 0.5158 | 0.1228 |
| post_return_lora_moe_head | second | 4,096 | 0.9847 | 7.9017e-07 | 5.5287e-04 | 0.5374 | 0.1246 |
| mid_token_asd_lora_moe_patchtst | second | 4,096 | 0.9876 | 7.9254e-07 | 5.6040e-04 | 0.5080 | 0.1268 |
| internal_after1_asd_lora_moe_patchtst | second | 4,096 | 0.9826 | 7.8853e-07 | 5.5353e-04 | 0.5381 | 0.1390 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9836 | 7.8931e-07 | 5.5623e-04 | 0.5190 | 0.1339 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9837 | 7.8937e-07 | 5.5336e-04 | 0.5386 | 0.1309 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9788 | 7.8547e-07 | 5.5374e-04 | 0.5224 | 0.1481 |
| raw_joint | minute | 3,224 | 0.9937 | 1.2345e-06 | 7.0643e-04 | 0.5263 | 0.0913 |
| post_return_lora_moe_head | minute | 3,224 | 1.0072 | 1.2513e-06 | 7.1480e-04 | 0.5014 | 0.0709 |
| mid_token_asd_lora_moe_patchtst | minute | 3,224 | 0.9955 | 1.2368e-06 | 7.0803e-04 | 0.5210 | 0.0865 |
| internal_after1_asd_lora_moe_patchtst | minute | 3,224 | 1.0003 | 1.2427e-06 | 7.1156e-04 | 0.5104 | 0.0922 |
| pre_return_asd_lora_moe_patchtst | minute | 3,224 | 1.0008 | 1.2433e-06 | 7.1027e-04 | 0.5111 | 0.0722 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 3,224 | 0.9955 | 1.2368e-06 | 7.0764e-04 | 0.5276 | 0.0865 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 3,224 | 0.9914 | 1.2317e-06 | 7.0657e-04 | 0.5263 | 0.0964 |
| raw_joint | hour | 620.0000 | 0.5239 | 1.8350e-05 | 2.7924e-03 | 0.7452 | 0.6918 |
| post_return_lora_moe_head | hour | 620.0000 | 0.5144 | 1.8015e-05 | 2.7722e-03 | 0.7581 | 0.6974 |
| mid_token_asd_lora_moe_patchtst | hour | 620.0000 | 0.5172 | 1.8112e-05 | 2.7568e-03 | 0.7565 | 0.6961 |
| internal_after1_asd_lora_moe_patchtst | hour | 620.0000 | 0.5179 | 1.8139e-05 | 2.7790e-03 | 0.7500 | 0.6948 |
| pre_return_asd_lora_moe_patchtst | hour | 620.0000 | 0.5180 | 1.8141e-05 | 2.7824e-03 | 0.7548 | 0.6967 |
| gated_pre_return_asd_lora_moe_patchtst | hour | 620.0000 | 0.5120 | 1.7932e-05 | 2.7455e-03 | 0.7597 | 0.6999 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 620.0000 | 0.5141 | 1.8004e-05 | 2.7564e-03 | 0.7565 | 0.6988 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gated_pre_return_asd_lora_moe_patchtst | hour | 620.0000 | 0.5120 | 1.7932e-05 | 2.7455e-03 | 0.7597 | 0.6999 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 620.0000 | 0.5141 | 1.8004e-05 | 2.7564e-03 | 0.7565 | 0.6988 |
| post_return_lora_moe_head | hour | 620.0000 | 0.5144 | 1.8015e-05 | 2.7722e-03 | 0.7581 | 0.6974 |
| mid_token_asd_lora_moe_patchtst | hour | 620.0000 | 0.5172 | 1.8112e-05 | 2.7568e-03 | 0.7565 | 0.6961 |
| internal_after1_asd_lora_moe_patchtst | hour | 620.0000 | 0.5179 | 1.8139e-05 | 2.7790e-03 | 0.7500 | 0.6948 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 3,224 | 0.9914 | 1.2317e-06 | 7.0657e-04 | 0.5263 | 0.0964 |
| raw_joint | minute | 3,224 | 0.9937 | 1.2345e-06 | 7.0643e-04 | 0.5263 | 0.0913 |
| mid_token_asd_lora_moe_patchtst | minute | 3,224 | 0.9955 | 1.2368e-06 | 7.0803e-04 | 0.5210 | 0.0865 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 3,224 | 0.9955 | 1.2368e-06 | 7.0764e-04 | 0.5276 | 0.0865 |
| internal_after1_asd_lora_moe_patchtst | minute | 3,224 | 1.0003 | 1.2427e-06 | 7.1156e-04 | 0.5104 | 0.0922 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9788 | 7.8547e-07 | 5.5374e-04 | 0.5224 | 0.1481 |
| internal_after1_asd_lora_moe_patchtst | second | 4,096 | 0.9826 | 7.8853e-07 | 5.5353e-04 | 0.5381 | 0.1390 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9836 | 7.8931e-07 | 5.5623e-04 | 0.5190 | 0.1339 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9837 | 7.8937e-07 | 5.5336e-04 | 0.5386 | 0.1309 |
| post_return_lora_moe_head | second | 4,096 | 0.9847 | 7.9017e-07 | 5.5287e-04 | 0.5374 | 0.1246 |

## Diagnostics

| model | scale | asd_gate_mean | asd_tau_mean | asd_mean_abs_delta | gate_mean | tau_mean | local_mask_mean | pre_adapter_gate_mean | pre_adapter_mean_abs_delta | final_gate_mean | final_mean_abs_delta | router_entropy | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | adapter_kind_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| post_return_lora_moe_head | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4645 | 0.3724 | 4.7172e-06 | 0.6135 | 0.0141 | nan |
| post_return_lora_moe_head | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4632 | 0.0000e+00 | 0.6112 | 1.1213e-03 | 0.3877 | nan |
| post_return_lora_moe_head | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4776 | 2.1316e-03 | 0.3571 | 0.1655 | 0.4752 | nan |
| mid_token_asd_lora_moe_patchtst | second | nan | nan | nan | 0.9742 | 3.2995 | 0.1551 | nan | nan | nan | nan | 0.3782 | 0.0111 | 0.2439 | 0.7443 | 6.9763e-04 | nan |
| mid_token_asd_lora_moe_patchtst | minute | nan | nan | nan | 7.9715e-03 | 1.3360 | 0.4173 | nan | nan | nan | nan | 0.4985 | 0.0000e+00 | 5.5189e-03 | 0.5057 | 0.4887 | nan |
| mid_token_asd_lora_moe_patchtst | hour | nan | nan | nan | 0.6067 | 1.5189 | 0.4559 | nan | nan | nan | nan | 0.4651 | 9.3643e-03 | 5.3363e-03 | 0.5931 | 0.3922 | nan |
| internal_after1_asd_lora_moe_patchtst | second | nan | nan | nan | 0.9779 | 3.7658 | 0.1190 | nan | nan | nan | nan | 0.2476 | 0.8756 | 0.0928 | 0.0112 | 0.0203 | nan |
| internal_after1_asd_lora_moe_patchtst | minute | nan | nan | nan | 0.0148 | 0.8194 | 0.5699 | nan | nan | nan | nan | 0.4518 | 3.4574e-03 | 2.4037e-03 | 0.3447 | 0.6494 | nan |
| internal_after1_asd_lora_moe_patchtst | hour | nan | nan | nan | 0.6855 | 0.8892 | 0.6927 | nan | nan | nan | nan | 0.3967 | 2.1439e-03 | 4.7552e-03 | 0.6729 | 0.3202 | nan |
| pre_return_asd_lora_moe_patchtst | second | 0.0269 | 1.6383 | 3.4753e-03 | 0.0269 | 1.6383 | nan | 0.0224 | 9.1930e-03 | 0.1192 | 4.2881e-03 | 0.4948 | 0.1111 | 0.0965 | 0.4330 | 0.3593 | nan |
| pre_return_asd_lora_moe_patchtst | minute | 0.0261 | 1.1808 | 8.0882e-03 | 0.0261 | 1.1808 | nan | 0.0226 | 4.2932e-03 | 0.1192 | 8.5860e-03 | 0.4947 | 0.2687 | 0.2810 | 0.2461 | 0.2043 | nan |
| pre_return_asd_lora_moe_patchtst | hour | 0.0159 | 1.8864 | 3.5939e-03 | 0.0159 | 1.8864 | nan | 6.5409e-03 | 4.5121e-03 | 0.1192 | 3.7046e-03 | 0.4937 | 0.2443 | 0.2116 | 0.2795 | 0.2646 | nan |
| gated_pre_return_asd_lora_moe_patchtst | second | 0.0418 | 7.9263 | 0.0157 | 0.0418 | 7.9263 | nan | 0.0263 | 0.0282 | 0.1426 | 2.3497e-03 | 0.4989 | 0.4219 | 0.0747 | 0.4046 | 0.0987 | nan |
| gated_pre_return_asd_lora_moe_patchtst | minute | 0.0124 | 1.3161 | 4.2357e-03 | 0.0124 | 1.3161 | nan | 0.0174 | 0.0110 | 0.1131 | 5.6019e-04 | 0.4945 | 7.5673e-03 | 0.4385 | 0.1561 | 0.3978 | nan |
| gated_pre_return_asd_lora_moe_patchtst | hour | 0.0216 | 9.9345 | 0.0144 | 0.0216 | 9.9345 | nan | 0.0320 | 0.0160 | 0.0863 | 1.3098e-03 | 0.4963 | 0.0625 | 0.0997 | 0.3982 | 0.4397 | nan |
| scale_specific_gated_pre_asd_moe_patchtst | second | 0.0280 | 3.3705 | 6.7976e-03 | 0.0280 | 3.3705 | nan | 4.1035e-03 | 4.5938e-03 | 2.3744e-03 | 1.6269e-05 | 0.4763 | 0.1283 | 4.0081e-06 | 0.4320 | 0.4397 | 0.0000e+00 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 0.0226 | 2.5965 | 0.0125 | 0.0226 | 2.5965 | nan | 0.0165 | 0.0168 | 0.1007 | 1.3575e-03 | 0.4981 | 0.2361 | 0.2589 | 0.2406 | 0.2644 | 1.0000 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 0.0649 | 14.2185 | 0.0452 | 0.0649 | 14.2185 | nan | 3.0232e-03 | 0.0143 | 0.3255 | 0.0148 | 0.4783 | 0.2102 | 0.2623 | 0.2457 | 0.2818 | 0.0000e+00 |

## Files

- summary: `outputs\singlechannel_patchtst_32stocks_more_data\summary.csv`
- diagnostics: `outputs\singlechannel_patchtst_32stocks_more_data\diagnostics.csv`
