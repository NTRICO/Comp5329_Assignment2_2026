# Gated Pre-ASD 32-Stock Multi-Seed Confirmation

This run checks whether `gated_pre_return_asd_lora_moe_patchtst` remains useful on the larger 32-stock cache across seeds. Training is balanced across `second/minute/hour`; no day data is included.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_32stocks_512t.npz`; seeds: `42, 43, 44`; patch preset: `short_second`; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096.

## Test Mean / Std

| model | scale | n_mean | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_pre_return_asd_lora_moe_patchtst | hour | 620.0000 | 0.5082 | 5.3007e-03 | 2.7405e-03 | 1.9595e-05 | 0.7538 | 4.0591e-03 | 0.7041 | 4.0672e-03 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 3,224 | 0.9915 | 5.6800e-04 | 7.0621e-04 | 4.3111e-07 | 0.5209 | 3.7415e-03 | 0.0977 | 3.1192e-03 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9851 | 1.6583e-03 | 5.5382e-04 | 2.0143e-06 | 0.5304 | 7.8502e-03 | 0.1296 | 6.1611e-03 |
| last_return | hour | 620.0000 | 3.0754 | 0.0000e+00 | 7.4308e-03 | 0.0000e+00 | 0.3097 | 0.0000e+00 | -0.5547 | 0.0000e+00 |
| last_return | minute | 3,224 | 2.2817 | 0.0000e+00 | 1.0833e-03 | 0.0000e+00 | 0.4696 | 0.0000e+00 | -0.0853 | 0.0000e+00 |
| last_return | second | 4,096 | 64.6097 | 0.0000e+00 | 2.7696e-03 | 0.0000e+00 | 0.4810 | 0.0000e+00 | -0.1016 | 0.0000e+00 |
| post_return_lora_moe_head | hour | 620.0000 | 0.5085 | 6.1427e-03 | 2.7473e-03 | 2.1693e-05 | 0.7554 | 3.3575e-03 | 0.7020 | 4.7033e-03 |
| post_return_lora_moe_head | minute | 3,224 | 1.0015 | 9.1610e-03 | 7.1237e-04 | 5.0084e-06 | 0.5096 | 0.0121 | 0.0895 | 0.0163 |
| post_return_lora_moe_head | second | 4,096 | 0.9846 | 1.0050e-03 | 5.5331e-04 | 7.8527e-07 | 0.5326 | 4.9052e-03 | 0.1290 | 3.8200e-03 |
| raw_joint | hour | 620.0000 | 0.5240 | 2.4070e-03 | 2.7933e-03 | 2.7938e-06 | 0.7419 | 0.0101 | 0.6962 | 4.2323e-03 |
| raw_joint | minute | 3,224 | 0.9927 | 1.7039e-03 | 7.0656e-04 | 3.2744e-07 | 0.5261 | 5.9199e-03 | 0.1006 | 9.0011e-03 |
| raw_joint | second | 4,096 | 0.9884 | 7.0890e-03 | 5.5509e-04 | 2.4634e-06 | 0.5234 | 7.5999e-03 | 0.1184 | 0.0212 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 620.0000 | 0.5099 | 4.0654e-03 | 2.7497e-03 | 2.4638e-05 | 0.7516 | 7.3913e-03 | 0.7042 | 3.5429e-03 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 3,224 | 0.9923 | 1.8691e-03 | 7.0693e-04 | 4.5754e-07 | 0.5281 | 4.2283e-03 | 0.1083 | 6.7231e-03 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9846 | 1.6259e-03 | 5.5491e-04 | 3.5136e-06 | 0.5285 | 0.0136 | 0.1271 | 7.9912e-03 |
| zero | hour | 620.0000 | 1.0000 | 0.0000e+00 | 4.1690e-03 | 0.0000e+00 | 0.5048 | 0.0000e+00 | nan | nan |
| zero | minute | 3,224 | 1.0009 | 0.0000e+00 | 7.0896e-04 | 0.0000e+00 | 0.5067 | 0.0000e+00 | nan | nan |
| zero | second | 4,096 | 1.0000 | 0.0000e+00 | 5.5585e-04 | 0.0000e+00 | 0.4869 | 0.0000e+00 | nan | nan |

## Seed-Level Test Rows

| seed | model | scale | n | nmse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.0000 | raw_joint | second | 4,096 | 0.9855 | 5.5585e-04 | 0.5158 | 0.1228 |
| 42.0000 | post_return_lora_moe_head | second | 4,096 | 0.9847 | 5.5287e-04 | 0.5374 | 0.1246 |
| 42.0000 | gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9836 | 5.5551e-04 | 0.5224 | 0.1298 |
| 42.0000 | scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9847 | 5.5871e-04 | 0.5134 | 0.1303 |
| 42.0000 | raw_joint | minute | 3,224 | 0.9937 | 7.0643e-04 | 0.5263 | 0.0913 |
| 42.0000 | post_return_lora_moe_head | minute | 3,224 | 1.0072 | 7.1480e-04 | 0.5014 | 0.0709 |
| 42.0000 | gated_pre_return_asd_lora_moe_patchtst | minute | 3,224 | 0.9916 | 7.0670e-04 | 0.5173 | 0.0951 |
| 42.0000 | scale_specific_gated_pre_asd_moe_patchtst | minute | 3,224 | 0.9908 | 7.0669e-04 | 0.5241 | 0.1088 |
| 42.0000 | raw_joint | hour | 620.0000 | 0.5239 | 2.7924e-03 | 0.7452 | 0.6918 |
| 42.0000 | post_return_lora_moe_head | hour | 620.0000 | 0.5144 | 2.7722e-03 | 0.7581 | 0.6974 |
| 42.0000 | gated_pre_return_asd_lora_moe_patchtst | hour | 620.0000 | 0.5135 | 2.7631e-03 | 0.7581 | 0.7013 |
| 42.0000 | scale_specific_gated_pre_asd_moe_patchtst | hour | 620.0000 | 0.5146 | 2.7712e-03 | 0.7581 | 0.7015 |
| 43.0000 | raw_joint | second | 4,096 | 0.9831 | 5.5234e-04 | 0.5310 | 0.1370 |
| 43.0000 | post_return_lora_moe_head | second | 4,096 | 0.9836 | 5.5422e-04 | 0.5276 | 0.1308 |
| 43.0000 | gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9869 | 5.5437e-04 | 0.5308 | 0.1356 |
| 43.0000 | scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9830 | 5.5425e-04 | 0.5322 | 0.1330 |
| 43.0000 | raw_joint | minute | 3,224 | 0.9936 | 7.0693e-04 | 0.5319 | 0.1093 |
| 43.0000 | post_return_lora_moe_head | minute | 3,224 | 1.0064 | 7.1570e-04 | 0.5039 | 0.0961 |
| 43.0000 | gated_pre_return_asd_lora_moe_patchtst | minute | 3,224 | 0.9920 | 7.0589e-04 | 0.5248 | 0.0968 |
| 43.0000 | scale_specific_gated_pre_asd_moe_patchtst | minute | 3,224 | 0.9918 | 7.0664e-04 | 0.5325 | 0.1147 |
| 43.0000 | raw_joint | hour | 620.0000 | 0.5216 | 2.7965e-03 | 0.7306 | 0.7003 |
| 43.0000 | post_return_lora_moe_head | hour | 620.0000 | 0.5091 | 2.7377e-03 | 0.7516 | 0.7019 |
| 43.0000 | gated_pre_return_asd_lora_moe_patchtst | hour | 620.0000 | 0.5082 | 2.7285e-03 | 0.7500 | 0.7023 |
| 43.0000 | scale_specific_gated_pre_asd_moe_patchtst | hour | 620.0000 | 0.5074 | 2.7228e-03 | 0.7532 | 0.7029 |
| 44.0000 | raw_joint | second | 4,096 | 0.9964 | 5.5709e-04 | 0.5234 | 0.0953 |
| 44.0000 | post_return_lora_moe_head | second | 4,096 | 0.9856 | 5.5284e-04 | 0.5327 | 0.1317 |
| 44.0000 | gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9849 | 5.5159e-04 | 0.5381 | 0.1233 |
| 44.0000 | scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9862 | 5.5177e-04 | 0.5398 | 0.1180 |
| 44.0000 | raw_joint | minute | 3,224 | 0.9907 | 7.0631e-04 | 0.5201 | 0.1012 |
| 44.0000 | post_return_lora_moe_head | minute | 3,224 | 0.9909 | 7.0661e-04 | 0.5235 | 0.1015 |
| 44.0000 | gated_pre_return_asd_lora_moe_patchtst | minute | 3,224 | 0.9909 | 7.0604e-04 | 0.5207 | 0.1012 |
| 44.0000 | scale_specific_gated_pre_asd_moe_patchtst | minute | 3,224 | 0.9944 | 7.0745e-04 | 0.5276 | 0.1013 |
| 44.0000 | raw_joint | hour | 620.0000 | 0.5264 | 2.7911e-03 | 0.7500 | 0.6966 |
| 44.0000 | post_return_lora_moe_head | hour | 620.0000 | 0.5021 | 2.7322e-03 | 0.7565 | 0.7068 |
| 44.0000 | gated_pre_return_asd_lora_moe_patchtst | hour | 620.0000 | 0.5029 | 2.7298e-03 | 0.7532 | 0.7088 |
| 44.0000 | scale_specific_gated_pre_asd_moe_patchtst | hour | 620.0000 | 0.5077 | 2.7550e-03 | 0.7435 | 0.7082 |

## Diagnostics Mean / Std

| model | scale | asd_gate_mean_mean | asd_gate_mean_std | asd_tau_mean_mean | asd_tau_mean_std | final_gate_mean_mean | final_gate_mean_std | final_mean_abs_delta_mean | final_mean_abs_delta_std | router_entropy_mean | router_entropy_std | expert_prob_0_mean | expert_prob_1_mean | expert_prob_2_mean | expert_prob_3_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_pre_return_asd_lora_moe_patchtst | hour | 0.0391 | 0.0503 | 9.8548 | 7.0763 | 0.1276 | 0.1153 | 6.2476e-03 | 0.0100 | 0.4795 | 0.0130 | 0.1768 | 0.2863 | 0.2749 | 0.2620 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 0.0404 | 0.0214 | 2.8381 | 1.9263 | 0.1342 | 0.0442 | 3.5900e-03 | 3.0597e-03 | 0.4831 | 0.0130 | 0.2686 | 0.2782 | 0.2572 | 0.1960 |
| gated_pre_return_asd_lora_moe_patchtst | second | 7.9432e-03 | 3.2354e-03 | 4.8726 | 1.9893 | 0.0825 | 9.3309e-03 | 3.3390e-04 | 2.6078e-04 | 0.4851 | 8.2490e-03 | 0.2645 | 0.3041 | 0.2921 | 0.1393 |
| post_return_lora_moe_head | hour | nan | nan | nan | nan | nan | nan | nan | nan | 0.4672 | 0.0199 | 0.2246 | 0.2953 | 0.2764 | 0.2038 |
| post_return_lora_moe_head | minute | nan | nan | nan | nan | nan | nan | nan | nan | 0.4523 | 0.0240 | 0.3026 | 0.4386 | 0.1097 | 0.1491 |
| post_return_lora_moe_head | second | nan | nan | nan | nan | nan | nan | nan | nan | 0.4193 | 0.0942 | 0.2625 | 0.0546 | 0.4834 | 0.1995 |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 0.0412 | 0.0656 | 6.4236 | 9.3194 | 0.1871 | 0.1802 | 0.0109 | 0.0183 | 0.4806 | 0.0159 | 0.3433 | 0.1687 | 0.3126 | 0.1753 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 0.0357 | 0.0270 | 2.8388 | 2.4495 | 0.0959 | 0.0313 | 2.1591e-03 | 1.7903e-03 | 0.4859 | 2.1178e-03 | 0.2804 | 0.2783 | 0.1663 | 0.2749 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 0.0296 | 0.0280 | 5.4199 | 2.0540 | 2.5415e-03 | 9.9735e-04 | 3.9665e-05 | 4.0507e-05 | 0.4888 | 4.6872e-03 | 0.0928 | 0.1661 | 0.3309 | 0.4102 |

## Files

- seed summaries: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_32stock_multiseed\seed_*\summary.csv`
- all summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_32stock_multiseed\summary_all.csv`
- aggregate: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_32stock_multiseed\aggregate.csv`
- diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_32stock_multiseed\diagnostics_all.csv`
- diagnostics aggregate: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_32stock_multiseed\diagnostics_aggregate.csv`
