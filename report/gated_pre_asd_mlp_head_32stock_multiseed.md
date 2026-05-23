# Gated Pre-ASD 32-Stock Multi-Seed Confirmation

This run checks whether `gated_pre_return_asd_lora_moe_patchtst` remains useful on the larger 32-stock cache across seeds. Training is balanced across `second/minute/hour`; no day data is included.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_32stocks_512t.npz`; seeds: `42, 43, 44`; patch preset: `short_second`; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096.

## Test Mean / Std

| model | scale | n_mean | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_pre_return_asd_lora_moe_mlp_head | hour | 620.0000 | 0.5178 | 9.1580e-03 | 2.8006e-03 | 3.5465e-05 | 0.7468 | 7.3913e-03 | 0.6973 | 7.6771e-03 |
| gated_pre_return_asd_lora_moe_mlp_head | minute | 3,224 | 0.9921 | 6.4099e-03 | 7.0812e-04 | 2.0819e-06 | 0.5160 | 3.3686e-03 | 0.1007 | 0.0544 |
| gated_pre_return_asd_lora_moe_mlp_head | second | 4,096 | 0.9891 | 6.6747e-03 | 5.5454e-04 | 1.7742e-06 | 0.5183 | 0.0170 | 0.1064 | 0.0340 |
| last_return | hour | 620.0000 | 3.0754 | 0.0000e+00 | 7.4308e-03 | 0.0000e+00 | 0.3097 | 0.0000e+00 | -0.5547 | 0.0000e+00 |
| last_return | minute | 3,224 | 2.2817 | 0.0000e+00 | 1.0833e-03 | 0.0000e+00 | 0.4696 | 0.0000e+00 | -0.0853 | 0.0000e+00 |
| last_return | second | 4,096 | 64.6097 | 0.0000e+00 | 2.7696e-03 | 0.0000e+00 | 0.4810 | 0.0000e+00 | -0.1016 | 0.0000e+00 |
| post_return_lora_moe_mlp_head | hour | 620.0000 | 0.5203 | 7.7486e-03 | 2.8080e-03 | 2.7822e-05 | 0.7478 | 4.6561e-03 | 0.6928 | 5.4781e-03 |
| post_return_lora_moe_mlp_head | minute | 3,224 | 0.9922 | 5.9020e-03 | 7.0811e-04 | 2.9354e-06 | 0.5114 | 0.0128 | 0.1220 | 0.0245 |
| post_return_lora_moe_mlp_head | second | 4,096 | 0.9917 | 7.2560e-03 | 5.5473e-04 | 2.2354e-06 | 0.5237 | 0.0183 | 0.0861 | 0.0475 |
| raw_joint | hour | 620.0000 | 0.5240 | 2.4070e-03 | 2.7933e-03 | 2.7938e-06 | 0.7419 | 0.0101 | 0.6962 | 4.2323e-03 |
| raw_joint | minute | 3,224 | 0.9927 | 1.7039e-03 | 7.0656e-04 | 3.2744e-07 | 0.5261 | 5.9199e-03 | 0.1006 | 9.0011e-03 |
| raw_joint | second | 4,096 | 0.9884 | 7.0890e-03 | 5.5509e-04 | 2.4634e-06 | 0.5234 | 7.5999e-03 | 0.1184 | 0.0212 |
| zero | hour | 620.0000 | 1.0000 | 0.0000e+00 | 4.1690e-03 | 0.0000e+00 | 0.5048 | 0.0000e+00 | nan | nan |
| zero | minute | 3,224 | 1.0009 | 0.0000e+00 | 7.0896e-04 | 0.0000e+00 | 0.5067 | 0.0000e+00 | nan | nan |
| zero | second | 4,096 | 1.0000 | 0.0000e+00 | 5.5585e-04 | 0.0000e+00 | 0.4869 | 0.0000e+00 | nan | nan |

## Seed-Level Test Rows

| seed | model | scale | n | nmse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 42.0000 | raw_joint | second | 4,096 | 0.9855 | 5.5585e-04 | 0.5158 | 0.1228 |
| 42.0000 | post_return_lora_moe_mlp_head | second | 4,096 | 0.9883 | 5.5674e-04 | 0.5026 | 0.1106 |
| 42.0000 | gated_pre_return_asd_lora_moe_mlp_head | second | 4,096 | 0.9833 | 5.5571e-04 | 0.5153 | 0.1366 |
| 42.0000 | raw_joint | minute | 3,224 | 0.9937 | 7.0643e-04 | 0.5263 | 0.0913 |
| 42.0000 | post_return_lora_moe_mlp_head | minute | 3,224 | 0.9952 | 7.1068e-04 | 0.5026 | 0.1091 |
| 42.0000 | gated_pre_return_asd_lora_moe_mlp_head | minute | 3,224 | 0.9973 | 7.1035e-04 | 0.5123 | 0.0596 |
| 42.0000 | raw_joint | hour | 620.0000 | 0.5239 | 2.7924e-03 | 0.7452 | 0.6918 |
| 42.0000 | post_return_lora_moe_mlp_head | hour | 620.0000 | 0.5292 | 2.8387e-03 | 0.7452 | 0.6865 |
| 42.0000 | gated_pre_return_asd_lora_moe_mlp_head | hour | 620.0000 | 0.5178 | 2.7986e-03 | 0.7484 | 0.6958 |
| 43.0000 | raw_joint | second | 4,096 | 0.9831 | 5.5234e-04 | 0.5310 | 0.1370 |
| 43.0000 | post_return_lora_moe_mlp_head | second | 4,096 | 0.9867 | 5.5232e-04 | 0.5347 | 0.1164 |
| 43.0000 | gated_pre_return_asd_lora_moe_mlp_head | second | 4,096 | 0.9875 | 5.5250e-04 | 0.5367 | 0.1130 |
| 43.0000 | raw_joint | minute | 3,224 | 0.9936 | 7.0693e-04 | 0.5319 | 0.1093 |
| 43.0000 | post_return_lora_moe_mlp_head | minute | 3,224 | 0.9959 | 7.0874e-04 | 0.5055 | 0.1068 |
| 43.0000 | gated_pre_return_asd_lora_moe_mlp_head | minute | 3,224 | 0.9941 | 7.0778e-04 | 0.5188 | 0.0801 |
| 43.0000 | raw_joint | hour | 620.0000 | 0.5216 | 2.7965e-03 | 0.7306 | 0.7003 |
| 43.0000 | post_return_lora_moe_mlp_head | hour | 620.0000 | 0.5169 | 2.7846e-03 | 0.7532 | 0.6953 |
| 43.0000 | gated_pre_return_asd_lora_moe_mlp_head | hour | 620.0000 | 0.5269 | 2.8370e-03 | 0.7387 | 0.6904 |
| 44.0000 | raw_joint | second | 4,096 | 0.9964 | 5.5709e-04 | 0.5234 | 0.0953 |
| 44.0000 | post_return_lora_moe_mlp_head | second | 4,096 | 1.0000 | 5.5514e-04 | 0.5340 | 0.0313 |
| 44.0000 | gated_pre_return_asd_lora_moe_mlp_head | second | 4,096 | 0.9964 | 5.5542e-04 | 0.5031 | 0.0695 |
| 44.0000 | raw_joint | minute | 3,224 | 0.9907 | 7.0631e-04 | 0.5201 | 0.1012 |
| 44.0000 | post_return_lora_moe_mlp_head | minute | 3,224 | 0.9854 | 7.0491e-04 | 0.5260 | 0.1502 |
| 44.0000 | gated_pre_return_asd_lora_moe_mlp_head | minute | 3,224 | 0.9850 | 7.0623e-04 | 0.5170 | 0.1623 |
| 44.0000 | raw_joint | hour | 620.0000 | 0.5264 | 2.7911e-03 | 0.7500 | 0.6966 |
| 44.0000 | post_return_lora_moe_mlp_head | hour | 620.0000 | 0.5149 | 2.8005e-03 | 0.7452 | 0.6966 |
| 44.0000 | gated_pre_return_asd_lora_moe_mlp_head | hour | 620.0000 | 0.5086 | 2.7661e-03 | 0.7532 | 0.7056 |

## Diagnostics Mean / Std

| model | scale | asd_gate_mean_mean | asd_gate_mean_std | asd_tau_mean_mean | asd_tau_mean_std | final_gate_mean_mean | final_gate_mean_std | final_mean_abs_delta_mean | final_mean_abs_delta_std | router_entropy_mean | router_entropy_std | expert_prob_0_mean | expert_prob_1_mean | expert_prob_2_mean | expert_prob_3_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gated_pre_return_asd_lora_moe_mlp_head | hour | 6.8776e-03 | 2.4949e-03 | 0.9215 | 0.4387 | 0.0338 | 5.3810e-03 | 7.0875e-05 | 5.5067e-05 | 0.4875 | 0.0100 | 0.1739 | 0.2861 | 0.2584 | 0.2816 |
| gated_pre_return_asd_lora_moe_mlp_head | minute | 0.0139 | 8.6435e-03 | 1.4526 | 0.5915 | 0.0601 | 0.0192 | 1.8722e-04 | 1.2931e-04 | 0.4836 | 0.0186 | 0.1764 | 0.2747 | 0.2805 | 0.2683 |
| gated_pre_return_asd_lora_moe_mlp_head | second | 4.5336e-03 | 1.1702e-03 | 1.9399 | 1.0850 | 0.0974 | 0.0194 | 1.2055e-03 | 1.2257e-03 | 0.4909 | 7.3759e-03 | 0.3940 | 0.3042 | 0.1900 | 0.1118 |
| post_return_lora_moe_mlp_head | hour | nan | nan | nan | nan | nan | nan | nan | nan | 0.4569 | 9.1000e-03 | 0.1957 | 0.0130 | 0.5094 | 0.2819 |
| post_return_lora_moe_mlp_head | minute | nan | nan | nan | nan | nan | nan | nan | nan | 0.4899 | 7.1265e-03 | 0.1670 | 0.3643 | 0.3114 | 0.1573 |
| post_return_lora_moe_mlp_head | second | nan | nan | nan | nan | nan | nan | nan | nan | 0.4802 | 3.3375e-03 | 0.2984 | 0.0763 | 0.2031 | 0.4222 |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |

## Files

- seed summaries: `outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_mlp_head_32stock_multiseed\seed_*\summary.csv`
- all summary: `outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_mlp_head_32stock_multiseed\summary_all.csv`
- aggregate: `outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_mlp_head_32stock_multiseed\aggregate.csv`
- diagnostics: `outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_mlp_head_32stock_multiseed\diagnostics_all.csv`
- diagnostics aggregate: `outputs\prepatch_asd_adapter_patchtst\gated_pre_asd_mlp_head_32stock_multiseed\diagnostics_aggregate.csv`
