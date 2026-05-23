# Direct Price Input PatchTST Experiment

This experiment feeds `raw_price` or `log_price` windows directly into PatchTST variants and keeps the target as future return. Inputs are normalized using train-split level statistics; targets use return statistics. No day data is included.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_32stocks_512t.npz`; price modes: `log_price, raw_price`; seeds: `42, 43, 44`; patch preset: `short_second`; epochs=5; steps/epoch=50; train cap=20000.

## Test Mean / Std

| price_mode | model | scale | n_mean | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4866 | 1.9350e-03 | 2.6483e-03 | 4.3388e-06 | 0.7435 | 8.0645e-03 | 0.7175 | 9.3709e-04 |
| log_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 0.9982 | 4.1425e-03 | 7.0975e-04 | 1.6583e-06 | 0.5086 | 0.0119 | 0.1007 | 0.0241 |
| log_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9962 | 3.0106e-03 | 5.5538e-04 | 8.9104e-07 | 0.5173 | 3.6114e-03 | 0.0755 | 0.0204 |
| log_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4877 | 3.3293e-03 | 2.6567e-03 | 8.8352e-06 | 0.7505 | 3.7248e-03 | 0.7171 | 9.9603e-04 |
| log_price | direct_price_lora_moe_head | minute | 3,224 | 1.0028 | 8.5267e-03 | 7.1317e-04 | 7.0011e-06 | 0.5049 | 8.1290e-03 | 0.1136 | 8.0661e-03 |
| log_price | direct_price_lora_moe_head | second | 4,096 | 0.9978 | 9.8135e-04 | 5.5567e-04 | 9.1295e-07 | 0.5154 | 6.3568e-03 | 0.0604 | 0.0234 |
| log_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4897 | 2.0372e-03 | 2.6681e-03 | 7.4967e-06 | 0.7532 | 4.2673e-03 | 0.7175 | 2.2596e-03 |
| log_price | direct_price_raw_patchtst | minute | 3,224 | 0.9943 | 1.8114e-03 | 7.0792e-04 | 4.0263e-07 | 0.5176 | 5.4300e-03 | 0.1153 | 0.0144 |
| log_price | direct_price_raw_patchtst | second | 4,096 | 0.9991 | 2.6320e-03 | 5.5721e-04 | 2.3970e-06 | 0.5061 | 0.0184 | 0.0456 | 0.0355 |
| log_price | last_return | hour | 620.0000 | 3.0754 | 0.0000e+00 | 7.4308e-03 | 0.0000e+00 | 0.3097 | 0.0000e+00 | -0.5547 | 0.0000e+00 |
| log_price | last_return | minute | 3,224 | 2.2817 | 0.0000e+00 | 1.0833e-03 | 0.0000e+00 | 0.4696 | 0.0000e+00 | -0.0853 | 0.0000e+00 |
| log_price | last_return | second | 4,096 | 64.6093 | 0.0000e+00 | 2.7695e-03 | 0.0000e+00 | 0.4810 | 0.0000e+00 | -0.1016 | 0.0000e+00 |
| log_price | zero | hour | 620.0000 | 1.0000 | 0.0000e+00 | 4.1690e-03 | 0.0000e+00 | 0.5048 | 0.0000e+00 | nan | nan |
| log_price | zero | minute | 3,224 | 1.0009 | 0.0000e+00 | 7.0896e-04 | 0.0000e+00 | 0.5067 | 0.0000e+00 | nan | nan |
| log_price | zero | second | 4,096 | 1.0000 | 0.0000e+00 | 5.5585e-04 | 0.0000e+00 | 0.4869 | 0.0000e+00 | nan | nan |
| raw_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4863 | 2.0276e-03 | 2.6475e-03 | 4.7299e-06 | 0.7435 | 7.3913e-03 | 0.7176 | 1.1449e-03 |
| raw_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 0.9982 | 4.1688e-03 | 7.0978e-04 | 1.6664e-06 | 0.5087 | 0.0115 | 0.1003 | 0.0244 |
| raw_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9962 | 3.0268e-03 | 5.5538e-04 | 8.9142e-07 | 0.5173 | 4.5404e-03 | 0.0757 | 0.0209 |
| raw_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4867 | 5.0740e-03 | 2.6504e-03 | 1.6699e-05 | 0.7489 | 4.0591e-03 | 0.7180 | 1.9880e-03 |
| raw_price | direct_price_lora_moe_head | minute | 3,224 | 0.9976 | 1.8935e-03 | 7.0904e-04 | 1.1574e-06 | 0.5064 | 3.1143e-03 | 0.1009 | 0.0230 |
| raw_price | direct_price_lora_moe_head | second | 4,096 | 1.0002 | 4.1133e-03 | 5.5709e-04 | 2.1136e-06 | 0.5146 | 2.1372e-03 | 0.0380 | 0.0193 |
| raw_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4896 | 2.0815e-03 | 2.6674e-03 | 6.1732e-06 | 0.7527 | 4.9275e-03 | 0.7176 | 2.2458e-03 |
| raw_price | direct_price_raw_patchtst | minute | 3,224 | 0.9943 | 1.7878e-03 | 7.0792e-04 | 4.0890e-07 | 0.5173 | 4.8547e-03 | 0.1150 | 0.0146 |
| raw_price | direct_price_raw_patchtst | second | 4,096 | 0.9992 | 2.6507e-03 | 5.5723e-04 | 2.4189e-06 | 0.5054 | 0.0175 | 0.0458 | 0.0361 |
| raw_price | last_return | hour | 620.0000 | 3.0754 | 0.0000e+00 | 7.4308e-03 | 0.0000e+00 | 0.3097 | 0.0000e+00 | -0.5547 | 0.0000e+00 |
| raw_price | last_return | minute | 3,224 | 2.2817 | 0.0000e+00 | 1.0833e-03 | 0.0000e+00 | 0.4696 | 0.0000e+00 | -0.0853 | 0.0000e+00 |
| raw_price | last_return | second | 4,096 | 64.6093 | 0.0000e+00 | 2.7695e-03 | 0.0000e+00 | 0.4810 | 0.0000e+00 | -0.1016 | 0.0000e+00 |
| raw_price | zero | hour | 620.0000 | 1.0000 | 0.0000e+00 | 4.1690e-03 | 0.0000e+00 | 0.5048 | 0.0000e+00 | nan | nan |
| raw_price | zero | minute | 3,224 | 1.0009 | 0.0000e+00 | 7.0896e-04 | 0.0000e+00 | 0.5067 | 0.0000e+00 | nan | nan |
| raw_price | zero | second | 4,096 | 1.0000 | 0.0000e+00 | 5.5585e-04 | 0.0000e+00 | 0.4869 | 0.0000e+00 | nan | nan |

## Seed-Level Test Rows

| seed | price_mode | model | scale | n | nmse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 42.0000 | log_price | direct_price_raw_patchtst | second | 4,096 | 1.0012 | 5.5949e-04 | 0.4879 | 0.0200 |
| 42.0000 | log_price | direct_price_lora_moe_head | second | 4,096 | 0.9977 | 5.5516e-04 | 0.5227 | 0.0858 |
| 42.0000 | log_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9994 | 5.5636e-04 | 0.5165 | 0.0539 |
| 42.0000 | log_price | direct_price_raw_patchtst | minute | 3,224 | 0.9945 | 7.0814e-04 | 0.5238 | 0.1001 |
| 42.0000 | log_price | direct_price_lora_moe_head | minute | 3,224 | 1.0124 | 7.2116e-04 | 0.4964 | 0.1043 |
| 42.0000 | log_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 1.0012 | 7.1076e-04 | 0.5030 | 0.0730 |
| 42.0000 | log_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4876 | 2.6717e-03 | 0.7581 | 0.7176 |
| 42.0000 | log_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4866 | 2.6554e-03 | 0.7484 | 0.7171 |
| 42.0000 | log_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4852 | 2.6503e-03 | 0.7516 | 0.7177 |
| 42.0000 | raw_price | direct_price_raw_patchtst | second | 4,096 | 1.0013 | 5.5954e-04 | 0.4879 | 0.0198 |
| 42.0000 | raw_price | direct_price_lora_moe_head | second | 4,096 | 1.0048 | 5.5937e-04 | 0.5121 | 0.0180 |
| 42.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9994 | 5.5636e-04 | 0.5170 | 0.0535 |
| 42.0000 | raw_price | direct_price_raw_patchtst | minute | 3,224 | 0.9945 | 7.0817e-04 | 0.5229 | 0.0996 |
| 42.0000 | raw_price | direct_price_lora_moe_head | minute | 3,224 | 0.9969 | 7.0888e-04 | 0.5095 | 0.0744 |
| 42.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 1.0013 | 7.1080e-04 | 0.5026 | 0.0724 |
| 42.0000 | raw_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4876 | 2.6715e-03 | 0.7581 | 0.7176 |
| 42.0000 | raw_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4826 | 2.6360e-03 | 0.7532 | 0.7199 |
| 42.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4852 | 2.6502e-03 | 0.7516 | 0.7178 |
| 43.0000 | log_price | direct_price_raw_patchtst | second | 4,096 | 1.0000 | 5.5743e-04 | 0.5058 | 0.0306 |
| 43.0000 | log_price | direct_price_lora_moe_head | second | 4,096 | 0.9989 | 5.5672e-04 | 0.5109 | 0.0395 |
| 43.0000 | log_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9934 | 5.5462e-04 | 0.5212 | 0.0945 |
| 43.0000 | log_price | direct_price_raw_patchtst | minute | 3,224 | 0.9961 | 7.0816e-04 | 0.5151 | 0.1288 |
| 43.0000 | log_price | direct_price_lora_moe_head | minute | 3,224 | 0.9996 | 7.1018e-04 | 0.5058 | 0.1176 |
| 43.0000 | log_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 0.9934 | 7.0784e-04 | 0.5223 | 0.1176 |
| 43.0000 | log_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4900 | 2.6731e-03 | 0.7500 | 0.7198 |
| 43.0000 | log_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4851 | 2.6487e-03 | 0.7548 | 0.7180 |
| 43.0000 | log_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4857 | 2.6433e-03 | 0.7355 | 0.7183 |
| 43.0000 | raw_price | direct_price_raw_patchtst | second | 4,096 | 1.0000 | 5.5744e-04 | 0.5053 | 0.0306 |
| 43.0000 | raw_price | direct_price_lora_moe_head | second | 4,096 | 0.9990 | 5.5671e-04 | 0.5161 | 0.0396 |
| 43.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9934 | 5.5462e-04 | 0.5219 | 0.0950 |
| 43.0000 | raw_price | direct_price_raw_patchtst | minute | 3,224 | 0.9960 | 7.0815e-04 | 0.5145 | 0.1285 |
| 43.0000 | raw_price | direct_price_lora_moe_head | minute | 3,224 | 0.9997 | 7.1026e-04 | 0.5033 | 0.1161 |
| 43.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 0.9934 | 7.0785e-04 | 0.5220 | 0.1176 |
| 43.0000 | raw_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4895 | 2.6703e-03 | 0.7484 | 0.7198 |
| 43.0000 | raw_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4851 | 2.6466e-03 | 0.7484 | 0.7180 |
| 43.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4851 | 2.6420e-03 | 0.7371 | 0.7187 |
| 44.0000 | log_price | direct_price_raw_patchtst | second | 4,096 | 0.9962 | 5.5471e-04 | 0.5246 | 0.0861 |
| 44.0000 | log_price | direct_price_lora_moe_head | second | 4,096 | 0.9969 | 5.5512e-04 | 0.5126 | 0.0558 |
| 44.0000 | log_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9958 | 5.5516e-04 | 0.5141 | 0.0782 |
| 44.0000 | log_price | direct_price_raw_patchtst | minute | 3,224 | 0.9924 | 7.0746e-04 | 0.5139 | 0.1171 |
| 44.0000 | log_price | direct_price_lora_moe_head | minute | 3,224 | 0.9963 | 7.0815e-04 | 0.5126 | 0.1189 |
| 44.0000 | log_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 0.9998 | 7.1065e-04 | 0.5005 | 0.1113 |
| 44.0000 | log_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4916 | 2.6594e-03 | 0.7516 | 0.7152 |
| 44.0000 | log_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4915 | 2.6662e-03 | 0.7484 | 0.7160 |
| 44.0000 | log_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4888 | 2.6513e-03 | 0.7435 | 0.7165 |
| 44.0000 | raw_price | direct_price_raw_patchtst | second | 4,096 | 0.9962 | 5.5471e-04 | 0.5229 | 0.0870 |
| 44.0000 | raw_price | direct_price_lora_moe_head | second | 4,096 | 0.9969 | 5.5519e-04 | 0.5156 | 0.0565 |
| 44.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | second | 4,096 | 0.9958 | 5.5516e-04 | 0.5129 | 0.0787 |
| 44.0000 | raw_price | direct_price_raw_patchtst | minute | 3,224 | 0.9925 | 7.0745e-04 | 0.5145 | 0.1169 |
| 44.0000 | raw_price | direct_price_lora_moe_head | minute | 3,224 | 0.9961 | 7.0796e-04 | 0.5064 | 0.1122 |
| 44.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | minute | 3,224 | 0.9998 | 7.1068e-04 | 0.5014 | 0.1110 |
| 44.0000 | raw_price | direct_price_raw_patchtst | hour | 620.0000 | 0.4918 | 2.6603e-03 | 0.7516 | 0.7153 |
| 44.0000 | raw_price | direct_price_lora_moe_head | hour | 620.0000 | 0.4924 | 2.6687e-03 | 0.7452 | 0.7160 |
| 44.0000 | raw_price | direct_price_gated_pre_asd_lora_moe | hour | 620.0000 | 0.4887 | 2.6502e-03 | 0.7419 | 0.7164 |

## Diagnostics Mean / Std

| price_mode | model | scale | asd_gate_mean_mean | final_gate_mean_mean | final_mean_abs_delta_mean | router_entropy_mean | expert_prob_0_mean | expert_prob_1_mean | expert_prob_2_mean | expert_prob_3_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| log_price | direct_price_gated_pre_asd_lora_moe | hour | 0.1636 | 0.4895 | 0.0589 | 0.4720 | 0.2813 | 0.3838 | 0.1134 | 0.2215 |
| log_price | direct_price_gated_pre_asd_lora_moe | minute | 0.0386 | 0.1339 | 1.8096e-03 | 0.4719 | 0.1906 | 0.3551 | 0.2510 | 0.2032 |
| log_price | direct_price_gated_pre_asd_lora_moe | second | 0.0163 | 0.1847 | 9.7693e-03 | 0.4546 | 0.2583 | 0.2609 | 0.1631 | 0.3177 |
| log_price | direct_price_lora_moe_head | hour | nan | nan | nan | 0.4831 | 0.0416 | 0.2019 | 0.4493 | 0.3072 |
| log_price | direct_price_lora_moe_head | minute | nan | nan | nan | 0.4656 | 1.6248e-05 | 0.3593 | 0.3481 | 0.2926 |
| log_price | direct_price_lora_moe_head | second | nan | nan | nan | 0.4323 | 0.1785 | 0.2068 | 0.2783 | 0.3363 |
| log_price | direct_price_raw_patchtst | hour | nan | nan | nan | nan | nan | nan | nan | nan |
| log_price | direct_price_raw_patchtst | minute | nan | nan | nan | nan | nan | nan | nan | nan |
| log_price | direct_price_raw_patchtst | second | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_price | direct_price_gated_pre_asd_lora_moe | hour | 0.1638 | 0.4943 | 0.0585 | 0.4742 | 0.2702 | 0.3842 | 0.1142 | 0.2314 |
| raw_price | direct_price_gated_pre_asd_lora_moe | minute | 0.0388 | 0.1326 | 1.7987e-03 | 0.4630 | 0.1604 | 0.2408 | 0.3280 | 0.2708 |
| raw_price | direct_price_gated_pre_asd_lora_moe | second | 0.0163 | 0.1781 | 8.3015e-03 | 0.4573 | 0.2293 | 0.1997 | 0.2281 | 0.3429 |
| raw_price | direct_price_lora_moe_head | hour | nan | nan | nan | 0.4732 | 0.1485 | 0.1761 | 0.2815 | 0.3940 |
| raw_price | direct_price_lora_moe_head | minute | nan | nan | nan | 0.4758 | 1.3439e-05 | 0.4200 | 0.2170 | 0.3629 |
| raw_price | direct_price_lora_moe_head | second | nan | nan | nan | 0.4909 | 0.3455 | 0.2032 | 0.2980 | 0.1533 |
| raw_price | direct_price_raw_patchtst | hour | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_price | direct_price_raw_patchtst | minute | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_price | direct_price_raw_patchtst | second | nan | nan | nan | nan | nan | nan | nan | nan |

## Files

- all summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\direct_price_input_patchtst_32stock_multiseed\summary_all.csv`
- aggregate: `E:\Working Area\Comp5329_Assignment2_2026\outputs\direct_price_input_patchtst_32stock_multiseed\aggregate.csv`
- diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\direct_price_input_patchtst_32stock_multiseed\diagnostics_all.csv`
- diagnostics aggregate: `E:\Working Area\Comp5329_Assignment2_2026\outputs\direct_price_input_patchtst_32stock_multiseed\diagnostics_aggregate.csv`
