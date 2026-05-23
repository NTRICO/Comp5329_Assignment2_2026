# Pre-PatchTST ASD Adapter Experiment

This small experiment tests whether a learnable front-end can clean/adapt price or return sequences before the frozen PatchTST backbone sees returns. The front-end is ASD plus either LoRA-MoE or MLP-MoE.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_hf_second_feature_cache_11stocks_512t.npz`; patch preset: `short_second`; epochs=3; balanced steps/epoch=12; rank=8; init_gate=-4.0.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 1,024 | 1.0023 | 9.6510e-07 | 6.1415e-04 | 0.4568 | nan |
| raw_joint | second | 1,024 | 1.0177 | 9.7993e-07 | 6.1876e-04 | 0.5029 | -0.0334 |
| post_return_lora_moe_head | second | 1,024 | 1.0240 | 9.8598e-07 | 6.2168e-04 | 0.5206 | -0.0316 |
| pre_return_asd_lora_moe_patchtst | second | 1,024 | 1.0368 | 9.9830e-07 | 6.2903e-04 | 0.4754 | -0.0397 |
| pre_return_asd_mlp_moe_patchtst | second | 1,024 | 1.0428 | 1.0041e-06 | 6.3820e-04 | 0.4705 | -0.0208 |
| pre_log_price_asd_lora_moe_to_return_patchtst | second | 1,024 | 1.0250 | 9.8694e-07 | 6.2423e-04 | 0.4676 | -0.0538 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | second | 1,024 | 1.0470 | 1.0081e-06 | 6.4083e-04 | 0.4666 | -0.0303 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | second | 1,024 | 1.0557 | 1.0165e-06 | 6.4502e-04 | 0.4558 | -0.0681 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | second | 1,024 | 1.0272 | 9.8905e-07 | 6.2216e-04 | 0.5157 | -0.0521 |
| zero | minute | 1,024 | 1.0030 | 1.5852e-06 | 7.9712e-04 | 0.4936 | nan |
| raw_joint | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| post_return_lora_moe_head | minute | 1,024 | 1.0064 | 1.5905e-06 | 7.9792e-04 | 0.5230 | 0.0225 |
| pre_return_asd_lora_moe_patchtst | minute | 1,024 | 1.0071 | 1.5918e-06 | 7.9790e-04 | 0.5288 | 0.0461 |
| pre_return_asd_mlp_moe_patchtst | minute | 1,024 | 1.0042 | 1.5871e-06 | 7.9806e-04 | 0.5044 | 0.0241 |
| pre_log_price_asd_lora_moe_to_return_patchtst | minute | 1,024 | 1.0137 | 1.6021e-06 | 8.0117e-04 | 0.5112 | 0.0517 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | minute | 1,024 | 1.0001 | 1.5806e-06 | 7.9564e-04 | 0.5132 | 0.0595 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | minute | 1,024 | 1.0055 | 1.5891e-06 | 7.9668e-04 | 0.5230 | 0.0336 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | minute | 1,024 | 1.0008 | 1.5817e-06 | 7.9574e-04 | 0.5230 | 0.0525 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| raw_joint | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| post_return_lora_moe_head | hour | 200.0000 | 0.8550 | 4.0062e-05 | 4.1683e-03 | 0.6950 | 0.5288 |
| pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.7580 | 3.5519e-05 | 3.8397e-03 | 0.6950 | 0.5403 |
| pre_return_asd_mlp_moe_patchtst | hour | 200.0000 | 0.8682 | 4.0683e-05 | 4.2109e-03 | 0.6750 | 0.5279 |
| pre_log_price_asd_lora_moe_to_return_patchtst | hour | 200.0000 | 0.8665 | 4.0599e-05 | 4.1951e-03 | 0.6950 | 0.5465 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | hour | 200.0000 | 0.8683 | 4.0684e-05 | 4.1963e-03 | 0.7150 | 0.5322 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | hour | 200.0000 | 0.8685 | 4.0695e-05 | 4.2118e-03 | 0.6550 | 0.5384 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | hour | 200.0000 | 0.8609 | 4.0339e-05 | 4.2012e-03 | 0.6200 | 0.5416 |

## Test NMSE Relative To Raw PatchTST

| model | scale | nmse | nmse_vs_raw_pct |
| --- | --- | --- | --- |
| post_return_lora_moe_head | second | 1.0240 | 0.6172 |
| pre_return_asd_lora_moe_patchtst | second | 1.0368 | 1.8743 |
| pre_return_asd_mlp_moe_patchtst | second | 1.0428 | 2.4675 |
| pre_log_price_asd_lora_moe_to_return_patchtst | second | 1.0250 | 0.7151 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | second | 1.0470 | 2.8791 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | second | 1.0557 | 3.7283 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | second | 1.0272 | 0.9299 |
| post_return_lora_moe_head | minute | 1.0064 | -1.0164 |
| pre_return_asd_lora_moe_patchtst | minute | 1.0071 | -0.9412 |
| pre_return_asd_mlp_moe_patchtst | minute | 1.0042 | -1.2281 |
| pre_log_price_asd_lora_moe_to_return_patchtst | minute | 1.0137 | -0.2978 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | minute | 1.0001 | -1.6346 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | minute | 1.0055 | -1.1044 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | minute | 1.0008 | -1.5689 |
| post_return_lora_moe_head | hour | 0.8550 | -8.5226 |
| pre_return_asd_lora_moe_patchtst | hour | 0.7580 | -18.8963 |
| pre_return_asd_mlp_moe_patchtst | hour | 0.8682 | -7.1058 |
| pre_log_price_asd_lora_moe_to_return_patchtst | hour | 0.8665 | -7.2977 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | hour | 0.8683 | -7.1023 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | hour | 0.8685 | -7.0783 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | hour | 0.8609 | -7.8902 |

## Front-End Diagnostics

| model | scale | asd_gate_mean | asd_tau_mean | asd_mean_abs_delta | pre_adapter_gate_mean | pre_adapter_mean_abs_delta | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | patch_input_abs_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| post_return_lora_moe_head | second | nan | nan | nan | nan | nan | 0.4597 | 0.0634 | 4.9371e-03 | 0.6552 | 0.0794 | 0.2604 | nan |
| post_return_lora_moe_head | minute | nan | nan | nan | nan | nan | 0.4755 | 0.0675 | 2.3460e-04 | 3.0417e-04 | 0.3986 | 0.6009 | nan |
| post_return_lora_moe_head | hour | nan | nan | nan | nan | nan | 0.0345 | 0.1833 | 2.4073e-03 | 4.1632e-06 | 6.0247e-03 | 0.9916 | nan |
| pre_return_asd_lora_moe_patchtst | second | 0.0117 | 4.8902 | 3.4620e-03 | 0.0215 | 0.0983 | 0.4844 | 0.0583 | 0.0000e+00 | 0.0221 | 0.5372 | 0.4407 | 0.3409 |
| pre_return_asd_lora_moe_patchtst | minute | 0.0113 | 0.4885 | 1.5307e-03 | 0.0645 | 0.3679 | 0.4068 | 0.0589 | 0.0000e+00 | 0.4214 | 0.5545 | 0.0241 | 0.7508 |
| pre_return_asd_lora_moe_patchtst | hour | 7.0142e-04 | 0.3709 | 3.4392e-05 | 0.3235 | 0.6655 | 0.3668 | 0.0666 | 0.0000e+00 | 0.4066 | 0.5925 | 8.1517e-04 | 1.9712 |
| pre_return_asd_mlp_moe_patchtst | second | 8.0153e-03 | 2.0323 | 1.2406e-03 | 0.0109 | 0.0118 | 0.4916 | 0.0443 | 0.4494 | 0.0746 | 0.4687 | 7.3779e-03 | 0.3310 |
| pre_return_asd_mlp_moe_patchtst | minute | 2.0224e-03 | 1.6586 | 8.1825e-04 | 0.0162 | 0.0367 | 0.4671 | 0.0567 | 3.7476e-04 | 0.4596 | 0.5147 | 0.0253 | 0.6149 |
| pre_return_asd_mlp_moe_patchtst | hour | 7.8846e-04 | 0.7723 | 7.8979e-05 | 0.0517 | 0.0865 | 0.4272 | 0.0750 | 0.0000e+00 | 0.0105 | 0.3209 | 0.6686 | 0.7323 |
| pre_log_price_asd_lora_moe_to_return_patchtst | second | 0.0149 | 1.2862e-03 | 1.3257e-06 | 0.0136 | 3.4272e-03 | 0.4983 | 0.0631 | 0.0000e+00 | 0.4652 | 0.5348 | 0.0000e+00 | 0.3241 |
| pre_log_price_asd_lora_moe_to_return_patchtst | minute | 0.0116 | 2.8138e-03 | 7.5389e-06 | 0.0141 | 5.4657e-03 | 0.4591 | 0.0764 | 0.0000e+00 | 0.6668 | 0.0000e+00 | 0.3332 | 0.5651 |
| pre_log_price_asd_lora_moe_to_return_patchtst | hour | 0.0118 | 3.6559e-03 | 5.2261e-06 | 1.2109e-03 | 0.0139 | 0.4985 | 0.0630 | 0.0000e+00 | 0.5328 | 0.0000e+00 | 0.4672 | 0.6917 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | second | 4.5047e-03 | 1.3890e-03 | 4.2371e-07 | 0.0100 | 3.5530e-03 | 0.4676 | 0.0736 | 0.0000e+00 | 0.3512 | 0.6488 | 0.0000e+00 | 0.3255 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | minute | 9.5801e-03 | 3.0163e-03 | 6.5080e-06 | 0.0513 | 8.0439e-03 | 0.4610 | 0.0758 | 0.0000e+00 | 0.0000e+00 | 0.3370 | 0.6630 | 0.5659 |
| pre_log_price_asd_mlp_moe_to_return_patchtst | hour | 9.4297e-04 | 4.0027e-03 | 4.5650e-07 | 0.1189 | 5.1179e-03 | 0.3231 | 0.1186 | 0.0000e+00 | 0.0000e+00 | 0.1651 | 0.8349 | 0.6928 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | second | 9.2369e-03 | 2.5070 | 3.6182e-04 | 8.5540e-03 | 3.9487e-03 | 0.3928 | 0.0978 | 0.2344 | 0.7656 | 0.0000e+00 | 0.0000e+00 | 0.3233 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | minute | 5.4403e-03 | 0.4996 | 3.3978e-04 | 0.0331 | 5.1535e-03 | 0.4827 | 0.0684 | 0.6090 | 0.3910 | 0.0000e+00 | 0.0000e+00 | 0.5664 |
| pre_raw_price_asd_lora_moe_to_return_patchtst | hour | 5.2607e-04 | 0.4193 | 7.0470e-06 | 0.0208 | 0.0146 | 0.3727 | 0.1039 | 0.2121 | 0.0000e+00 | 0.0000e+00 | 0.7879 | 0.6918 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | second | 6.9205e-03 | 1.6629 | 1.7981e-04 | 0.0149 | 5.7921e-03 | 0.4660 | 0.0741 | 0.0000e+00 | 0.6524 | 0.0000e+00 | 0.3476 | 0.3244 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | minute | 4.0876e-03 | 2.3874 | 1.2199e-03 | 0.0451 | 0.0203 | 0.3709 | 0.1045 | 0.7899 | 0.2101 | 0.0000e+00 | 0.0000e+00 | 0.5704 |
| pre_raw_price_asd_mlp_moe_to_return_patchtst | hour | 1.7919e-03 | 5.9907 | 3.3547e-04 | 0.0322 | 0.0187 | 0.4872 | 0.0669 | 0.4060 | 0.5940 | 0.0000e+00 | 0.0000e+00 | 0.6942 |

## Files

- summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\summary.csv`
- diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\prepatch_asd_adapter_patchtst\diagnostics.csv`
