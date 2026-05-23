# Pre-PatchTST Targeted More-Data Run

This run compares targeted architectures with larger sample caps and more balanced training steps. `mid_token_asd_lora_moe_patchtst` places token-level spectral denoising between PatchTST and LoRA-MoE. `internal_after1_asd_lora_moe_patchtst` places it after the first encoder layer, before later encoder layers.

patch preset: `short_second`; layers=3; epochs=5; steps/epoch=50; train cap=20000; eval cap=4096; seed=42.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 4,096 | 1.0000 | 9.4161e-07 | 6.1005e-04 | 0.4877 | nan |
| zero | minute | 1,040 | 1.0036 | 1.5905e-06 | 7.9701e-04 | 0.4957 | nan |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| raw_joint | second | 4,096 | 0.9949 | 9.3685e-07 | 6.0970e-04 | 0.5289 | 0.0838 |
| post_return_lora_moe_head | second | 4,096 | 0.9950 | 9.3689e-07 | 6.0978e-04 | 0.5400 | 0.0846 |
| mid_token_asd_lora_moe_patchtst | second | 4,096 | 1.0025 | 9.4399e-07 | 6.1970e-04 | 0.5022 | 0.0806 |
| internal_after1_asd_lora_moe_patchtst | second | 4,096 | 1.0019 | 9.4338e-07 | 6.1729e-04 | 0.4998 | 0.0786 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9946 | 9.3653e-07 | 6.0895e-04 | 0.5429 | 0.0857 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9957 | 9.3757e-07 | 6.0931e-04 | 0.5397 | 0.0810 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9959 | 9.3774e-07 | 6.0987e-04 | 0.5375 | 0.0787 |
| raw_joint | minute | 1,040 | 1.0170 | 1.6116e-06 | 8.0412e-04 | 0.5038 | 0.0846 |
| post_return_lora_moe_head | minute | 1,040 | 0.9970 | 1.5800e-06 | 7.9584e-04 | 0.5298 | 0.1061 |
| mid_token_asd_lora_moe_patchtst | minute | 1,040 | 1.0038 | 1.5907e-06 | 7.9723e-04 | 0.5279 | 0.0888 |
| internal_after1_asd_lora_moe_patchtst | minute | 1,040 | 0.9974 | 1.5805e-06 | 7.9546e-04 | 0.5337 | 0.1074 |
| pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0010 | 1.5863e-06 | 7.9640e-04 | 0.5356 | 0.0829 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0025 | 1.5887e-06 | 7.9664e-04 | 0.5365 | 0.0920 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 1,040 | 1.0139 | 1.6068e-06 | 8.0273e-04 | 0.5115 | 0.0912 |
| raw_joint | hour | 200.0000 | 0.5625 | 2.6359e-05 | 3.3238e-03 | 0.6800 | 0.6762 |
| post_return_lora_moe_head | hour | 200.0000 | 0.5430 | 2.5445e-05 | 3.2542e-03 | 0.7100 | 0.6794 |
| mid_token_asd_lora_moe_patchtst | hour | 200.0000 | 0.5458 | 2.5576e-05 | 3.2348e-03 | 0.7050 | 0.6763 |
| internal_after1_asd_lora_moe_patchtst | hour | 200.0000 | 0.5526 | 2.5892e-05 | 3.2901e-03 | 0.7200 | 0.6740 |
| pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5431 | 2.5447e-05 | 3.2301e-03 | 0.7000 | 0.6775 |
| gated_pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5352 | 2.5079e-05 | 3.2276e-03 | 0.7100 | 0.6830 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 200.0000 | 0.5429 | 2.5440e-05 | 3.2660e-03 | 0.6800 | 0.6808 |

## Test Ranking By Scale

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gated_pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5352 | 2.5079e-05 | 3.2276e-03 | 0.7100 | 0.6830 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 200.0000 | 0.5429 | 2.5440e-05 | 3.2660e-03 | 0.6800 | 0.6808 |
| post_return_lora_moe_head | hour | 200.0000 | 0.5430 | 2.5445e-05 | 3.2542e-03 | 0.7100 | 0.6794 |
| pre_return_asd_lora_moe_patchtst | hour | 200.0000 | 0.5431 | 2.5447e-05 | 3.2301e-03 | 0.7000 | 0.6775 |
| mid_token_asd_lora_moe_patchtst | hour | 200.0000 | 0.5458 | 2.5576e-05 | 3.2348e-03 | 0.7050 | 0.6763 |
| post_return_lora_moe_head | minute | 1,040 | 0.9970 | 1.5800e-06 | 7.9584e-04 | 0.5298 | 0.1061 |
| internal_after1_asd_lora_moe_patchtst | minute | 1,040 | 0.9974 | 1.5805e-06 | 7.9546e-04 | 0.5337 | 0.1074 |
| pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0010 | 1.5863e-06 | 7.9640e-04 | 0.5356 | 0.0829 |
| gated_pre_return_asd_lora_moe_patchtst | minute | 1,040 | 1.0025 | 1.5887e-06 | 7.9664e-04 | 0.5365 | 0.0920 |
| mid_token_asd_lora_moe_patchtst | minute | 1,040 | 1.0038 | 1.5907e-06 | 7.9723e-04 | 0.5279 | 0.0888 |
| pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9946 | 9.3653e-07 | 6.0895e-04 | 0.5429 | 0.0857 |
| raw_joint | second | 4,096 | 0.9949 | 9.3685e-07 | 6.0970e-04 | 0.5289 | 0.0838 |
| post_return_lora_moe_head | second | 4,096 | 0.9950 | 9.3689e-07 | 6.0978e-04 | 0.5400 | 0.0846 |
| gated_pre_return_asd_lora_moe_patchtst | second | 4,096 | 0.9957 | 9.3757e-07 | 6.0931e-04 | 0.5397 | 0.0810 |
| scale_specific_gated_pre_asd_moe_patchtst | second | 4,096 | 0.9959 | 9.3774e-07 | 6.0987e-04 | 0.5375 | 0.0787 |

## Diagnostics

| model | scale | asd_gate_mean | asd_tau_mean | asd_mean_abs_delta | gate_mean | tau_mean | local_mask_mean | pre_adapter_gate_mean | pre_adapter_mean_abs_delta | final_gate_mean | final_mean_abs_delta | router_entropy | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | adapter_kind_id |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| post_return_lora_moe_head | second | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4724 | 2.3234e-03 | 0.3958 | 0.5818 | 0.0201 | nan |
| post_return_lora_moe_head | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4369 | 0.6686 | 0.2727 | 1.0857e-04 | 0.0586 | nan |
| post_return_lora_moe_head | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan | 0.4724 | 0.2508 | 0.3529 | 0.3406 | 0.0557 | nan |
| mid_token_asd_lora_moe_patchtst | second | nan | nan | nan | 0.0323 | 3.2338 | 0.1555 | nan | nan | nan | nan | 0.3839 | 3.0783e-04 | 0.2316 | 0.7664 | 1.7068e-03 | nan |
| mid_token_asd_lora_moe_patchtst | minute | nan | nan | nan | 0.0202 | 2.6266 | 0.2537 | nan | nan | nan | nan | 0.4783 | 0.4463 | 0.5311 | 0.0226 | 0.0000e+00 | nan |
| mid_token_asd_lora_moe_patchtst | hour | nan | nan | nan | 0.1083 | 2.3446 | 0.2683 | nan | nan | nan | nan | 0.4692 | 0.4776 | 0.3960 | 0.0595 | 0.0669 | nan |
| internal_after1_asd_lora_moe_patchtst | second | nan | nan | nan | 2.5144e-03 | 3.8059 | 0.1355 | nan | nan | nan | nan | 0.4368 | 0.6512 | 0.3166 | 3.6661e-06 | 0.0323 | nan |
| internal_after1_asd_lora_moe_patchtst | minute | nan | nan | nan | 0.1309 | 0.7813 | 0.5773 | nan | nan | nan | nan | 0.4732 | 0.0209 | 9.1101e-03 | 0.4553 | 0.5146 | nan |
| internal_after1_asd_lora_moe_patchtst | hour | nan | nan | nan | 0.9349 | 1.2591 | 0.5175 | nan | nan | nan | nan | 0.4632 | 0.0871 | 0.1068 | 0.4401 | 0.3660 | nan |
| pre_return_asd_lora_moe_patchtst | second | 6.0204e-03 | 3.9037 | 1.4797e-03 | 6.0204e-03 | 3.9037 | nan | 6.9543e-03 | 0.0614 | 0.1192 | 2.2765e-03 | 0.4783 | 0.4404 | 7.3739e-05 | 0.4388 | 0.1208 | nan |
| pre_return_asd_lora_moe_patchtst | minute | 0.0132 | 1.1173 | 3.8609e-03 | 0.0132 | 1.1173 | nan | 0.0127 | 0.0771 | 0.1192 | 4.7913e-03 | 0.4625 | 0.3381 | 0.3308 | 0.3311 | 7.9718e-06 | nan |
| pre_return_asd_lora_moe_patchtst | hour | 0.0135 | 12.1499 | 9.5146e-03 | 0.0135 | 12.1499 | nan | 0.0179 | 0.2072 | 0.1192 | 0.0180 | 0.4379 | 0.4703 | 4.4085e-06 | 0.2565 | 0.2733 | nan |
| gated_pre_return_asd_lora_moe_patchtst | second | 5.5155e-03 | 4.6982 | 1.5197e-03 | 5.5155e-03 | 4.6982 | nan | 0.0135 | 0.1274 | 0.0730 | 7.9126e-04 | 0.4833 | 0.4461 | 0.0927 | 0.0591 | 0.4021 | nan |
| gated_pre_return_asd_lora_moe_patchtst | minute | 9.4143e-03 | 0.8871 | 2.2388e-03 | 9.4143e-03 | 0.8871 | nan | 0.0345 | 0.2334 | 0.1186 | 5.7552e-03 | 0.4783 | 0.2711 | 0.2241 | 0.2269 | 0.2779 | nan |
| gated_pre_return_asd_lora_moe_patchtst | hour | 0.0122 | 14.8983 | 8.6496e-03 | 0.0122 | 14.8983 | nan | 0.0325 | 0.6850 | 0.1802 | 0.0235 | 0.4417 | 0.3296 | 0.3775 | 9.7430e-03 | 0.2831 | nan |
| scale_specific_gated_pre_asd_moe_patchtst | second | 0.0232 | 5.7745 | 7.0564e-03 | 0.0232 | 5.7745 | nan | 0.0292 | 0.1836 | 2.3165e-03 | 7.6891e-05 | 0.4929 | 0.0848 | 0.3891 | 0.0940 | 0.4321 | 0.0000e+00 |
| scale_specific_gated_pre_asd_moe_patchtst | minute | 4.7524e-03 | 1.0158 | 1.2777e-03 | 4.7524e-03 | 1.0158 | nan | 0.0349 | 0.0690 | 0.0628 | 7.7570e-04 | 0.4704 | 0.0000e+00 | 0.3294 | 0.3360 | 0.3346 | 1.0000 |
| scale_specific_gated_pre_asd_moe_patchtst | hour | 2.3503e-03 | 10.4255 | 1.6063e-03 | 2.3503e-03 | 10.4255 | nan | 0.0225 | 0.4786 | 0.2713 | 0.0150 | 0.4736 | 0.3459 | 0.3899 | 0.2412 | 0.0229 | 0.0000e+00 |

## Files

- summary: `outputs\prepatch_asd_adapter_patchtst\targeted_internal_after1_3layer_more_data\summary.csv`
- diagnostics: `outputs\prepatch_asd_adapter_patchtst\targeted_internal_after1_3layer_more_data\diagnostics.csv`
