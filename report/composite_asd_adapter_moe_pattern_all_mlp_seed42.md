# Routed Composite ASD-Adapter MoE Quick Ablation

This quick run tests a routed composite expert design: each expert owns its own ASD module and either a LoRA or MLP value enhancer. The router operates per return position with scale and position information, then a final residual gate mixes the adapted window back with raw returns.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz`; patch preset: `balanced_60_45_24`; seed=42; epochs=2; steps/epoch=8; train cap=2048; eval cap=512.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr | architecture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 512.0000 | 1.0012 | 9.2639e-08 | 2.0502e-04 | 0.5128 | nan | baseline |
| last_return | second | 512.0000 | 22.5323 | 2.0849e-06 | 7.3272e-04 | 0.4635 | -0.1430 | baseline |
| zero | minute | 512.0000 | 1.0033 | 3.2139e-07 | 4.0862e-04 | 0.5156 | nan | baseline |
| last_return | minute | 512.0000 | 2.1858 | 7.0019e-07 | 6.2857e-04 | 0.4629 | 0.0566 | baseline |
| zero | hour | 252.0000 | 1.0000 | 4.4683e-05 | 4.4187e-03 | 0.4563 | nan | baseline |
| last_return | hour | 252.0000 | 1.5654 | 6.9944e-05 | 5.9218e-03 | 0.5675 | 0.2153 | baseline |
| raw_joint | second | 512.0000 | 1.0821 | 1.0013e-07 | 2.2461e-04 | 0.5010 | -0.0115 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | second | 512.0000 | 1.0667 | 9.8698e-08 | 2.1619e-04 | 0.5049 | 0.0193 | routed_composite_asd_adapter_moe |
| raw_joint | minute | 512.0000 | 0.9902 | 3.1721e-07 | 4.0970e-04 | 0.5312 | 0.1233 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | minute | 512.0000 | 1.0222 | 3.2744e-07 | 4.1341e-04 | 0.5039 | 0.1194 | routed_composite_asd_adapter_moe |
| raw_joint | hour | 252.0000 | 1.1113 | 4.9655e-05 | 4.7376e-03 | 0.5278 | 0.0293 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | hour | 252.0000 | 1.0206 | 4.5600e-05 | 4.5597e-03 | 0.5079 | 8.0424e-03 | routed_composite_asd_adapter_moe |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | final_gate_mean | final_mean_abs_delta | composite_mean_abs_delta | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| routed_composite_asd_adapter_moe_patchtst | second | 0.4964 | 0.0336 | 0.1235 | 7.1054e-04 | 5.7529e-03 | 0.4877 | 0.0175 | 0.1382 | 0.3566 |
| routed_composite_asd_adapter_moe_patchtst | minute | 0.4985 | 7.0822e-03 | 0.1192 | 6.3603e-04 | 5.3345e-03 | 0.1581 | 0.1760 | 0.3150 | 0.3509 |
| routed_composite_asd_adapter_moe_patchtst | hour | 0.4841 | 0.0436 | 0.1095 | 5.1097e-03 | 0.0467 | 7.0582e-04 | 0.2960 | 0.5630 | 0.1403 |

## Parameter Counts

| model | total | trainable |
| --- | --- | --- |
| raw_joint | 75,971 | 75,971 |
| routed_composite_asd_adapter_moe_patchtst | 86,712 | 12,792 |

## Files

- summary: `outputs\composite_asd_adapter_moe_pattern_all_mlp_seed42\summary.csv`
- diagnostics: `outputs\composite_asd_adapter_moe_pattern_all_mlp_seed42\diagnostics.csv`
