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
| routed_composite_asd_adapter_moe_patchtst | second | 512.0000 | 1.0281 | 9.5131e-08 | 2.0817e-04 | 0.5187 | 3.0269e-03 | routed_composite_asd_adapter_moe |
| raw_joint | minute | 512.0000 | 0.9902 | 3.1721e-07 | 4.0970e-04 | 0.5312 | 0.1233 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | minute | 512.0000 | 1.0681 | 3.4214e-07 | 4.2857e-04 | 0.4844 | 0.1290 | routed_composite_asd_adapter_moe |
| raw_joint | hour | 252.0000 | 1.1113 | 4.9655e-05 | 4.7376e-03 | 0.5278 | 0.0293 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | hour | 252.0000 | 1.0113 | 4.5185e-05 | 4.5198e-03 | 0.5079 | 0.0307 | routed_composite_asd_adapter_moe |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | final_gate_mean | final_mean_abs_delta | composite_mean_abs_delta | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| routed_composite_asd_adapter_moe_patchtst | second | 0.4994 | 0.0485 | 0.1442 | 1.2924e-03 | 8.9609e-03 | 0.0265 | 0.0332 | 0.4650 | 0.4753 |
| routed_composite_asd_adapter_moe_patchtst | minute | 0.4923 | 0.0369 | 0.1221 | 8.8849e-04 | 7.2794e-03 | 0.4536 | 0.4196 | 0.1260 | 6.8578e-04 |
| routed_composite_asd_adapter_moe_patchtst | hour | 0.4822 | 0.0658 | 0.1347 | 1.4605e-03 | 0.0108 | 1.2904e-03 | 0.5849 | 0.4138 | 0.0000e+00 |

## Parameter Counts

| model | total | trainable |
| --- | --- | --- |
| raw_joint | 75,971 | 75,971 |
| routed_composite_asd_adapter_moe_patchtst | 86,640 | 12,720 |

## Files

- summary: `outputs\composite_asd_adapter_moe_pattern_three_mlp_seed42\summary.csv`
- diagnostics: `outputs\composite_asd_adapter_moe_pattern_three_mlp_seed42\diagnostics.csv`
