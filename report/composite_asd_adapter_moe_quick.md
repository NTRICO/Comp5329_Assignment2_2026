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
| routed_composite_asd_adapter_moe_patchtst | second | 512.0000 | 1.0365 | 9.5904e-08 | 2.1022e-04 | 0.4990 | 0.0117 | routed_composite_asd_adapter_moe |
| raw_joint | minute | 512.0000 | 0.9902 | 3.1721e-07 | 4.0970e-04 | 0.5312 | 0.1233 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | minute | 512.0000 | 1.0310 | 3.3026e-07 | 4.1662e-04 | 0.4922 | 0.1107 | routed_composite_asd_adapter_moe |
| raw_joint | hour | 252.0000 | 1.1113 | 4.9655e-05 | 4.7376e-03 | 0.5278 | 0.0293 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | hour | 252.0000 | 1.0200 | 4.5574e-05 | 4.4578e-03 | 0.5397 | 0.0241 | routed_composite_asd_adapter_moe |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | final_gate_mean | final_mean_abs_delta | composite_mean_abs_delta | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| routed_composite_asd_adapter_moe_patchtst | second | 0.4970 | 0.0297 | 0.1411 | 1.6982e-03 | 0.0120 | 0.1409 | 0.3825 | 0.4494 | 0.0271 |
| routed_composite_asd_adapter_moe_patchtst | minute | 0.4921 | 0.0313 | 0.1287 | 8.7353e-04 | 6.7870e-03 | 0.3688 | 0.4657 | 0.0178 | 0.1477 |
| routed_composite_asd_adapter_moe_patchtst | hour | 0.4928 | 0.0373 | 0.1189 | 8.4472e-04 | 7.1055e-03 | 1.4941e-03 | 0.5242 | 0.3141 | 0.1602 |

## Parameter Counts

| model | total | trainable |
| --- | --- | --- |
| raw_joint | 75,971 | 75,971 |
| routed_composite_asd_adapter_moe_patchtst | 86,568 | 12,648 |

## Files

- summary: `outputs\composite_asd_adapter_moe_quick\summary.csv`
- diagnostics: `outputs\composite_asd_adapter_moe_quick\diagnostics.csv`
