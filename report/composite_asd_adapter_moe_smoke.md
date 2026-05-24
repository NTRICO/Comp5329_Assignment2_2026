# Routed Composite ASD-Adapter MoE Quick Ablation

This quick run tests a routed composite expert design: each expert owns its own ASD module and either a LoRA or MLP value enhancer. The router operates per return position with scale and position information, then a final residual gate mixes the adapted window back with raw returns.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz`; patch preset: `balanced_60_45_24`; seed=42; epochs=1; steps/epoch=2; train cap=512; eval cap=128.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr | architecture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 128.0000 | 1.0140 | 8.0312e-08 | 1.9272e-04 | 0.4766 | nan | baseline |
| last_return | second | 128.0000 | 28.9713 | 2.2947e-06 | 7.6526e-04 | 0.4688 | -0.1557 | baseline |
| zero | minute | 128.0000 | 1.0244 | 4.1599e-07 | 4.7813e-04 | 0.5703 | nan | baseline |
| last_return | minute | 128.0000 | 1.9321 | 7.8461e-07 | 6.6659e-04 | 0.4375 | -0.1260 | baseline |
| zero | hour | 128.0000 | 1.0002 | 5.8706e-05 | 4.9439e-03 | 0.4219 | nan | baseline |
| last_return | hour | 128.0000 | 1.1431 | 6.7090e-05 | 5.7454e-03 | 0.5859 | 0.2641 | baseline |
| raw_joint | second | 128.0000 | 2.5071 | 1.9857e-07 | 3.8740e-04 | 0.4766 | -0.1341 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | second | 128.0000 | 1.2132 | 9.6088e-08 | 2.0762e-04 | 0.5625 | -0.0943 | routed_composite_asd_adapter_moe |
| raw_joint | minute | 128.0000 | 1.0325 | 4.1930e-07 | 4.8304e-04 | 0.5156 | 0.0153 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | minute | 128.0000 | 1.0875 | 4.4162e-07 | 4.9577e-04 | 0.4453 | 0.0350 | routed_composite_asd_adapter_moe |
| raw_joint | hour | 128.0000 | 1.0478 | 6.1495e-05 | 4.9860e-03 | 0.5781 | 0.0485 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | hour | 128.0000 | 1.0579 | 6.2091e-05 | 5.3244e-03 | 0.4609 | 0.0447 | routed_composite_asd_adapter_moe |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | final_gate_mean | final_mean_abs_delta | composite_mean_abs_delta | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| routed_composite_asd_adapter_moe_patchtst | second | 0.4978 | 0.0331 | 0.1157 | 5.1070e-04 | 4.4127e-03 | 0.0179 | 0.3639 | 0.1379 | 0.4803 |
| routed_composite_asd_adapter_moe_patchtst | minute | 0.4941 | 0.0315 | 0.1279 | 7.1896e-04 | 5.6225e-03 | 0.4613 | 0.0892 | 0.0598 | 0.3897 |
| routed_composite_asd_adapter_moe_patchtst | hour | 0.4545 | 0.0705 | 0.1397 | 4.9801e-03 | 0.0356 | 0.6501 | 0.3299 | 0.0200 | 0.0000e+00 |

## Parameter Counts

| model | total | trainable | lora | total_parameters | trainable_parameters |
| --- | --- | --- | --- | --- | --- |
| raw_joint | 75,971 | 75,971 | 0.0000e+00 | nan | nan |
| routed_composite_asd_adapter_moe_patchtst | nan | nan | nan | 86,568 | 12,648 |

## Files

- summary: `outputs\composite_asd_adapter_moe_smoke\summary.csv`
- diagnostics: `outputs\composite_asd_adapter_moe_smoke\diagnostics.csv`
