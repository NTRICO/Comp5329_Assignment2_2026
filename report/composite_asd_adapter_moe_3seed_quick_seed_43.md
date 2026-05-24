# Routed Composite ASD-Adapter MoE Quick Ablation

This quick run tests a routed composite expert design: each expert owns its own ASD module and either a LoRA or MLP value enhancer. The router operates per return position with scale and position information, then a final residual gate mixes the adapted window back with raw returns.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz`; patch preset: `balanced_60_45_24`; seed=43; epochs=2; steps/epoch=8; train cap=2048; eval cap=512.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr | architecture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 512.0000 | 1.0012 | 9.2639e-08 | 2.0502e-04 | 0.5128 | nan | baseline |
| last_return | second | 512.0000 | 22.5323 | 2.0849e-06 | 7.3272e-04 | 0.4635 | -0.1430 | baseline |
| zero | minute | 512.0000 | 1.0033 | 3.2139e-07 | 4.0862e-04 | 0.5156 | nan | baseline |
| last_return | minute | 512.0000 | 2.1858 | 7.0019e-07 | 6.2857e-04 | 0.4629 | 0.0566 | baseline |
| zero | hour | 252.0000 | 1.0000 | 4.4683e-05 | 4.4187e-03 | 0.4563 | nan | baseline |
| last_return | hour | 252.0000 | 1.5654 | 6.9944e-05 | 5.9218e-03 | 0.5675 | 0.2153 | baseline |
| raw_joint | second | 512.0000 | 1.1234 | 1.0395e-07 | 2.2682e-04 | 0.5247 | 0.0903 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | second | 512.0000 | 1.0062 | 9.3102e-08 | 2.0867e-04 | 0.4970 | 0.0540 | routed_composite_asd_adapter_moe |
| raw_joint | minute | 512.0000 | 1.0466 | 3.3525e-07 | 4.1983e-04 | 0.5059 | -0.0745 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | minute | 512.0000 | 1.0374 | 3.3233e-07 | 4.1676e-04 | 0.4844 | -0.0588 | routed_composite_asd_adapter_moe |
| raw_joint | hour | 252.0000 | 1.2446 | 5.5612e-05 | 5.1037e-03 | 0.5437 | -0.0945 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | hour | 252.0000 | 1.0369 | 4.6329e-05 | 4.6101e-03 | 0.4524 | -0.1009 | routed_composite_asd_adapter_moe |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | final_gate_mean | final_mean_abs_delta | composite_mean_abs_delta | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| routed_composite_asd_adapter_moe_patchtst | second | 0.4871 | 0.0577 | 0.1298 | 7.5294e-04 | 5.8012e-03 | 0.0202 | 0.4745 | 1.3749e-04 | 0.5052 |
| routed_composite_asd_adapter_moe_patchtst | minute | 0.4950 | 0.0249 | 0.1097 | 7.1123e-04 | 6.4839e-03 | 0.2455 | 0.3241 | 0.4292 | 1.2310e-03 |
| routed_composite_asd_adapter_moe_patchtst | hour | 0.4993 | 0.0267 | 0.1000 | 4.4783e-04 | 4.4799e-03 | 0.4744 | 0.3050 | 0.1956 | 0.0251 |

## Parameter Counts

| model | total | trainable |
| --- | --- | --- |
| raw_joint | 75,971 | 75,971 |
| routed_composite_asd_adapter_moe_patchtst | 86,568 | 12,648 |

## Files

- summary: `outputs\composite_asd_adapter_moe_3seed_quick\seed_43\summary.csv`
- diagnostics: `outputs\composite_asd_adapter_moe_3seed_quick\seed_43\diagnostics.csv`
