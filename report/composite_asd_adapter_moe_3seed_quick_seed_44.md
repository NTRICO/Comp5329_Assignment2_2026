# Routed Composite ASD-Adapter MoE Quick Ablation

This quick run tests a routed composite expert design: each expert owns its own ASD module and either a LoRA or MLP value enhancer. The router operates per return position with scale and position information, then a final residual gate mixes the adapted window back with raw returns.

cache: `E:\Working Area\Comp5329_Assignment2_2026\data\cache\position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz`; patch preset: `balanced_60_45_24`; seed=44; epochs=2; steps/epoch=8; train cap=2048; eval cap=512.

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr | architecture |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 512.0000 | 1.0012 | 9.2639e-08 | 2.0502e-04 | 0.5128 | nan | baseline |
| last_return | second | 512.0000 | 22.5323 | 2.0849e-06 | 7.3272e-04 | 0.4635 | -0.1430 | baseline |
| zero | minute | 512.0000 | 1.0033 | 3.2139e-07 | 4.0862e-04 | 0.5156 | nan | baseline |
| last_return | minute | 512.0000 | 2.1858 | 7.0019e-07 | 6.2857e-04 | 0.4629 | 0.0566 | baseline |
| zero | hour | 252.0000 | 1.0000 | 4.4683e-05 | 4.4187e-03 | 0.4563 | nan | baseline |
| last_return | hour | 252.0000 | 1.5654 | 6.9944e-05 | 5.9218e-03 | 0.5675 | 0.2153 | baseline |
| raw_joint | second | 512.0000 | 1.0628 | 9.8343e-08 | 2.2111e-04 | 0.5010 | 0.0278 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | second | 512.0000 | 1.0544 | 9.7564e-08 | 2.2150e-04 | 0.4911 | 0.0477 | routed_composite_asd_adapter_moe |
| raw_joint | minute | 512.0000 | 0.9981 | 3.1972e-07 | 4.0883e-04 | 0.4844 | 0.0808 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | minute | 512.0000 | 1.0398 | 3.3308e-07 | 4.1781e-04 | 0.5137 | 0.0190 | routed_composite_asd_adapter_moe |
| raw_joint | hour | 252.0000 | 1.0229 | 4.5706e-05 | 4.5854e-03 | 0.4841 | 0.0977 | raw_patchtst |
| routed_composite_asd_adapter_moe_patchtst | hour | 252.0000 | 0.9985 | 4.4616e-05 | 4.3291e-03 | 0.6032 | 0.0723 | routed_composite_asd_adapter_moe |

## Diagnostics

| model | scale | router_entropy | router_balance_loss | final_gate_mean | final_mean_abs_delta | composite_mean_abs_delta | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| raw_joint | second | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | minute | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| raw_joint | hour | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| routed_composite_asd_adapter_moe_patchtst | second | 0.4990 | 0.0567 | 0.1286 | 1.1248e-03 | 8.7446e-03 | 8.4993e-04 | 0.4695 | 0.0240 | 0.5056 |
| routed_composite_asd_adapter_moe_patchtst | minute | 0.4725 | 0.0639 | 0.1748 | 9.8209e-03 | 0.0562 | 0.5541 | 0.0000e+00 | 0.4456 | 2.7900e-04 |
| routed_composite_asd_adapter_moe_patchtst | hour | 0.4888 | 0.0551 | 0.1635 | 0.0114 | 0.0697 | 0.5601 | 0.0000e+00 | 0.0469 | 0.3929 |

## Parameter Counts

| model | total | trainable |
| --- | --- | --- |
| raw_joint | 75,971 | 75,971 |
| routed_composite_asd_adapter_moe_patchtst | 86,568 | 12,648 |

## Files

- summary: `outputs\composite_asd_adapter_moe_3seed_quick\seed_44\summary.csv`
- diagnostics: `outputs\composite_asd_adapter_moe_3seed_quick\seed_44\diagnostics.csv`
