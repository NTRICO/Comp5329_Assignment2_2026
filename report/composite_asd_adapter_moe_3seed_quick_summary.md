# composite_asd_adapter_moe_3seed_quick

This report aggregates routed composite ASD-adapter MoE runs. The main metric is `relative_nmse_improvement_pct = (raw_joint_nmse - candidate_nmse) / raw_joint_nmse * 100`.

## Decision

- verdict: **keep as promising challenger**
- run count: `3`

## Relative NMSE Improvement

| split | scale | seed_count | relative_nmse_improvement_pct_mean | relative_nmse_improvement_pct_std | positive_seed_rate | raw_joint_nmse_mean | candidate_nmse_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| validation | second | 3 | 4.7664 | 4.1631 | 1.0000 | 1.1021 | 1.0481 |
| validation | minute | 3 | -1.2253 | 1.8247 | 0.3333 | 1.0335 | 1.0461 |
| validation | hour | 3 | 6.0207 | 3.5892 | 1.0000 | 1.1053 | 1.0375 |
| test | second | 3 | 5.1492 | 4.8883 | 1.0000 | 1.0894 | 1.0323 |
| test | minute | 3 | -2.4728 | 2.8971 | 0.3333 | 1.0116 | 1.0361 |
| test | hour | 3 | 9.0993 | 7.1938 | 1.0000 | 1.1263 | 1.0184 |
| zero_shot | second | 3 | 7.8404 | 9.7864 | 0.6667 | 1.1552 | 1.0598 |
| zero_shot | minute | 3 | -1.3131 | 1.9544 | 0.3333 | 1.0183 | 1.0316 |
| zero_shot | hour | 3 | 13.5972 | 7.1944 | 1.0000 | 1.1798 | 1.0137 |

## Diagnostics

| scale | router_entropy_mean | router_entropy_std | router_balance_loss_mean | final_gate_mean_mean | final_gate_mean_std | expert_prob_0_mean | expert_prob_1_mean | expert_prob_2_mean | expert_prob_3_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| second | 0.4944 | 0.0064 | 0.0480 | 0.1332 | 0.0069 | 0.0540 | 0.4422 | 0.1579 | 0.3460 |
| minute | 0.4865 | 0.0123 | 0.0400 | 0.1377 | 0.0335 | 0.3895 | 0.2633 | 0.2975 | 0.0497 |
| hour | 0.4936 | 0.0053 | 0.0397 | 0.1275 | 0.0326 | 0.3453 | 0.2764 | 0.1855 | 0.1927 |

## Model Metric Aggregate

| split | scale | model | n_mean | nmse_mean | nmse_std | mae_mean | mae_std | direction_accuracy_nonzero_mean | direction_accuracy_nonzero_std | corr_mean | corr_std |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| validation | second | raw_joint | 512.0000 | 1.1021 | 0.0569 | 2.1672e-04 | 9.7750e-06 | 0.4971 | 0.0126 | -0.0016 | 0.1078 |
| validation | second | routed_composite_asd_adapter_moe_patchtst | 512.0000 | 1.0481 | 0.0154 | 2.0590e-04 | 4.4785e-06 | 0.4957 | 0.0165 | -0.0018 | 0.1127 |
| validation | minute | raw_joint | 512.0000 | 1.0335 | 0.0080 | 4.1131e-04 | 1.6352e-06 | 0.5042 | 0.0382 | -0.0354 | 0.0410 |
| validation | minute | routed_composite_asd_adapter_moe_patchtst | 512.0000 | 1.0461 | 0.0144 | 4.1569e-04 | 5.8561e-06 | 0.4938 | 0.0255 | -0.0326 | 0.0416 |
| validation | hour | raw_joint | 243.0000 | 1.1053 | 0.0567 | 0.0051 | 1.3466e-04 | 0.5240 | 0.0582 | -0.1068 | 0.0962 |
| validation | hour | routed_composite_asd_adapter_moe_patchtst | 243.0000 | 1.0375 | 0.0178 | 0.0048 | 1.0080e-04 | 0.5103 | 0.0464 | -0.1204 | 0.0681 |
| test | second | raw_joint | 512.0000 | 1.0894 | 0.0310 | 2.2418e-04 | 2.8827e-06 | 0.5089 | 0.0137 | 0.0355 | 0.0513 |
| test | second | routed_composite_asd_adapter_moe_patchtst | 512.0000 | 1.0323 | 0.0244 | 2.1346e-04 | 7.0020e-06 | 0.4957 | 0.0041 | 0.0378 | 0.0228 |
| test | minute | raw_joint | 512.0000 | 1.0116 | 0.0305 | 4.1279e-04 | 6.1142e-06 | 0.5072 | 0.0235 | 0.0432 | 0.1041 |
| test | minute | routed_composite_asd_adapter_moe_patchtst | 512.0000 | 1.0361 | 0.0046 | 4.1706e-04 | 6.5119e-07 | 0.4967 | 0.0152 | 0.0236 | 0.0848 |
| test | hour | raw_joint | 252.0000 | 1.1263 | 0.1116 | 0.0048 | 2.6638e-04 | 0.5185 | 0.0308 | 0.0108 | 0.0974 |
| test | hour | routed_composite_asd_adapter_moe_patchtst | 252.0000 | 1.0184 | 0.0192 | 0.0045 | 1.4068e-04 | 0.5317 | 0.0757 | -0.0015 | 0.0894 |
| zero_shot | second | raw_joint | 512.0000 | 1.1552 | 0.0790 | 1.6301e-04 | 7.5120e-06 | 0.4928 | 0.0181 | 0.0418 | 0.0212 |
| zero_shot | second | routed_composite_asd_adapter_moe_patchtst | 512.0000 | 1.0598 | 0.0527 | 1.4968e-04 | 7.9064e-06 | 0.4941 | 0.0141 | 0.0310 | 0.0208 |
| zero_shot | minute | raw_joint | 512.0000 | 1.0183 | 0.0283 | 3.4514e-04 | 5.9634e-06 | 0.5157 | 0.0137 | 0.0439 | 0.0629 |
| zero_shot | minute | routed_composite_asd_adapter_moe_patchtst | 512.0000 | 1.0316 | 0.0349 | 3.4471e-04 | 2.9637e-06 | 0.5111 | 0.0142 | 0.0329 | 0.0723 |
| zero_shot | hour | raw_joint | 488.0000 | 1.1798 | 0.1186 | 0.0041 | 3.8607e-04 | 0.5150 | 0.0432 | 0.0287 | 0.0171 |
| zero_shot | hour | routed_composite_asd_adapter_moe_patchtst | 488.0000 | 1.0137 | 0.0163 | 0.0035 | 1.1255e-04 | 0.5260 | 0.0426 | 0.0636 | 0.0107 |

## Run Directories

- `outputs\composite_asd_adapter_moe_3seed_quick\seed_42`
- `outputs\composite_asd_adapter_moe_3seed_quick\seed_43`
- `outputs\composite_asd_adapter_moe_3seed_quick\seed_44`

## Output Files

- `aggregate_model_metrics.csv`
- `aggregate_relative_improvement.csv`
- `seed_relative_improvement.csv`
- `aggregate_diagnostics.csv`
