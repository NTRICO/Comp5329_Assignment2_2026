# ASD 与 LoRA-MoE 冲突诊断

本轮目标是验证 ASD 是否通过改变 patch/token 表示导致 LoRA-MoE router 分工漂移，并测试 router 稳定化、side denoising、post-patch spectral filtering 三类修复。

配置：patch preset `short_second`，rank=8，ASD init gate=-4.0，epochs=3，balanced steps/epoch=12。

## Test Metrics

| model | scale | n | nmse | mse | mae | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- |
| zero | second | 1,024 | 1.0023 | 9.6510e-07 | 6.1415e-04 | 0.4568 | nan |
| raw_joint | second | 1,024 | 1.0177 | 9.7993e-07 | 6.1876e-04 | 0.5029 | -0.0334 |
| lora_moe_head | second | 1,024 | 1.0240 | 9.8598e-07 | 6.2168e-04 | 0.5206 | -0.0316 |
| asd_lora_moe | second | 1,024 | 1.0301 | 9.9186e-07 | 6.2427e-04 | 0.4794 | -0.0464 |
| asd_lora_moe_router_frozen | second | 1,024 | 1.0296 | 9.9136e-07 | 6.2306e-04 | 0.5354 | -0.0321 |
| asd_lora_moe_router_kl | second | 1,024 | 1.0407 | 1.0020e-06 | 6.2841e-04 | 0.4784 | -0.0441 |
| asd_side_feature_lora_moe | second | 1,024 | 1.0056 | 9.6830e-07 | 6.1735e-04 | 0.5206 | 0.0169 |
| patch_token_asd_lora_moe | second | 1,024 | 1.0320 | 9.9369e-07 | 6.2725e-04 | 0.5334 | -0.0325 |
| zero | minute | 1,024 | 1.0030 | 1.5852e-06 | 7.9712e-04 | 0.4936 | nan |
| raw_joint | minute | 1,024 | 1.0167 | 1.6069e-06 | 8.0126e-04 | 0.5064 | 0.0179 |
| lora_moe_head | minute | 1,024 | 1.0064 | 1.5905e-06 | 7.9792e-04 | 0.5230 | 0.0225 |
| asd_lora_moe | minute | 1,024 | 1.0075 | 1.5923e-06 | 7.9808e-04 | 0.5279 | 0.0145 |
| asd_lora_moe_router_frozen | minute | 1,024 | 1.0089 | 1.5945e-06 | 7.9803e-04 | 0.5396 | 0.0402 |
| asd_lora_moe_router_kl | minute | 1,024 | 1.0007 | 1.5816e-06 | 7.9746e-04 | 0.5015 | 0.0493 |
| asd_side_feature_lora_moe | minute | 1,024 | 1.0045 | 1.5876e-06 | 7.9649e-04 | 0.5152 | 0.0413 |
| patch_token_asd_lora_moe | minute | 1,024 | 1.0018 | 1.5834e-06 | 7.9956e-04 | 0.5112 | 0.0304 |
| zero | hour | 200.0000 | 1.0000 | 4.6856e-05 | 4.5733e-03 | 0.4750 | nan |
| raw_joint | hour | 200.0000 | 0.9347 | 4.3795e-05 | 4.3949e-03 | 0.6000 | 0.4649 |
| lora_moe_head | hour | 200.0000 | 0.8550 | 4.0062e-05 | 4.1683e-03 | 0.6950 | 0.5288 |
| asd_lora_moe | hour | 200.0000 | 0.8478 | 3.9723e-05 | 4.1510e-03 | 0.6850 | 0.5455 |
| asd_lora_moe_router_frozen | hour | 200.0000 | 0.7614 | 3.5677e-05 | 3.9199e-03 | 0.7200 | 0.5704 |
| asd_lora_moe_router_kl | hour | 200.0000 | 0.7544 | 3.5347e-05 | 3.8900e-03 | 0.7000 | 0.5673 |
| asd_side_feature_lora_moe | hour | 200.0000 | 0.9483 | 4.4436e-05 | 4.4185e-03 | 0.5750 | 0.3518 |
| patch_token_asd_lora_moe | hour | 200.0000 | 0.7522 | 3.5246e-05 | 3.8899e-03 | 0.7050 | 0.5633 |

## Router / ASD Diagnostics

| model | scale | gate_mean | tau_mean | mean_abs_delta | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 | side_residual_abs_mean | local_mask_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lora_moe_head | second | nan | nan | 0.0237 | 0.4597 | 0.0634 | 4.9371e-03 | 0.6552 | 0.0794 | 0.2604 | nan | nan |
| lora_moe_head | minute | nan | nan | 0.0387 | 0.4755 | 0.0675 | 2.3460e-04 | 3.0417e-04 | 0.3986 | 0.6009 | nan | nan |
| lora_moe_head | hour | nan | nan | 0.0390 | 0.0345 | 0.1833 | 2.4073e-03 | 4.1632e-06 | 6.0247e-03 | 0.9916 | nan | nan |
| asd_lora_moe | second | 0.0161 | 3.4743 | 0.0181 | 0.4900 | 0.0637 | 3.2544e-04 | 0.5530 | 3.7816e-04 | 0.4463 | nan | nan |
| asd_lora_moe | minute | 0.0148 | 0.7038 | 0.0169 | 0.4754 | 0.0673 | 0.6165 | 0.3731 | 0.0104 | 0.0000e+00 | nan | nan |
| asd_lora_moe | hour | 5.4623e-03 | 0.2416 | 0.0326 | 0.0270 | 0.1843 | 0.9936 | 6.0360e-03 | 1.5852e-05 | 3.8218e-04 | nan | nan |
| asd_lora_moe_router_frozen | second | 0.0169 | 2.2111 | 0.0523 | 0.4597 | 0.0634 | 4.8179e-03 | 0.6553 | 0.0790 | 0.2609 | nan | nan |
| asd_lora_moe_router_frozen | minute | 7.0145e-03 | 0.2420 | 0.1287 | 0.4755 | 0.0675 | 2.3442e-04 | 3.0407e-04 | 0.3986 | 0.6009 | nan | nan |
| asd_lora_moe_router_frozen | hour | 9.6883e-03 | 0.1566 | 0.0940 | 0.0345 | 0.1833 | 2.4068e-03 | 4.1632e-06 | 6.0245e-03 | 0.9916 | nan | nan |
| asd_lora_moe_router_kl | second | 0.0172 | 1.4752 | 0.0590 | 0.4634 | 0.0617 | 9.6210e-03 | 0.6445 | 0.0693 | 0.2766 | nan | nan |
| asd_lora_moe_router_kl | minute | 4.7451e-03 | 0.3084 | 0.1404 | 0.4770 | 0.0663 | 5.4171e-04 | 9.2635e-05 | 0.4106 | 0.5888 | nan | nan |
| asd_lora_moe_router_kl | hour | 1.7405e-03 | 0.6940 | 0.0985 | 0.0274 | 0.1843 | 4.9867e-03 | 7.3545e-07 | 1.4087e-03 | 0.9936 | nan | nan |
| asd_side_feature_lora_moe | second | 8.0387e-03 | 9.0742 | 0.0417 | 0.4454 | 0.0790 | 6.9701e-04 | 0.6865 | 4.7872e-03 | 0.3080 | 2.7023e-03 | nan |
| asd_side_feature_lora_moe | minute | 0.2568 | 1.3175 | 0.0244 | 0.4979 | 0.0511 | 0.0506 | 0.4999 | 0.0000e+00 | 0.4495 | 0.0867 | nan |
| asd_side_feature_lora_moe | hour | 0.7862 | 2.4817 | 0.0167 | 0.4930 | 0.0639 | 0.0000e+00 | 0.4465 | 0.0000e+00 | 0.5535 | 0.2291 | nan |
| patch_token_asd_lora_moe | second | 6.4313e-03 | 0.8202 | 0.0282 | 0.4925 | 0.0571 | 0.0225 | 0.5275 | 0.4469 | 3.1490e-03 | nan | 0.6326 |
| patch_token_asd_lora_moe | minute | 0.0468 | 2.6237 | 0.1055 | 0.3722 | 0.1002 | 0.0112 | 0.0000e+00 | 0.7788 | 0.2100 | nan | 0.3581 |
| patch_token_asd_lora_moe | hour | 0.3064 | 5.3092 | 0.0959 | 0.0641 | 0.1785 | 0.0152 | 0.0000e+00 | 3.0155e-03 | 0.9818 | nan | 0.1647 |

## Conflict Metrics vs LoRA-MoE Teacher

| model | scale | prediction_delta_abs_mean | router_kl_to_teacher | router_l1_to_teacher | token_cosine_to_teacher | token_l2_to_teacher |
| --- | --- | --- | --- | --- | --- | --- |
| asd_lora_moe | second | 6.6009e-05 | 0.2409 | 0.1336 | 1.0000 | 4.8648e-03 |
| asd_lora_moe | minute | 1.0549e-05 | 1.6380 | 0.3692 | 1.0000 | 2.6862e-03 |
| asd_lora_moe | hour | 6.8650e-04 | 4.9820 | 0.4926 | 1.0000 | 1.4810e-04 |
| asd_lora_moe_router_frozen | second | 2.8196e-05 | 4.4600e-07 | 1.3380e-04 | 1.0000 | 3.4235e-03 |
| asd_lora_moe_router_frozen | minute | 3.7859e-05 | 1.1737e-08 | 2.3081e-05 | 1.0000 | 4.4910e-04 |
| asd_lora_moe_router_frozen | hour | 6.9520e-04 | -4.2430e-10 | 8.0295e-07 | 1.0000 | 1.7048e-04 |
| asd_lora_moe_router_kl | second | 8.0296e-05 | 9.1514e-04 | 6.9083e-03 | 1.0000 | 2.3769e-03 |
| asd_lora_moe_router_kl | minute | 2.5991e-05 | 8.8271e-04 | 8.1322e-03 | 1.0000 | 3.8615e-04 |
| asd_lora_moe_router_kl | hour | 1.1373e-03 | 8.5439e-04 | 2.3420e-03 | 1.0000 | 1.3376e-04 |
| asd_side_feature_lora_moe | second | 6.0201e-05 | 0.0668 | 0.0727 | 1.0000 | 1.4373e-07 |
| asd_side_feature_lora_moe | minute | 2.1751e-05 | 0.8974 | 0.2594 | 1.0000 | 1.6720e-07 |
| asd_side_feature_lora_moe | hour | 1.3711e-03 | 2.0004 | 0.2574 | 1.0000 | 1.8964e-07 |
| patch_token_asd_lora_moe | second | 5.1604e-05 | 0.0649 | 0.0806 | 1.0000 | 1.8829e-04 |
| patch_token_asd_lora_moe | minute | 5.2614e-05 | 0.2951 | 0.1840 | 1.0000 | 1.3297e-03 |
| patch_token_asd_lora_moe | hour | 1.0183e-03 | 6.9582e-03 | 7.7302e-03 | 1.0000 | 0.0118 |
| lora_moe_head | second | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000 | 0.0000e+00 |
| lora_moe_head | minute | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000 | 0.0000e+00 |
| lora_moe_head | hour | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 | 1.0000 | 0.0000e+00 |

## Gradient Diagnostics

| model | asd_grad_norm | moe_grad_norm | head_grad_norm | encoder_spectral_grad_norm | asd_moe_grad_ratio | asd_moe_grad_cosine_truncated |
| --- | --- | --- | --- | --- | --- | --- |
| asd_lora_moe | 2.7620e-04 | 0.0239 | 0.3254 | 0.0000e+00 | 0.0116 | 0.2408 |
| asd_lora_moe_router_frozen | 3.3693e-04 | 0.0537 | 0.6963 | 0.0000e+00 | 6.2691e-03 | 0.2077 |
| asd_lora_moe_router_kl | 2.4748e-04 | 0.0396 | 0.4026 | 0.0000e+00 | 6.2461e-03 | 0.3965 |
| asd_side_feature_lora_moe | 0.0101 | 0.0245 | 0.9460 | 0.0000e+00 | 0.4125 | 0.3542 |
| patch_token_asd_lora_moe | 0.0000e+00 | 0.0413 | 0.7735 | 3.5848e-03 | 0.0000e+00 | nan |
| lora_moe_head | 0.0000e+00 | 0.0298 | 0.5408 | 0.0000e+00 | 0.0000e+00 | nan |

## Decision Notes

- `router_frozen` / `router_kl` 若恢复 hour，说明主要问题是 router distribution shift。
- `asd_side_feature_lora_moe` 若更稳，说明 ASD 有信息价值但不应替换 raw path。
- `patch_token_asd_lora_moe` 若更稳，说明 ASD 应后移到 patch/token 表示。
- 若所有 ASD 变体都弱于 `lora_moe_head`，主模型应保持 no-ASD LoRA-MoE。

## Files

- summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\asd_moe_conflict_diagnosis\summary.csv`
- diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\asd_moe_conflict_diagnosis\diagnostics.csv`
- conflict diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\asd_moe_conflict_diagnosis\conflict_diagnostics.csv`
- gradient diagnostics: `E:\Working Area\Comp5329_Assignment2_2026\outputs\asd_moe_conflict_diagnosis\gradient_diagnostics.csv`
