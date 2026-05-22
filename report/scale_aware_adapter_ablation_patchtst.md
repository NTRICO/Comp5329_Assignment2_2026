# PatchTST Adapter Ablation

本轮只做后接 adapter 消融：所有 adapter 都插在 PatchTST encoder 后、scale-specific head 前；无 ASD 的 LoRA-only/MLP-MoE/LoRA-MoE 前面没有 ASD；`asd_*` 行用于测试加 ASD 后结论是否改变。

small setting: patch presets=['short_second'], ranks=[4, 8], epochs=3, steps/epoch=12.

## 1. Small Result

summary: `E:\Working Area\Comp5329_Assignment2_2026\outputs\scale_aware_adapter_ablation_patchtst\round1_small_summary.csv`

| patch_preset | model | adapter_rank | scale | n | mse | mae | nmse | direction_accuracy_nonzero | corr |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| short_second | zero | nan | second | 1,024 | 9.6510e-07 | 6.1415e-04 | 1.0023 | 0.4568 | nan |
| short_second | asd_lora_only_frozen_base_train_adapter_head | 4.0000 | second | 1,024 | 9.8236e-07 | 6.1913e-04 | 1.0202 | 0.5334 | -0.0255 |
| short_second | asd_mlp_moe_frozen_base_train_moe_head | 4.0000 | second | 1,024 | 9.8421e-07 | 6.2163e-04 | 1.0222 | 0.5432 | -0.0261 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 4.0000 | second | 1,024 | 9.9238e-07 | 6.2486e-04 | 1.0307 | 0.4902 | -0.0371 |
| short_second | lora_only_frozen_base_train_adapter_only | 4.0000 | second | 1,024 | 9.8177e-07 | 6.2023e-04 | 1.0196 | 0.4882 | -0.0341 |
| short_second | lora_only_frozen_base_train_adapter_head | 4.0000 | second | 1,024 | 9.8927e-07 | 6.2425e-04 | 1.0274 | 0.4646 | -0.0320 |
| short_second | mlp_moe_frozen_base_train_moe_only | 4.0000 | second | 1,024 | 9.8254e-07 | 6.2088e-04 | 1.0204 | 0.4813 | -0.0353 |
| short_second | mlp_moe_frozen_base_train_moe_head | 4.0000 | second | 1,024 | 9.9497e-07 | 6.2420e-04 | 1.0333 | 0.4823 | -0.0543 |
| short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | second | 1,024 | 9.8497e-07 | 6.2122e-04 | 1.0230 | 0.4931 | -0.0232 |
| short_second | asd_lora_only_frozen_base_train_adapter_head | 8.0000 | second | 1,024 | 9.9393e-07 | 6.2683e-04 | 1.0323 | 0.4735 | -0.0336 |
| short_second | asd_mlp_moe_frozen_base_train_moe_head | 8.0000 | second | 1,024 | 9.9669e-07 | 6.2903e-04 | 1.0351 | 0.4686 | -0.0356 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | second | 1,024 | 1.0031e-06 | 6.3814e-04 | 1.0418 | 0.4617 | -0.0222 |
| short_second | lora_only_frozen_base_train_adapter_only | 8.0000 | second | 1,024 | 9.8131e-07 | 6.2011e-04 | 1.0192 | 0.4853 | -0.0321 |
| short_second | lora_only_frozen_base_train_adapter_head | 8.0000 | second | 1,024 | 9.9903e-07 | 6.3283e-04 | 1.0376 | 0.4656 | -0.0287 |
| short_second | mlp_moe_frozen_base_train_moe_only | 8.0000 | second | 1,024 | 9.8170e-07 | 6.2023e-04 | 1.0196 | 0.4892 | -0.0346 |
| short_second | mlp_moe_frozen_base_train_moe_head | 8.0000 | second | 1,024 | 9.8840e-07 | 6.2284e-04 | 1.0265 | 0.4902 | -0.0346 |
| short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | second | 1,024 | 9.9261e-07 | 6.2848e-04 | 1.0309 | 0.5344 | -0.0440 |
| short_second | asd_frozen_encoder_train_head | nan | second | 1,024 | 9.8925e-07 | 6.2427e-04 | 1.0274 | 0.4686 | -0.0334 |
| short_second | raw_joint | nan | second | 1,024 | 9.7993e-07 | 6.1876e-04 | 1.0177 | 0.5029 | -0.0334 |
| short_second | raw_frozen_base_train_head | nan | second | 1,024 | 9.9125e-07 | 6.2188e-04 | 1.0295 | 0.4892 | -0.0496 |
| short_second | zero | nan | minute | 1,024 | 1.5852e-06 | 7.9712e-04 | 1.0030 | 0.4936 | nan |
| short_second | asd_lora_only_frozen_base_train_adapter_head | 4.0000 | minute | 1,024 | 1.5952e-06 | 7.9845e-04 | 1.0093 | 0.5298 | 0.0212 |
| short_second | asd_mlp_moe_frozen_base_train_moe_head | 4.0000 | minute | 1,024 | 1.5868e-06 | 7.9714e-04 | 1.0040 | 0.5142 | 0.0357 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 4.0000 | minute | 1,024 | 1.5832e-06 | 8.0242e-04 | 1.0018 | 0.5083 | 0.0223 |
| short_second | lora_only_frozen_base_train_adapter_only | 4.0000 | minute | 1,024 | 1.5900e-06 | 7.9873e-04 | 1.0060 | 0.5024 | 0.0106 |
| short_second | lora_only_frozen_base_train_adapter_head | 4.0000 | minute | 1,024 | 1.5861e-06 | 7.9935e-04 | 1.0036 | 0.4985 | 0.0219 |
| short_second | mlp_moe_frozen_base_train_moe_only | 4.0000 | minute | 1,024 | 1.5924e-06 | 7.9819e-04 | 1.0076 | 0.5318 | 0.0148 |
| short_second | mlp_moe_frozen_base_train_moe_head | 4.0000 | minute | 1,024 | 1.5916e-06 | 7.9839e-04 | 1.0070 | 0.5200 | 9.6656e-03 |
| short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | minute | 1,024 | 1.5871e-06 | 7.9857e-04 | 1.0042 | 0.4985 | 0.0136 |
| short_second | asd_lora_only_frozen_base_train_adapter_head | 8.0000 | minute | 1,024 | 1.5949e-06 | 7.9853e-04 | 1.0091 | 0.5249 | 0.0266 |
| short_second | asd_mlp_moe_frozen_base_train_moe_head | 8.0000 | minute | 1,024 | 1.5876e-06 | 7.9957e-04 | 1.0045 | 0.5044 | 0.0192 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | minute | 1,024 | 1.5780e-06 | 7.9950e-04 | 0.9985 | 0.5152 | 0.0480 |
| short_second | lora_only_frozen_base_train_adapter_only | 8.0000 | minute | 1,024 | 1.5884e-06 | 7.9829e-04 | 1.0050 | 0.4927 | 0.0155 |
| short_second | lora_only_frozen_base_train_adapter_head | 8.0000 | minute | 1,024 | 1.6036e-06 | 8.0010e-04 | 1.0146 | 0.5249 | 0.0184 |
| short_second | mlp_moe_frozen_base_train_moe_only | 8.0000 | minute | 1,024 | 1.5923e-06 | 7.9810e-04 | 1.0075 | 0.5308 | 0.0162 |
| short_second | mlp_moe_frozen_base_train_moe_head | 8.0000 | minute | 1,024 | 1.5921e-06 | 7.9759e-04 | 1.0073 | 0.5415 | 0.0271 |
| short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | minute | 1,024 | 1.5967e-06 | 7.9888e-04 | 1.0103 | 0.5259 | 0.0155 |
| short_second | asd_frozen_encoder_train_head | nan | minute | 1,024 | 1.5952e-06 | 7.9830e-04 | 1.0094 | 0.5357 | 0.0394 |
| short_second | raw_joint | nan | minute | 1,024 | 1.6069e-06 | 8.0126e-04 | 1.0167 | 0.5064 | 0.0179 |
| short_second | raw_frozen_base_train_head | nan | minute | 1,024 | 1.5855e-06 | 7.9853e-04 | 1.0032 | 0.5044 | 0.0306 |
| short_second | zero | nan | hour | 200.0000 | 4.6856e-05 | 4.5733e-03 | 1.0000 | 0.4750 | nan |
| short_second | asd_lora_only_frozen_base_train_adapter_head | 4.0000 | hour | 200.0000 | 4.0232e-05 | 4.1631e-03 | 0.8586 | 0.7050 | 0.5438 |
| short_second | asd_mlp_moe_frozen_base_train_moe_head | 4.0000 | hour | 200.0000 | 4.0535e-05 | 4.2041e-03 | 0.8651 | 0.6450 | 0.5422 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 4.0000 | hour | 200.0000 | 3.9264e-05 | 4.1487e-03 | 0.8380 | 0.6350 | 0.5517 |
| short_second | lora_only_frozen_base_train_adapter_only | 4.0000 | hour | 200.0000 | 4.3418e-05 | 4.3727e-03 | 0.9266 | 0.6500 | 0.4853 |
| short_second | lora_only_frozen_base_train_adapter_head | 4.0000 | hour | 200.0000 | 3.9775e-05 | 4.1478e-03 | 0.8489 | 0.7000 | 0.5479 |
| short_second | mlp_moe_frozen_base_train_moe_only | 4.0000 | hour | 200.0000 | 4.3350e-05 | 4.3580e-03 | 0.9252 | 0.6800 | 0.4733 |
| short_second | mlp_moe_frozen_base_train_moe_head | 4.0000 | hour | 200.0000 | 4.0341e-05 | 4.1782e-03 | 0.8610 | 0.6950 | 0.5377 |
| short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | hour | 200.0000 | 3.8995e-05 | 4.0863e-03 | 0.8322 | 0.6950 | 0.5437 |
| short_second | asd_lora_only_frozen_base_train_adapter_head | 8.0000 | hour | 200.0000 | 4.0597e-05 | 4.1914e-03 | 0.8664 | 0.6800 | 0.5371 |
| short_second | asd_mlp_moe_frozen_base_train_moe_head | 8.0000 | hour | 200.0000 | 3.9241e-05 | 4.1067e-03 | 0.8375 | 0.7000 | 0.5500 |
| short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | hour | 200.0000 | 3.9368e-05 | 4.1164e-03 | 0.8402 | 0.6750 | 0.5361 |
| short_second | lora_only_frozen_base_train_adapter_only | 8.0000 | hour | 200.0000 | 4.3033e-05 | 4.3448e-03 | 0.9184 | 0.6400 | 0.4995 |
| short_second | lora_only_frozen_base_train_adapter_head | 8.0000 | hour | 200.0000 | 3.9952e-05 | 4.1641e-03 | 0.8527 | 0.6700 | 0.5445 |
| short_second | mlp_moe_frozen_base_train_moe_only | 8.0000 | hour | 200.0000 | 4.2991e-05 | 4.3323e-03 | 0.9175 | 0.7050 | 0.4928 |
| short_second | mlp_moe_frozen_base_train_moe_head | 8.0000 | hour | 200.0000 | 4.0676e-05 | 4.2093e-03 | 0.8681 | 0.6550 | 0.5427 |
| short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | hour | 200.0000 | 3.8096e-05 | 4.0485e-03 | 0.8130 | 0.7150 | 0.5508 |
| short_second | asd_frozen_encoder_train_head | nan | hour | 200.0000 | 4.1026e-05 | 4.2233e-03 | 0.8756 | 0.7000 | 0.5338 |
| short_second | raw_joint | nan | hour | 200.0000 | 4.3795e-05 | 4.3949e-03 | 0.9347 | 0.6000 | 0.4649 |
| short_second | raw_frozen_base_train_head | nan | hour | 200.0000 | 4.0853e-05 | 4.2104e-03 | 0.8719 | 0.6850 | 0.5349 |

## 2. Full Result

本次没有运行 full；如 small 结果出现明确候选，再用 `--run-adapter-ablation-full` 补跑。

## 3. Diagnostics

| round | patch_preset | training_regime | adapter_rank | scale | mean_abs_delta | router_entropy | router_balance_loss | expert_prob_0 | expert_prob_1 | expert_prob_2 | expert_prob_3 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | 4.0000 | hour | 0.1015 | 0.4712 | 0.0636 | 0.4534 | 0.5466 | 0.0000e+00 | 0.0000e+00 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | 4.0000 | minute | 0.0266 | 0.4887 | 0.0617 | 0.5137 | 3.5184e-03 | 0.4828 | 0.0000e+00 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | 4.0000 | second | 0.0287 | 0.4912 | 0.0617 | 0.5522 | 6.8819e-04 | 0.4378 | 9.3275e-03 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | hour | 0.0998 | 0.4686 | 0.0647 | 0.5668 | 0.4332 | 0.0000e+00 | 0.0000e+00 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | minute | 0.0285 | 0.4213 | 0.0849 | 0.7119 | 0.0000e+00 | 0.2878 | 2.7726e-04 |
| round1_small | short_second | asd_lora_moe_frozen_base_train_adapters_head | 8.0000 | second | 0.0116 | 0.4916 | 0.0631 | 0.4498 | 5.2643e-04 | 0.5477 | 1.9059e-03 |
| round1_small | short_second | asd_lora_only_frozen_base_train_adapter_head | 4.0000 | hour | 0.0327 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | asd_lora_only_frozen_base_train_adapter_head | 4.0000 | minute | 0.0241 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | asd_lora_only_frozen_base_train_adapter_head | 4.0000 | second | 0.0167 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | asd_lora_only_frozen_base_train_adapter_head | 8.0000 | hour | 0.0229 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | asd_lora_only_frozen_base_train_adapter_head | 8.0000 | minute | 0.0248 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | asd_lora_only_frozen_base_train_adapter_head | 8.0000 | second | 0.0159 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | asd_mlp_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.0522 | 0.3232 | 0.1158 | 0.1710 | 0.8271 | 9.5280e-04 | 9.3977e-04 |
| round1_small | short_second | asd_mlp_moe_frozen_base_train_moe_head | 4.0000 | minute | 7.5084e-03 | 0.4938 | 0.0601 | 0.0104 | 1.2197e-04 | 0.5121 | 0.4774 |
| round1_small | short_second | asd_mlp_moe_frozen_base_train_moe_head | 4.0000 | second | 5.3252e-03 | 0.4922 | 0.0632 | 0.4491 | 3.8878e-04 | 0.5486 | 1.8874e-03 |
| round1_small | short_second | asd_mlp_moe_frozen_base_train_moe_head | 8.0000 | hour | 0.0470 | 0.1313 | 0.1653 | 0.0465 | 0.9535 | 0.0000e+00 | 0.0000e+00 |
| round1_small | short_second | asd_mlp_moe_frozen_base_train_moe_head | 8.0000 | minute | 0.0296 | 0.4843 | 0.0419 | 0.5180 | 0.3794 | 0.0471 | 0.0555 |
| round1_small | short_second | asd_mlp_moe_frozen_base_train_moe_head | 8.0000 | second | 0.0193 | 0.4785 | 0.0671 | 0.3899 | 1.1353e-03 | 0.6054 | 3.5567e-03 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.0912 | 0.4691 | 0.0631 | 0.5359 | 0.4641 | 0.0000e+00 | 0.0000e+00 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | minute | 0.0254 | 0.4401 | 0.0779 | 0.6760 | 1.2394e-04 | 0.3237 | 2.0685e-04 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 4.0000 | second | 0.0240 | 0.4740 | 0.0640 | 0.6181 | 7.1630e-04 | 0.3510 | 0.0302 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | hour | 0.1180 | 0.4693 | 0.0625 | 0.5057 | 0.4943 | 0.0000e+00 | 0.0000e+00 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | minute | 0.0524 | 0.4876 | 0.0504 | 0.3670 | 0.0000e+00 | 0.0750 | 0.5580 |
| round1_small | short_second | lora_moe_frozen_base_train_moe_head | 8.0000 | second | 0.0126 | 0.4898 | 0.0636 | 0.4336 | 8.5212e-04 | 0.5626 | 2.9436e-03 |
| round1_small | short_second | lora_only_frozen_base_train_adapter_head | 4.0000 | hour | 0.0425 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_head | 4.0000 | minute | 0.0288 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_head | 4.0000 | second | 0.0123 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_head | 8.0000 | hour | 0.0292 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_head | 8.0000 | minute | 0.0434 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_head | 8.0000 | second | 0.0192 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_only | 4.0000 | hour | 0.1314 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_only | 4.0000 | minute | 0.0958 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_only | 4.0000 | second | 0.0330 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_only | 8.0000 | hour | 0.1025 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_only | 8.0000 | minute | 0.0842 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | lora_only_frozen_base_train_adapter_only | 8.0000 | second | 0.0366 | nan | nan | nan | nan | nan | nan |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_head | 4.0000 | hour | 0.0288 | 0.4755 | 0.0623 | 1.3948e-03 | 0.4833 | 0.5153 | 0.0000e+00 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_head | 4.0000 | minute | 7.9390e-03 | 0.4765 | 0.0689 | 2.8937e-04 | 0.0000e+00 | 0.6138 | 0.3859 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_head | 4.0000 | second | 6.1690e-03 | 0.4918 | 0.0631 | 0.4482 | 7.6540e-04 | 0.5492 | 1.8753e-03 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_head | 8.0000 | hour | 0.0116 | 0.4556 | 0.0753 | 0.6608 | 0.0000e+00 | 5.7413e-04 | 0.3387 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_head | 8.0000 | minute | 0.0128 | 0.4884 | 0.0629 | 7.1641e-03 | 0.0000e+00 | 0.5628 | 0.4300 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_head | 8.0000 | second | 0.0131 | 0.4921 | 0.0597 | 0.4375 | 6.5704e-03 | 0.5453 | 0.0106 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_only | 4.0000 | hour | 0.0597 | 0.2918 | 0.1255 | 0.0000e+00 | 0.8548 | 0.1452 | 0.0000e+00 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_only | 4.0000 | minute | 0.0470 | 0.3689 | 0.1033 | 0.0000e+00 | 3.7853e-05 | 0.7858 | 0.2141 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_only | 4.0000 | second | 0.0260 | 0.4822 | 0.0667 | 0.4005 | 4.0704e-04 | 0.5970 | 2.1066e-03 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_only | 8.0000 | hour | 0.0998 | 0.0508 | 0.1809 | 0.0133 | 0.9866 | 0.0000e+00 | 1.0363e-04 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_only | 8.0000 | minute | 0.0462 | 0.4964 | 0.0501 | 0.4140 | 0.0612 | 0.0000e+00 | 0.5248 |
| round1_small | short_second | mlp_moe_frozen_base_train_moe_only | 8.0000 | second | 0.0395 | 0.4608 | 0.0729 | 0.3466 | 3.7104e-04 | 0.6492 | 3.8400e-03 |

## Interpretation

- `raw_frozen_base_train_head` 测的是 head recalibration 本身，不含 ASD/LoRA/MoE。
- `lora_only_*` 测的是共享低秩金融域适配，不含 MoE routing。
- `mlp_moe_*` 测的是 MoE routing + 更强 MLP expert，不含 LoRA low-rank 约束。
- `lora_moe_*` 是已有 LoRA expert + MoE router；`asd_*` 版本是在同一 adapter 前加入 ASD。
- 如果 `asd_*` 没有超过对应无 ASD 行，说明 ASD 不改变 adapter 消融结论。
