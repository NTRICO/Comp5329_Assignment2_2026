# ASD Utility Verdict

## 实验目的

本报告只回答一个问题：`Scale-Aware ASD` 对 PatchTST 是否真的有帮助。

对照来自 `outputs/scale_aware_asd_patchtst_ablation/`，不包含 LoRA、MoE、ASB 或 day 数据。核心比较是：

- `raw_joint`: 只训练 shared PatchTST。
- `asd_frozen_encoder_train_head`: 从 raw PatchTST checkpoint 加载，冻结 encoder 和 patch projection，只训练 ASD + scale-specific heads。

这种设置比 joint training 更能隔离 ASD 的作用，因为它测试的是：在已有 PatchTST 表征上，ASD 是否能通过输入端 scale-aware denoising 带来增益。

## Full Test 单 seed 结果

| scale | model | n | MSE | NMSE | Direction | Corr |
|---|---:|---:|---:|---:|---:|---:|
| second | raw | 263640 | 9.485918e-07 | 0.987225 | 0.530123 | 0.114268 |
| second | ASD gate=-4 | 263640 | 9.497298e-07 | 0.988410 | 0.522597 | 0.114254 |
| second | ASD gate=-2 | 263640 | 9.496059e-07 | 0.988281 | 0.521845 | 0.115537 |
| minute | raw | 3120 | 1.583550e-06 | 0.995514 | 0.499037 | 0.082794 |
| minute | ASD gate=-4 | 3120 | 1.581154e-06 | 0.994007 | 0.517020 | 0.078428 |
| minute | ASD gate=-2 | 3120 | 1.581193e-06 | 0.994032 | 0.516699 | 0.078277 |
| hour | raw | 200 | 2.693793e-05 | 0.574909 | 0.730000 | 0.663325 |
| hour | ASD gate=-4 | 200 | 2.640490e-05 | 0.563533 | 0.740000 | 0.670939 |
| hour | ASD gate=-2 | 200 | 2.639513e-05 | 0.563324 | 0.740000 | 0.671090 |

单 seed full test 显示：ASD 在 minute/hour 有帮助，但 second 上不稳定。

## Robustness 结果

Robustness 使用 seeds `42, 43, 44`，比较 `compact` preset 下 raw 与 ASD gate=-4。

| scale | Raw NMSE mean | ASD NMSE mean | ASD NMSE change | Raw Direction | ASD Direction | Direction change |
|---|---:|---:|---:|---:|---:|---:|
| second | 0.988327 | 0.987681 | -0.065% | 0.533133 | 0.530944 | -0.219 pp |
| minute | 0.994920 | 0.995348 | +0.043% | 0.521516 | 0.529544 | +0.803 pp |
| hour | 0.565408 | 0.553066 | -2.183% | 0.715000 | 0.713333 | -0.167 pp |

Robustness 显示：ASD 对 hour 的 MSE/NMSE 改善最明显；second 上只有极小 NMSE 改善，direction 略降；minute 上 NMSE 基本不变，但 direction 有提升。

## ASD 行为诊断

ASD 的平均诊断值如下：

| scale | gate_mean | tau_mean | mean_abs_delta |
|---|---:|---:|---:|
| second | 0.084978 | 9.133528 | 0.025589 |
| minute | 0.025924 | 1.930866 | 0.013547 |
| hour | 0.010338 | 8.879438 | 0.006109 |

这说明模型确实学到了 scale-dependent denoising strength：second 的 gate 最大，hour 的 gate 最小。也就是说 ASD 没有退化成完全 identity，但它整体仍然是保守的轻量模块。

## 结论

ASD 不是完全没用，但收益有限。

更准确的结论是：

> ASD is useful as a lightweight scale-aware regularizer, especially for hour-scale point forecasting, but it is not strong enough to be treated as the main source of predictive power.

按 scale 看：

- **second**: ASD 的 NMSE 略好，但 direction 略差；结论是弱收益，不稳定。
- **minute**: ASD 的 direction 明显更好，但 NMSE 基本不变；结论是改善方向性，不改善点预测。
- **hour**: ASD 的 NMSE/MSE 改善最清楚；结论是当前最支持 ASD 的 scale。

因此，报告中不应写成 “ASD significantly improves all scales”。更稳的说法是：

> Scale-aware ASD provides modest and scale-dependent gains. It helps most clearly on the hour/proxy scale, gives mixed results on minute, and only marginal effects on second. This suggests that spectral denoising is a useful inductive bias, but stronger financial time-series denoising or better task design is still needed.

## 下一步建议

如果继续研究 ASD，优先做：

1. 重新设计 second/minute 的 context-horizon，例如 `30s/60s -> 1s/5s/10s` 和 `4/6/8min -> 1min`。
2. 对比更强 denoiser：wavelet-band gate、learnable spectral filter、moving-average decomposition。
3. 保留 ASD 作为 baseline，而不是把它当作最终主贡献。
