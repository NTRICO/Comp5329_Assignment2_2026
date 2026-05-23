# Current Project Progress

## 当前主线

项目当前聚焦于 **PatchTST for multi-scale intraday financial return forecasting**，不再把 FinCast trader、position controller 或 day-level 任务作为主线。

当前推荐候选模型：

```text
return window
-> per-scale normalization
-> scale-aware ASD denoising
-> scale-aware LoRA-MoE sequence adapter
-> gated residual back to raw return input
-> scale-specific patch embedding
-> shared PatchTST encoder
-> scale-specific linear head
-> future return prediction
```

解释口径：

- ASD：做 scale-aware denoising。
- LoRA-MoE：在 PatchTST 前做金融任务适配和尺度/频率 specialization。
- PatchTST：保留共享 temporal representation。
- Scale-specific head：把共享表示映射到各 scale 的 return target。

## 数据协议更新

旧实验主要使用 600 秒匿名 Optiver bucket cache，因此 `hour` 更准确地说是 low-frequency / time_id proxy。

现在新增了 additional data true-hour cache builder：

```text
scripts/build_optiver_additional_second_feature_cache.py
```

它把 additional data 中的两段 order book 拼成真实小时 episode：

```text
order_book_feature.csv: seconds_in_bucket 0-1799
order_book_target.csv:  seconds_in_bucket 1800-3599
        -> one 3600-second true-hour episode
```

本地 additional data 实际可用 10 个 dense stocks：

```text
stock_0 ... stock_9
```

当前默认 split：

```text
train stocks: 0-8
zero-shot:    9
```

默认 cache：

```text
data/cache/position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz
```

## 当前任务长度

当前推荐 patch preset 是：

```text
balanced_60_30_10
```

含义：

| scale | input context | target |
| --- | ---: | --- |
| second | 60 seconds | next 10-second cumulative return |
| minute | 30 minutes | next 1-minute return |
| hour | 10 true-hour steps | next true-hour return |

Patch 设置：

| scale | context | patch | stride |
| --- | ---: | ---: | ---: |
| second | 60 | 10 | 5 |
| minute | 30 | 5 | 2 |
| hour | 10 | 2 | 1 |

## Small Sanity Result

已在 additional true-hour cache 上跑了一个轻量 3-seed sanity check：

```text
seeds: 42, 43, 44
epochs: 2
steps/epoch: 6
train cap: 1024 per scale
eval cap: 512 per scale
models: raw PatchTST vs gated pre-ASD + LoRA-MoE
```

相对 raw PatchTST 的 test 结果：

| scale | MSE improvement | MAE improvement | Direction change |
| --- | ---: | ---: | ---: |
| second | +4.73% +/- 8.99% | +3.72% +/- 7.51% | -0.99 pp |
| minute | +2.16% +/- 7.64% | +1.26% +/- 6.20% | -0.00 pp |
| hour | -0.39% +/- 3.55% | -0.93% +/- 4.70% | -3.35 pp |

Interpretation:

- The new true-hour 60/30/10 protocol runs end-to-end.
- Second and minute show positive average error improvement, but seed variance is high.
- Hour does not yet improve under this very short training run.
- This is not a final result; it is a sanity check confirming that the new data protocol and main architecture are runnable.

Output folder:

```text
outputs/prepatch_asd_adapter_patchtst/gated_pre_asd_true_hour_60_30_10_h10_sanity_10stocks_3seed/
```

## Current Recommendation

Do not claim the 60/30/10 model is final yet. The next step should be a more realistic confirmation run:

```text
epochs: 5
steps/epoch: 30 or 50
train cap: at least 4096 per scale
seeds: 42, 43, 44
```

If hour remains weak, run a small context-length sweep inspired by FinCast:

```text
A: second 60,  minute 30,  hour 10
B: second 60,  minute 60,  hour 24
C: second 120, minute 60,  hour 48
```

The most likely issue is that `hour=10` is too short for stable low-frequency prediction.
