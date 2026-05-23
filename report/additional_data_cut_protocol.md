# Additional Data Cut Protocol

## 目的

这一步把 Optiver additional data 切成更适合当前 PatchTST 三尺度任务的
true-hour cache。旧 cache 的 `time_id` 是 600 秒匿名 bucket；additional data
明确说明 `1 time_id = 1 hour`，所以更适合解释 `second / minute / hour`。

## 切法

对每个 selected stock 和 selected `time_id`：

```text
order_book_feature.csv: seconds_in_bucket 0-1799
order_book_target.csv:  seconds_in_bucket 1800-3599
        ↓
combine into one 3600-second hourly episode
        ↓
compute WAP1/WAP2, spread, imbalance, update flags, 1-second log returns
        ↓
save as existing-runner-compatible npz cache
```

生成后的三尺度含义：

| scale | 当前切法 |
| --- | --- |
| second | 同一个真实小时内的 60 个 1-second return，默认预测未来 10 秒累计 return |
| minute | 同一个真实小时内聚合 60 个 1-minute WAP level，使用 45 个 minute return 预测下一分钟 |
| hour | 按 sequential `time_id` 的真实小时级 WAP level，使用 24 个 hour return 预测下一小时 |

当前推荐 patch preset 是 `balanced_60_45_24`：

| scale | context | patch | stride |
| --- | ---: | ---: | ---: |
| second | 60 | 10 | 5 |
| minute | 45 | 9 | 4 |
| hour | 24 | 4 | 2 |

## 使用命令

```powershell
.\.conda-fincast\python.exe scripts\build_optiver_additional_second_feature_cache.py
```

默认输出：

```text
data/cache/position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz
```

小样本验证命令：

```powershell
.\.conda-fincast\python.exe scripts\build_optiver_additional_second_feature_cache.py `
  --max-stocks 3 `
  --max-time-ids-per-stock 512 `
  --output data\cache\position_optiver_additional_true_hour_second_feature_cache_3stocks_512h_validation.npz
```

## 已验证

- `py_compile` 通过。
- `3 stocks x 512 hours` 验证 cache 成功生成。
- 现有 dataset builder 可以从该 cache 生成 `second / minute / hour` 的
  `train / validation / test / zero-shot` windows。
- 该 cache 写入 `seconds_per_bucket=3600`，因此 minute scale 会自动按 60
  分钟聚合；旧 600 秒 cache 仍按 10 分钟聚合。

## 注意

这一步只是把数据协议切清楚，还没有重跑最终模型。旧报告中的结果仍来自旧
600 秒 bucket cache；如果使用 additional true-hour cache，需要重新跑主模型
和 baseline，不能直接把旧结果当成新数据协议的结论。
