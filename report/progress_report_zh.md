# COMP5329 FinCast Position Trader 项目进度报告

日期：2026-05-14  
目的：向 tutor 说明当前项目总体规划、已经完成的工作、目前实验暴露的问题，并请求帮助判断问题主要来自数据/信号、目标函数、评估方式还是模型架构。

## 1. 项目总体目标

本项目希望在 frozen FinCast 时间序列预测模型之上，构建一个 position-aware daily trader。

FinCast 本身不做交易决策，而是给出未来价格路径的预测分布；我们在外层额外训练一个 decision model，把 FinCast 的预测分布转化为每日 ETF 仓位。整体目标是评估：

- FinCast 预测分布是否包含可交易的 next-day alpha；
- 一个带状态记忆的 trader head 能否利用该分布做 daily position decision；
- 该策略相对 buy-and-hold、Markowitz baseline、cash baseline 是否有风险收益优势；
- 如果当前结果不好，判断是 forecast signal 不足、目标函数不合适、评估方式不公平，还是模型结构问题。

当前设计中，FinCast-fts 保持 frozen，不修改上游 FinCast 源码。外层项目负责数据处理、cache 构建、trader head、训练、baseline 和 backtest。

## 2. 当前交易设定

当前交易场景是 daily transaction：

- 数据：8 个 ETF 日频 close price，包含 SPY、QQQ、IWM、TLT、GLD、EEM、USO、UUP；
- FinCast context length：过去 128 个交易日；
- FinCast forecast horizon：未来 32 个交易日；
- FinCast output：每个 rolling window 输出 `[32, 10]`，其中 10 个 channel 是 mean 和 q10 到 q90；
- 当前 trader input：默认只使用前 5 个 forecast horizon，即 `[5, 10]`；
- label：`holding_horizon = 1`，即下一交易日 return；
- decision：每天做一次仓位决策，`position_t` 持有到下一交易日；
- 仓位约束：long-only，`position in [0, 1]`；
- 调仓约束：`max_trade = 0.25`，每天最多调整 25% 仓位；
- 离散仓位：`round_step = 0.01`。

需要强调：模型不是一次预测未来 32 天后固定用同一组预测做 32 天交易。实际流程是每天 rolling 一次，用最新 128 天生成新的未来 32 天预测，再取前 5 天预测信息做下一天仓位决策。

## 3. 已完成工作

### 3.1 数据和 cache

已经完成 ETF 日频数据整理和 EDA：

- 原始 close 数据路径：`data/raw/etf_daily_close.csv`
- 数据日期范围：2007-03-01 到 2026-05-08
- 基础数据质量检查显示没有明显 missing、infinite、non-positive price 问题

已经构建 full daily FinCast cache：

- cache 路径：`data/cache/position_fincast_daily_cache.npz`
- `full_outputs = (37608, 32, 10)`
- 每个 ETF 有 4701 个 rolling samples
- `holding_horizon = 1`
- cache 用于 daily next-day trading task

### 3.2 模型和训练

当前主要 trader head 是 `encoder_transformer`：

```text
FinCast forecast patch -> TransformerEncoder -> GRUCell -> bounded position
```

当前也保留了旧的 `cnn_gru` 版本：

```text
FinCast forecast patch -> Conv1D encoder -> GRU -> position
```

训练目标是 mean-variance-turnover loss：

```text
loss = -mean_return
     + lambda_variance * variance
     + lambda_turnover * turnover
```

其中：

```text
portfolio_return_t = position_t * realized_return_t
```

所以当前目标不是直接最大化 whole-period wealth，也不是直接最大化 cumulative return，而是在 sequence 内优化平均日收益，同时惩罚方差和换手。

### 3.3 Backtest 和 sanity checks

目前已经实现：

- pooled sequence-level smoke backtest；
- cash baseline；
- buy-and-hold baseline；
- Markowitz mean-variance baseline；
- perfect-foresight oracle baseline；
- constrained oracle baseline；
- continuous evaluation sanity check；
- FinCast predicted signal 与 realized return 的 IC / directional accuracy 检查；
- financial metrics：annualized return、annualized volatility、Sharpe、max drawdown、hit rate、average position、turnover 等。

诊断脚本：

```powershell
& ".\.conda-fincast\python.exe" scripts\run_sanity_checks.py
```

输出目录：

```text
outputs/sanity_checks/
```

## 4. 当前实验结果摘要

### 4.1 主要策略表现

当前 full test steps 为 7424 个 pooled daily decisions。注意这里仍然是 smoke / diagnostic evaluation，不是最终正式 per-asset continuous live backtest。

| strategy | annualized return | Sharpe | max drawdown | average position | average turnover |
|---|---:|---:|---:|---:|---:|
| cash | 0.00% | 0.000 | 0.000 | 0.000 | 0.000 |
| buy-and-hold | 14.80% | 0.733 | 0.333 | 1.000 | 0.031 |
| Markowitz | 1.78% | 0.342 | 0.221 | 0.187 | 0.124 |
| policy | 8.25% | 0.690 | 0.202 | 0.587 | 0.019 |
| oracle_trade_cap | 61.26% | 5.354 | 0.096 | 0.488 | 0.203 |
| oracle_binary | 223.43% | 17.573 | 0.000 | 0.529 | 0.510 |

解释：

- 当前 policy 的收益低于 buy-and-hold；
- policy 的风险和 drawdown 低于 buy-and-hold；
- policy 更像一个 conservative exposure controller，而不是明显 alpha strategy；
- oracle_trade_cap 明显高于 policy，说明如果 daily direction 全对，理论空间很大；
- 但当前模型没有学到这种 next-day timing 能力。

### 4.2 同一个训练 loss 下的 constant position 检查

为了判断是不是 loss 不鼓励满仓收益最大化，我们直接用训练 loss 评估常数仓位。

| strategy | loss | mean_return | variance | turnover |
|---|---:|---:|---:|---:|
| constant p=0 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| constant p=0.25 | -0.000121 | 0.000139 | 0.000010 | 0.007812 |
| constant p=0.50 | -0.000223 | 0.000279 | 0.000040 | 0.015625 |
| constant p=0.75 | -0.000304 | 0.000418 | 0.000091 | 0.023438 |
| constant p=1.00 | -0.000365 | 0.000557 | 0.000161 | 0.031250 |
| policy | -0.000246 | 0.000322 | 0.000056 | 0.019375 |

loss 越低越好。结果显示 constant p=1.0 的 loss 比 policy 更低。因此当前问题不只是 “training loss 不喜欢 buy-and-hold”。即使在同一个 loss 下，policy 也没有赢过满仓常数策略。

### 4.3 Window-reset vs continuous evaluation

当前训练和 smoke backtest 使用 32 天 sequence，每条 sequence 初始仓位 reset。为检查 reset 是否导致不公平，我们加入 continuous evaluation：每个 ETF 的 test period 只初始化一次，然后逐日传递 previous position 和 hidden state。

| setting | mean_return | annualized return | Sharpe | max drawdown | average turnover |
|---|---:|---:|---:|---:|---:|
| window-reset policy | 0.000315 | 8.25% | 0.690 | 0.202 | 0.0194 |
| continuous policy | 0.000328 | 8.61% | 0.712 | 0.245 | 0.0007 |

continuous evaluation 有小幅改善，但不足以改变结论。因此 sequence reset 会影响结果，但不是主要原因。

### 4.4 初始仓位和 max_trade 公平性检查

我们检查了让 model 使用 `initial_position=1.0`，以及把 `max_trade` 临时放宽到 1.0 的效果。

| setting | mean_return | annualized return | Sharpe | max drawdown | average position |
|---|---:|---:|---:|---:|---:|
| buy_hold init1 | 0.000551 | 14.89% | 0.737 | 0.332 | 1.000 |
| policy init0 max_trade=0.25 | 0.000315 | 8.25% | 0.690 | 0.202 | 0.587 |
| policy init1 max_trade=0.25 | 0.000340 | 8.93% | 0.728 | 0.213 | 0.608 |
| policy init0 max_trade=1.0 | 0.000328 | 8.60% | 0.709 | 0.209 | 0.600 |

公平性调整能提升 policy 一点，尤其 initial_position=1.0 时 Sharpe 接近 buy-and-hold，但 mean return 仍明显低于 buy-and-hold。

### 4.5 FinCast signal IC 检查

为了判断 FinCast forecast patch 是否包含 next-day trading signal，我们直接检查 forecast return signal 与 realized next-day return 的 correlation 和 direction accuracy。

在 test steps 上：

| signal | IC | direction accuracy |
|---|---:|---:|
| mean_h1 | -0.019 | 49.54% |
| median_h1 | -0.025 | 50.05% |
| mean_avg_h5 | -0.013 | 49.42% |
| median_avg_h5 | -0.019 | 49.45% |

在 all cache 上，IC 也只有约 0.016 到 0.024，direction accuracy 约 49.5% 到 50.1%。

这说明当前 FinCast forecast distribution 对 next-day trading direction 的 alpha 非常弱，甚至在 test set 上略为负相关。这可能是目前模型无法打赢 buy-and-hold 的主要原因。

## 5. 当前怀疑的问题

### 问题 A：FinCast 的预测目标和交易目标可能不一致

FinCast 可能擅长预测未来价格路径的 distribution / MSE，但这不等于它对 one-day trading direction 有强 alpha。当前 IC 检查显示 next-day direction signal 很弱。因此模型可能不是没有学会，而是输入信号本身对当前交易任务不够有效。

### 问题 B：当前 trading target 可能过短

当前 label 是 next-day return，但 FinCast 输出的是未来 32 天路径分布。使用 next-day return 作为交易目标可能没有充分利用 FinCast 的 horizon information。也许应该考虑：

- 预测并交易 5-day return；
- 使用 multi-day holding horizon；
- 让 decision target 与 `input_horizon=5` 或更长 horizon 对齐；
- 或者把目标改为 sequence-level return / log wealth。

### 问题 C：当前 objective 不是 whole-period return objective

当前 loss 是 mean-variance-turnover，不是 terminal wealth 或 cumulative log return。它会鼓励更稳定、更保守的 exposure，因此 policy 学成平均约 0.59 仓位是合理的。但由于 constant p=1.0 在同一 loss 下仍更好，objective mismatch 不是唯一解释。

### 问题 D：模型架构可能不是第一优先问题

当前模型结构已经包含 TransformerEncoder 和 GRU state，也考虑 previous position。sanity checks 说明更可能的问题是信号弱和目标设定不匹配。继续换更复杂模型前，可能更应该先确认 forecast signal 和 target horizon。

### 问题 E：正式 backtest 仍需完善

目前有 continuous sanity check，但最终报告还需要更正式的 per-asset continuous backtest：

- 每个 ETF 从 test period 开始连续运行；
- hidden state 和 previous position 跨天传递；
- buy-and-hold 只在第一天建仓；
- 输出每个 asset 的 equity curve、drawdown、turnover；
- 再汇总为 portfolio-level 结果。

## 6. 希望 tutor 帮忙判断的问题

我希望 tutor 重点帮忙看以下几个问题：

1. 用 FinCast 的 one-day forecast signal 去做 next-day trading decision 是否合理？
2. 当前 IC 接近 0，是否说明这个任务本身不适合做 next-day alpha，还是我的 signal extraction 方式不对？
3. 是否应该把 target 从 next-day return 改成 5-day 或 multi-day return，以匹配 FinCast 的 forecast horizon？
4. 当前 mean-variance-turnover loss 是否适合本项目，还是应该改成 log wealth / terminal wealth objective？
5. 当前 policy 没有赢过 constant p=1.0，是模型训练不足，还是 forecast signal 不足？
6. 对于 assignment/report，是否可以把 “FinCast distribution 对 one-day trading alpha 不足” 作为一个合理的 negative result？
7. 下一步应该优先改 objective、改 target horizon，还是换 decision model 架构？

## 7. 下一步计划

如果继续推进，我建议按以下顺序：

1. 完成正式 per-asset continuous backtest，并生成 equity curve；
2. 做不同 target horizon 的实验：1-day、5-day、10-day；
3. 做更系统的 input horizon ablation：1、3、5、10、16、32；
4. 尝试 log-wealth 或 terminal wealth objective；
5. 保留 oracle upper baseline，作为“理论上限”参照；
6. 若 IC 仍接近 0，则把项目结论转为 negative result：FinCast forecast distribution 并不天然产生 next-day ETF trading alpha。

## 8. 当前阶段总结

当前项目工程链路已经基本跑通：数据、FinCast cache、trader model、training、baseline、oracle 和 sanity checks 都已经完成。

目前最大问题不是代码无法运行，而是实验结果显示：

- 当前 policy 没有打赢 buy-and-hold；
- 在训练 loss 下也没有打赢 constant p=1；
- continuous evaluation 和公平初始仓位只能小幅改善；
- FinCast forecast signal 对 next-day return 的 IC 很弱；
- 因此主要问题可能在 forecast signal 与 trading target 的 mismatch，而不是单纯模型架构错误。

希望 tutor 能帮助判断：当前方向是否应该继续优化 trader head，还是应该重新定义 trading target / objective，或者把结果整理成一个关于 FinCast forecast distribution 缺乏 one-day trading alpha 的 negative finding。
