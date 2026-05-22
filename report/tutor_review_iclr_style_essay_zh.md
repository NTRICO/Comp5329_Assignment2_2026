# Position-Aware Trading on Frozen FinCast Forecasts: An ICLR-style Diagnostic Essay for Tutor Review

## 摘要

本项目尝试把 frozen FinCast 时间序列预测模型扩展为一个可交易的 position-aware trader。FinCast 本身只输出未来价格路径的预测分布或 hidden representation，并不直接给出交易仓位。因此，本项目在 FinCast 外层增加一个独立的 decision model：它读取 FinCast forecast patch 或 encoder features，同时记住 previous position，并输出下一步 long-only position。目标是检验一个更实际的问题：一个已经训练好的 forecasting model 是否能被有效转化为 trading policy。

目前代码层面已经完成了较完整的实验链路，包括 ETF daily data pipeline、FinCast forecast cache、FinCast encoder feature cache、Optiver high-frequency feature cache、position-aware policy heads、mean-variance-turnover training objective、baseline strategies、oracle sanity check、backtest metrics 和若干诊断脚本。工程上，这个项目已经可以支撑一个完整 report。但是，从当前结果来看，我还不能诚实地把它写成一个 strong positive result。Daily ETF 实验中，learned policy 没有稳定超过 buy-and-hold，也没有明显超过 matched random baseline；FinCast forecast signal 对 next-day return 的 IC 接近 0。Optiver high-frequency 分支虽然出现了一些 model-vs-random 的改善，但 EDA 显示当前 target 与 price normalization/reset proxy 高度相关，因此这部分结果很可能受到 target artifact 影响，不能直接解释为真实可交易 alpha。

因此，我目前更倾向于把项目写成一个 diagnostic / negative empirical study：frozen forecasting representation 并不会自然变成 profitable trading signal；如果 trading target、forecast horizon、objective function 和 backtest protocol 没有对齐，即使模型结构更复杂，也不一定能产生正向交易结果。希望 tutor review 重点帮我判断：当前项目是否还可以通过修改 target horizon、objective 或 evaluation protocol 写成 positive result；如果不行，是否可以把它整理成一个有价值的 negative result。

## 1. Introduction

Recent forecasting models for time series are often evaluated by prediction accuracy, such as whether they can reconstruct or forecast future values. However, in a trading setting, accurate forecasting and profitable decision-making are not equivalent. A forecast model may capture distributional patterns in price paths, but a trading model must decide how much position to hold, under turnover limits, transaction costs, drawdown risk, and changing market regimes.

This project is motivated by that gap. FinCast is treated as a frozen upstream forecasting model. I do not modify FinCast itself. Instead, I build a separate downstream decision layer that converts FinCast outputs into trading positions. The project therefore asks:

**Can a frozen forecasting model such as FinCast provide useful signals for a position-aware trading policy?**

More concretely, the current project investigates three questions:

1. Does the FinCast forecast distribution or encoder representation contain useful short-horizon trading signal?
2. Can a stateful decision model, which remembers previous position and recurrent state, use that signal better than simple baselines?
3. If the model fails to beat simple baselines, is the problem caused by weak forecast signal, mismatched target horizon, unsuitable loss function, unfair evaluation, or model architecture?

This framing is important because my current results are not simply “model works” or “model fails”. The codebase now gives enough diagnostics to explain why the current policy is weak. This may still be a meaningful project outcome, but it changes the paper/report narrative: instead of claiming that the model produces a profitable strategy, the stronger and more honest claim may be that converting a frozen forecaster into a trader requires careful target and evaluation design.

## 2. Project Design

The project has two main experimental branches.

### 2.1 Daily ETF Branch

The first branch uses daily ETF close prices. The current universe contains eight ETFs: SPY, QQQ, IWM, TLT, GLD, EEM, USO, and UUP. For each asset, a rolling context window is passed through frozen FinCast. FinCast produces a forecast distribution over the next 32 trading days. The cached output has shape:

```text
full_outputs = (37608, 32, 10)
```

Here, each sample contains a forecast horizon of 32 days and 10 channels, including the mean and quantile forecasts. The current trading label is next-day return:

```text
holding_horizon = 1
```

The daily branch also has an encoder-feature path, where frozen FinCast produces 1280-dimensional features:

```text
encoder_features = (37608, 1280)
```

The train/validation/test split is chronological within each ETF, rather than random. This is necessary because random splitting would leak future market regimes into training.

### 2.2 Optiver High-frequency Branch

The second branch uses Optiver high-frequency order-book data. The project builds two kinds of caches:

1. A `time_id`-level engineered feature cache, where each row represents one stock and one `time_id` bucket.
2. A per-second feature cache, where each row represents a within-bucket second-level state.

The current `time_id` engineered feature cache has:

```text
rows = 30632
stocks = 8
features = 73
```

The per-second feature cache is much larger:

```text
rows = 18353360
features = 17
```

I also build FinCast-ready WAP1 context windows from Optiver data:

```text
contexts = (428960, 128)
future_values = (428960, 32)
```

These windows do not cross anonymous Optiver `time_id` boundaries. This is important because `time_id` buckets are not necessarily continuous in real calendar time.

## 3. Model

The core model idea is to make the trader position-aware. Instead of predicting return directly, the model predicts a bounded target position. The policy always satisfies:

```text
position_t in [0, 1]
```

so the current setting is long-only. The model also has a maximum daily trade size:

```text
max_trade = 0.25 or 0.05, depending on experiment
```

This prevents the policy from unrealistically jumping between 0% and 100% exposure every step.

The current codebase contains several policy heads:

### 3.1 CNN-GRU Policy

The first model processes the FinCast forecast patch through a Conv1D residual encoder, then passes the encoded signal through a GRU:

```text
FinCast forecast patch -> Conv1D encoder -> GRU -> position
```

This is the original position-aware controller direction.

### 3.2 TransformerEncoder-GRU Policy

The stronger structured model treats the forecast horizon as a sequence of tokens:

```text
FinCast forecast patch -> TransformerEncoder -> GRUCell -> position
```

The TransformerEncoder reads the `[H, C]` forecast patch. Its pooled representation is combined with an embedding of the previous position, then passed through a GRUCell. This means the decision at time `t` depends on both the current forecast and the model’s previous state.

### 3.3 Encoder-feature Policy

The encoder-feature branch skips the forecast distribution and uses frozen FinCast encoder features directly:

```text
FinCast encoder features + previous position -> MLP -> target position
```

This branch tests whether FinCast hidden representations contain more useful trading information than the decoded forecast distribution.

## 4. Objective Function

The current training objective is mean-variance-turnover loss:

```text
loss = -mean_return
     + lambda_variance * variance
     + lambda_turnover * turnover
     + lambda_forecast_risk * forecast_risk
```

where:

```text
portfolio_return_t = position_t * realized_return_t
```

This objective is reasonable for a risk-aware trading policy, but it is not the same as directly maximizing terminal wealth or cumulative log return. It encourages the model to trade off return, variance, and turnover. Therefore, a policy trained under this objective may choose conservative exposure.

However, this does not fully explain the current underperformance. In earlier sanity checks, constant full exposure can outperform the learned policy even under the same loss. This suggests that the problem is not only that the objective is too conservative. It is more likely that the input signal and trading target are not well aligned.

## 5. Baselines and Evaluation

The project currently compares the learned policy against several baselines:

| Baseline | Purpose |
|---|---|
| Cash | zero-position lower reference |
| Buy-and-hold | always-long market exposure |
| FinCast Markowitz | closed-form mean-variance position from FinCast mean and quantile spread |
| Random position | matched random long-only policy |
| Rolling AR-GARCH-like rule | simple historical-return baseline using only past returns |
| Oracle binary | perfect foresight upper bound |
| Oracle trade-cap | perfect foresight with max-trade constraint |

The oracle baselines intentionally use future returns, so they are not valid trading strategies. They only show the upper-bound space available if direction prediction were perfect.

Evaluation metrics include mean return, volatility, Sharpe-like score, annualized return, cumulative return, max drawdown, hit rate, average position, turnover, and transaction cost. This is a useful set of metrics because it shows not only whether the model makes money, but also how it takes risk.

## 6. Current Results

### 6.1 Daily ETF Results

The daily ETF branch is currently the most important evidence against writing a strong positive result.

In the available baseline comparison, buy-and-hold is much stronger than the learned policy:

| Strategy | Cumulative Return | Sharpe-like |
|---|---:|---:|
| Buy-and-hold | 31.046 | 0.684 |
| Random trade-cap 0.05, MC100 mean | 3.083 | 0.635 |
| Policy encoder-GRU, max_trade 0.05 | 2.988 | 0.493 |
| FinCast Markowitz | 0.617 | 0.339 |
| Rolling AR-GARCH | 0.450 | 0.263 |

This table should be interpreted carefully, because the current evaluation is still a pooled sequence-level smoke backtest rather than a final live-trading backtest. However, even as a diagnostic result, it is not positive for the learned policy. The policy does not beat buy-and-hold, and it is also not clearly better than matched random position strategies.

The model behaves more like a conservative exposure controller than a strong alpha strategy. It can reduce risk and avoid some drawdown, but it does not show enough timing ability to justify a positive trading claim.

### 6.2 FinCast Signal Diagnosis

A key sanity check is whether FinCast forecasts contain next-day trading signal. The current signal IC results are weak:

| Signal | IC | Direction Accuracy |
|---|---:|---:|
| mean_h1 | -0.019 | 49.54% |
| median_h1 | -0.025 | 50.05% |
| mean_avg_h5 | -0.013 | 49.42% |
| median_avg_h5 | -0.019 | 49.45% |

These numbers are close to random direction prediction. This is probably the main reason why the policy cannot beat buy-and-hold. If the forecast patch does not contain useful next-day direction information, then a more complex trader head cannot create alpha from nothing.

This also suggests that the current target may be too short. FinCast predicts a 32-step future path distribution, but the trading label is next-day return. It may be more reasonable to test 5-day or multi-day returns, or to align the input horizon and holding horizon more carefully.

### 6.3 Optiver Engineered-feature Results

The Optiver branch gives a different picture. Some high-frequency experiments show that simple historical baselines can perform well, but the learned policy is still not consistently strong.

For the `time_id` engineered feature branch:

| Strategy | Mean Return | Sharpe-like | Cumulative Return |
|---|---:|---:|---:|
| Rolling AR-GARCH | `4.83e-05` | 1.063 | 0.327 |
| Policy encoder-GRU | `4.31e-06` | 0.025 | 0.0039 |
| Buy-and-hold | `2.30e-05` | 0.053 | -0.0069 |

Here, the rolling AR-GARCH-like rule is much stronger than the learned policy. This suggests that if the Optiver engineered-feature branch is used, the learned model must first beat simple historical-return rules, not just random baselines.

### 6.4 Optiver FinCast Mean-only Results

The FinCast mean-only smoke experiment is slightly more encouraging but still not strong enough.

| Strategy | Mean Return | Sharpe-like | Cumulative Return |
|---|---:|---:|---:|
| Buy-and-hold | `6.96e-05` | 1.285 | 0.137 |
| Policy FinCast-mean encoder | `1.54e-05` | 1.137 | 0.0289 |
| Random seeds | around `8.6e-06` to `1.4e-05` | 0.813 to 1.316 | up to 0.0262 |

The policy is slightly above many random seeds in cumulative return, but it remains far below buy-and-hold on this subset. Also, this is still a smoke-level experiment. It does not yet prove that FinCast mean forecasts provide robust tradable signal.

### 6.5 Target Artifact in Optiver Data

The biggest problem in the Optiver branch is target validity. The EDA finds very high correlations between the target and price-level reset proxies:

| Diagnostic | Correlation |
|---|---:|
| target vs `wap1_last` | -0.7088 |
| target vs `1 / wap1_last - 1` | 0.7091 |
| target vs same-bucket WAP1 return | -0.5839 |

These correlations are too large to treat as normal financial alpha. They suggest that the current cross-`time_id` target may be capturing normalization or reset structure, rather than a real tradable relationship. Therefore, even if a model beats random on this task, the result may not be meaningful.

This is the strongest reason I hesitate to write the current high-frequency results as positive. A tutor’s advice is especially needed here: should I redefine the target, switch to within-bucket prediction, or avoid using this branch as the main evidence?

## 7. Can This Be Written as a Positive Result?

At the current stage, I think the honest answer is: **not yet**.

The daily ETF branch cannot support a positive result because the learned policy does not beat simple baselines. More importantly, the FinCast forecast signal itself has near-zero IC for next-day returns. This means the failure is probably not just a model architecture issue. The input signal and the trading target may not match.

The Optiver branch also cannot yet support a clean positive result because the target artifact is too serious. If the model is exploiting a normalization reset effect, then a high Sharpe or positive cumulative return would be misleading.

However, the project can still become positive under certain changes:

1. If a multi-day target better aligned with FinCast horizon produces stronger IC.
2. If continuous per-asset backtest shows the policy beats buy-and-hold and matched random baselines after transaction costs.
3. If Optiver target is redefined to remove reset artifact and the model still beats strong baselines.
4. If the encoder-feature branch can beat both random and rolling AR-GARCH under the same max-trade and cost assumptions.

Without these improvements, the safer report narrative is diagnostic rather than positive.

## 8. What I Think the Main Problem Is

My current interpretation is that the main problem is not simply “the model is too weak”. The model architecture already includes previous position, bounded target position, and recurrent state. The more likely problems are:

### 8.1 Forecast-target mismatch

FinCast produces a 32-step future distribution, but the current ETF label is next-day return. If FinCast is better at medium-horizon path distribution than one-day directional timing, then using only next-day return may throw away its strengths.

### 8.2 Objective mismatch

The mean-variance-turnover loss is reasonable, but it is not the same as maximizing cumulative wealth. It may encourage stable exposure instead of aggressive timing. Still, because constant full exposure can outperform the policy under the same loss, objective mismatch is probably not the only issue.

### 8.3 Evaluation reset

Some current evaluations use independent non-overlapping sequences. This resets previous position and hidden state at sequence boundaries. A final trading evaluation should carry position and hidden state continuously through each asset’s test period.

### 8.4 Target artifact in high-frequency data

The Optiver branch currently has a serious target-definition risk. If this is not fixed, high-frequency results should not be used as primary evidence.

## 9. Suggested Next Experiments

Before writing the final report, I think the next experiments should be:

1. **Formal continuous ETF backtest.** Run each ETF continuously through the test period, carrying previous position and hidden state across days. Report per-asset and pooled results.
2. **Target horizon ablation.** Compare 1-day, 5-day, 10-day, and possibly mean-return-over-horizon targets.
3. **Input horizon ablation.** Compare using first 1, 5, 10, 16, and 32 FinCast forecast steps.
4. **Objective ablation.** Compare mean-variance-turnover loss against log-wealth or cumulative-return-oriented objectives.
5. **Random baseline distribution.** Report median, 25/75 quantiles, and best/worst random baselines under the same max_trade and transaction cost.
6. **Optiver target repair.** Avoid cross-`time_id` target unless its continuity is justified. Consider within-bucket next-second return, realized volatility, or another microstructure target.
7. **Architecture ablation.** Compare no-state MLP, previous-position-only MLP, GRU, and Transformer-GRU under the same data split.

These experiments would make the final report much more defensible, even if the final result remains negative.

## 10. Questions for Tutor

I would like tutor feedback on the following points:

1. Is it reasonable to use frozen FinCast forecasts for next-day trading, or is the current target horizon too short?
2. Should the main report focus on daily ETF results, Optiver high-frequency results, or both?
3. If FinCast signal IC is close to zero for next-day return, is that enough evidence to frame the current result as a negative finding?
4. Should I change the objective from mean-variance-turnover loss to log wealth or cumulative return?
5. For the Optiver branch, does the target artifact mean I should discard current high-frequency results from the main claim?
6. Is beating buy-and-hold required for this project to be considered successful, or is beating matched random and showing lower drawdown sufficient?
7. Would a diagnostic report about the difficulty of turning frozen forecasts into trading policies be acceptable, even without a positive trading result?
8. Which next experiment would be most valuable before final submission: horizon ablation, objective ablation, continuous backtest, or target repair?

## 11. Current Conclusion

The project has made substantial engineering progress. It has a working data pipeline, frozen FinCast cache builders, multiple policy heads, baseline strategies, evaluation metrics, and diagnostic scripts. This is enough to write a coherent technical report.

But the current evidence does not support a strong positive claim. The daily ETF policy does not beat buy-and-hold or matched random baselines, and the FinCast next-day signal appears weak. The Optiver branch has possible target artifact, so its positive-looking results cannot yet be trusted.

My preferred current framing is therefore:

**This project shows that a frozen forecasting model can be wrapped into a position-aware trading pipeline, but also shows that forecast representations do not automatically produce tradable alpha. The main bottleneck is likely not only model architecture, but the alignment between forecast horizon, trading target, objective function, and evaluation protocol.**

If tutor thinks this negative/diagnostic framing is acceptable, I can write the final report around this argument. If tutor thinks a positive result is still required, then the most urgent next step is to repair the target/evaluation setup and run horizon/objective ablations before making any final claim.
