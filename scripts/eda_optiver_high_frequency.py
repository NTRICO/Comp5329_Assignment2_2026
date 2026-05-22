from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run EDA on the Optiver high-frequency feature cache.")
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_optiver_hf_feature_cache_8stocks.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "report" / "optiver_hf_eda"),
    )
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = Path(args.cache)
    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(cache_path, allow_pickle=True)
    feature_matrix = data["encoder_features"].astype(np.float64)
    returns = data["realized_returns"].astype(np.float64)
    asset_names = data["asset_names"].astype(str)
    time_ids = data["time_ids"].astype(np.int64)
    feature_names = data["feature_names"].astype(str).tolist()
    source_files = data["source_files"].astype(str).tolist() if "source_files" in data else []

    features = pd.DataFrame(feature_matrix, columns=feature_names)
    frame = features.copy()
    frame.insert(0, "asset", asset_names)
    frame.insert(1, "time_id", time_ids)
    frame["realized_return"] = returns

    health = build_health_summary(frame, feature_names)
    asset_summary = build_asset_summary(frame)
    feature_stats = build_feature_stats(frame, feature_names)
    feature_ic = build_feature_ic(frame, feature_names)
    autocorr = build_autocorr_summary(frame)
    target_artifact = build_target_artifact_summary(frame)
    split_summary = build_split_summary(frame, args.seq_len, args.stride, args.validation_fraction, args.test_fraction)
    source_summary = pd.DataFrame(
        {
            "source_file": source_files,
            "asset": [Path(path).stem for path in source_files],
        }
    )

    health.to_csv(output_dir / "health_summary.csv", index=False)
    asset_summary.to_csv(output_dir / "asset_summary.csv", index=False)
    feature_stats.to_csv(output_dir / "feature_stats.csv", index=False)
    feature_ic.to_csv(output_dir / "feature_ic.csv", index=False)
    autocorr.to_csv(output_dir / "return_autocorr.csv", index=False)
    target_artifact.to_csv(output_dir / "target_artifact_summary.csv", index=False)
    split_summary.to_csv(output_dir / "sequence_split_summary.csv", index=False)
    source_summary.to_csv(output_dir / "source_files.csv", index=False)

    plot_return_distribution(frame, figures_dir / "return_distribution.png")
    plot_asset_return_summary(asset_summary, figures_dir / "asset_return_summary.png")
    plot_coverage_summary(frame, figures_dir / "coverage_summary.png")
    plot_top_feature_ic(feature_ic, figures_dir / "top_feature_ic.png")
    plot_autocorr(autocorr, figures_dir / "return_autocorr.png")
    plot_target_artifact(frame, figures_dir / "target_artifact.png")

    write_report(
        output_path=output_dir / "README.md",
        cache_path=cache_path,
        figures_dir=figures_dir,
        health=health,
        asset_summary=asset_summary,
        feature_ic=feature_ic,
        autocorr=autocorr,
        target_artifact=target_artifact,
        split_summary=split_summary,
        source_files=source_files,
        feature_count=len(feature_names),
    )

    print(f"EDA report written -> {output_dir / 'README.md'}")
    print(f"Tables written     -> {output_dir}")
    print(f"Figures written    -> {figures_dir}")


def build_health_summary(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    feature_values = frame[feature_names]
    return_values = frame["realized_return"]
    rows = [
        ("rows", len(frame)),
        ("assets", frame["asset"].nunique()),
        ("time_ids", frame["time_id"].nunique()),
        ("features", len(feature_names)),
        ("feature_nan_cells", int(feature_values.isna().to_numpy().sum())),
        ("feature_inf_cells", int(np.isinf(feature_values.to_numpy()).sum())),
        ("return_nan_cells", int(return_values.isna().sum())),
        ("return_inf_cells", int(np.isinf(return_values.to_numpy()).sum())),
        ("duplicate_asset_time_ids", int(frame.duplicated(["asset", "time_id"]).sum())),
        ("return_mean", float(return_values.mean())),
        ("return_std", float(return_values.std(ddof=0))),
        ("return_positive_rate", float((return_values > 0).mean())),
        ("return_p01", float(return_values.quantile(0.01))),
        ("return_p99", float(return_values.quantile(0.99))),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def build_asset_summary(frame: pd.DataFrame) -> pd.DataFrame:
    grouped = frame.groupby("asset", sort=True)
    rows = []
    for asset, group in grouped:
        row = {
            "asset": asset,
            "rows": int(len(group)),
            "time_id_min": int(group["time_id"].min()),
            "time_id_max": int(group["time_id"].max()),
            "time_id_unique": int(group["time_id"].nunique()),
            "return_mean": float(group["realized_return"].mean()),
            "return_std": float(group["realized_return"].std(ddof=0)),
            "return_min": float(group["realized_return"].min()),
            "return_p01": float(group["realized_return"].quantile(0.01)),
            "return_p05": float(group["realized_return"].quantile(0.05)),
            "return_median": float(group["realized_return"].median()),
            "return_p95": float(group["realized_return"].quantile(0.95)),
            "return_p99": float(group["realized_return"].quantile(0.99)),
            "return_max": float(group["realized_return"].max()),
            "positive_rate": float((group["realized_return"] > 0).mean()),
        }
        for col in ["n_updates", "coverage_ratio", "seconds_span", "wap1_realized_vol", "rel_spread1_mean"]:
            if col in group.columns:
                row[f"{col}_mean"] = float(group[col].mean())
                row[f"{col}_median"] = float(group[col].median())
        rows.append(row)
    return pd.DataFrame(rows)


def build_feature_stats(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    rows = []
    for name in feature_names:
        values = frame[name]
        rows.append(
            {
                "feature": name,
                "mean": float(values.mean()),
                "std": float(values.std(ddof=0)),
                "min": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "median": float(values.median()),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
                "zero_fraction": float((values == 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def build_feature_ic(frame: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    y = frame["realized_return"]
    rows = []
    for name in feature_names:
        x = frame[name]
        if x.std(ddof=0) == 0 or y.std(ddof=0) == 0:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = x.corr(y, method="pearson")
            spearman = x.corr(y, method="spearman")
        rows.append(
            {
                "feature": name,
                "pearson_ic": float(pearson) if pd.notna(pearson) else np.nan,
                "spearman_ic": float(spearman) if pd.notna(spearman) else np.nan,
                "abs_pearson_ic": abs(float(pearson)) if pd.notna(pearson) else np.nan,
                "abs_spearman_ic": abs(float(spearman)) if pd.notna(spearman) else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("abs_pearson_ic", ascending=False)


def build_autocorr_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset, group in frame.sort_values(["asset", "time_id"]).groupby("asset"):
        returns = group["realized_return"].reset_index(drop=True)
        for lag in [1, 2, 5, 10, 20]:
            rows.append(
                {
                    "asset": asset,
                    "lag": lag,
                    "autocorr": float(returns.autocorr(lag=lag)),
                }
            )
    return pd.DataFrame(rows)


def build_target_artifact_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    y = frame["realized_return"]
    if "wap1_last" in frame.columns:
        reset_to_one = 1.0 / frame["wap1_last"].clip(lower=1e-12) - 1.0
        rows.append(
            {
                "diagnostic": "target_vs_wap1_last",
                "correlation": float(frame["wap1_last"].corr(y)),
                "interpretation": "Large negative values suggest current price level predicts a cross-time_id normalization reset.",
            }
        )
        rows.append(
            {
                "diagnostic": "target_vs_1_over_wap1_last_minus_1",
                "correlation": float(reset_to_one.corr(y)),
                "interpretation": "Large positive values suggest the target is close to a reset-to-one effect.",
            }
        )
    if "wap1_bucket_return" in frame.columns:
        rows.append(
            {
                "diagnostic": "target_vs_same_bucket_wap1_return",
                "correlation": float(frame["wap1_bucket_return"].corr(y)),
                "interpretation": "Large values indicate short-horizon continuation/reversal rather than independent next-bucket signal.",
            }
        )
    return pd.DataFrame(rows)


def build_split_summary(
    frame: pd.DataFrame,
    seq_len: int,
    stride: int,
    validation_fraction: float,
    test_fraction: float,
) -> pd.DataFrame:
    rows = []
    for asset, group in frame.groupby("asset", sort=True):
        n_rows = len(group)
        n_seq = 0 if n_rows < seq_len else len(range(0, n_rows - seq_len + 1, stride))
        n_test = max(1, int(n_seq * test_fraction)) if n_seq > 0 else 0
        n_validation = max(1, int(n_seq * validation_fraction)) if n_seq > 0 and validation_fraction > 0 else 0
        n_train = max(0, n_seq - n_validation - n_test)
        rows.append(
            {
                "asset": asset,
                "rows": n_rows,
                "seq_len": seq_len,
                "stride": stride,
                "sequences": n_seq,
                "train_sequences": n_train,
                "validation_sequences": n_validation,
                "test_sequences": n_test,
            }
        )
    total = pd.DataFrame(rows)
    rows.append(
        {
            "asset": "__total__",
            "rows": int(total["rows"].sum()),
            "seq_len": seq_len,
            "stride": stride,
            "sequences": int(total["sequences"].sum()),
            "train_sequences": int(total["train_sequences"].sum()),
            "validation_sequences": int(total["validation_sequences"].sum()),
            "test_sequences": int(total["test_sequences"].sum()),
        }
    )
    return pd.DataFrame(rows)


def plot_return_distribution(frame: pd.DataFrame, output: Path) -> None:
    returns = frame["realized_return"]
    lo, hi = returns.quantile([0.005, 0.995])
    clipped = returns.clip(lo, hi)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.hist(clipped, bins=80, color="#2F6F6D", alpha=0.85)
    ax.axvline(0, color="#111111", linewidth=1)
    ax.axvline(returns.mean(), color="#C44536", linewidth=1.5, label=f"mean={returns.mean():.2e}")
    ax.set_title("Optiver HF next-bucket return distribution")
    ax.set_xlabel("realized return (clipped 0.5%-99.5%)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_asset_return_summary(asset_summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    order = asset_summary["asset"].tolist()
    axes[0].bar(order, asset_summary["return_mean"], color="#356AA0")
    axes[0].axhline(0, color="#111111", linewidth=1)
    axes[0].set_title("Mean next-bucket return by stock")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_ylabel("mean return")

    axes[1].bar(order, asset_summary["positive_rate"], color="#6C8E3F")
    axes[1].axhline(0.5, color="#111111", linewidth=1)
    axes[1].set_title("Positive return rate by stock")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_ylim(0.45, 0.55)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_coverage_summary(frame: pd.DataFrame, output: Path) -> None:
    if "coverage_ratio" not in frame.columns:
        return
    data = [g["coverage_ratio"].to_numpy() for _, g in frame.groupby("asset", sort=True)]
    labels = [asset for asset, _ in frame.groupby("asset", sort=True)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.boxplot(data, tick_labels=labels, showfliers=False)
    ax.set_title("Order-book update coverage by stock")
    ax.set_ylabel("updates per 600-second bucket")
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_top_feature_ic(feature_ic: pd.DataFrame, output: Path) -> None:
    top = feature_ic.dropna(subset=["pearson_ic"]).head(20).iloc[::-1]
    colors = ["#2F6F6D" if value >= 0 else "#C44536" for value in top["pearson_ic"]]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top["feature"], top["pearson_ic"], color=colors)
    ax.axvline(0, color="#111111", linewidth=1)
    ax.set_title("Top 20 feature Pearson IC vs next-bucket return")
    ax.set_xlabel("Pearson correlation")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_autocorr(autocorr: pd.DataFrame, output: Path) -> None:
    pivot = autocorr.pivot(index="asset", columns="lag", values="autocorr")
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    im = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-0.1, vmax=0.1, aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Return autocorrelation by stock and lag")
    ax.set_xlabel("lag")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_target_artifact(frame: pd.DataFrame, output: Path) -> None:
    if "wap1_last" not in frame.columns:
        return
    reset_to_one = 1.0 / frame["wap1_last"].clip(lower=1e-12) - 1.0
    x = reset_to_one.to_numpy()
    y = frame["realized_return"].to_numpy()
    lo_x, hi_x = np.quantile(x, [0.005, 0.995])
    lo_y, hi_y = np.quantile(y, [0.005, 0.995])
    mask = (x >= lo_x) & (x <= hi_x) & (y >= lo_y) & (y <= hi_y)
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    ax.hexbin(x[mask], y[mask], gridsize=45, cmap="viridis", mincnt=1)
    ax.axhline(0, color="#111111", linewidth=0.8)
    ax.axvline(0, color="#111111", linewidth=0.8)
    ax.set_title("Target vs reset-to-one diagnostic")
    ax.set_xlabel("1 / wap1_last - 1")
    ax.set_ylabel("next-bucket realized return")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def write_report(
    *,
    output_path: Path,
    cache_path: Path,
    figures_dir: Path,
    health: pd.DataFrame,
    asset_summary: pd.DataFrame,
    feature_ic: pd.DataFrame,
    autocorr: pd.DataFrame,
    target_artifact: pd.DataFrame,
    split_summary: pd.DataFrame,
    source_files: list[str],
    feature_count: int,
) -> None:
    metrics = dict(zip(health["metric"], health["value"], strict=False))
    total_split = split_summary[split_summary["asset"] == "__total__"].iloc[0]
    top_ic = feature_ic.head(8)
    lag1_mean = autocorr[autocorr["lag"] == 1]["autocorr"].mean()
    artifact_corr = {
        row["diagnostic"]: row["correlation"]
        for _, row in target_artifact.iterrows()
    }
    strongest_asset = asset_summary.iloc[asset_summary["return_mean"].abs().argmax()]
    coverage_col = "coverage_ratio_mean"
    coverage_note = ""
    if coverage_col in asset_summary.columns:
        coverage_note = (
            f"- 平均 bucket 覆盖率约 `{asset_summary[coverage_col].mean():.3f}`，"
            f"不同股票中位覆盖率范围 `{asset_summary['coverage_ratio_median'].min():.3f}` 到 "
            f"`{asset_summary['coverage_ratio_median'].max():.3f}`。\n"
        )

    lines = [
        "# Optiver 高频数据 EDA",
        "",
        f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 数据来源与当前任务",
        "",
        f"- 使用 cache：`{cache_path}`",
        f"- source files：`{len(source_files)}` 个 stock CSV",
        f"- 当前训练实际使用的是聚合后的 `time_id` 级别 engineered feature cache，而不是逐笔/逐秒原始行。",
        f"- 每一行样本表示一个 stock 的一个 `time_id` bucket；目标是下一 bucket 的 WAP return。",
        "",
        "## 数据规模与健康度",
        "",
        f"- 样本行数：`{int(metrics['rows'])}`",
        f"- 股票数量：`{int(metrics['assets'])}`",
        f"- 特征数：`{feature_count}`",
        f"- 唯一 `time_id` 数：`{int(metrics['time_ids'])}`",
        f"- feature NaN / Inf：`{int(metrics['feature_nan_cells'])}` / `{int(metrics['feature_inf_cells'])}`",
        f"- return NaN / Inf：`{int(metrics['return_nan_cells'])}` / `{int(metrics['return_inf_cells'])}`",
        f"- duplicate `(asset, time_id)`：`{int(metrics['duplicate_asset_time_ids'])}`",
        "",
        "## Return 分布",
        "",
        f"- next-bucket return 均值：`{metrics['return_mean']:.3e}`",
        f"- next-bucket return 标准差：`{metrics['return_std']:.3e}`",
        f"- 正收益比例：`{metrics['return_positive_rate']:.3%}`",
        f"- 1% / 99% 分位：`{metrics['return_p01']:.3e}` / `{metrics['return_p99']:.3e}`",
        f"- 绝对均值最大的股票是 `{strongest_asset['asset']}`，mean return = `{strongest_asset['return_mean']:.3e}`。",
        "",
        f"![Return distribution](figures/{(figures_dir / 'return_distribution.png').name})",
        "",
        f"![Asset return summary](figures/{(figures_dir / 'asset_return_summary.png').name})",
        "",
        "## 重要诊断：当前 target 可能有归一化 reset artifact",
        "",
        f"- `realized_return` 与 `wap1_last` 的相关性：`{artifact_corr.get('target_vs_wap1_last', np.nan):.4f}`",
        f"- `realized_return` 与 `1 / wap1_last - 1` 的相关性：`{artifact_corr.get('target_vs_1_over_wap1_last_minus_1', np.nan):.4f}`",
        "- 这非常不像正常的弱 alpha，更像 Optiver 价格在不同 `time_id` bucket 之间被归一化后，当前 bucket 的最后价格水平会机械地预测下一 bucket 相对回到 1 附近。",
        "- 因此，目前高频 backtest 里 model 能赢 random，可能部分来自学习这个 normalization/reset 结构，而不是学到真实可交易的跨 bucket return。",
        "",
        f"![Target artifact](figures/{(figures_dir / 'target_artifact.png').name})",
        "",
        "## Microstructure 覆盖与活跃度",
        "",
        coverage_note.rstrip(),
        f"![Coverage summary](figures/{(figures_dir / 'coverage_summary.png').name})",
        "",
        "## 特征与目标的单变量关系",
        "",
        "Pearson IC 绝对值最高的特征如下：",
        "",
        "| feature | pearson IC | spearman IC |",
        "|---|---:|---:|",
    ]
    for _, row in top_ic.iterrows():
        lines.append(f"| `{row['feature']}` | `{row['pearson_ic']:.4f}` | `{row['spearman_ic']:.4f}` |")
    lines.extend(
        [
            "",
        f"![Top feature IC](figures/{(figures_dir / 'top_feature_ic.png').name})",
        "",
        "解读：这里的 IC 不是几个百分点，而是大到异常；这更支持上面的 target artifact 诊断。后续如果要把高频结果写进报告，需要先重新定义目标，例如使用同一 `time_id` 内的 realized volatility、同一 bucket 内可解释的微结构目标，或确认 `time_id` 的真实时间连续性后再构造跨 bucket return。",
            "",
            "## Return 自相关",
            "",
            f"- 各股票 lag-1 autocorrelation 平均值：`{lag1_mean:.4f}`",
            "- 若 lag autocorrelation 接近 0，说明简单 momentum/mean-reversion 规则不一定稳定；若某些股票显著偏正/偏负，可作为后续 raw-data baseline 的候选特征。",
            "",
            f"![Return autocorrelation](figures/{(figures_dir / 'return_autocorr.png').name})",
            "",
            "## 当前 train/validation/test 序列规模",
            "",
            f"- `seq_len={int(total_split['seq_len'])}`，`stride={int(total_split['stride'])}`",
            f"- total sequences：`{int(total_split['sequences'])}`",
            f"- train / validation / test：`{int(total_split['train_sequences'])}` / `{int(total_split['validation_sequences'])}` / `{int(total_split['test_sequences'])}`",
            "",
            "## 对当前 trading 实验的含义",
            "",
            "- 高频数据的 return 分布更接近零中心，少了 ETF 日频里明显的长期市场 beta；这解释了为什么 random baseline 在高频任务里平均偏弱。",
            "- 当前 cache 没有明显 NaN/Inf/重复键问题，可以继续用于 model-vs-random 对比。",
            "- 但当前 target 很可能混入价格归一化 reset artifact；因此高频 model-vs-random 胜利只能作为工程 smoke result，不能直接当作真实交易 alpha 结论。",
            "- 报告里的 random 对照若要更严谨，应继续使用相同 `max_trade`、transaction cost、seq split，并报告 random seeds 的分位数，而不是单个 seed。",
            "- 下一步最有价值的是先修正/验证高频 target，再补正式 continuous per-stock backtest，确认 sequence reset 不会夸大 model 对 random 的优势。",
            "",
            "## 输出文件",
            "",
            "- `asset_summary.csv`：按 stock 的 return、覆盖率和活跃度摘要",
            "- `feature_ic.csv`：每个特征与 next-bucket return 的 Pearson/Spearman IC",
            "- `target_artifact_summary.csv`：target 与价格归一化 reset proxy 的相关性诊断",
            "- `return_autocorr.csv`：按 stock/lag 的 return autocorrelation",
            "- `sequence_split_summary.csv`：当前 seq_len/stride 下的 train/validation/test 序列数",
        ]
    )
    output_path.write_text("\n".join(line for line in lines if line is not None) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
