from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


BOOK_COLUMNS = [
    "time_id",
    "seconds_in_bucket",
    "bid_price1",
    "ask_price1",
    "bid_size1",
    "ask_size1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build FinCast-ready scalar context windows from Optiver per-second WAP1 series."
    )
    parser.add_argument(
        "--input-dir",
        default=str(WORKSPACE_ROOT / "data" / "high-frequency" / "Optiver" / "individual_book_train"),
    )
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "data" / "fincast_inputs" / "optiver_8stocks_wap1_second_fincast_inputs.npz"),
    )
    parser.add_argument(
        "--sample-csv",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "fincast_inputs"
            / "optiver_8stocks_wap1_second_fincast_sample_contexts.csv"
        ),
    )
    parser.add_argument(
        "--report-dir",
        default=str(WORKSPACE_ROOT / "report" / "optiver_fincast_inputs"),
    )
    parser.add_argument("--max-stocks", type=int, default=8)
    parser.add_argument(
        "--max-time-ids-per-stock",
        type=int,
        default=0,
        help="Use 0 for all time_ids in each selected stock.",
    )
    parser.add_argument("--seconds-per-bucket", type=int, default=600)
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--horizon-len", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--sample-rows", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_time_ids = None if args.max_time_ids_per_stock == 0 else args.max_time_ids_per_stock
    cache = build_optiver_fincast_inputs(
        input_dir=Path(args.input_dir),
        max_stocks=args.max_stocks,
        max_time_ids_per_stock=max_time_ids,
        seconds_per_bucket=args.seconds_per_bucket,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        stride=args.stride,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cache)

    sample_csv = Path(args.sample_csv)
    sample_csv.parent.mkdir(parents=True, exist_ok=True)
    write_sample_contexts_csv(cache, sample_csv, max_rows=args.sample_rows)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_report(cache, report_dir, output_path=output_path, sample_csv=sample_csv)

    print("Optiver FinCast input cache")
    print("---------------------------")
    print(f"output:             {output_path}")
    print(f"sample_csv:         {sample_csv}")
    print(f"report_dir:         {report_dir}")
    print(f"contexts:           {cache['contexts'].shape}")
    print(f"future_values:      {cache['future_values'].shape}")
    print(f"frequency:          {str(cache['data_frequency'])}")
    print(f"context_len:        {int(cache['context_len'])}")
    print(f"horizon_len:        {int(cache['horizon_len'])}")
    print(f"stocks:             {len(np.unique(cache['stock_ids']))}")
    print(f"time_ids:           {len(np.unique(cache['episode_ids']))}")
    print(f"realized_return μ/σ:{cache['realized_returns'].mean():.6g} / {cache['realized_returns'].std():.6g}")


def build_optiver_fincast_inputs(
    *,
    input_dir: Path,
    max_stocks: int,
    max_time_ids_per_stock: int | None,
    seconds_per_bucket: int,
    context_len: int,
    horizon_len: int,
    stride: int,
) -> dict[str, np.ndarray]:
    _validate_window_args(
        seconds_per_bucket=seconds_per_bucket,
        context_len=context_len,
        horizon_len=horizon_len,
        stride=stride,
    )
    stock_files = sorted(input_dir.glob("stock_*.csv"), key=_stock_file_sort_key)[:max_stocks]
    if not stock_files:
        raise ValueError(f"No stock_*.csv files found in {input_dir}.")

    contexts: list[np.ndarray] = []
    future_values: list[np.ndarray] = []
    last_values: list[np.ndarray] = []
    realized_returns: list[np.ndarray] = []
    stock_ids: list[np.ndarray] = []
    time_ids: list[np.ndarray] = []
    episode_ids: list[np.ndarray] = []
    window_end_seconds: list[np.ndarray] = []
    asset_names: list[np.ndarray] = []
    source_files: list[str] = []

    for path in stock_files:
        stock_id = _stock_file_sort_key(path)
        frame = pd.read_csv(path, usecols=BOOK_COLUMNS)
        series = per_second_wap1_series(
            frame,
            max_time_ids=max_time_ids_per_stock,
            seconds_per_bucket=seconds_per_bucket,
        )
        block = rolling_context_windows(
            series,
            stock_id=stock_id,
            context_len=context_len,
            horizon_len=horizon_len,
            stride=stride,
        )
        if block["contexts"].shape[0] == 0:
            continue

        contexts.append(block["contexts"])
        future_values.append(block["future_values"])
        last_values.append(block["last_values"])
        realized_returns.append(block["realized_returns"])
        stock_ids.append(block["stock_ids"])
        time_ids.append(block["time_ids"])
        episode_ids.append(block["episode_ids"])
        window_end_seconds.append(block["window_end_seconds"])
        asset_names.append(np.asarray([f"stock_{stock_id}"] * block["contexts"].shape[0], dtype=object))
        source_files.append(str(path))

    if not contexts:
        raise ValueError("No FinCast input windows were built.")

    return {
        "contexts": np.concatenate(contexts, axis=0).astype(np.float32),
        "future_values": np.concatenate(future_values, axis=0).astype(np.float32),
        "last_values": np.concatenate(last_values, axis=0).astype(np.float32),
        "realized_returns": np.concatenate(realized_returns, axis=0).astype(np.float32),
        "stock_ids": np.concatenate(stock_ids, axis=0).astype(np.int16),
        "time_ids": np.concatenate(time_ids, axis=0).astype(np.int64),
        "episode_ids": np.concatenate(episode_ids, axis=0).astype(np.int64),
        "window_end_seconds": np.concatenate(window_end_seconds, axis=0).astype(np.int16),
        "asset_names": np.concatenate(asset_names, axis=0),
        "source_files": np.asarray(source_files, dtype=object),
        "context_len": np.asarray(context_len, dtype=np.int64),
        "horizon_len": np.asarray(horizon_len, dtype=np.int64),
        "stride": np.asarray(stride, dtype=np.int64),
        "seconds_per_bucket": np.asarray(seconds_per_bucket, dtype=np.int64),
        "data_frequency": np.asarray("S"),
        "price_source": np.asarray("wap1"),
        "input_contract": np.asarray("FinCast model_api.forecast(list(contexts), freq=['S'])"),
    }


def per_second_wap1_series(
    frame: pd.DataFrame,
    *,
    max_time_ids: int | None,
    seconds_per_bucket: int,
) -> pd.Series:
    missing = [col for col in BOOK_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing Optiver book columns: {missing}")

    df = frame.sort_values(["time_id", "seconds_in_bucket"]).copy()
    if max_time_ids is not None:
        keep_time_ids = np.sort(df["time_id"].unique())[: int(max_time_ids)]
        df = df[df["time_id"].isin(keep_time_ids)].copy()
    df = df[
        (df["seconds_in_bucket"] >= 0)
        & (df["seconds_in_bucket"] < int(seconds_per_bucket))
    ].copy()
    if df.empty:
        return pd.Series(dtype=np.float32)

    eps = 1e-12
    df["wap1"] = (
        df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]
    ) / (df["bid_size1"] + df["ask_size1"]).clip(lower=eps)
    last_wap = df.groupby(["time_id", "seconds_in_bucket"], sort=True)["wap1"].last()
    time_ids = np.sort(df["time_id"].unique()).astype(np.int64)
    full_index = pd.MultiIndex.from_product(
        [time_ids, np.arange(int(seconds_per_bucket), dtype=np.int16)],
        names=["time_id", "seconds_in_bucket"],
    )
    series = last_wap.reindex(full_index)
    return series.groupby(level="time_id", group_keys=False).ffill().astype(np.float32)


def rolling_context_windows(
    series: pd.Series,
    *,
    stock_id: int,
    context_len: int,
    horizon_len: int,
    stride: int,
) -> dict[str, np.ndarray]:
    context_rows: list[np.ndarray] = []
    future_rows: list[np.ndarray] = []
    last_values: list[float] = []
    realized_returns: list[float] = []
    stock_ids: list[int] = []
    time_ids: list[int] = []
    episode_ids: list[int] = []
    window_end_seconds: list[int] = []

    if series.empty:
        return _empty_window_block(context_len=context_len, horizon_len=horizon_len)

    for time_id, group in series.groupby(level="time_id", sort=True):
        values = group.to_numpy(dtype=np.float32)
        first_end = context_len - 1
        last_end = len(values) - horizon_len - 1
        if last_end < first_end:
            continue
        for end_second in range(first_end, last_end + 1, stride):
            context = values[end_second - context_len + 1 : end_second + 1]
            future = values[end_second + 1 : end_second + 1 + horizon_len]
            if not np.isfinite(context).all() or not np.isfinite(future).all():
                continue
            last_value = float(context[-1])
            if abs(last_value) < 1e-8:
                continue
            context_rows.append(context)
            future_rows.append(future)
            last_values.append(last_value)
            realized_returns.append(float(future[-1] / last_value - 1.0))
            stock_ids.append(stock_id)
            time_ids.append(int(time_id))
            episode_ids.append(int(stock_id * 1_000_000 + int(time_id)))
            window_end_seconds.append(int(end_second))

    if not context_rows:
        return _empty_window_block(context_len=context_len, horizon_len=horizon_len)
    return {
        "contexts": np.stack(context_rows, axis=0).astype(np.float32),
        "future_values": np.stack(future_rows, axis=0).astype(np.float32),
        "last_values": np.asarray(last_values, dtype=np.float32),
        "realized_returns": np.asarray(realized_returns, dtype=np.float32),
        "stock_ids": np.asarray(stock_ids, dtype=np.int16),
        "time_ids": np.asarray(time_ids, dtype=np.int64),
        "episode_ids": np.asarray(episode_ids, dtype=np.int64),
        "window_end_seconds": np.asarray(window_end_seconds, dtype=np.int16),
    }


def write_sample_contexts_csv(cache: dict[str, np.ndarray], path: Path, *, max_rows: int) -> None:
    n = min(max_rows, cache["contexts"].shape[0])
    contexts = cache["contexts"][:n]
    futures = cache["future_values"][:n]
    rows = {
        "asset": cache["asset_names"][:n],
        "stock_id": cache["stock_ids"][:n],
        "time_id": cache["time_ids"][:n],
        "window_end_second": cache["window_end_seconds"][:n],
        "last_value": cache["last_values"][:n],
        "realized_return_horizon": cache["realized_returns"][:n],
    }
    for i in range(contexts.shape[1]):
        rows[f"context_{i + 1:03d}"] = contexts[:, i]
    for i in range(futures.shape[1]):
        rows[f"future_{i + 1:03d}"] = futures[:, i]
    pd.DataFrame(rows).to_csv(path, index=False)


def write_report(
    cache: dict[str, np.ndarray],
    report_dir: Path,
    *,
    output_path: Path,
    sample_csv: Path,
) -> None:
    figures_dir = report_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    _plot_context_examples(cache, figures_dir / "context_examples_by_stock.png")
    _plot_context_heatmap(cache, figures_dir / "context_heatmap_sample.png")
    _plot_return_distribution(cache, figures_dir / "realized_return_distribution.png")
    _plot_window_counts(cache, figures_dir / "window_counts_by_stock.png")

    counts = pd.Series(cache["asset_names"].astype(str)).value_counts().sort_index()
    summary = pd.DataFrame(
        {
            "metric": [
                "contexts",
                "context_len",
                "horizon_len",
                "stride",
                "stocks",
                "episodes",
                "return_mean",
                "return_std",
                "return_q01",
                "return_q50",
                "return_q99",
            ],
            "value": [
                str(cache["contexts"].shape[0]),
                str(int(cache["context_len"])),
                str(int(cache["horizon_len"])),
                str(int(cache["stride"])),
                str(len(np.unique(cache["stock_ids"]))),
                str(len(np.unique(cache["episode_ids"]))),
                f"{cache['realized_returns'].mean():.8g}",
                f"{cache['realized_returns'].std():.8g}",
                f"{np.quantile(cache['realized_returns'], 0.01):.8g}",
                f"{np.quantile(cache['realized_returns'], 0.50):.8g}",
                f"{np.quantile(cache['realized_returns'], 0.99):.8g}",
            ],
        }
    )
    summary.to_csv(report_dir / "summary.csv", index=False)
    counts.rename_axis("asset").rename("windows").reset_index().to_csv(
        report_dir / "window_counts_by_stock.csv",
        index=False,
    )

    readme = report_dir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Optiver FinCast Input Preview",
                "",
                "This artifact converts the first 8 Optiver stocks into FinCast-ready scalar context windows.",
                "",
                f"- input cache: `{output_path}`",
                f"- sample CSV: `{sample_csv}`",
                f"- contexts shape: `{tuple(cache['contexts'].shape)}`",
                f"- future_values shape: `{tuple(cache['future_values'].shape)}`",
                f"- data_frequency: `{str(cache['data_frequency'])}`",
                f"- price_source: `{str(cache['price_source'])}`",
                "",
                "The direct FinCast API contract is:",
                "",
                "```python",
                "model_api.forecast([row for row in contexts], freq=[freq_value] * len(contexts))",
                "```",
                "",
                "Each row is a single-variable WAP1 price history from one stock-time_id episode.",
                "Windows never cross anonymous Optiver time_id boundaries.",
            ]
        ),
        encoding="utf-8",
    )


def _plot_context_examples(cache: dict[str, np.ndarray], path: Path) -> None:
    contexts = cache["contexts"]
    futures = cache["future_values"]
    stock_ids = cache["stock_ids"]
    unique_stocks = np.unique(stock_ids)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), constrained_layout=True, sharex=True)
    axes_flat = axes.ravel()
    for ax, stock_id in zip(axes_flat, unique_stocks, strict=False):
        idx = int(np.flatnonzero(stock_ids == stock_id)[0])
        context = contexts[idx]
        future = futures[idx]
        x_context = np.arange(-len(context) + 1, 1)
        x_future = np.arange(1, len(future) + 1)
        ax.plot(x_context, context, color="#2f6fbb", linewidth=1.8, label="FinCast context")
        ax.plot(x_future, future, color="#d65f2f", linewidth=1.8, label="held-out future")
        ax.axvline(0, color="#333333", linewidth=1.0, alpha=0.7)
        ax.set_title(f"stock_{int(stock_id)}")
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="y", useOffset=False)
    for ax in axes_flat[len(unique_stocks) :]:
        ax.axis("off")
    axes_flat[0].legend(loc="best")
    fig.suptitle("FinCast-ready WAP1 contexts with held-out future horizon")
    fig.supxlabel("seconds relative to forecast origin")
    fig.supylabel("WAP1")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_context_heatmap(cache: dict[str, np.ndarray], path: Path, *, max_rows: int = 256) -> None:
    contexts = cache["contexts"]
    n = min(max_rows, contexts.shape[0])
    if n == 0:
        return
    sample_idx = np.linspace(0, contexts.shape[0] - 1, n).round().astype(int)
    sample = contexts[sample_idx].astype(np.float64)
    mean = sample.mean(axis=1, keepdims=True)
    std = sample.std(axis=1, keepdims=True)
    normalized = (sample - mean) / np.clip(std, 1e-8, None)
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    im = ax.imshow(normalized, aspect="auto", cmap="coolwarm", vmin=-3, vmax=3)
    ax.set_title("Sample of normalized FinCast input contexts")
    ax.set_xlabel("context step")
    ax.set_ylabel("sampled window")
    fig.colorbar(im, ax=ax, label="within-window z-score")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_return_distribution(cache: dict[str, np.ndarray], path: Path) -> None:
    returns = cache["realized_returns"].astype(np.float64)
    fig, ax = plt.subplots(figsize=(10, 5.5), constrained_layout=True)
    lo, hi = np.quantile(returns, [0.005, 0.995])
    ax.hist(returns, bins=120, range=(lo, hi), color="#4c7f7b", alpha=0.85)
    ax.axvline(0.0, color="#222222", linewidth=1.0)
    ax.set_title("Realized return over held-out horizon")
    ax.set_xlabel("future[-1] / context[-1] - 1")
    ax.set_ylabel("window count")
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_window_counts(cache: dict[str, np.ndarray], path: Path) -> None:
    counts = pd.Series(cache["asset_names"].astype(str)).value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(counts.index, counts.values, color="#7a5c99")
    ax.set_title("FinCast input windows by stock")
    ax.set_xlabel("asset")
    ax.set_ylabel("windows")
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _empty_window_block(*, context_len: int, horizon_len: int) -> dict[str, np.ndarray]:
    return {
        "contexts": np.empty((0, context_len), dtype=np.float32),
        "future_values": np.empty((0, horizon_len), dtype=np.float32),
        "last_values": np.empty((0,), dtype=np.float32),
        "realized_returns": np.empty((0,), dtype=np.float32),
        "stock_ids": np.empty((0,), dtype=np.int16),
        "time_ids": np.empty((0,), dtype=np.int64),
        "episode_ids": np.empty((0,), dtype=np.int64),
        "window_end_seconds": np.empty((0,), dtype=np.int16),
    }


def _validate_window_args(
    *,
    seconds_per_bucket: int,
    context_len: int,
    horizon_len: int,
    stride: int,
) -> None:
    if seconds_per_bucket <= 0:
        raise ValueError("seconds_per_bucket must be positive.")
    if context_len <= 0:
        raise ValueError("context_len must be positive.")
    if horizon_len <= 0:
        raise ValueError("horizon_len must be positive.")
    if stride <= 0:
        raise ValueError("stride must be positive.")
    if context_len + horizon_len > seconds_per_bucket:
        raise ValueError("context_len + horizon_len must fit within a time_id bucket.")


def _stock_file_sort_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])


if __name__ == "__main__":
    main()
