from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen FinCast on Optiver contexts and save mean-only point forecasts."
    )
    parser.add_argument(
        "--input",
        default=str(WORKSPACE_ROOT / "data" / "fincast_inputs" / "optiver_8stocks_wap1_second_fincast_inputs.npz"),
    )
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "optiver_8stocks_fincast_mean_smoke_512.npz"),
    )
    parser.add_argument(
        "--figure",
        default=str(WORKSPACE_ROOT / "report" / "optiver_fincast_inputs" / "figures" / "fincast_mean_only_smoke.png"),
    )
    parser.add_argument("--model", default=str(WORKSPACE_ROOT / "models" / "FinCast" / "v1.pth"))
    parser.add_argument("--fincast-root", default=str(WORKSPACE_ROOT / "FinCast-fts"))
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--progress-every-batches", type=int, default=50)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=512,
        help="Default is a quick smoke subset. Use 0 to forecast every input window.",
    )
    parser.add_argument(
        "--selection",
        choices=["per_stock", "first"],
        default="per_stock",
        help="How to choose the smoke subset when max-samples is positive.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = np.load(args.input, allow_pickle=True)
    selected = select_sample_indices(
        source,
        max_samples=max(0, int(args.max_samples)),
        mode=args.selection,
    )
    contexts = source["contexts"][selected].astype(np.float32)
    horizon_len = int(source["horizon_len"])
    frequency = str(source["data_frequency"])

    mean_outputs = run_fincast_mean_forecast(
        contexts,
        model_path=Path(args.model),
        fincast_root=Path(args.fincast_root),
        backend=args.backend,
        horizon_len=horizon_len,
        context_len=int(source["context_len"]),
        frequency=frequency,
        batch_size=max(1, int(args.batch_size)),
        progress_every_batches=max(1, int(args.progress_every_batches)),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean_outputs=mean_outputs.astype(np.float32),
        full_outputs=mean_outputs[:, :, None].astype(np.float32),
        predicted_returns=(mean_outputs / source["last_values"][selected, None] - 1.0).astype(np.float32),
        last_values=source["last_values"][selected].astype(np.float32),
        realized_returns=source["realized_returns"][selected].astype(np.float32),
        asset_names=source["asset_names"][selected],
        stock_ids=source["stock_ids"][selected],
        time_ids=source["time_ids"][selected],
        episode_ids=source["episode_ids"][selected],
        window_end_seconds=source["window_end_seconds"][selected],
        selected_source_indices=selected.astype(np.int64),
        context_len=source["context_len"],
        horizon_len=source["horizon_len"],
        data_frequency=source["data_frequency"],
        price_source=source["price_source"],
        forecast_channels=np.asarray(["mean"]),
        feature_source=np.asarray("frozen_fincast_mean_forecast"),
    )

    figure_path = Path(args.figure)
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    plot_mean_only_examples(
        contexts=contexts,
        futures=source["future_values"][selected],
        mean_outputs=mean_outputs,
        stock_ids=source["stock_ids"][selected],
        figure_path=figure_path,
    )

    print("FinCast mean-only cache")
    print("-----------------------")
    print(f"input:          {Path(args.input)}")
    print(f"output:         {output_path}")
    print(f"figure:         {figure_path}")
    print(f"samples:        {len(selected)} / {source['contexts'].shape[0]}")
    print(f"mean_outputs:   {mean_outputs.shape}")
    print(f"full_outputs:   {(mean_outputs[:, :, None]).shape}")
    print(f"frequency:      {frequency}")
    print(f"channels:       ['mean']")


def select_sample_indices(source: np.lib.npyio.NpzFile, *, max_samples: int, mode: str) -> np.ndarray:
    n_total = source["contexts"].shape[0]
    if max_samples == 0 or max_samples >= n_total:
        return np.arange(n_total, dtype=np.int64)
    if mode == "first":
        return np.arange(max_samples, dtype=np.int64)
    if mode != "per_stock":
        raise ValueError(f"Unsupported selection mode: {mode}")

    stock_ids = source["stock_ids"]
    unique_stocks = np.unique(stock_ids)
    per_stock = max(1, max_samples // len(unique_stocks))
    chunks: list[np.ndarray] = []
    for stock_id in unique_stocks:
        idx = np.flatnonzero(stock_ids == stock_id)
        chunks.append(idx[:per_stock])
    selected = np.concatenate(chunks, axis=0)
    if selected.size < max_samples:
        used = np.zeros(n_total, dtype=bool)
        used[selected] = True
        filler = np.flatnonzero(~used)[: max_samples - selected.size]
        selected = np.concatenate([selected, filler], axis=0)
    return np.sort(selected[:max_samples].astype(np.int64))


def run_fincast_mean_forecast(
    contexts: np.ndarray,
    *,
    model_path: Path,
    fincast_root: Path,
    backend: str,
    horizon_len: int,
    context_len: int,
    frequency: str,
    batch_size: int,
    progress_every_batches: int,
) -> np.ndarray:
    fincast_src = fincast_root / "src"
    if str(fincast_src) not in sys.path:
        sys.path.insert(0, str(fincast_src))

    from tools.inference_utils import freq_reader_inference, get_model_api

    config = SimpleNamespace(
        model_path=str(model_path),
        backend=backend,
        horizon_len=horizon_len,
        context_len=context_len,
        num_experts=4,
        gating_top_n=2,
        load_from_compile=True,
        forecast_mode="mean",
    )
    model_api = get_model_api(config)
    freq_value = freq_reader_inference(frequency)

    chunks: list[np.ndarray] = []
    for batch_index, start in enumerate(range(0, len(contexts), batch_size), start=1):
        end = min(start + batch_size, len(contexts))
        batch_contexts = [row for row in contexts[start:end]]
        mean, _ = model_api.forecast(batch_contexts, freq=[freq_value] * len(batch_contexts))
        mean_arr = np.asarray(mean, dtype=np.float32)
        chunks.append(mean_arr[:, -horizon_len:])
        if batch_index % progress_every_batches == 0 or end == len(contexts):
            print(f"forecasted {end}/{len(contexts)}")
    return np.concatenate(chunks, axis=0).astype(np.float32)


def plot_mean_only_examples(
    *,
    contexts: np.ndarray,
    futures: np.ndarray,
    mean_outputs: np.ndarray,
    stock_ids: np.ndarray,
    figure_path: Path,
) -> None:
    unique_stocks = np.unique(stock_ids)
    chosen = []
    for stock_id in unique_stocks[:8]:
        chosen.append(int(np.flatnonzero(stock_ids == stock_id)[0]))
    if not chosen:
        return

    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True, sharex=True)
    axes_flat = axes.ravel()
    for ax, idx in zip(axes_flat, chosen, strict=False):
        context = contexts[idx]
        future = futures[idx]
        mean = mean_outputs[idx]
        x_context = np.arange(-len(context) + 1, 1)
        x_future = np.arange(1, len(future) + 1)
        ax.plot(x_context, context, color="#2f6fbb", linewidth=1.5, label="context")
        ax.plot(x_future, future, color="#111111", linewidth=1.8, label="actual future")
        ax.plot(x_future, mean, color="#d65f2f", linewidth=1.8, label="FinCast mean")
        ax.axvline(0, color="#555555", linewidth=1.0)
        ax.set_title(f"stock_{int(stock_ids[idx])}")
        ax.grid(alpha=0.25)
        ax.ticklabel_format(axis="y", useOffset=False)
    for ax in axes_flat[len(chosen) :]:
        ax.axis("off")
    axes_flat[0].legend(loc="best")
    fig.suptitle("FinCast point forecast only: mean output")
    fig.supxlabel("seconds relative to forecast origin")
    fig.supylabel("WAP1")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
