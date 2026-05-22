from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.sources import DEFAULT_ETF_TICKERS
from src.fincast_io.simple_forecaster import SimpleForecastConfig, build_simple_distribution_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a rolling-Gaussian Simple-FinCast baseline cache."
    )
    parser.add_argument("--csv", default=str(WORKSPACE_ROOT / "data" / "raw" / "etf_daily_close.csv"))
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_simple_fincast_daily_cache.npz"),
    )
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_ETF_TICKERS))
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--horizon-len", type=int, default=32)
    parser.add_argument("--holding-horizon", type=int, default=1)
    parser.add_argument("--frequency", default="D")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows-per-asset", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=["random_walk", "rolling_mean", "ewma_vol"],
        default="random_walk",
        help="Naive forecast family used to fill the FinCast-shaped cache.",
    )
    parser.add_argument("--volatility-window", type=int, default=None)
    parser.add_argument("--ewma-alpha", type=float, default=0.06)
    parser.add_argument("--min-sigma", type=float, default=1e-6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_simple_distribution_cache(
        csv_path=args.csv,
        output_path=args.output,
        tickers=args.tickers,
        context_len=args.context_len,
        horizon_len=args.horizon_len,
        holding_horizon=args.holding_horizon,
        data_frequency=args.frequency,
        stride=args.stride,
        max_windows_per_asset=args.max_windows_per_asset,
        forecast_config=SimpleForecastConfig(
            mode=args.mode,
            volatility_window=args.volatility_window,
            ewma_alpha=args.ewma_alpha,
            min_sigma=args.min_sigma,
        ),
    )
    print(f"saved Simple-FinCast cache -> {output}")
    print(f"mode={args.mode} context_len={args.context_len} horizon_len={args.horizon_len}")


if __name__ == "__main__":
    main()
