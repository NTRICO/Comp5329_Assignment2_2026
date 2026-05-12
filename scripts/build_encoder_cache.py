from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.sources import DEFAULT_ETF_TICKERS
from src.fincast_io.encoder_features import build_fincast_encoder_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build cached frozen FinCast encoder features.")
    parser.add_argument("--csv", default=str(WORKSPACE_ROOT / "data" / "raw" / "etf_daily_close.csv"))
    parser.add_argument("--model", default=str(WORKSPACE_ROOT / "models" / "FinCast" / "v1.pth"))
    parser.add_argument("--output", default=str(WORKSPACE_ROOT / "data" / "cache" / "position_fincast_encoder_daily_cache.npz"))
    parser.add_argument("--fincast-root", default=str(WORKSPACE_ROOT / "FinCast-fts"))
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_ETF_TICKERS))
    parser.add_argument("--context-len", type=int, default=128)
    parser.add_argument("--holding-horizon", type=int, default=1)
    parser.add_argument("--frequency", default="D")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-windows-per-asset", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--backend", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--pool", choices=["last", "mean"], default="last")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = build_fincast_encoder_cache(
        csv_path=args.csv,
        model_path=args.model,
        output_path=args.output,
        fincast_root=args.fincast_root,
        tickers=args.tickers,
        context_len=args.context_len,
        holding_horizon=args.holding_horizon,
        data_frequency=args.frequency,
        stride=args.stride,
        max_windows_per_asset=args.max_windows_per_asset,
        batch_size=args.batch_size,
        backend=args.backend,
        pool=args.pool,
    )
    print(f"saved FinCast encoder cache -> {output}")


if __name__ == "__main__":
    main()
