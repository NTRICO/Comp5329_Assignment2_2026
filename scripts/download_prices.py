from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.sources import DEFAULT_ETF_TICKERS, download_yfinance_close, save_close_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download daily ETF close data for the position-aware controller.")
    parser.add_argument("--tickers", nargs="+", default=list(DEFAULT_ETF_TICKERS))
    parser.add_argument("--start", default="2007-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--output", default=str(WORKSPACE_ROOT / "data" / "raw" / "etf_daily_close.csv"))
    parser.add_argument("--no-auto-adjust", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = download_yfinance_close(
        args.tickers,
        start=args.start,
        end=args.end,
        auto_adjust=not args.no_auto_adjust,
    )
    output = save_close_csv(df, args.output)
    print(f"saved {len(df)} rows x {len(df.columns) - 1} assets -> {output}")
    print("columns:", ", ".join(df.columns))


if __name__ == "__main__":
    main()
