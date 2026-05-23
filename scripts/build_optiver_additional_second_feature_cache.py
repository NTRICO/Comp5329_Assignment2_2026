from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.optiver_features import (  # noqa: E402
    build_optiver_additional_second_feature_cache_from_dir,
    save_optiver_feature_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a true-hour per-second cache from Optiver additional data. "
            "The builder combines order_book_feature seconds 0-1799 and "
            "order_book_target seconds 1800-3599 into one 3600-second time_id."
        )
    )
    parser.add_argument(
        "--input-dir",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "high-frequency"
            / "Optiver_additional data"
            / "Optiver_additional data"
        ),
    )
    parser.add_argument(
        "--output",
        default=str(
            WORKSPACE_ROOT
            / "data"
            / "cache"
            / "position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz"
        ),
    )
    parser.add_argument("--max-stocks", type=int, default=10)
    parser.add_argument(
        "--max-time-ids-per-stock",
        type=int,
        default=512,
        help=(
            "Default is large enough for the current hour context=32 validation/test split. "
            "Use a smaller value for cache-building tests or 0 to build all available time_ids."
        ),
    )
    parser.add_argument(
        "--stock-ids",
        default="",
        help="Optional comma-separated raw additional-data stock ids. Overrides --max-stocks selection.",
    )
    parser.add_argument("--seconds-per-bucket", type=int, default=3600)
    parser.add_argument("--chunksize", type=int, default=1_000_000)
    parser.add_argument(
        "--preserve-raw-stock-ids",
        action="store_true",
        help="Use raw additional-data stock ids in asset_names instead of dense stock_0... mapping.",
    )
    parser.add_argument(
        "--no-assume-stock-sorted",
        action="store_true",
        help="Scan entire CSV files instead of stopping once sorted stock ids pass the selected range.",
    )
    return parser.parse_args()


def parse_stock_ids(value: str) -> list[int] | None:
    if not value.strip():
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    args = parse_args()
    max_time_ids = None if args.max_time_ids_per_stock == 0 else args.max_time_ids_per_stock
    cache = build_optiver_additional_second_feature_cache_from_dir(
        args.input_dir,
        max_stocks=args.max_stocks,
        max_time_ids_per_stock=max_time_ids,
        stock_ids=parse_stock_ids(args.stock_ids),
        seconds_per_bucket=args.seconds_per_bucket,
        chunksize=args.chunksize,
        map_stock_ids_to_dense=not args.preserve_raw_stock_ids,
        assume_stock_sorted=not args.no_assume_stock_sorted,
    )
    output = save_optiver_feature_cache(cache, args.output)
    print(f"saved Optiver additional true-hour cache -> {output}")
    print(f"features:           {cache.features.shape}")
    print(f"realized_returns:   {cache.realized_returns.shape}")
    print(f"assets:             {len(set(cache.asset_names.tolist()))}")
    print(f"feature_dim:        {len(cache.feature_names)}")
    print(f"source_files:       {len(cache.source_files)}")
    print(f"seconds_in_bucket:  {cache.seconds_in_bucket.shape if cache.seconds_in_bucket is not None else None}")
    print(f"episode_ids:        {cache.episode_ids.shape if cache.episode_ids is not None else None}")
    print(f"seconds_per_bucket: {cache.seconds_per_bucket}")
    print(f"data_frequency:     {cache.data_frequency}")
    if cache.stock_id_map is not None:
        print("stock_id_map dense->raw:")
        for dense_id, raw_id in cache.stock_id_map.tolist():
            print(f"  stock_{dense_id} -> {raw_id}")


if __name__ == "__main__":
    main()
