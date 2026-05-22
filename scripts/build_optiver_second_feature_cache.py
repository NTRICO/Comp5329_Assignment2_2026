from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.optiver_features import (
    build_optiver_second_feature_cache_from_dir,
    save_optiver_feature_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build per-second engineered features from Optiver book CSV files.")
    parser.add_argument(
        "--input-dir",
        default=str(WORKSPACE_ROOT / "data" / "high-frequency" / "Optiver" / "individual_book_train"),
    )
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_optiver_hf_second_feature_cache_8stocks_256t.npz"),
    )
    parser.add_argument("--max-stocks", type=int, default=8)
    parser.add_argument(
        "--max-time-ids-per-stock",
        type=int,
        default=256,
        help="Default keeps the output manageable. Use 0 to build all time_ids.",
    )
    parser.add_argument(
        "--seconds-per-bucket",
        type=int,
        default=600,
        help="Optiver book buckets are 600 seconds; this is mainly exposed for small tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    max_time_ids = None if args.max_time_ids_per_stock == 0 else args.max_time_ids_per_stock
    cache = build_optiver_second_feature_cache_from_dir(
        args.input_dir,
        max_stocks=args.max_stocks,
        max_time_ids_per_stock=max_time_ids,
        seconds_per_bucket=args.seconds_per_bucket,
    )
    output = save_optiver_feature_cache(cache, args.output)
    print(f"saved Optiver per-second feature cache -> {output}")
    print(f"features:           {cache.features.shape}")
    print(f"realized_returns:   {cache.realized_returns.shape}")
    print(f"assets:             {len(set(cache.asset_names.tolist()))}")
    print(f"feature_dim:        {len(cache.feature_names)}")
    print(f"source_files:       {len(cache.source_files)}")
    print(f"seconds_in_bucket:  {cache.seconds_in_bucket.shape if cache.seconds_in_bucket is not None else None}")
    print(f"episode_ids:        {cache.episode_ids.shape if cache.episode_ids is not None else None}")
    print(f"data_frequency:     {cache.data_frequency}")


if __name__ == "__main__":
    main()
