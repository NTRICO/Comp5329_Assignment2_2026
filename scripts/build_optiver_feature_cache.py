from __future__ import annotations

import argparse
from pathlib import Path
import sys


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.datasets.optiver_features import (
    build_optiver_feature_cache_from_dir,
    save_optiver_feature_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build engineered feature cache from Optiver book CSV files.")
    parser.add_argument(
        "--input-dir",
        default=str(WORKSPACE_ROOT / "data" / "high-frequency" / "Optiver" / "individual_book_train"),
    )
    parser.add_argument(
        "--output",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_optiver_hf_feature_cache.npz"),
    )
    parser.add_argument("--max-stocks", type=int, default=8)
    parser.add_argument("--max-time-ids-per-stock", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache = build_optiver_feature_cache_from_dir(
        args.input_dir,
        max_stocks=args.max_stocks,
        max_time_ids_per_stock=args.max_time_ids_per_stock,
    )
    output = save_optiver_feature_cache(cache, args.output)
    print(f"saved Optiver feature cache -> {output}")
    print(f"features:         {cache.features.shape}")
    print(f"realized_returns: {cache.realized_returns.shape}")
    print(f"assets:           {len(set(cache.asset_names.tolist()))}")
    print(f"feature_dim:      {len(cache.feature_names)}")
    print(f"source_files:     {len(cache.source_files)}")


if __name__ == "__main__":
    main()
