from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BOOK_COLUMNS = [
    "time_id",
    "seconds_in_bucket",
    "bid_price1",
    "ask_price1",
    "bid_price2",
    "ask_price2",
    "bid_size1",
    "ask_size1",
    "bid_size2",
    "ask_size2",
    "stock_id",
]


@dataclass(frozen=True)
class OptiverFeatureCache:
    features: np.ndarray
    realized_returns: np.ndarray
    asset_names: np.ndarray
    time_ids: np.ndarray
    feature_names: list[str]
    source_files: list[str]
    seconds_in_bucket: np.ndarray | None = None
    episode_ids: np.ndarray | None = None
    seconds_per_bucket: int | None = None
    stock_id_map: np.ndarray | None = None
    data_frequency: str = "optiver_time_id"
    feature_source: str = "optiver_engineered_book_features"


def build_optiver_feature_cache_from_dir(
    input_dir: str | Path,
    *,
    max_stocks: int | None = None,
    max_time_ids_per_stock: int | None = None,
) -> OptiverFeatureCache:
    input_dir = Path(input_dir)
    stock_files = sorted(input_dir.glob("stock_*.csv"), key=_stock_file_sort_key)
    if max_stocks is not None:
        stock_files = stock_files[: int(max_stocks)]
    if not stock_files:
        raise ValueError(f"No stock_*.csv files found in {input_dir}.")

    feature_blocks: list[np.ndarray] = []
    return_blocks: list[np.ndarray] = []
    asset_blocks: list[np.ndarray] = []
    time_id_blocks: list[np.ndarray] = []
    feature_names: list[str] | None = None
    source_files: list[str] = []

    for path in stock_files:
        stock_frame = pd.read_csv(path, usecols=BOOK_COLUMNS)
        stock_features, stock_returns, stock_time_ids, names = build_optiver_stock_features(
            stock_frame,
            max_time_ids=max_time_ids_per_stock,
        )
        if len(stock_returns) == 0:
            continue
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names changed across stocks.")

        stock_id = _stock_file_sort_key(path)
        feature_blocks.append(stock_features)
        return_blocks.append(stock_returns)
        asset_blocks.append(np.asarray([f"stock_{stock_id}"] * len(stock_returns)))
        time_id_blocks.append(stock_time_ids)
        source_files.append(str(path))

    if not feature_blocks:
        raise ValueError("No usable stock feature rows were built.")

    return OptiverFeatureCache(
        features=np.concatenate(feature_blocks, axis=0).astype(np.float32),
        realized_returns=np.concatenate(return_blocks, axis=0).astype(np.float32),
        asset_names=np.concatenate(asset_blocks, axis=0),
        time_ids=np.concatenate(time_id_blocks, axis=0).astype(np.int64),
        feature_names=feature_names or [],
        source_files=source_files,
    )


def build_optiver_second_feature_cache_from_dir(
    input_dir: str | Path,
    *,
    max_stocks: int | None = None,
    max_time_ids_per_stock: int | None = 256,
    seconds_per_bucket: int = 600,
) -> OptiverFeatureCache:
    """Build a one-row-per-second cache from Optiver order-book CSV files.

    The target is the next-second WAP1 return within the same `time_id`; rows at
    the end of a `time_id` are dropped so the target never crosses the anonymous
    Optiver bucket boundary.
    """

    input_dir = Path(input_dir)
    if seconds_per_bucket <= 1:
        raise ValueError("seconds_per_bucket must be greater than 1.")
    stock_files = sorted(input_dir.glob("stock_*.csv"), key=_stock_file_sort_key)
    if max_stocks is not None:
        stock_files = stock_files[: int(max_stocks)]
    if not stock_files:
        raise ValueError(f"No stock_*.csv files found in {input_dir}.")

    feature_blocks: list[np.ndarray] = []
    return_blocks: list[np.ndarray] = []
    asset_blocks: list[np.ndarray] = []
    time_id_blocks: list[np.ndarray] = []
    second_blocks: list[np.ndarray] = []
    episode_blocks: list[np.ndarray] = []
    feature_names: list[str] | None = None
    source_files: list[str] = []

    for path in stock_files:
        stock_frame = pd.read_csv(path, usecols=BOOK_COLUMNS)
        stock_id = _stock_file_sort_key(path)
        (
            stock_features,
            stock_returns,
            stock_time_ids,
            stock_seconds,
            names,
        ) = build_optiver_stock_second_features(
            stock_frame,
            stock_id=stock_id,
            max_time_ids=max_time_ids_per_stock,
            seconds_per_bucket=seconds_per_bucket,
        )
        if len(stock_returns) == 0:
            continue
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names changed across stocks.")

        feature_blocks.append(stock_features)
        return_blocks.append(stock_returns)
        asset_blocks.append(np.asarray([f"stock_{stock_id}"] * len(stock_returns)))
        time_id_blocks.append(stock_time_ids)
        second_blocks.append(stock_seconds)
        episode_blocks.append((stock_id * 1_000_000 + stock_time_ids).astype(np.int64))
        source_files.append(str(path))

    if not feature_blocks:
        raise ValueError("No usable per-second feature rows were built.")

    return OptiverFeatureCache(
        features=np.concatenate(feature_blocks, axis=0).astype(np.float32),
        realized_returns=np.concatenate(return_blocks, axis=0).astype(np.float32),
        asset_names=np.concatenate(asset_blocks, axis=0),
        time_ids=np.concatenate(time_id_blocks, axis=0).astype(np.int64),
        seconds_in_bucket=np.concatenate(second_blocks, axis=0).astype(np.int16),
        episode_ids=np.concatenate(episode_blocks, axis=0).astype(np.int64),
        seconds_per_bucket=int(seconds_per_bucket),
        feature_names=feature_names or [],
        source_files=source_files,
        data_frequency="optiver_second",
        feature_source="optiver_second_engineered_book_features",
    )


def build_optiver_additional_second_feature_cache_from_dir(
    input_dir: str | Path,
    *,
    max_stocks: int | None = 10,
    max_time_ids_per_stock: int | None = 512,
    stock_ids: list[int] | None = None,
    seconds_per_bucket: int = 3600,
    chunksize: int = 1_000_000,
    map_stock_ids_to_dense: bool = True,
    assume_stock_sorted: bool = True,
) -> OptiverFeatureCache:
    """Build a true-hour per-second cache from Optiver additional data.

    The additional data stores each hour in two files: `order_book_feature`
    contains seconds 0-1799 and `order_book_target` contains seconds
    1800-3599. This builder combines them back into one hour-long episode so
    the existing second/minute/hour PatchTST runners can slice windows without
    crossing an hourly `time_id` boundary.
    """

    input_dir = Path(input_dir)
    feature_path = input_dir / "order_book_feature.csv"
    target_path = input_dir / "order_book_target.csv"
    train_path = input_dir / "train.csv"
    if not feature_path.exists() or not target_path.exists():
        raise ValueError(f"Missing order_book_feature.csv/order_book_target.csv in {input_dir}.")
    if seconds_per_bucket <= 1:
        raise ValueError("seconds_per_bucket must be greater than 1.")

    selected_raw_stock_ids = _select_additional_stock_ids(
        train_path=train_path,
        stock_ids=stock_ids,
        max_stocks=max_stocks,
    )
    selected_time_ids_by_stock = _select_additional_time_ids_by_stock(
        train_path=train_path,
        stock_ids=selected_raw_stock_ids,
        max_time_ids_per_stock=max_time_ids_per_stock,
    )
    dense_by_raw = {
        raw_stock_id: dense_id if map_stock_ids_to_dense else raw_stock_id
        for dense_id, raw_stock_id in enumerate(selected_raw_stock_ids)
    }

    frames_by_stock = _read_additional_order_book_frames(
        paths=[feature_path, target_path],
        selected_time_ids_by_stock=selected_time_ids_by_stock,
        chunksize=chunksize,
        assume_stock_sorted=assume_stock_sorted,
    )

    feature_blocks: list[np.ndarray] = []
    return_blocks: list[np.ndarray] = []
    asset_blocks: list[np.ndarray] = []
    time_id_blocks: list[np.ndarray] = []
    second_blocks: list[np.ndarray] = []
    episode_blocks: list[np.ndarray] = []
    feature_names: list[str] | None = None

    for raw_stock_id in selected_raw_stock_ids:
        parts = frames_by_stock.get(raw_stock_id, [])
        if not parts:
            continue
        stock_frame = pd.concat(parts, ignore_index=True)
        dense_stock_id = dense_by_raw[raw_stock_id]
        (
            stock_features,
            stock_returns,
            stock_time_ids,
            stock_seconds,
            names,
        ) = build_optiver_stock_second_features(
            stock_frame,
            stock_id=int(dense_stock_id),
            max_time_ids=None,
            seconds_per_bucket=seconds_per_bucket,
        )
        if len(stock_returns) == 0:
            continue
        if feature_names is None:
            feature_names = names
        elif feature_names != names:
            raise ValueError("Feature names changed across stocks.")

        feature_blocks.append(stock_features)
        return_blocks.append(stock_returns)
        asset_blocks.append(np.asarray([f"stock_{dense_stock_id}"] * len(stock_returns)))
        time_id_blocks.append(stock_time_ids)
        second_blocks.append(stock_seconds)
        episode_blocks.append((int(dense_stock_id) * 1_000_000 + stock_time_ids).astype(np.int64))

    if not feature_blocks:
        raise ValueError("No usable additional-data feature rows were built.")

    stock_id_map = np.asarray(
        [[int(dense_by_raw[raw_stock_id]), int(raw_stock_id)] for raw_stock_id in selected_raw_stock_ids],
        dtype=np.int64,
    )
    return OptiverFeatureCache(
        features=np.concatenate(feature_blocks, axis=0).astype(np.float32),
        realized_returns=np.concatenate(return_blocks, axis=0).astype(np.float32),
        asset_names=np.concatenate(asset_blocks, axis=0),
        time_ids=np.concatenate(time_id_blocks, axis=0).astype(np.int64),
        seconds_in_bucket=np.concatenate(second_blocks, axis=0).astype(np.int16),
        episode_ids=np.concatenate(episode_blocks, axis=0).astype(np.int64),
        seconds_per_bucket=int(seconds_per_bucket),
        stock_id_map=stock_id_map,
        feature_names=feature_names or [],
        source_files=[str(feature_path), str(target_path), str(train_path)],
        data_frequency="optiver_additional_true_hour_second",
        feature_source="optiver_additional_combined_feature_target_book_features",
    )


def build_optiver_stock_features(
    frame: pd.DataFrame,
    *,
    max_time_ids: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    missing = [col for col in BOOK_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing Optiver book columns: {missing}")

    df = frame.sort_values(["time_id", "seconds_in_bucket"]).copy()
    if max_time_ids is not None:
        keep_time_ids = np.sort(df["time_id"].unique())[: int(max_time_ids)]
        df = df[df["time_id"].isin(keep_time_ids)].copy()

    _add_microstructure_columns(df)
    grouped = df.groupby("time_id", sort=True)

    base_cols = [
        "wap1",
        "wap2",
        "mid1",
        "mid2",
        "rel_spread1",
        "rel_spread2",
        "imbalance1",
        "imbalance2",
        "log_total_size1",
        "log_total_size2",
        "total_imbalance",
    ]
    feature_parts: list[pd.DataFrame] = []
    for agg_name, agg_func in [
        ("mean", "mean"),
        ("std", "std"),
        ("min", "min"),
        ("max", "max"),
        ("first", "first"),
        ("last", "last"),
    ]:
        part = grouped[base_cols].agg(agg_func)
        part.columns = [f"{col}_{agg_name}" for col in part.columns]
        feature_parts.append(part)

    extra = pd.DataFrame(index=feature_parts[0].index)
    extra["n_updates"] = grouped.size().astype(float)
    extra["coverage_ratio"] = extra["n_updates"] / 600.0
    extra["wap1_bucket_return"] = grouped["log_wap1"].last() - grouped["log_wap1"].first()
    extra["wap2_bucket_return"] = grouped["log_wap2"].last() - grouped["log_wap2"].first()
    extra["wap1_realized_vol"] = grouped["log_wap1"].apply(_realized_volatility)
    extra["wap2_realized_vol"] = grouped["log_wap2"].apply(_realized_volatility)
    extra["seconds_span"] = grouped["seconds_in_bucket"].max() - grouped["seconds_in_bucket"].min()
    feature_parts.append(extra)

    features = pd.concat(feature_parts, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    time_ids = features.index.to_numpy(dtype=np.int64)

    last_wap = grouped["wap1"].last().reindex(features.index).to_numpy(dtype=np.float64)
    next_last_wap = np.roll(last_wap, -1)
    realized_returns = next_last_wap / np.clip(last_wap, 1e-12, None) - 1.0

    # The final row has no next bucket return, so it cannot be used as a
    # supervised trading decision.
    features_np = features.iloc[:-1].to_numpy(dtype=np.float32)
    returns_np = realized_returns[:-1].astype(np.float32)
    time_ids_np = time_ids[:-1]
    return features_np, returns_np, time_ids_np, list(features.columns)


def build_optiver_stock_second_features(
    frame: pd.DataFrame,
    *,
    stock_id: int = 0,
    max_time_ids: int | None = 256,
    seconds_per_bucket: int = 600,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    missing = [col for col in BOOK_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Missing Optiver book columns: {missing}")
    if seconds_per_bucket <= 1:
        raise ValueError("seconds_per_bucket must be greater than 1.")

    df = frame.sort_values(["time_id", "seconds_in_bucket"]).copy()
    if max_time_ids is not None:
        keep_time_ids = np.sort(df["time_id"].unique())[: int(max_time_ids)]
        df = df[df["time_id"].isin(keep_time_ids)].copy()
    df = df[
        (df["seconds_in_bucket"] >= 0)
        & (df["seconds_in_bucket"] < int(seconds_per_bucket))
    ].copy()
    if df.empty:
        empty_features = np.empty((0, 0), dtype=np.float32)
        empty_int = np.empty((0,), dtype=np.int64)
        empty_seconds = np.empty((0,), dtype=np.int16)
        return empty_features, np.empty((0,), dtype=np.float32), empty_int, empty_seconds, []

    _add_microstructure_columns(df)
    base_cols = [
        "wap1",
        "wap2",
        "mid1",
        "mid2",
        "rel_spread1",
        "rel_spread2",
        "imbalance1",
        "imbalance2",
        "log_total_size1",
        "log_total_size2",
        "total_imbalance",
        "log_wap1",
        "log_wap2",
    ]
    grouped = df.groupby(["time_id", "seconds_in_bucket"], sort=True)
    last_state = grouped[base_cols].last()
    updates = grouped.size().rename("updates_in_second")

    time_ids = np.sort(df["time_id"].unique()).astype(np.int64)
    full_index = pd.MultiIndex.from_product(
        [time_ids, np.arange(int(seconds_per_bucket), dtype=np.int16)],
        names=["time_id", "seconds_in_bucket"],
    )
    state = last_state.reindex(full_index)
    state = state.groupby(level="time_id", group_keys=False).ffill()
    update_counts = updates.reindex(full_index).fillna(0.0).astype(float)
    state["updates_in_second"] = update_counts
    state["is_observed_update"] = (update_counts > 0.0).astype(float)
    state["second_frac"] = (
        state.index.get_level_values("seconds_in_bucket").to_numpy(dtype=np.float64)
        / float(seconds_per_bucket - 1)
    )
    state["seconds_since_update"] = _seconds_since_update_series(
        update_counts,
        seconds_per_bucket=seconds_per_bucket,
    )

    by_time = state.groupby(level="time_id", group_keys=False)
    state["wap1_log_return_1s"] = by_time["log_wap1"].diff().fillna(0.0)
    state["wap2_log_return_1s"] = by_time["log_wap2"].diff().fillna(0.0)
    next_wap1 = by_time["wap1"].shift(-1)
    realized_returns = next_wap1 / state["wap1"].clip(lower=1e-12) - 1.0

    feature_cols = [
        "wap1",
        "wap2",
        "mid1",
        "mid2",
        "rel_spread1",
        "rel_spread2",
        "imbalance1",
        "imbalance2",
        "log_total_size1",
        "log_total_size2",
        "total_imbalance",
        "updates_in_second",
        "is_observed_update",
        "seconds_since_update",
        "second_frac",
        "wap1_log_return_1s",
        "wap2_log_return_1s",
    ]
    valid = state["wap1"].notna() & next_wap1.notna() & realized_returns.notna()
    features = state.loc[valid, feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    index = features.index
    time_ids_np = index.get_level_values("time_id").to_numpy(dtype=np.int64)
    seconds_np = index.get_level_values("seconds_in_bucket").to_numpy(dtype=np.int16)
    returns_np = realized_returns.loc[valid].to_numpy(dtype=np.float32)
    return (
        features.to_numpy(dtype=np.float32),
        returns_np,
        time_ids_np,
        seconds_np,
        feature_cols,
    )


def save_optiver_feature_cache(cache: OptiverFeatureCache, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "encoder_features": cache.features,
        "realized_returns": cache.realized_returns,
        "asset_names": cache.asset_names,
        "time_ids": cache.time_ids,
        "feature_names": np.asarray(cache.feature_names),
        "source_files": np.asarray(cache.source_files),
        "data_frequency": np.asarray(cache.data_frequency),
        "feature_source": np.asarray(cache.feature_source),
    }
    if cache.seconds_in_bucket is not None:
        arrays["seconds_in_bucket"] = cache.seconds_in_bucket
    if cache.episode_ids is not None:
        arrays["episode_ids"] = cache.episode_ids
    if cache.seconds_per_bucket is not None:
        arrays["seconds_per_bucket"] = np.asarray(cache.seconds_per_bucket, dtype=np.int64)
    if cache.stock_id_map is not None:
        arrays["stock_id_map"] = cache.stock_id_map
    np.savez(
        output_path,
        **arrays,
    )
    return output_path


def _select_additional_stock_ids(
    *,
    train_path: Path,
    stock_ids: list[int] | None,
    max_stocks: int | None,
) -> list[int]:
    if stock_ids:
        selected = sorted({int(stock_id) for stock_id in stock_ids})
    else:
        if not train_path.exists():
            raise ValueError(f"Missing train.csv for stock selection: {train_path}")
        train = pd.read_csv(train_path, usecols=["stock_id"])
        selected = sorted(int(stock_id) for stock_id in train["stock_id"].dropna().unique())
        if max_stocks is not None:
            selected = selected[: int(max_stocks)]
    if not selected:
        raise ValueError("No additional-data stock ids selected.")
    return selected


def _select_additional_time_ids_by_stock(
    *,
    train_path: Path,
    stock_ids: list[int],
    max_time_ids_per_stock: int | None,
) -> dict[int, set[int]]:
    if not train_path.exists():
        raise ValueError(f"Missing train.csv for time_id selection: {train_path}")
    train = pd.read_csv(train_path, usecols=["stock_id", "time_id"])
    out: dict[int, set[int]] = {}
    for stock_id in stock_ids:
        stock_times = np.sort(train.loc[train["stock_id"] == stock_id, "time_id"].dropna().unique())
        if max_time_ids_per_stock is not None:
            stock_times = stock_times[: int(max_time_ids_per_stock)]
        out[int(stock_id)] = {int(time_id) for time_id in stock_times}
    return out


def _read_additional_order_book_frames(
    *,
    paths: list[Path],
    selected_time_ids_by_stock: dict[int, set[int]],
    chunksize: int,
    assume_stock_sorted: bool,
) -> dict[int, list[pd.DataFrame]]:
    selected_stock_ids = sorted(selected_time_ids_by_stock)
    selected_set = set(selected_stock_ids)
    max_selected_stock = max(selected_stock_ids)
    frames_by_stock: dict[int, list[pd.DataFrame]] = {stock_id: [] for stock_id in selected_stock_ids}
    dtype = {
        "stock_id": "int64",
        "time_id": "int64",
        "seconds_in_bucket": "float64",
        "bid_price1": "float64",
        "ask_price1": "float64",
        "bid_price2": "float64",
        "ask_price2": "float64",
        "bid_size1": "float64",
        "ask_size1": "float64",
        "bid_size2": "float64",
        "ask_size2": "float64",
    }
    for path in paths:
        for chunk in pd.read_csv(
            path,
            sep="\t",
            usecols=BOOK_COLUMNS,
            dtype=dtype,
            chunksize=int(chunksize),
        ):
            if assume_stock_sorted and len(chunk) and int(chunk["stock_id"].min()) > max_selected_stock:
                break
            chunk = chunk[chunk["stock_id"].isin(selected_set)].copy()
            if chunk.empty:
                continue
            chunk["seconds_in_bucket"] = chunk["seconds_in_bucket"].round().astype(np.int64)
            for stock_id, stock_chunk in chunk.groupby("stock_id", sort=False):
                keep_times = selected_time_ids_by_stock.get(int(stock_id), set())
                if not keep_times:
                    continue
                stock_chunk = stock_chunk[stock_chunk["time_id"].isin(keep_times)]
                if not stock_chunk.empty:
                    frames_by_stock[int(stock_id)].append(stock_chunk)
    return frames_by_stock


def _add_microstructure_columns(df: pd.DataFrame) -> None:
    eps = 1e-12
    df["wap1"] = (
        df["bid_price1"] * df["ask_size1"] + df["ask_price1"] * df["bid_size1"]
    ) / (df["bid_size1"] + df["ask_size1"]).clip(lower=eps)
    df["wap2"] = (
        df["bid_price2"] * df["ask_size2"] + df["ask_price2"] * df["bid_size2"]
    ) / (df["bid_size2"] + df["ask_size2"]).clip(lower=eps)
    df["mid1"] = 0.5 * (df["bid_price1"] + df["ask_price1"])
    df["mid2"] = 0.5 * (df["bid_price2"] + df["ask_price2"])
    df["rel_spread1"] = (df["ask_price1"] - df["bid_price1"]) / df["mid1"].clip(lower=eps)
    df["rel_spread2"] = (df["ask_price2"] - df["bid_price2"]) / df["mid2"].clip(lower=eps)
    df["imbalance1"] = (
        df["bid_size1"] - df["ask_size1"]
    ) / (df["bid_size1"] + df["ask_size1"]).clip(lower=eps)
    df["imbalance2"] = (
        df["bid_size2"] - df["ask_size2"]
    ) / (df["bid_size2"] + df["ask_size2"]).clip(lower=eps)
    df["log_total_size1"] = np.log1p(df["bid_size1"] + df["ask_size1"])
    df["log_total_size2"] = np.log1p(df["bid_size2"] + df["ask_size2"])
    total_bid = df["bid_size1"] + df["bid_size2"]
    total_ask = df["ask_size1"] + df["ask_size2"]
    df["total_imbalance"] = (total_bid - total_ask) / (total_bid + total_ask).clip(lower=eps)
    df["log_wap1"] = np.log(df["wap1"].clip(lower=eps))
    df["log_wap2"] = np.log(df["wap2"].clip(lower=eps))


def _realized_volatility(log_prices: pd.Series) -> float:
    diff = log_prices.diff().dropna()
    if diff.empty:
        return 0.0
    return float(np.sqrt(np.square(diff.to_numpy(dtype=np.float64)).sum()))


def _seconds_since_update_series(
    update_counts: pd.Series,
    *,
    seconds_per_bucket: int = 600,
) -> np.ndarray:
    values = []
    for _, group in update_counts.groupby(level="time_id", sort=False):
        observed = group.to_numpy() > 0.0
        last_update_second = -1
        seconds = group.index.get_level_values("seconds_in_bucket").to_numpy(dtype=np.int16)
        for second, has_update in zip(seconds, observed, strict=False):
            if has_update:
                last_update_second = int(second)
                values.append(0.0)
            elif last_update_second >= 0:
                values.append(float(int(second) - last_update_second))
            else:
                values.append(float(seconds_per_bucket))
    return np.asarray(values, dtype=np.float64)


def _stock_file_sort_key(path: Path) -> int:
    return int(path.stem.split("_")[-1])
