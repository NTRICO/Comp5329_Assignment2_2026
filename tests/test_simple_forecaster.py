from __future__ import annotations

import numpy as np
import pandas as pd

from src.fincast_io.cache_builder import load_distribution_cache
from src.fincast_io.simple_forecaster import (
    SimpleForecastConfig,
    build_simple_distribution_cache,
    simple_gaussian_level_forecasts,
)


def test_random_walk_forecasts_match_fincast_shape() -> None:
    contexts = np.asarray(
        [
            [100.0, 101.0, 100.5, 102.0, 103.0],
            [50.0, 49.5, 50.5, 51.0, 50.75],
        ],
        dtype=np.float32,
    )

    forecasts = simple_gaussian_level_forecasts(
        contexts,
        horizon_len=4,
        config=SimpleForecastConfig(mode="random_walk"),
    )

    assert forecasts.shape == (2, 4, 10)
    assert np.isfinite(forecasts).all()
    expected_mean = np.repeat(contexts[:, -1, None], repeats=4, axis=1)
    np.testing.assert_allclose(forecasts[:, :, 0], expected_mean, rtol=1e-6)
    assert np.all(forecasts[:, :, 1] <= forecasts[:, :, 5])
    assert np.all(forecasts[:, :, 5] <= forecasts[:, :, 9])


def test_build_simple_distribution_cache_roundtrip(tmp_path) -> None:
    prices = pd.DataFrame(
        {
            "Date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "AAA": np.linspace(100.0, 111.0, 12),
            "BBB": np.linspace(50.0, 52.2, 12),
        }
    )
    csv_path = tmp_path / "prices.csv"
    out_path = tmp_path / "simple_cache.npz"
    prices.to_csv(csv_path, index=False)

    build_simple_distribution_cache(
        csv_path=csv_path,
        output_path=out_path,
        tickers=["AAA", "BBB"],
        context_len=5,
        horizon_len=3,
        holding_horizon=1,
        stride=2,
        max_windows_per_asset=None,
        forecast_config=SimpleForecastConfig(mode="rolling_mean"),
    )

    cache = load_distribution_cache(out_path)
    assert cache["full_outputs"].shape == (8, 3, 10)
    assert cache["last_values"].shape == (8,)
    assert cache["realized_returns"].shape == (8,)
    assert set(cache["asset_names"].tolist()) == {"AAA", "BBB"}
