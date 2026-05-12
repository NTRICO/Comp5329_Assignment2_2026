"""Position-aware portfolio controller modules built on frozen FinCast outputs."""

from .baselines import closed_form_markowitz_position
from .config import PositionControllerConfig, PositionLossConfig
from .features import (
    forecast_to_return_patch,
    make_forward_returns,
    quantile_summary_features,
)
from .data_sources import DEFAULT_ETF_TICKERS, download_yfinance_close, load_close_csv
from .losses import MeanVarianceTurnoverLoss
from .model import DistributionPatchEncoder, PositionAwareGRUPolicy
from .datasets import CachedDistributionDataset
from .fincast_cache import (
    build_fincast_distribution_cache,
    load_distribution_cache,
    make_rolling_forecast_samples,
    save_distribution_cache,
)

__all__ = [
    "PositionControllerConfig",
    "PositionLossConfig",
    "DistributionPatchEncoder",
    "PositionAwareGRUPolicy",
    "MeanVarianceTurnoverLoss",
    "CachedDistributionDataset",
    "closed_form_markowitz_position",
    "forecast_to_return_patch",
    "make_forward_returns",
    "quantile_summary_features",
    "DEFAULT_ETF_TICKERS",
    "download_yfinance_close",
    "load_close_csv",
    "build_fincast_distribution_cache",
    "load_distribution_cache",
    "make_rolling_forecast_samples",
    "save_distribution_cache",
]
