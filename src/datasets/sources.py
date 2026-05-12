from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


DEFAULT_ETF_TICKERS: tuple[str, ...] = (
    "SPY",
    "QQQ",
    "IWM",
    "TLT",
    "GLD",
    "EEM",
    "USO",
    "UUP",
)


def load_close_csv(path: str | Path) -> pd.DataFrame:
    """Load a wide close-price CSV with a Date column."""

    df = pd.read_csv(path)
    if "Date" not in df.columns:
        raise ValueError("Expected a Date column.")
    value_cols = [col for col in df.columns if col != "Date"]
    if not value_cols:
        raise ValueError("Expected at least one asset column.")

    out = df[["Date", *value_cols]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="raise")
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.sort_values("Date").dropna(axis=0, how="any").reset_index(drop=True)
    return out


def save_close_csv(df: pd.DataFrame, path: str | Path) -> Path:
    """Write a validated wide close-price CSV."""

    out = load_close_frame(df)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return path


def load_close_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Validate a wide close-price dataframe and normalize Date formatting."""

    if "Date" not in df.columns:
        raise ValueError("Expected a Date column.")
    value_cols = [col for col in df.columns if col != "Date"]
    if not value_cols:
        raise ValueError("Expected at least one asset column.")
    out = df[["Date", *value_cols]].copy()
    out["Date"] = pd.to_datetime(out["Date"], errors="raise").dt.strftime("%Y-%m-%d")
    for col in value_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(axis=0, how="any").reset_index(drop=True)
    return out


def download_yfinance_close(
    tickers: Iterable[str] = DEFAULT_ETF_TICKERS,
    *,
    start: str = "2007-01-01",
    end: str | None = None,
    auto_adjust: bool = True,
) -> pd.DataFrame:
    """Download daily close prices through yfinance.

    Returns a wide dataframe: Date plus one column per ticker. The import is
    kept inside the function so the rest of the package works without network
    dependencies.
    """

    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is required to download market data.") from exc

    ticker_list = [str(ticker).upper() for ticker in tickers]
    if not ticker_list:
        raise ValueError("At least one ticker is required.")

    raw = yf.download(
        ticker_list,
        start=start,
        end=end,
        auto_adjust=auto_adjust,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise ValueError("yfinance returned no rows.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" not in raw.columns.get_level_values(0):
            raise ValueError("Downloaded data does not contain Close prices.")
        close = raw["Close"].copy()
    else:
        if "Close" not in raw.columns:
            raise ValueError("Downloaded data does not contain Close prices.")
        close = raw[["Close"]].copy()
        close.columns = ticker_list[:1]

    close = close.reindex(columns=ticker_list)
    close = close.dropna(axis=0, how="any")
    close = close.reset_index()
    date_col = "Date" if "Date" in close.columns else close.columns[0]
    close = close.rename(columns={date_col: "Date"})
    return load_close_frame(close)
