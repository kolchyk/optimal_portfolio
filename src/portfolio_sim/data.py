"""Data loading: ticker universe from JSON, price fetching via yfinance, Parquet cache."""

import json

import pandas as pd
import structlog
import yfinance as yf

from src.portfolio_sim.config import CACHE_DIR, SAFE_HAVEN_TICKER, TICKERS_JSON_PATH

log = structlog.get_logger(__name__)

CLOSE_CACHE = CACHE_DIR / "close_prices.parquet"
OPEN_CACHE = CACHE_DIR / "open_prices.parquet"


def load_tickers(path=TICKERS_JSON_PATH) -> tuple[list[str], dict]:
    """Load ticker universe from JSON.

    Returns:
        (tickers_list, original_portfolio_dict)
    """
    with open(path) as f:
        data = json.load(f)
    tickers = list(set(data.get("tickers_600", [])))
    original_portfolio = data.get("original_portfolio", {})
    return tickers, original_portfolio


def fetch_price_data(
    tickers: list[str],
    period: str = "5y",
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Close and Open prices for all tickers.

    Uses Parquet cache. Returns (close_prices, open_prices) DataFrames
    with DatetimeIndex rows and ticker columns.
    """
    if CLOSE_CACHE.exists() and OPEN_CACHE.exists() and not refresh:
        log.info("Loading prices from Parquet cache")
        close_df = pd.read_parquet(CLOSE_CACHE)
        open_df = pd.read_parquet(OPEN_CACHE)
        return close_df, open_df

    # Ensure SPY and SHV are included
    full_list = list(set(tickers + ["SPY", SAFE_HAVEN_TICKER]))
    log.info("Downloading prices via yfinance", n_tickers=len(full_list), period=period)

    close_df, open_df = _download_from_yfinance(full_list, period)

    # Save to cache
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    close_df.to_parquet(CLOSE_CACHE)
    open_df.to_parquet(OPEN_CACHE)
    log.info("Prices cached to Parquet", path=str(CACHE_DIR))

    return close_df, open_df


def _download_from_yfinance(
    tickers: list[str], period: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OHLCV data via yfinance and extract Close/Open."""
    raw = yf.download(
        tickers,
        period=period,
        group_by="ticker",
        auto_adjust=True,
        threads=True,
        progress=True,
    )

    # yfinance returns MultiIndex columns (ticker, field) when multiple tickers
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw.xs("Close", axis=1, level=1) if "Close" in raw.columns.get_level_values(1) else pd.DataFrame()
        open_df = raw.xs("Open", axis=1, level=1) if "Open" in raw.columns.get_level_values(1) else pd.DataFrame()
    else:
        # Single ticker fallback
        close_df = raw[["Close"]].rename(columns={"Close": tickers[0]})
        open_df = raw[["Open"]].rename(columns={"Open": tickers[0]})

    # Clean up
    close_df = close_df.ffill().dropna(axis=1, how="all")
    open_df = open_df[close_df.columns].ffill().bfill()

    # Ensure timezone-naive DatetimeIndex
    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)
        open_df.index = open_df.index.tz_localize(None)

    log.info("Download complete", tickers_received=len(close_df.columns), rows=len(close_df))
    return close_df, open_df
