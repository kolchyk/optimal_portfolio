"""Data loading: tickers + price fetching via yfinance, Parquet cache.

Supports two universes:
  - S&P 500 (from local CSV or fallback URL)
  - Cross-asset ETFs (hardcoded in config.py)

Cache behavior:
  - If close_prices{suffix}.parquet and open_prices{suffix}.parquet exist in
    output/cache/, data is loaded from disk and yfinance is NOT called.
  - Use refresh=True to force re-download and overwrite cache.
"""

import re
from datetime import datetime, timedelta

import pandas as pd
import structlog
import yfinance as yf
from tqdm import tqdm

from src.portfolio_sim.config import CACHE_DIR, ETF_UNIVERSE, SPY_TICKER

log = structlog.get_logger(__name__)


def _period_to_start_date(period: str) -> datetime | None:
    """Convert yfinance-style period ('2y', '5y', '6mo', etc.) to start date.
    Returns None for max/all (e.g. 'max')."""
    m = re.match(r"^(\d+)(d|mo|y)$", period.strip().lower())
    if not m:
        return None
    n, unit = int(m.group(1)), m.group(2)
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "d":
        return today - timedelta(days=n)
    if unit == "mo":
        return today - timedelta(days=n * 31)
    if unit == "y":
        return today - timedelta(days=n * 365)
    return None

CLOSE_CACHE = CACHE_DIR / "close_prices.parquet"
OPEN_CACHE = CACHE_DIR / "open_prices.parquet"


def fetch_etf_tickers() -> list[str]:
    """Return the hardcoded cross-asset ETF universe from config.

    No network call or CSV read needed. SPY is included as both
    a tradable asset and benchmark.
    """
    log.info("Using cross-asset ETF universe", n_tickers=len(ETF_UNIVERSE))
    return sorted(ETF_UNIVERSE)


def fetch_price_data(
    tickers: list[str],
    period: str = "5y",
    refresh: bool = False,
    cache_suffix: str = "",
    min_rows: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Close and Open prices for all tickers.

    Returns (close_prices, open_prices) DataFrames with DatetimeIndex rows
    and ticker columns.

    Args:
        tickers: list of ticker symbols to download.
        period: yfinance period string (default: "5y").
        refresh: force re-download and overwrite cache.
        cache_suffix: appended to cache filenames to separate ETF vs S&P 500
                      caches (e.g. "_etf").
        min_rows: if > 0 and the cached data has fewer rows, automatically
                  re-download with the requested *period*.

    Cache: if output/cache/ contains the parquet files, loads from disk and
    does not download. Pass refresh=True to force re-download.
    """
    close_cache = CACHE_DIR / f"close_prices{cache_suffix}.parquet"
    open_cache = CACHE_DIR / f"open_prices{cache_suffix}.parquet"

    if close_cache.exists() and open_cache.exists() and not refresh:
        close_df = pd.read_parquet(close_cache)
        open_df = pd.read_parquet(open_cache)
        if min_rows and len(close_df) < min_rows:
            log.warning(
                "Cache has fewer rows than required, re-downloading",
                cached_rows=len(close_df),
                min_rows=min_rows,
                period=period,
            )
        else:
            start_date = _period_to_start_date(period)
            if start_date is not None:
                start_ts = pd.Timestamp(start_date)
                close_df = close_df.loc[close_df.index >= start_ts]
                open_df = open_df.loc[open_df.index >= start_ts]
                if min_rows and len(close_df) < min_rows:
                    log.warning(
                        "Cache has insufficient rows after period trim, re-downloading",
                        trimmed_rows=len(close_df),
                        min_rows=min_rows,
                        period=period,
                    )
                else:
                    log.info(
                        "Loading prices from Parquet cache (trimmed to period)",
                        suffix=cache_suffix,
                        period=period,
                        rows=len(close_df),
                    )
                    return close_df, open_df
            else:
                log.info("Loading prices from Parquet cache (skip download)", suffix=cache_suffix)
                return close_df, open_df

        # Fall through to re-download when trim left too few rows

    full_list = list(set(tickers + [SPY_TICKER]))
    log.info("Downloading prices via yfinance", n_tickers=len(full_list), period=period)

    close_df, open_df = _download_from_yfinance(full_list, period)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    close_df.to_parquet(close_cache)
    open_df.to_parquet(open_cache)
    log.info("Prices cached to Parquet — future runs will use cache", path=str(CACHE_DIR))

    return close_df, open_df


def _download_from_yfinance(
    tickers: list[str], period: str, batch_size: int = 100
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Download OHLCV data via yfinance and extract Close/Open.

    Downloads in batches of batch_size to improve stability.
    """
    all_close = []
    all_open = []

    # Process in batches
    n_batches = (len(tickers) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(n_batches), desc="Downloading batches", unit="batch"):
        i = batch_idx * batch_size
        batch_tickers = tickers[i : i + batch_size]

        raw = yf.download(
            batch_tickers,
            period=period,
            group_by="ticker",
            auto_adjust=True,
            threads=True,
            progress=True,
        )

        if isinstance(raw.columns, pd.MultiIndex):
            close_batch = (
                raw.xs("Close", axis=1, level=1)
                if "Close" in raw.columns.get_level_values(1)
                else pd.DataFrame()
            )
            open_batch = (
                raw.xs("Open", axis=1, level=1)
                if "Open" in raw.columns.get_level_values(1)
                else pd.DataFrame()
            )
        else:
            # Case for single ticker
            close_batch = raw[["Close"]].rename(columns={"Close": batch_tickers[0]})
            open_batch = raw[["Open"]].rename(columns={"Open": batch_tickers[0]})

        all_close.append(close_batch)
        all_open.append(open_batch)

    # Combine results
    close_df = pd.concat(all_close, axis=1)
    open_df = pd.concat(all_open, axis=1)

    close_df = close_df.ffill().dropna(axis=1, how="all")
    open_df = open_df[close_df.columns].ffill()

    if close_df.index.tz is not None:
        close_df.index = close_df.index.tz_localize(None)
        open_df.index = open_df.index.tz_localize(None)

    log.info(
        "Download complete", tickers_received=len(close_df.columns), rows=len(close_df)
    )
    return close_df, open_df
