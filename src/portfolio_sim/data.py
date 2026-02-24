"""Data loading: S&P 500 tickers from Wikipedia, price fetching via yfinance, Parquet cache.

Cache behavior:
  - If close_prices.parquet and open_prices.parquet exist in output/cache/,
    data is loaded from disk and yfinance is NOT called. No network requests.
  - Use refresh=True to force re-download and overwrite the cache.
"""

import pandas as pd
import structlog
import yfinance as yf
from pathlib import Path

from src.portfolio_sim.config import CACHE_DIR, SPY_TICKER

log = structlog.get_logger(__name__)

CLOSE_CACHE = CACHE_DIR / "close_prices.parquet"
OPEN_CACHE = CACHE_DIR / "open_prices.parquet"


def fetch_sp500_tickers() -> list[str]:
    """Fetch current S&P 500 constituent tickers from Wikipedia or local CSV.

    Returns sorted list of ticker symbols. Dots in symbols (e.g. BRK.B)
    are replaced with hyphens for yfinance compatibility.
    """
    csv_path = Path("sp500_companies.csv")
    if csv_path.exists():
        log.info("Loading tickers from sp500_companies.csv")
        df = pd.read_csv(csv_path)
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    else:
        log.info("Fetching S&P 500 constituents from Wikipedia")
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        tables = pd.read_html(url)
        df = tables[0]
        tickers = df["Symbol"].str.replace(".", "-", regex=False).tolist()

    log.info("Fetched S&P 500 constituents", n_tickers=len(tickers))
    return sorted(tickers)


def fetch_price_data(
    tickers: list[str],
    period: str = "5y",
    refresh: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Close and Open prices for all tickers.

    Returns (close_prices, open_prices) DataFrames with DatetimeIndex rows
    and ticker columns.

    Cache: if output/cache/ contains close_prices.parquet and open_prices.parquet,
    loads from disk and does not download. Pass refresh=True to force re-download
    and overwrite cache.
    """
    if CLOSE_CACHE.exists() and OPEN_CACHE.exists() and not refresh:
        log.info("Loading prices from Parquet cache (skip download)")
        close_df = pd.read_parquet(CLOSE_CACHE)
        open_df = pd.read_parquet(OPEN_CACHE)
        return close_df, open_df

    full_list = list(set(tickers + [SPY_TICKER]))
    log.info("Downloading prices via yfinance", n_tickers=len(full_list), period=period)

    close_df, open_df = _download_from_yfinance(full_list, period)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    close_df.to_parquet(CLOSE_CACHE)
    open_df.to_parquet(OPEN_CACHE)
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
    for i in range(0, len(tickers), batch_size):
        batch_tickers = tickers[i : i + batch_size]
        log.info(
            "Downloading batch",
            current=i // batch_size + 1,
            total=(len(tickers) + batch_size - 1) // batch_size,
            n_tickers=len(batch_tickers),
        )

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
