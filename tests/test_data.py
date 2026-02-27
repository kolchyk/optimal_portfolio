"""Tests for data module."""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.portfolio_sim.data import (
    fetch_etf_tickers,
    fetch_price_data,
    _download_from_yfinance,
)
from src.portfolio_sim.config import ETF_UNIVERSE


def test_fetch_etf_tickers():
    """Verify ETF tickers are loaded from config (SPY is benchmark-only, not in universe)."""
    tickers = fetch_etf_tickers()
    assert isinstance(tickers, list)
    assert len(tickers) == len(ETF_UNIVERSE)
    assert tickers == sorted(ETF_UNIVERSE)


@patch("src.portfolio_sim.data._download_from_yfinance")
def test_fetch_price_data_cache_hit(mock_download, tmp_path):
    """If cache exists, should skip download."""
    dates = pd.date_range("2023-01-01", periods=1)
    mock_close = pd.DataFrame({"AAPL": [150.0]}, index=dates)
    mock_open = pd.DataFrame({"AAPL": [149.0]}, index=dates)
    
    # Create files in tmp_path
    close_cache = tmp_path / "close_prices.parquet"
    open_cache = tmp_path / "open_prices.parquet"
    mock_close.to_parquet(close_cache)
    mock_open.to_parquet(open_cache)
    
    with patch("src.portfolio_sim.data.CACHE_DIR", tmp_path):
        c, o = fetch_price_data(["AAPL"], refresh=False)
        
        assert c.equals(mock_close)
        assert o.equals(mock_open)
        mock_download.assert_not_called()


@patch("src.portfolio_sim.data._download_from_yfinance")
def test_fetch_price_data_refresh(mock_download, tmp_path):
    """If refresh=True, should download even if cache exists."""
    dates = pd.date_range("2023-01-01", periods=1)
    mock_close = pd.DataFrame({"AAPL": [150.0]}, index=dates)
    mock_open = pd.DataFrame({"AAPL": [149.0]}, index=dates)
    mock_download.return_value = (mock_close, mock_open)
    
    with patch("src.portfolio_sim.data.CACHE_DIR", tmp_path):
        c, o = fetch_price_data(["AAPL"], refresh=True)
        
        assert mock_download.called
        assert c.equals(mock_close)
        assert o.equals(mock_open)
        assert (tmp_path / "close_prices.parquet").exists()


@patch("src.portfolio_sim.data.yf.download")
def test_download_from_yfinance_multi(mock_yf_download):
    """Test yfinance download with multiple tickers (MultiIndex)."""
    # Create mock MultiIndex DataFrame
    iterables = [["AAPL", "MSFT"], ["Open", "Close"]]
    columns = pd.MultiIndex.from_product(iterables)
    dates = pd.date_range("2023-01-01", periods=2)
    mock_data = pd.DataFrame(
        [[150, 151, 250, 251], [152, 153, 252, 253]],
        index=dates,
        columns=columns
    )
    mock_yf_download.return_value = mock_data

    close_df, open_df = _download_from_yfinance(["AAPL", "MSFT"], period="1mo", batch_size=100)
    
    assert "AAPL" in close_df.columns
    assert "MSFT" in close_df.columns
    assert len(close_df) == 2
    assert close_df.iloc[0]["AAPL"] == 151


@patch("src.portfolio_sim.data.yf.download")
def test_download_from_yfinance_single(mock_yf_download):
    """Test yfinance download with single ticker (Flat columns)."""
    dates = pd.date_range("2023-01-01", periods=2)
    mock_data = pd.DataFrame(
        {"Open": [150, 152], "Close": [151, 153]},
        index=dates
    )
    mock_yf_download.return_value = mock_data

    close_df, open_df = _download_from_yfinance(["AAPL"], period="1mo")
    
    assert list(close_df.columns) == ["AAPL"]
    assert close_df.iloc[0]["AAPL"] == 151
