# Design: Replace Crypto Tickers with Top 5 Crypto ETFs

**Date:** 2026-02-26  
**Topic:** Configuration Update  
**Status:** Approved

## Overview
Replace the current list of 20 individual cryptocurrency tickers (Yahoo Finance format, e.g., `BTC-USD`) with the top 5 Crypto ETFs by AUM as of February 2026. This aligns the strategy with standard tradable assets and reduces the number of crypto-related instruments in the universe while maintaining exposure to both Bitcoin and Ethereum.

## Selected Tickers (Option B: Mixed)
- **IBIT**: iShares Bitcoin Trust (BlackRock)
- **FBTC**: Fidelity Wise Origin Bitcoin Fund
- **GBTC**: Grayscale Bitcoin Trust
- **ARKB**: ARK 21Shares Bitcoin ETF
- **ETHA**: iShares Ethereum Trust (BlackRock)

## Changes

### 1. `ETF_UNIVERSE` in `src/portfolio_sim/config.py`
Remove:
```python
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "TRX-USD", "DOT-USD", "LINK-USD", "AVAX-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
    "SHIB-USD", "SUI-USD", "NEAR-USD", "APT-USD", "HBAR-USD", "ONDO-USD",
```
Add:
```python
    "IBIT", "FBTC", "GBTC", "ARKB", "ETHA",
```

### 2. `ASSET_CLASS_MAP` in `src/portfolio_sim/config.py`
Remove entries for `BTC-USD` through `ONDO-USD`.
Add:
```python
    "IBIT": "Crypto", "FBTC": "Crypto", "GBTC": "Crypto", "ARKB": "Crypto", "ETHA": "Crypto",
```

## Testing Plan
- Run `uv run python src/portfolio_sim/data.py` (if it has a main block) or a simple script to verify `yfinance` can download these tickers.
- Run `uv run pytest` to ensure no configuration-related regressions.
