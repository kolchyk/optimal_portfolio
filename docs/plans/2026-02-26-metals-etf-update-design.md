# Design Doc: 2026-02-26 - Metals ETF Update

## Overview
Update `src/portfolio_sim/config.py` to replace the mixed commodities block with a focused list of pure Metals ETFs.

## Design
Replace the multi-asset commodity list in `ETF_UNIVERSE` with the following physical/liquid metals tickers:
- **GLD**: Gold
- **SLV**: Silver
- **PPLT**: Platinum
- **PALL**: Palladium
- **CPER**: Copper (Futures-based ETF)
- **LIT**: Lithium (Equity-based ETF)
- **JJU**: Aluminum (ETN tracking price)

## Component Updates
- `ETF_UNIVERSE`: Lines 60-62 will be replaced with the above tickers.
- `ASSET_CLASS_MAP`: Lines 117-121 will be replaced with mappings for these 7 tickers to "Metals".

## Testing
- Verify that `yfinance` can download 5 years of history for all new tickers.
- Run `uv run streamlit run dashboard.py` to ensure the new categories display correctly.

## Trade-offs
- Improves focus on the metals sector for KAMA Momentum strategy.
- Reduces exposure to Energy and Agriculture sectors within the "Commodities" category.
