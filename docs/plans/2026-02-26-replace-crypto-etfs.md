# Replace Crypto Tickers with Top 5 Crypto ETFs Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace 20 individual crypto tickers (e.g., BTC-USD) with top 5 Crypto ETFs (IBIT, FBTC, GBTC, ARKB, ETHA) in the configuration.

**Architecture:** Update `src/portfolio_sim/config.py` to reflect the new universe and asset class mapping. Maintain the "Crypto" category for tracking.

**Tech Stack:** Python 3.14, `yfinance`.

---

### Task 1: Update ETF_UNIVERSE in config.py

**Files:**
- Modify: `src/portfolio_sim/config.py:68-72`

**Step 1: Write the minimal implementation**

Modify `src/portfolio_sim/config.py`:
```python
<<<<
    # Crypto (Yahoo Finance format)
    "BTC-USD", "ETH-USD", "SOL-USD", "BNB-USD", "XRP-USD", "ADA-USD", "DOGE-USD",
    "TRX-USD", "DOT-USD", "LINK-USD", "AVAX-USD", "MATIC-USD", "LTC-USD", "BCH-USD",
    "SHIB-USD", "SUI-USD", "NEAR-USD", "APT-USD", "HBAR-USD", "ONDO-USD",
====
    # Crypto ETFs
    "IBIT", "FBTC", "GBTC", "ARKB", "ETHA",
>>>>
```

**Step 2: Commit**

```bash
git add src/portfolio_sim/config.py
git commit -m "config: replace individual crypto tickers with top 5 crypto ETFs in ETF_UNIVERSE"
```

### Task 2: Update ASSET_CLASS_MAP in config.py

**Files:**
- Modify: `src/portfolio_sim/config.py:129-134`

**Step 1: Write the minimal implementation**

Modify `src/portfolio_sim/config.py`:
```python
<<<<
    # Crypto
    "BTC-USD": "Crypto", "ETH-USD": "Crypto", "SOL-USD": "Crypto", "BNB-USD": "Crypto", "XRP-USD": "Crypto",
    "ADA-USD": "Crypto", "DOGE-USD": "Crypto", "TRX-USD": "Crypto", "DOT-USD": "Crypto", "LINK-USD": "Crypto",
    "AVAX-USD": "Crypto", "MATIC-USD": "Crypto", "LTC-USD": "Crypto", "BCH-USD": "Crypto", "SHIB-USD": "Crypto",
    "SUI-USD": "Crypto", "NEAR-USD": "Crypto", "APT-USD": "Crypto", "HBAR-USD": "Crypto", "ONDO-USD": "Crypto",
====
    # Crypto
    "IBIT": "Crypto", "FBTC": "Crypto", "GBTC": "Crypto", "ARKB": "Crypto", "ETHA": "Crypto",
>>>>
```

**Step 2: Commit**

```bash
git add src/portfolio_sim/config.py
git commit -m "config: update ASSET_CLASS_MAP with crypto ETFs"
```

### Task 3: Verify Data Download

**Files:**
- Create: `verify_etf_data.py`

**Step 1: Write verification script**

```python
import yfinance as yf
from src.portfolio_sim.config import ETF_UNIVERSE

def verify():
    crypto_etfs = ["IBIT", "FBTC", "GBTC", "ARKB", "ETHA"]
    print(f"Verifying download for: {crypto_etfs}")
    for ticker in crypto_etfs:
        data = yf.download(ticker, period="5d", progress=False)
        if data.empty:
            print(f"FAILED: {ticker} has no data")
        else:
            print(f"SUCCESS: {ticker} downloaded {len(data)} rows")

if __name__ == "__main__":
    verify()
```

**Step 2: Run verification**

Run: `uv run python verify_etf_data.py`
Expected: SUCCESS for all 5 tickers.

**Step 3: Cleanup and Commit**

```bash
rm verify_etf_data.py
git add docs/plans/2026-02-26-replace-crypto-etfs-design.md docs/plans/2026-02-26-replace-crypto-etfs.md
git commit -m "docs: add design and implementation plan for crypto ETF replacement"
```
