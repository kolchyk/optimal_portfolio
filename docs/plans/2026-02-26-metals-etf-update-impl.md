# Metals ETF Update Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace mixed commodity ETFs with a focused list of 7 pure metals ETFs in the strategy configuration.

**Architecture:** Update the `ETF_UNIVERSE` list and `ASSET_CLASS_MAP` dictionary in `src/portfolio_sim/config.py` with the new tickers.

**Tech Stack:** Python, `yfinance`.

---

### Task 1: Update Tickers in config.py

**Files:**
- Modify: `src/portfolio_sim/config.py:60-62`
- Modify: `src/portfolio_sim/config.py:117-121`

**Step 1: Replace ETF_UNIVERSE block**

Modify `src/portfolio_sim/config.py`:
Replace:
```python
    # Commodities & Metals
    "GLD", "SLV", "IAU", "SGOL", "SIVR", "PPLT", "USO", "BNO", "DBC", "GSG",
    "PDBC", "DJP", "DBA", "UNG", "CORN", "WEAT", "SOYB", "JO", "JJI", "GCC",
```
With:
```python
    # Metals (Pure ETFs)
    "GLD", "SLV", "PPLT", "PALL", "CPER", "LIT", "JJU",
```

**Step 2: Replace ASSET_CLASS_MAP block**

Modify `src/portfolio_sim/config.py`:
Replace:
```python
    # Commodities
    "GLD": "Commodities", "SLV": "Commodities", "IAU": "Commodities", "SGOL": "Commodities", "SIVR": "Commodities",
    "PPLT": "Commodities", "USO": "Commodities", "BNO": "Commodities", "DBC": "Commodities", "GSG": "Commodities",
    "PDBC": "Commodities", "DJP": "Commodities", "DBA": "Commodities", "UNG": "Commodities", "CORN": "Commodities",
    "WEAT": "Commodities", "SOYB": "Commodities", "JO": "Commodities", "JJI": "Commodities", "GCC": "Commodities",
```
With:
```python
    # Metals
    "GLD": "Metals", "SLV": "Metals", "PPLT": "Metals", "PALL": "Metals", "CPER": "Metals", "LIT": "Metals", "JJU": "Metals",
```

**Step 3: Commit**

```bash
git add src/portfolio_sim/config.py
git commit -m "feat: replace mixed commodities with pure metals ETFs"
```

### Task 2: Verify Data Availability

**Files:**
- Create: `scripts/verify_metals_data.py`

**Step 1: Write verification script**

Create `scripts/verify_metals_data.py`:
```python
import yfinance as yf
import sys
import os

# Add src to path to import config
sys.path.append(os.getcwd())
from src.portfolio_sim.config import ETF_UNIVERSE

def verify():
    metals = ["GLD", "SLV", "PPLT", "PALL", "CPER", "LIT", "JJU"]
    all_ok = True
    for ticker in metals:
        print(f"Checking {ticker}...")
        data = yf.download(ticker, period="5y", progress=False)
        if data.empty:
            print(f"ERROR: No data for {ticker}")
            all_ok = False
        else:
            print(f"OK: {len(data)} rows found for {ticker}")
    
    if all_ok:
        print("\nAll metals ETFs are available and have data.")
    else:
        print("\nSome tickers failed verification.")

if __name__ == "__main__":
    verify()
```

**Step 2: Run verification**

Run: `uv run python scripts/verify_metals_data.py`
Expected: "OK" for all 7 tickers with several hundred/thousand rows each.

**Step 3: Commit and Cleanup**

```bash
rm scripts/verify_metals_data.py
git commit -am "test: verify data availability for new metals tickers" --allow-empty
```
