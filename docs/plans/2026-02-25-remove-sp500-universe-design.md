# 2026-02-25-remove-sp500-universe-design.md Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Remove S&P 500 as a selectable universe from the sidebar, while keeping it as a benchmark for performance comparison.

**Architecture:**
- `dashboard.py`: Remove the universe selector radio button. Hardcode `is_etf_mode` to `True`.
- `pages/strategy_rules.py`: Update content to only describe the ETF universe and remove references to the S&P 500 selectable mode.
- S&P 500 (`SPY`) will continue to be fetched and used as a benchmark for comparison in charts and metrics.

**Tech Stack:** Streamlit, Pandas, Plotly

---

### Task 1: Update dashboard.py

**Files:**
- Modify: `dashboard.py:734-744`
- Modify: `dashboard.py:854-871`
- Modify: `dashboard.py:883-918`

**Step 1: Remove Universe selector from sidebar**
Remove lines 734-743 and replace with:
```python
        is_etf_mode = True
        universe_mode = "ETF Cross-Asset"
```

**Step 2: Update returned dict from _render_sidebar**
Remove `universe_mode` and set `is_etf_mode` as always True.

**Step 3: Simplify main() logic**
- Remove the `if sidebar["is_etf_mode"]:` branches and just use the ETF logic.
- Remove the `valid` tickers filtering logic that excludes `SPY` in S&P 500 mode.

### Task 2: Update pages/strategy_rules.py

**Files:**
- Modify: `pages/strategy_rules.py:37`
- Modify: `pages/strategy_rules.py:60-125`
- Modify: `pages/strategy_rules.py:173-190`
- Modify: `pages/strategy_rules.py:213`
- Modify: `pages/strategy_rules.py:406-445`

**Step 1: Update page title and caption**
Remove "S&P 500" from the caption.

**Step 2: Remove "Two universes" section**
Remove the info block and the two-column universe comparison.

**Step 3: Update ranking description**
Remove the "Raw momentum (S&P 500)" tab and only show "Risk-adjusted (ETF)".

**Step 4: Update parameters table**
Remove the "S&P 500" column from the advanced parameters table.
