"""Strategy explanation page — Hybrid R\u00b2 Momentum + Vol-Targeting (EN / UK)."""

from collections import Counter

import streamlit as st

from src.portfolio_sim.config import (
    ASSET_CLASS_MAP,
    COMMISSION_RATE,
    DEFAULT_N_TRIALS,
    ETF_UNIVERSE,
    INITIAL_CAPITAL,
    RISK_FREE_RATE,
    SEARCH_SPACE,
    SLIPPAGE_RATE,
)
from src.portfolio_sim.params import StrategyParams

_DEFAULTS = StrategyParams()
_BUFFER_PCT = _DEFAULTS.kama_buffer * 100
_GAP_PCT = _DEFAULTS.gap_threshold * 100
_N_ETF = len(ETF_UNIVERSE)
_ASSET_CLASS_COUNTS = Counter(ASSET_CLASS_MAP.values())
_N_ASSET_CLASSES = len(_ASSET_CLASS_COUNTS)
_N_SENSITIVITY = DEFAULT_N_TRIALS
_PARAM_NAMES = list(SEARCH_SPACE.keys())


def _page_en():
    st.title("How It Works")
    st.caption(
        "Hybrid R\u00b2 Momentum + Vol-Targeting | Long/Cash | Cross-Asset ETF | "
        "ATR Risk Parity | Lazy-Hold Rebalancing"
    )

    # ------------------------------------------------------------------
    # 1. Overview
    # ------------------------------------------------------------------
    st.header("1. What Is This Strategy?")
    st.markdown(
        """
**In simple terms:** this strategy automatically picks the best-performing assets
(stocks, bonds, gold, real estate, crypto) and holds them while their trend is
strong. When the trend weakens, it sells and moves to cash. Think of it as a
**smart autopilot for your portfolio** \u2014 it only rides waves that are clean and strong,
like a skilled surfer.

---

This is a **hybrid systematic momentum strategy** combining Andreas Clenow's
methodology ("Stocks on the Move") with a **vol-targeting overlay**.

**Core idea:** we fit a linear regression to the log of each asset's price over the
last N days. The slope of this line represents the daily trend return, and R\u00b2
measures how consistently the price follows that trend. Their product
`annualized_slope \u00d7 R\u00b2` is the **R\u00b2 Momentum Score**.

On top of ATR risk parity positioning, a **vol-targeting overlay** scales the
portfolio so that realized volatility stays near a target level. This automatically
reduces exposure in turbulent periods and increases it in calm ones.

**Strategy type:** Long/Cash \u2014 we either hold assets or cash.
We never short (bet against an asset).
        """
    )

    st.info(
        f"**Cross-Asset ETF** \u2014 {_N_ETF} assets across {_N_ASSET_CLASSES} classes "
        "(stocks, bonds, commodities, real estate, crypto). "
        "Tactical all-weather allocation with ATR risk parity + vol-targeting."
    )

    # ------------------------------------------------------------------
    # 2. Asset universe
    # ------------------------------------------------------------------
    st.header("2. Asset Universe")

    st.markdown(
        "The strategy invests across multiple asset classes to avoid putting "
        "all eggs in one basket. Here's what we trade:"
    )

    with st.expander("Cross-Asset ETF breakdown by asset class"):
        class_data = sorted(_ASSET_CLASS_COUNTS.items(), key=lambda x: -x[1])
        st.table(
            {
                "Asset Class": [c for c, _ in class_data],
                "Count": [n for _, n in class_data],
            }
        )

    # ------------------------------------------------------------------
    # 3. KAMA indicator
    # ------------------------------------------------------------------
    st.header("3. KAMA Indicator (Trend Filter)")
    st.markdown(
        f"""
**KAMA** (Kaufman's Adaptive Moving Average) is a smart moving average that
automatically speeds up in trending markets and slows down in sideways markets.

- **Period for assets:** `{_DEFAULTS.kama_asset_period}` trading days

KAMA uses an *Efficiency Ratio* (the ratio of net price movement to total
volatility). If price moves steadily in one direction, KAMA follows quickly.
In a choppy, sideways market, KAMA barely moves.

> **For beginners:** think of KAMA as a "smart trend line" that follows the price
> but doesn't jerk around with every small fluctuation. It's like noise-canceling
> headphones for market data.

In the strategy, KAMA is used as a **trend filter** (entry/exit signals),
not for ranking momentum.
        """
    )

    # ------------------------------------------------------------------
    # 4. Candidate selection pipeline
    # ------------------------------------------------------------------
    st.header("4. How We Pick Assets (Selection Pipeline)")

    st.markdown(
        f"""
An asset enters the portfolio through four sequential steps:

**Step 1 \u2014 Asset KAMA Filter (Absolute Momentum):**
Asset closing price > KAMA \u00d7 (1 + {_BUFFER_PCT:.1f}%). Each asset must be
above its own trend to be considered. No global regime filter \u2014 each asset
answers for itself.

> *In plain English:* each asset must be in an uptrend. If stocks are falling,
> they fail their own trend filter. But crypto or commodities can stay if
> their own trend holds \u2014 true cross-asset diversification.

**Step 2 \u2014 Gap Filter:**
No single-day move > {_GAP_PCT:.0f}% in the last `{max(_DEFAULTS.r2_windows)}` days.
This filters out assets with distorted trends due to stock splits, earnings gaps, etc.

> *In plain English:* we avoid assets that had sudden, dramatic jumps \u2014
> they mess up our trend calculations.

        """
    )

    st.subheader("Step 3 \u2014 R\u00b2 Momentum Scoring & Top-N Selection")
    st.markdown(
        f"""
**Ranking by R\u00b2 Momentum Score (Clenow method).**

Formula: `score = annualized_slope \u00d7 R\u00b2`

For each asset, we fit OLS regressions to the log of its price over multiple
windows (`{_DEFAULTS.r2_windows}` days) and blend the scores with weights
`{_DEFAULTS.r2_weights}`:
- **annualized_slope** \u2014 regression slope \u00d7 252 (annual trend return)
- **R\u00b2** \u2014 coefficient of determination (0..1), measures trend "smoothness"

**The R\u00b2 multiplier penalizes choppy movement:**
- R\u00b2 = 0.9 (perfect trend) \u2192 keeps 90% of the score
- R\u00b2 = 0.5 (noisy trend) \u2192 cuts 50%
- R\u00b2 = 0.2 (chaotic movement) \u2192 only 20% remains

We select the **Top-{_DEFAULTS.top_n}** assets with the highest score > 0.

> **Analogy:** if two assets both went up 30% over a period, the one whose chart
> looks like a straight line gets priority over the one that zigzags wildly.
        """
    )

    # ------------------------------------------------------------------
    # 5. Sell rules
    # ------------------------------------------------------------------
    st.header("5. When We Sell (Lazy Hold)")
    st.markdown(
        f"""
A position is sold when any of these conditions is met:

- **KAMA stop:** closing price < KAMA \u00d7 (1 \u2212 {_BUFFER_PCT:.1f}%)
- **Gap exit:** single-day move > {_GAP_PCT:.0f}% \u2014 immediate exit
- **R\u00b2 Score < 0** or asset fails filters on rebalancing date

A position is **NOT** sold just because the asset dropped out of the Top-{_DEFAULTS.top_n}
by momentum. As long as its own trend holds, we keep it.

Rebalancing (signal check) happens every **{_DEFAULTS.rebal_period_weeks} weeks**,
not every day. This reduces turnover and trading costs.

> **For beginners:** it's like having individual protection for each asset.
> We check the portfolio regularly, but don't panic every day. If an asset
> is still going up, we hold it even if newer ones look slightly better.
        """
    )

    # ------------------------------------------------------------------
    # 6. Risk management — 4 pillars
    # ------------------------------------------------------------------
    st.header("6. Four Pillars of Risk Management")

    st.markdown(
        "The strategy has four layers of protection for your capital. "
        "Each works independently from the others:"
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("Hysteresis")
        st.markdown(
            f"""
**"Market breathing room"**

A \u00b1{_BUFFER_PCT:.1f}% buffer around KAMA prevents false switches
in sideways markets.

**Analogy:** like a thermostat with tolerance \u2014 it doesn't
turn the heater on and off with every 0.1\u00b0 fluctuation.
            """
        )

    with col2:
        st.subheader("Lazy Hold")
        st.markdown(
            """
**"Patient" stop-loss**

Positions are held as long as their own trend holds
(price > KAMA). We don't sell an asset just because
it dropped from Top-N, if it's still rising.

_See Section 5 for details._
            """
        )

    with col3:
        st.subheader("ATR Risk Parity")
        st.markdown(
            f"""
**Position sizing by ATR**

Each asset's weight is proportional to:
`risk_factor / (ATR / price)`

Where ATR is the average daily price movement
over `{_DEFAULTS.atr_period}` days.

Calm assets (bonds) get more capital,
volatile ones (crypto) get less.

> **Analogy:** each asset "risks" roughly the same amount of money.
            """
        )

    with col4:
        st.subheader("Vol-Targeting")
        st.markdown(
            f"""
**Volatility overlay**

The portfolio is scaled daily:
`scale = target_vol / realised_vol`

Target volatility: **{_DEFAULTS.target_vol:.0%}** annual.
Max leverage: **{_DEFAULTS.max_leverage}x**.

If the market is calm, positions increase.
If volatile, they shrink automatically.

> **Analogy:** autopilot that keeps the "speed of risk" steady.
            """
        )

    # ------------------------------------------------------------------
    # 7. Parameters table
    # ------------------------------------------------------------------
    st.header("7. Strategy Parameters")

    tab_core, tab_adv = st.tabs(["Core Parameters", "Advanced Parameters"])

    with tab_core:
        st.table(
            {
                "Parameter": [
                    "Data Period",
                    "Initial Capital",
                    "R\u00b2 Lookback",
                    "KAMA Period (asset)",
                    "KAMA Buffer (hysteresis)",
                    "Number of Positions (Top N)",
                    "Rebalancing Period",
                    "Commission per Trade",
                    "Slippage",
                    "Risk-Free Rate",
                ],
                "Default": [
                    "3 years",
                    f"${INITIAL_CAPITAL:,.0f}",
                    f"{_DEFAULTS.r2_windows} days (blended)",
                    f"{_DEFAULTS.kama_asset_period} days",
                    f"\u00b1{_BUFFER_PCT:.1f}%",
                    f"{_DEFAULTS.top_n} assets",
                    f"{_DEFAULTS.rebal_period_weeks} weeks",
                    f"{COMMISSION_RATE:.2%} ({COMMISSION_RATE * 10_000:.0f} bps)",
                    f"{SLIPPAGE_RATE:.2%} ({SLIPPAGE_RATE * 10_000:.0f} bps)",
                    f"{RISK_FREE_RATE:.0%} annual",
                ],
                "Range": [
                    "3 \u2013 5 years",
                    "$1,000 \u2013 $10,000,000",
                    "60 \u2013 120 days",
                    "10 \u2013 50 days",
                    "0.5% \u2013 3.0%",
                    "5 \u2013 15",
                    "2 \u2013 4 weeks",
                    "(fixed)",
                    "(fixed)",
                    "(fixed)",
                ],
                "What It Does": [
                    "How many years of historical data to load",
                    "Starting amount for the simulation",
                    "Window for OLS regression to score momentum",
                    "Window for individual asset trend filter",
                    "Protection against false bull/bear switches",
                    "Maximum assets in portfolio at once",
                    "How often we check signals (lazy-hold)",
                    "Broker cost per trade",
                    "Difference between expected and actual price",
                    "Used for calculating the Sharpe Ratio",
                ],
            }
        )

    with tab_adv:
        st.table(
            {
                "Parameter": [
                    "Gap Threshold",
                    "ATR Period",
                    "Risk Factor",
                    "Target Vol",
                    "Max Leverage",
                    "Vol Lookback",
                ],
                "Default": [
                    f"{_DEFAULTS.gap_threshold:.0%}",
                    f"{_DEFAULTS.atr_period} days",
                    f"{_DEFAULTS.risk_factor} ({_DEFAULTS.risk_factor * 10_000:.0f} bps)",
                    f"{_DEFAULTS.target_vol:.0%}",
                    f"{_DEFAULTS.max_leverage}x",
                    f"{_DEFAULTS.portfolio_vol_lookback} days",
                ],
                "What It Does": [
                    "Exclude assets with a single-day move exceeding this",
                    "Window for calculating ATR (position sizing)",
                    "Daily risk per position (for ATR risk parity)",
                    "Target annual portfolio volatility",
                    "Maximum scaling factor",
                    "Window for estimating realized volatility",
                ],
            }
        )

    st.info(
        "All parameters can be adjusted on the dashboard via the sidebar. "
        "Enable **Walk-Forward** optimization for automatic parameter tuning "
        "(Optuna TPE sampler)."
    )

    # ------------------------------------------------------------------
    # 8. Execution model
    # ------------------------------------------------------------------
    st.header("8. How Trades Are Executed")
    st.markdown(
        f"""
| Step | Timing | Action |
|------|--------|--------|
| 1 | Every {_DEFAULTS.rebal_period_weeks} weeks, close of day T | Analyze signals: asset KAMA, gap filter, R\u00b2 scoring |
| 2 | Close of day T | Determine sells (KAMA breach, failed filters) and buys (new top-N candidates) |
| 3 | Open of day T+1 | Execute trades at opening price |
| 4 | Every day | Vol-targeting overlay: scale positions by target_vol / realised_vol |
| 5 | Every day | Gap exit: immediate exit if |daily_return| > threshold |

> **Why not at close?** Signals form at close, but trading at the same price
> is impossible in real life. Executing at the next day's open is a more
> realistic model.

> **Lazy-Hold:** between rebalancing dates, positions are held unchanged
> (except for vol-targeting and gap exits). This reduces turnover.
        """
    )

    # ------------------------------------------------------------------
    # 9. Walk-Forward Optimization
    # ------------------------------------------------------------------
    st.header("9. Walk-Forward Parameter Optimization")

    st.warning(
        "**For beginners:** this section is for advanced users. Walk-Forward "
        "optimization tests whether the strategy works not just on past data, "
        "but also on new, unseen data. If you're just getting started, you can "
        "skip this section."
    )

    st.markdown(
        f"""
**Goal:** find parameters that work not just on historical data,
but also on new, unseen data (out-of-sample).

**How it works:** data is split into rolling windows:
- **In-Sample (IS):** training window \u2014 optimize parameters
- **Out-of-Sample (OOS):** validation window \u2014 test quality

**Objective function:** CAGR (maximize) with MaxDD \u2264 25% constraint.

Search space ({_N_SENSITIVITY} Optuna TPE trials per step):
        """
    )

    grid_data = {
        "Parameter": [],
        "Search Space": [],
    }
    for name in _PARAM_NAMES:
        spec = SEARCH_SPACE[name]
        grid_data["Parameter"].append(name)
        if spec.get("type") == "categorical":
            grid_data["Search Space"].append(
                ", ".join(str(v) for v in spec["choices"])
            )
        else:
            grid_data["Search Space"].append(
                f"{spec['low']} \u2013 {spec['high']}, step {spec['step']}"
            )
    st.table(grid_data)

    st.markdown(
        """
**IS \u2192 OOS degradation:** if OOS CAGR < 50% of IS CAGR, a warning
about possible overfitting is shown.

Run walk-forward optimization:
`uv run python -m src.portfolio_sim walk-forward`
        """
    )

    # ------------------------------------------------------------------
    # 10. Efficient frontier
    # ------------------------------------------------------------------
    st.header("10. Markowitz Efficient Frontier")
    st.markdown(
        """
The dashboard includes a **Risk-Return Distribution & Efficient Frontier** chart.

The **Efficient Frontier** is a set of portfolios that offer the
**highest return for each level of risk**.

**How to read the chart:**
- **Each dot** \u2014 one asset (volatility on X-axis, return on Y-axis)
- **Gold curve** \u2014 efficient frontier
- **Cyan star** \u2014 your portfolio's position
- **Gray diamond** \u2014 S&P 500 (benchmark)

> **Analogy:** the efficient frontier is like a car's speed limit.
> You can go slower, but you can't go faster without increasing risk.
        """
    )

    # ------------------------------------------------------------------
    # 11. Advantages
    # ------------------------------------------------------------------
    st.header("11. Advantages of This Approach")
    st.markdown(
        """
**Why is this hybrid strategy better than naive "buy and hold"?**

- **Trend quality, not just returns.** The R\u00b2 multiplier strictly penalizes
  chaotic movement \u2014 the strategy picks assets with smooth, reliable trends.

- **Vol-targeting overlay.** Automatic portfolio scaling to target volatility.
  In turbulent periods, exposure decreases; in calm ones, it increases.

- **Protection from large drawdowns.** Per-asset KAMA filters exit each
  position individually when its trend breaks \u2014 no global kill switch needed.

- **True cross-asset diversification.** Each asset answers for itself via its
  own KAMA filter. Crypto and commodities can stay in portfolio during equity
  bear markets if their own trends hold. Per-class sector limits prevent
  excessive concentration.

- **ATR Risk Parity.** Position sizes are proportional to actual volatility
  (ATR), not equal weights. Calm assets receive more capital.

- **Low turnover.** Lazy-Hold rebalancing every N weeks reduces the number
  of trades = lower commissions and slippage.
        """
    )

    # ------------------------------------------------------------------
    # 12. Glossary
    # ------------------------------------------------------------------
    with st.expander("Glossary of Terms for Beginners"):
        st.markdown(
            "Don't know what a term means? Find it here:"
        )
        st.markdown(
            f"""
- **ATR (Average True Range)** \u2014 the average daily price movement of an asset.
  Used for position sizing. Higher ATR = more volatile = smaller position.
- **Benchmark** \u2014 a reference point (S&P 500 in our case) to compare your
  strategy against. If you can't beat the benchmark, you're better off
  just buying an index fund.
- **CAGR (Compound Annual Growth Rate)** \u2014 the average annual return,
  as if your portfolio grew at a steady rate each year.
- **Calmar Ratio** \u2014 annual return divided by maximum drawdown.
  Higher is better.
- **Drawdown** \u2014 the peak-to-trough decline in portfolio value.
  Max drawdown shows the worst-case scenario you would have experienced.
- **Efficient Frontier** \u2014 the Markowitz curve showing optimal portfolios
  for each level of risk.
- **ETF (Exchange-Traded Fund)** \u2014 a fund that trades on an exchange
  like a stock, tracking an index, commodity, or basket of assets.
- **Gap** \u2014 a sudden single-day price jump (>{_GAP_PCT:.0f}%).
  Gaps distort regression scores, so those assets are excluded.
- **Hysteresis** \u2014 a buffer that prevents frequent false switches between
  bull and bear modes.
- **KAMA** \u2014 Kaufman's Adaptive Moving Average. Reacts to trend strength.
  Used as a trend filter in the strategy.
- **Lazy Hold** \u2014 a position is kept as long as its own trend holds,
  even if it drops from Top-N. Reduces turnover.
- **Long/Cash** \u2014 a strategy that either holds assets (long) or cash.
  No short-selling.
- **Momentum** \u2014 the tendency of assets that have been rising recently
  to continue rising.
- **OLS Regression** \u2014 ordinary least squares. We fit a straight line
  to the log of price over N days. The slope shows the average daily
  trend return.
- **R\u00b2 (Coefficient of Determination)** \u2014 measures how well a linear
  regression fits the data. R\u00b2 = 1 means a perfect straight line,
  R\u00b2 = 0 means random movement. Used to penalize unstable trends.
- **R\u00b2 Momentum Score** \u2014 `annualized_slope \u00d7 R\u00b2`. The main metric for
  ranking assets. High score = strong and stable trend.
- **Rebalancing** \u2014 the periodic process of reviewing and adjusting the
  portfolio. We do it every {_DEFAULTS.rebal_period_weeks} weeks.
- **Risk Factor** \u2014 daily risk per position ({_DEFAULTS.risk_factor * 10_000:.0f} bps).
  Determines how much capital is allocated to each asset relative to its ATR.
- **Sharpe Ratio** \u2014 a measure of risk-adjusted return. It shows how much
  extra return (above the risk-free rate of {RISK_FREE_RATE:.0%}) you get per
  unit of risk. Above 1 is considered good, above 2 is excellent.
- **SPY** \u2014 an ETF that tracks the S&P 500 index. Used as our benchmark
  and regime filter.
- **Vol-Targeting** \u2014 an overlay that scales the portfolio to a target
  volatility. target_vol / realised_vol determines the multiplier.
- **Walk-Forward Optimization** \u2014 a rolling optimization process: train
  parameters on an in-sample window, validate on an out-of-sample window.
  Protects against overfitting.
            """
        )


def _page_uk():
    st.title("Як це працює")
    st.caption(
        "Гібридна стратегія R² Momentum + Vol-Targeting | Long/Cash | Cross-Asset ETF | "
        "ATR Risk Parity | Lazy-Hold Ребалансування"
    )

    # ------------------------------------------------------------------
    # 1. Огляд
    # ------------------------------------------------------------------
    st.header("1. Що це за стратегія?")
    st.markdown(
        """
**Простими словами:** ця стратегія автоматично обирає найкращі активи
(акції, облігації, золото, нерухомість, крипто) і тримає їх, поки тренд
сильний. Коли тренд слабшає — продає і переходить у кеш. Уявіть собі
**розумний автопілот для вашого портфеля** — він ловить лише чисті й сильні
хвилі, як досвідчений серфер.

---

Це **гібридна систематична моментум-стратегія**, що поєднує методологію
Андреаса Кленова ("Stocks on the Move") з **vol-targeting оверлеєм**.

**Основна ідея:** ми будуємо лінійну регресію логарифму ціни кожного активу
за останні N днів. Нахил цієї лінії — це денна трендова дохідність, а R²
вимірює, наскільки послідовно ціна рухається за трендом. Їхній добуток
`annualized_slope × R²` — це **R² Momentum Score**.

Поверх ATR risk parity позиціонування, **vol-targeting оверлей** масштабує
портфель так, щоб реалізована волатильність залишалася поблизу цільового рівня.
Це автоматично зменшує експозицію у турбулентні періоди й збільшує у спокійні.

**Тип стратегії:** Long/Cash — ми або тримаємо активи, або кеш.
Ми ніколи не шортимо (не ставимо проти активу).
        """
    )

    st.info(
        f"**Cross-Asset ETF** — {_N_ETF} активів у {_N_ASSET_CLASSES} класах "
        "(акції, облігації, товари, нерухомість, крипто). "
        "Тактична всепогодна алокація з ATR risk parity + vol-targeting."
    )

    # ------------------------------------------------------------------
    # 2. Всесвіт активів
    # ------------------------------------------------------------------
    st.header("2. Всесвіт активів")

    st.markdown(
        "Стратегія інвестує в різні класи активів, щоб не класти "
        "всі яйця в один кошик. Ось чим ми торгуємо:"
    )

    with st.expander("Розподіл Cross-Asset ETF за класами активів"):
        class_data = sorted(_ASSET_CLASS_COUNTS.items(), key=lambda x: -x[1])
        st.table(
            {
                "Клас активів": [c for c, _ in class_data],
                "Кількість": [n for _, n in class_data],
            }
        )

    # ------------------------------------------------------------------
    # 3. Індикатор KAMA
    # ------------------------------------------------------------------
    st.header("3. Індикатор KAMA (фільтр тренду)")
    st.markdown(
        f"""
**KAMA** (Kaufman's Adaptive Moving Average) — це розумна ковзна середня, яка
автоматично прискорюється на трендових ринках і сповільнюється на бокових.

- **Період для активів:** `{_DEFAULTS.kama_asset_period}` торгових днів

KAMA використовує *Efficiency Ratio* (відношення чистого руху ціни до загальної
волатильності). Якщо ціна рухається стабільно в одному напрямку, KAMA слідує
швидко. На хаотичному, боковому ринку KAMA майже не рухається.

> **Для початківців:** уявіть KAMA як "розумну лінію тренду", що слідує за ціною,
> але не смикається від кожного дрібного коливання. Це як шумоподавлення
> для ринкових даних.

У стратегії KAMA використовується як **фільтр тренду** (сигнали входу/виходу),
а не для ранжування моментуму.
        """
    )

    # ------------------------------------------------------------------
    # 4. Відбір кандидатів
    # ------------------------------------------------------------------
    st.header("4. Як ми обираємо активи (конвеєр відбору)")

    st.markdown(
        f"""
Актив потрапляє в портфель через чотири послідовних кроки:

**Крок 1 — Фільтр KAMA активу (Абсолютний Моментум):**
Ціна закриття активу > KAMA × (1 + {_BUFFER_PCT:.1f}%). Кожен актив повинен бути
вище свого власного тренду. Глобального фільтра режиму немає — кожен актив
відповідає сам за себе.

> *Простою мовою:* кожен актив має бути у висхідному тренді. Якщо акції падають,
> вони не проходять свій власний фільтр тренду. Але крипта чи сировина можуть
> залишитися, якщо їхній тренд тримається — справжня міжкласова диверсифікація.

**Крок 2 — Фільтр гепів:**
Жодного одноденного руху > {_GAP_PCT:.0f}% за останні `{max(_DEFAULTS.r2_windows)}` днів.
Це відфільтровує активи зі спотвореними трендами через спліти акцій, гепи на звітах тощо.

> *Простою мовою:* ми уникаємо активів з раптовими, різкими стрибками —
> вони псують наші розрахунки тренду.

        """
    )

    st.subheader("Крок 3 — R² Momentum Scoring та вибір Top-N")
    st.markdown(
        f"""
**Ранжування за R² Momentum Score (метод Кленова).**

Формула: `score = annualized_slope × R²`

Для кожного активу ми будуємо OLS-регресії логарифму його ціни за кілька вікон
(`{_DEFAULTS.r2_windows}` днів) та змішуємо оцінки з вагами
`{_DEFAULTS.r2_weights}`:
- **annualized_slope** — нахил регресії × 252 (річна трендова дохідність)
- **R²** — коефіцієнт детермінації (0..1), вимірює "гладкість" тренду

**Множник R² штрафує хаотичний рух:**
- R² = 0.9 (ідеальний тренд) → зберігає 90% оцінки
- R² = 0.5 (зашумлений тренд) → зрізає 50%
- R² = 0.2 (хаотичний рух) → залишається лише 20%

Ми обираємо **Top-{_DEFAULTS.top_n}** активів з найвищим score > 0.

> **Аналогія:** якщо два активи зросли на 30% за період, той, чий графік
> виглядає як пряма лінія, має пріоритет над тим, що рухається зигзагами.
        """
    )

    # ------------------------------------------------------------------
    # 5. Правила продажу
    # ------------------------------------------------------------------
    st.header("5. Коли ми продаємо (Lazy Hold)")
    st.markdown(
        f"""
Позиція продається, коли виконується будь-яка з цих умов:

- **KAMA стоп:** ціна закриття < KAMA × (1 − {_BUFFER_PCT:.1f}%)
- **Вихід за гепом:** одноденний рух > {_GAP_PCT:.0f}% — негайний вихід
- **R² Score < 0** або актив не проходить фільтри на дату ребалансування

Позиція **НЕ** продається лише тому, що актив випав з Top-{_DEFAULTS.top_n}
за моментумом. Поки його власний тренд тримається, ми його зберігаємо.

Ребалансування (перевірка сигналів) відбувається кожні **{_DEFAULTS.rebal_period_weeks} тижні**,
а не щодня. Це зменшує оборот і торгові витрати.

> **Для початківців:** це як індивідуальний захист для кожного активу.
> Ми перевіряємо портфель регулярно, але не панікуємо щодня. Якщо актив
> все ще зростає, ми його тримаємо, навіть якщо новіші виглядають трохи краще.
        """
    )

    # ------------------------------------------------------------------
    # 6. Управління ризиками — 4 стовпи
    # ------------------------------------------------------------------
    st.header("6. Чотири стовпи управління ризиками")

    st.markdown(
        "Стратегія має чотири рівні захисту вашого капіталу. "
        "Кожен працює незалежно від інших:"
    )

    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)

    with col1:
        st.subheader("Гістерезис")
        st.markdown(
            f"""
**"Простір для дихання ринку"**

Буфер ±{_BUFFER_PCT:.1f}% навколо KAMA запобігає хибним
перемиканням на бокових ринках.

**Аналогія:** як термостат з допуском — він не вмикає
і не вимикає обігрівач від кожного коливання на 0.1°.
            """
        )

    with col2:
        st.subheader("Lazy Hold")
        st.markdown(
            """
**"Терплячий" стоп-лосс**

Позиції утримуються, поки їхній власний тренд тримається
(ціна > KAMA). Ми не продаємо актив лише тому, що
він випав з Top-N, якщо він все ще зростає.

_Деталі в Розділі 5._
            """
        )

    with col3:
        st.subheader("ATR Risk Parity")
        st.markdown(
            f"""
**Розмір позицій за ATR**

Вага кожного активу пропорційна:
`risk_factor / (ATR / price)`

Де ATR — середній денний рух ціни
за `{_DEFAULTS.atr_period}` днів.

Спокійні активи (облігації) отримують більше капіталу,
волатильні (крипто) — менше.

> **Аналогія:** кожен актив "ризикує" приблизно однаковою сумою грошей.
            """
        )

    with col4:
        st.subheader("Vol-Targeting")
        st.markdown(
            f"""
**Оверлей волатильності**

Портфель масштабується щодня:
`scale = target_vol / realised_vol`

Цільова волатильність: **{_DEFAULTS.target_vol:.0%}** річних.
Максимальне плече: **{_DEFAULTS.max_leverage}x**.

Якщо ринок спокійний — позиції збільшуються.
Якщо волатильний — автоматично зменшуються.

> **Аналогія:** автопілот, що тримає "швидкість ризику" стабільною.
            """
        )

    # ------------------------------------------------------------------
    # 7. Таблиця параметрів
    # ------------------------------------------------------------------
    st.header("7. Параметри стратегії")

    tab_core, tab_adv = st.tabs(["Основні параметри", "Додаткові параметри"])

    with tab_core:
        st.table(
            {
                "Параметр": [
                    "Період даних",
                    "Початковий капітал",
                    "R² Lookback",
                    "Період KAMA (актив)",
                    "Буфер KAMA (гістерезис)",
                    "Кількість позицій (Top N)",
                    "Період ребалансування",
                    "Комісія за угоду",
                    "Прослизання",
                    "Безризикова ставка",
                ],
                "За замовчуванням": [
                    "3 роки",
                    f"${INITIAL_CAPITAL:,.0f}",
                    f"{_DEFAULTS.r2_windows} днів (блендинг)",
                    f"{_DEFAULTS.kama_asset_period} днів",
                    f"±{_BUFFER_PCT:.1f}%",
                    f"{_DEFAULTS.top_n} активів",
                    f"{_DEFAULTS.rebal_period_weeks} тижні",
                    f"{COMMISSION_RATE:.2%} ({COMMISSION_RATE * 10_000:.0f} bps)",
                    f"{SLIPPAGE_RATE:.2%} ({SLIPPAGE_RATE * 10_000:.0f} bps)",
                    f"{RISK_FREE_RATE:.0%} річних",
                ],
                "Діапазон": [
                    "3 – 5 років",
                    "$1,000 – $10,000,000",
                    "60 – 120 днів",
                    "10 – 50 днів",
                    "0.5% – 3.0%",
                    "5 – 15",
                    "2 – 4 тижні",
                    "(фіксовано)",
                    "(фіксовано)",
                    "(фіксовано)",
                ],
                "Що це робить": [
                    "Скільки років історичних даних завантажувати",
                    "Стартова сума для симуляції",
                    "Вікно OLS-регресії для оцінки моментуму",
                    "Вікно фільтра тренду окремого активу",
                    "Захист від хибних перемикань bull/bear",
                    "Максимум активів у портфелі одночасно",
                    "Як часто перевіряємо сигнали (lazy-hold)",
                    "Витрати брокера за угоду",
                    "Різниця між очікуваною та фактичною ціною",
                    "Використовується для розрахунку Sharpe Ratio",
                ],
            }
        )

    with tab_adv:
        st.table(
            {
                "Параметр": [
                    "Gap Threshold",
                    "Період ATR",
                    "Risk Factor",
                    "Цільова волатильність",
                    "Максимальне плече",
                    "Vol Lookback",
                ],
                "За замовчуванням": [
                    f"{_DEFAULTS.gap_threshold:.0%}",
                    f"{_DEFAULTS.atr_period} днів",
                    f"{_DEFAULTS.risk_factor} ({_DEFAULTS.risk_factor * 10_000:.0f} bps)",
                    f"{_DEFAULTS.target_vol:.0%}",
                    f"{_DEFAULTS.max_leverage}x",
                    f"{_DEFAULTS.portfolio_vol_lookback} днів",
                ],
                "Що це робить": [
                    "Виключити активи з одноденним рухом, що перевищує цей поріг",
                    "Вікно для розрахунку ATR (розмір позицій)",
                    "Денний ризик на позицію (для ATR risk parity)",
                    "Цільова річна волатильність портфеля",
                    "Максимальний коефіцієнт масштабування",
                    "Вікно для оцінки реалізованої волатильності",
                ],
            }
        )

    st.info(
        "Усі параметри можна налаштувати на дашборді через бокову панель. "
        "Увімкніть **Walk-Forward** оптимізацію для автоматичного підбору параметрів "
        "(Optuna TPE sampler)."
    )

    # ------------------------------------------------------------------
    # 8. Модель виконання
    # ------------------------------------------------------------------
    st.header("8. Як виконуються угоди")
    st.markdown(
        f"""
| Крок | Час | Дія |
|------|-----|-----|
| 1 | Кожні {_DEFAULTS.rebal_period_weeks} тижні, закриття дня T | Аналіз сигналів: KAMA активу, фільтр гепів, R² scoring |
| 2 | Закриття дня T | Визначити продажі (пробій KAMA, невдалі фільтри) та покупки (нові кандидати top-N) |
| 3 | Відкриття дня T+1 | Виконати угоди за ціною відкриття |
| 4 | Щодня | Vol-targeting оверлей: масштабувати позиції за target_vol / realised_vol |
| 5 | Щодня | Вихід за гепом: негайний вихід, якщо |daily_return| > поріг |

> **Чому не на закритті?** Сигнали формуються на закритті, але торгувати за тією ж
> ціною неможливо в реальному житті. Виконання на відкритті наступного дня —
> більш реалістична модель.

> **Lazy-Hold:** між датами ребалансування позиції залишаються незмінними
> (окрім vol-targeting та виходу за гепом). Це зменшує оборот.
        """
    )

    # ------------------------------------------------------------------
    # 9. Walk-Forward оптимізація
    # ------------------------------------------------------------------
    st.header("9. Walk-Forward оптимізація параметрів")

    st.warning(
        "**Для початківців:** цей розділ для досвідчених користувачів. Walk-Forward "
        "оптимізація перевіряє, чи працює стратегія не лише на минулих даних, "
        "а й на нових, невідомих. Якщо ви тільки починаєте, можете "
        "пропустити цей розділ."
    )

    st.markdown(
        f"""
**Мета:** знайти параметри, які працюють не лише на історичних даних,
а й на нових, невідомих даних (out-of-sample).

**Як це працює:** дані розбиваються на ковзні вікна:
- **In-Sample (IS):** тренувальне вікно — оптимізація параметрів
- **Out-of-Sample (OOS):** валідаційне вікно — перевірка якості

**Цільова функція:** CAGR (максимізація) з обмеженням MaxDD ≤ 25%.

Простір пошуку ({_N_SENSITIVITY} Optuna TPE спроб на крок):
        """
    )

    grid_data = {
        "Параметр": [],
        "Простір пошуку": [],
    }
    for name in _PARAM_NAMES:
        spec = SEARCH_SPACE[name]
        grid_data["Параметр"].append(name)
        if spec.get("type") == "categorical":
            grid_data["Простір пошуку"].append(
                ", ".join(str(v) for v in spec["choices"])
            )
        else:
            grid_data["Простір пошуку"].append(
                f"{spec['low']} – {spec['high']}, крок {spec['step']}"
            )
    st.table(grid_data)

    st.markdown(
        """
**Деградація IS → OOS:** якщо OOS CAGR < 50% від IS CAGR, з'являється
попередження про можливе перенавчання.

Запустити walk-forward оптимізацію:
`uv run python -m src.portfolio_sim walk-forward`
        """
    )

    # ------------------------------------------------------------------
    # 10. Ефективна границя
    # ------------------------------------------------------------------
    st.header("10. Ефективна границя Марковіца")
    st.markdown(
        """
Дашборд включає графік **Risk-Return Distribution & Efficient Frontier**.

**Ефективна границя** — це набір портфелів, які пропонують
**найвищу дохідність для кожного рівня ризику**.

**Як читати графік:**
- **Кожна точка** — один актив (волатильність по осі X, дохідність по осі Y)
- **Золота крива** — ефективна границя
- **Блакитна зірка** — позиція вашого портфеля
- **Сірий ромб** — S&P 500 (бенчмарк)

> **Аналогія:** ефективна границя — як обмеження швидкості автомобіля.
> Можна їхати повільніше, але не можна швидше без збільшення ризику.
        """
    )

    # ------------------------------------------------------------------
    # 11. Переваги
    # ------------------------------------------------------------------
    st.header("11. Переваги цього підходу")
    st.markdown(
        """
**Чому ця гібридна стратегія краща за наївне "купи і тримай"?**

- **Якість тренду, а не просто дохідність.** Множник R² суворо штрафує
  хаотичний рух — стратегія обирає активи з плавними, надійними трендами.

- **Vol-targeting оверлей.** Автоматичне масштабування портфеля до цільової
  волатильності. У турбулентні періоди експозиція зменшується, у спокійні — зростає.

- **Захист від великих просадок.** Індивідуальні KAMA-фільтри виходять
  з кожної позиції окремо, коли її тренд ламається — без глобального kill switch.

- **Справжня міжкласова диверсифікація.** Кожен актив відповідає сам за себе
  через власний KAMA-фільтр. Крипта та сировина можуть залишатися в портфелі
  під час ведмежого ринку акцій, якщо їхні власні тренди тримаються.
  Ліміти по секторах запобігають надмірній концентрації.

- **ATR Risk Parity.** Розміри позицій пропорційні фактичній волатильності
  (ATR), а не рівним вагам. Спокійні активи отримують більше капіталу.

- **Низький оборот.** Lazy-Hold ребалансування кожні N тижнів зменшує кількість
  угод = менші комісії та прослизання.
        """
    )

    # ------------------------------------------------------------------
    # 12. Словник
    # ------------------------------------------------------------------
    with st.expander("Словник термінів для початківців"):
        st.markdown(
            "Не знаєте, що означає термін? Знайдіть його тут:"
        )
        st.markdown(
            f"""
- **ATR (Average True Range)** — середній денний рух ціни активу.
  Використовується для розміру позицій. Вищий ATR = більш волатильний = менша позиція.
- **Бенчмарк** — точка відліку (S&P 500 у нашому випадку) для порівняння вашої
  стратегії. Якщо ви не можете перевершити бенчмарк, краще просто
  купити індексний фонд.
- **CAGR (Compound Annual Growth Rate)** — середньорічна дохідність,
  ніби ваш портфель зростав стабільними темпами щороку.
- **Calmar Ratio** — річна дохідність, поділена на максимальну просадку.
  Чим вище — тим краще.
- **Просадка (Drawdown)** — зниження вартості портфеля від піку до дна.
  Максимальна просадка показує найгірший сценарій, який ви б пережили.
- **Ефективна границя** — крива Марковіца, що показує оптимальні портфелі
  для кожного рівня ризику.
- **ETF (Exchange-Traded Fund)** — фонд, що торгується на біржі
  як акція, відстежуючи індекс, товар або кошик активів.
- **Геп** — раптовий одноденний стрибок ціни (>{_GAP_PCT:.0f}%).
  Гепи спотворюють регресійні оцінки, тому такі активи виключаються.
- **Гістерезис** — буфер, що запобігає частим хибним перемиканням між
  бичачим і ведмежим режимами.
- **KAMA** — Kaufman's Adaptive Moving Average. Реагує на силу тренду.
  Використовується як фільтр тренду в стратегії.
- **Lazy Hold** — позиція утримується, поки її власний тренд тримається,
  навіть якщо вона випала з Top-N. Зменшує оборот.
- **Long/Cash** — стратегія, яка або тримає активи (long), або кеш.
  Без коротких продажів.
- **Моментум** — тенденція активів, що нещодавно зростали,
  продовжувати зростати.
- **OLS-регресія** — метод найменших квадратів. Ми будуємо пряму лінію
  по логарифму ціни за N днів. Нахил показує середню денну
  трендову дохідність.
- **R² (коефіцієнт детермінації)** — вимірює, наскільки добре лінійна
  регресія відповідає даним. R² = 1 означає ідеальну пряму лінію,
  R² = 0 — випадковий рух. Використовується для штрафування нестабільних трендів.
- **R² Momentum Score** — `annualized_slope × R²`. Основна метрика для
  ранжування активів. Високий score = сильний і стабільний тренд.
- **Ребалансування** — періодичний процес перегляду та коригування
  портфеля. Ми робимо це кожні {_DEFAULTS.rebal_period_weeks} тижні.
- **Risk Factor** — денний ризик на позицію ({_DEFAULTS.risk_factor * 10_000:.0f} bps).
  Визначає, скільки капіталу алокується на кожен актив відносно його ATR.
- **Sharpe Ratio** — міра дохідності з поправкою на ризик. Показує, скільки
  додаткової дохідності (понад безризикову ставку {RISK_FREE_RATE:.0%}) ви отримуєте
  на одиницю ризику. Вище 1 — добре, вище 2 — відмінно.
- **SPY** — ETF, що відстежує індекс S&P 500. Використовується як наш бенчмарк
  і фільтр режиму.
- **Vol-Targeting** — оверлей, що масштабує портфель до цільової
  волатильності. target_vol / realised_vol визначає множник.
- **Walk-Forward оптимізація** — ковзний процес оптимізації: тренування
  параметрів на in-sample вікні, валідація на out-of-sample вікні.
  Захищає від перенавчання.
            """
        )


def page():
    lang = st.segmented_control(
        "lang",
        options=["English", "Українська"],
        default="English",
        key="lang",
        label_visibility="collapsed",
    )

    if lang == "Українська":
        _page_uk()
    else:
        _page_en()
