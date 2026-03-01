Сергей, я внимательно изучил весь код. Проблема "полсрока проигрыш SP500" — системная, и корень в нескольких архитектурных решениях. Вот детальный разбор и конкретные улучшения.

---

## Корневые причины отставания

**1. Вселенная на ~80% состоит из equity — а стратегия Long/Cash**

Из ~160 тикеров около 128 — акции или equity ETF. Даже с `corr_threshold=0.7` портфель top_n=10 скорее всего 7-8 позиций equity + 2-3 остальное. Результат: ты платишь транзакционные издержки за фактически equity exposure, который проще получить через B&H SPY бесплатно. Когда стратегия в кэше — ты упускаешь ралли. Когда в позициях — ты по сути SPY с костами.

**2. KAMA whipsaw = главный убийца альфы**

В боковом рынке (весь 2023 Q1-Q2, большая часть 2022) стратегия делает так:
- Сигнал Close < KAMA*0.99 → выход (с проскальзыванием + комиссией)
- Через 2 недели Close > KAMA*1.01 → вход (снова косты)
- Повторить 3-5 раз за квартал

При COST_RATE = 0.07% на round trip для 10 позиций это ~1.4% за один цикл whipsaw. 4 цикла в год = 5-6% чистого drag'а. SPY B&H платит 0%.

**3. V2 убрал SPY regime filter — а замены нет**

V1 engine имеет `risk_off` режим (SPY < KAMA → ликвидация всего). V2 **полностью убрал это**, оставив только индивидуальные KAMA стоп-лоссы. В результате V2 держит позиции в медвежьем рынке дольше, выходит поздно и поштучно.

**4. Momentum lookback слишком короткий**

`lookback_period` в V2 ищется от 20 до 100 дней. Академическая литература (Jegadeesh & Titman, Asness) показывает, что equity momentum работает оптимально на 6-12 месяцев (126-252 дня). Короткий lookback ловит noise, а не trend.

**5. WFO переоптимизирован**

10 свободных параметров, 126 дней IS, 50-200 Optuna trials. Это классический overfitting: Optuna находит параметры, которые идеально подходят к последним 6 месяцам, но degradation на OOS закономерна.

---

## Конкретные улучшения

### A. Asset-class level allocation constraint (критично)

Сейчас `get_buy_candidates` ранжирует всех по score и берёт top_n. Нужен **asset-class budget**:

```python
# В alpha.py или engine.py
ASSET_CLASS_LIMITS = {
    "equity": 0.50,    # max 50% в equity (US + Intl + EM + sector ETFs)
    "bonds": 0.30,     # max 30% в bonds
    "commodities": 0.15, # metals + commodities
    "alternatives": 0.20, # real estate + crypto
}

def get_buy_candidates_diversified(
    candidates_ranked: list[str],
    current_holdings: dict[str, float],
    asset_class_map: dict[str, str],
    class_limits: dict[str, float],
    top_n: int,
) -> list[str]:
    """Выбирает кандидатов с учётом лимитов на asset class."""
    class_counts = Counter()
    # Считаем текущую аллокацию по классам
    for ticker in current_holdings:
        cls = _broad_class(asset_class_map.get(ticker, "equity"))
        class_counts[cls] += 1
    
    max_per_class = {
        cls: max(1, int(top_n * limit))
        for cls, limit in class_limits.items()
    }
    
    result = []
    for t in candidates_ranked:
        cls = _broad_class(asset_class_map.get(t, "equity"))
        if class_counts[cls] < max_per_class.get(cls, top_n):
            result.append(t)
            class_counts[cls] += 1
        if len(result) >= top_n:
            break
    return result
```

Это **одно изменение** даст самый большой эффект: в периоды equity drawdown стратегия будет держать bonds/gold, что сгладит кривую.

### B. Улучшенный regime filter для V2

V2 убрал SPY filter — нужно вернуть, но умнее:

```python
# Вместо бинарного risk_on/risk_off:
# Используй градуированный risk budget
def compute_risk_budget(spy_price, spy_kama, spy_er, kama_buffer):
    """Возвращает float 0.0-1.0: доля капитала для risky assets."""
    if spy_price > spy_kama * (1 + kama_buffer):
        # Явный uptrend — полный бюджет
        return 1.0
    elif spy_price > spy_kama * (1 - kama_buffer):
        # Зона неопределённости — частичный бюджет  
        # Линейная интерполяция внутри буфера
        position_in_buffer = (spy_price - spy_kama * (1 - kama_buffer)) / (
            spy_kama * 2 * kama_buffer
        )
        return max(0.3, position_in_buffer)  # минимум 30% exposure
    else:
        # Ниже буфера — минимальный бюджет
        # НО: не ноль! Позволяет держать bonds/gold
        return 0.3

# В engine:
risk_budget = compute_risk_budget(spy_close_val, spy_kama_val, ...)
# equity_allocation = risk_budget для equity класса
# bonds/metals = 1.0 (всегда доступны)
```

Это решает проблему полного выхода в кэш: стратегия **снижает equity** exposure, но сохраняет bonds/gold, что часто растут в кризисы.

### C. Увеличить momentum lookback + dual momentum

```python
# config.py / V2 search space
# Было:
"lookback_period": {"type": "int", "low": 20, "high": 100, "step": 20},

# Надо:
"lookback_period": {"type": "int", "low": 60, "high": 252, "step": 21},
```

А ещё лучше — **dual momentum** (Antonacci style): сравнивать absolute momentum (>0) И relative momentum (vs. risk-free rate):

```python
def score_dual_momentum(
    raw_return: float,  # return за lookback
    er_squared: float,  # ER² quality
    risk_free_return: float,  # T-bill return за тот же период
) -> float:
    """Dual momentum: возвращает score или -1 если absolute momentum < 0."""
    # Absolute momentum filter: актив должен бить risk-free
    if raw_return < risk_free_return:
        return -1.0  # не покупать
    excess_return = raw_return - risk_free_return
    return excess_return * er_squared
```

Сейчас `get_buy_candidates` фильтрует только `raw_return > 0`. При 4% risk-free rate актив с +2% return — это отрицательная excess alpha. Нет смысла покупать.

### D. Антиwhipsaw: минимальный holding period

```python
# В engine state:
entry_dates: dict[str, pd.Timestamp] = {}  # когда вошли

# В sell logic:
MIN_HOLD_DAYS = 10  # минимум 2 недели

for t in list(shares.keys()):
    # Проверяем KAMA stop-loss
    if daily_close.get(t, 0.0) < t_kama * (1 - p.kama_buffer):
        # НО: продаём только если держали >= MIN_HOLD_DAYS
        if (date - entry_dates.get(t, date)).days >= MIN_HOLD_DAYS:
            sells[t] = 0.0
```

Это резко снижает whipsaw в боковых рынках. Можно добавить `min_hold_days` в search space (5-20).

### E. Уменьшить размерность оптимизации

Сейчас V2 ищет 10 параметров. Зафиксируй стабильные:

```python
# Зафиксировать (не оптимизировать):
FIXED_V2 = {
    "weighting_mode": "risk_parity",  # уже fixed
    "kama_buffer": 0.01,              # мало влияет
    "oos_days": 21,                   # WFO schedule, не strategy param
    "max_leverage": 1.0,              # убрать leverage для стабильности
}

# Оптимизировать только 5-6 ключевых:
V2_SEARCH_SPACE_LEAN = {
    "kama_period": {"type": "categorical", "choices": [20, 30, 40]},
    "lookback_period": {"type": "int", "low": 63, "high": 252, "step": 21},
    "top_n": {"type": "int", "low": 6, "high": 15, "step": 3},
    "corr_threshold": {"type": "float", "low": 0.5, "high": 0.8, "step": 0.1},
    "target_vol": {"type": "float", "low": 0.08, "high": 0.15, "step": 0.01},
}
```

5 параметров вместо 10 = экспоненциально меньше overfitting при том же числе trials.

### F. Сократить и перебалансировать вселенную

Текущий ETF_UNIVERSE раздут. Много мелких, неликвидных, дублирующих ETF. Предлагаю:

```python
ETF_UNIVERSE_LEAN = [
    # US Equity (10 крупнейших + QQQ)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B", "JPM", "LLY",
    "QQQ",
    
    # US Sectors (6 ключевых)
    "XLK", "XLF", "XLE", "XLV", "XLI", "SMH",
    
    # International (5)
    "VEA", "VWO", "ASML", "TSM", "MELI",
    
    # Bonds (6 — разные дюрации)
    "TLT", "IEF", "SHY", "LQD", "HYG", "EMB",
    
    # Metals & Commodities (4)
    "GLD", "SLV", "CPER", "LIT",
    
    # Real Estate (3)
    "VNQ", "VNQI", "REM",
    
    # Crypto (2)
    "IBIT", "ETHA",
]
# Итого ~37 тикеров вместо 160
```

Преимущества: быстрее WFO, меньше noise в корреляционной матрице, каждый тикер представляет уникальный exposure.

### G. Benchmark-aware objective

Сейчас объективная функция — Calmar или Sharpe абсолютный. Но ты хочешь бить SPY. Добавь **information ratio** или **excess Sharpe**:

```python
def compute_objective_vs_benchmark(
    equity: pd.Series,
    spy_equity: pd.Series,
    max_dd_limit: float = 0.20,
    min_n_days: int = 60,
) -> float:
    """Maximize excess return per unit of tracking error."""
    if len(equity) < min_n_days:
        return -999.0
    
    metrics = compute_metrics(equity)
    if metrics["max_drawdown"] > max_dd_limit:
        return -999.0
    
    # Information Ratio = (strategy_return - benchmark_return) / tracking_error
    strat_returns = equity.pct_change().dropna()
    spy_returns = spy_equity.pct_change().dropna()
    
    # Align
    common = strat_returns.index.intersection(spy_returns.index)
    excess = strat_returns.loc[common] - spy_returns.loc[common]
    
    if excess.std() < 1e-8:
        return -999.0
    
    ir = (excess.mean() * 252) / (excess.std() * np.sqrt(252))
    return ir if ir > 0 else -999.0
```

Это заставит оптимизатор искать параметры, которые **именно обыгрывают** SPY, а не просто имеют хороший абсолютный Sharpe.

---

## Приоритет реализации

По impact/effort:

1. **Asset-class limits** (A) — максимальный эффект, ~2 часа работы
2. **Увеличить lookback + dual momentum** (C) — простое изменение, сильный эффект  
3. **Вернуть градуированный regime filter в V2** (B) — умеренная сложность
4. **Сократить вселенную** (F) — тривиально, сразу ускорит WFO
5. **Сократить search space** (E) — тривиально
6. **Min holding period** (D) — простое, антиwhipsaw
7. **Benchmark-aware objective** (G) — средняя сложность, но точно нужен

Хочешь, чтобы я реализовал какой-то из этих пунктов в коде?