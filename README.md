# Optimal Portfolio — Hybrid R² Momentum + Vol-Targeting Strategy

Гибридная систематическая моментум-стратегия: R² Momentum scoring (Кленов) +
vol-targeting overlay, KAMA тренд-фильтры, ATR risk parity sizing,
кореляционный фильтр, Walk-Forward оптимизация.

**Тип стратегии:** Long/Cash — покупаем активы с сильным, стабильным трендом,
выходим в кэш при развороте. Без шортов.

## Содержание

- [Вселенная активов](#вселенная-активов)
- [Пайплайн поиска оптимального портфеля](#пайплайн-поиска-оптимального-портфеля)
- [Алгоритмы отбора активов](#алгоритмы-отбора-активов)
- [Параметры стратегии](#параметры-стратегии)
- [Метрики производительности](#метрики-производительности)
- [Использование](#использование)
- [Технологический стек](#технологический-стек)

---

## Вселенная активов

37 тикеров из 7 классов активов (тактическая всепогодная аллокация):

| Класс актива | Кол-во | Примеры |
|---|---|---|
| US Equity | 11 | AAPL, MSFT, NVDA, GOOGL, QQQ |
| US Sector ETFs | 6 | XLK, XLF, XLE, SMH |
| International | 5 | VEA, VWO, ASML, TSM, MELI |
| Bonds | 6 | TLT, IEF, SHY, LQD, HYG, EMB |
| Metals & Commodities | 4 | GLD, SLV, CPER, LIT |
| Real Estate | 3 | VNQ, VNQI, REM |
| Crypto ETFs | 2 | IBIT, ETHA |

Бенчмарк: **S&P 500 (SPY)** — buy-and-hold.

---

## Пайплайн поиска оптимального портфеля

```
1. Загрузка данных (yfinance → Parquet-кэш)
   ↓
2. Расчёт индикаторов (KAMA для тренд-фильтров)
   ↓
3. Генерация WFO-расписания (скользящие окна IS/OOS)
   ↓
4. Для каждого WFO-шага:
   ├─ 4a. Оптимизация на IS-данных (Optuna TPE → макс. CAGR)
   ├─ 4b. Симуляция на IS с лучшими параметрами
   └─ 4c. Валидация на OOS (параметры НЕ оптимизированы на OOS)
   ↓
5. Сшивка OOS-эквити → агрегированная производительность
   ↓
6. Итоговые параметры (из последнего IS-окна)
```

---

## Алгоритмы отбора активов

### R² Momentum scoring (метод Кленова)

Для каждого актива фитим OLS-регрессию к лог-ценам за последние N дней:

```
log_prices = log(close[-N:])
slope, intercept = polyfit(x, log_prices, 1)
annualized_return = exp(slope × 252) − 1
R² = 1 − SS_res / SS_tot
score = annualized_return × R²
```

### Каскад фильтров

1. **SPY KAMA режим:** `SPY_close > KAMA(SPY) × (1 − buffer)` — бычий рынок
2. **Asset KAMA тренд:** `close > KAMA(asset) × (1 − buffer)` — актив в тренде
3. **Gap фильтр:** никаких однодневных движений > gap_threshold
4. **Кореляционный фильтр:** новые входы проверяются на корреляцию с текущими позициями
5. **R² Momentum score > 0** — положительный и стабильный тренд

### Vol-Targeting Overlay

Портфель масштабируется ежедневно: `scale = target_vol / realised_vol`,
ограничен `max_leverage`. Автоматически уменьшает экспозицию в волатильные
периоды и увеличивает в спокойные.

### Размер позиций — ATR Risk Parity

```
weight[i] = (risk_factor / (ATR[i] / price[i])) / Σ(...)
```

Низко-ATR активы (облигации) получают больше капитала.

---

## Параметры стратегии

### Основные параметры

| Параметр | По умолч. | Диапазон WFO | Описание |
|---|---|---|---|
| `r2_lookback` | 90 | 60–120 (шаг 20) | Окно OLS-регрессии для R² скоринга |
| `kama_asset_period` | 10 | 10, 20, 30, 40, 50 | KAMA для тренд-фильтра актива |
| `kama_spy_period` | 40 | 20, 30, 40, 50 | KAMA для режимного фильтра SPY |
| `kama_buffer` | 0.005 | 0.005–0.03 (шаг 0.005) | Буфер гистерезиса |
| `top_n` | 5 | 5–15 (шаг 5) | Макс. позиций в портфеле |
| `rebal_period_weeks` | 3 | 2–4 (шаг 1) | Период ребалансирования (недели) |

### Vol-Targeting параметры

| Параметр | По умолч. | Диапазон WFO | Описание |
|---|---|---|---|
| `target_vol` | 0.10 (10%) | 0.05–0.20 | Целевая годовая волатильность |
| `max_leverage` | 1.5 | 1.0, 1.25, 1.5, 2.0 | Макс. масштабный коэффициент |
| `portfolio_vol_lookback` | 21 | 15–35 (шаг 10) | Окно оценки реализованной вол. |
| `corr_threshold` | 0.7 | 0.5–1.0 (шаг 0.1) | Порог корреляции для новых входов |

### Фиксированные параметры

| Параметр | Значение | Описание |
|---|---|---|
| `INITIAL_CAPITAL` | $10,000 | Стартовый капитал |
| `COMMISSION_RATE` | 0.02% (2 bps) | Комиссия брокера |
| `SLIPPAGE_RATE` | 0.05% (5 bps) | Проскальзывание |
| `RISK_FREE_RATE` | 4% годовых | Для расчёта Sharpe Ratio |
| `risk_factor` | 0.001 (10 bps) | Риск на позицию в день |

---

## Метрики производительности

| Метрика | Формула | Что показывает |
|---|---|---|
| Total Return | equity_end / equity_start − 1 | Кумулятивная доходность |
| CAGR | (equity_end / equity_start)^(252/days) − 1 | Среднегодовая доходность |
| Max Drawdown | max(peak − trough) / peak | Макс. просадка от пика |
| Sharpe Ratio | (CAGR − risk_free) / ann_vol | Доходность на единицу риска |
| Calmar Ratio | CAGR / max_drawdown | Доходность на единицу просадки |
| Annualized Vol | std(daily_returns) × √252 | Годовая волатильность |
| Win Rate | дней с return > 0 / всего дней | Доля прибыльных дней |

---

## Использование

### CLI: Walk-Forward оптимизация

```bash
# Установка
uv sync

# Запуск WFO (параметры по умолчанию)
uv run python -m src.portfolio_sim walk-forward

# С указанием параметров
uv run python -m src.portfolio_sim walk-forward \
    --period 3y \
    --n-trials 100 \
    --n-workers 8 \
    --refresh
```

| Флаг | По умолч. | Описание |
|---|---|---|
| `--period` | 3y | Период данных (формат yfinance) |
| `--n-trials` | 100 | Кол-во триалов Optuna на каждый IS-шаг |
| `--n-workers` | cpu_count − 1 | Параллельные воркеры |
| `--refresh` | — | Принудительная перезагрузка данных |

### Streamlit Dashboard

```bash
uv run streamlit run app.py
```

Две страницы:

- **Backtest** — интерактивный бэктест с настройкой параметров через слайдеры
- **Правила стратегії** — документация по стратегии

---

## Технологический стек

| Технология | Назначение |
|---|---|
| **Python 3.12+** | Язык |
| **Numba** | JIT-компиляция KAMA/ER (100x ускорение) |
| **Optuna** | Байесовская оптимизация параметров (TPE) |
| **Pandas / NumPy** | Работа с временными рядами |
| **yfinance** | Загрузка рыночных данных |
| **Streamlit** | Интерактивный дашборд |
| **Parquet** | Кэш данных (быстрый, компактный) |
| **ProcessPoolExecutor** | Параллелизм (обход GIL) |

---

## Структура проекта

```
src/portfolio_sim/
├── cli.py              # CLI-диспетчер
├── command.py          # CLI-обработчик команды walk-forward
├── config.py           # Параметры, вселенная активов, пространство поиска
├── engine.py           # Hybrid bar-by-bar simulation engine
├── params.py           # StrategyParams (frozen dataclass)
├── parallel.py         # Parallel execution utilities (ProcessPoolExecutor)
├── optimizer.py        # KAMA precomputation, objective functions, sensitivity
├── walk_forward.py     # WFO schedule generation, walk-forward runner
├── models.py           # SimulationResult, WFOStep, WFOResult
├── data.py             # Загрузка данных (yfinance + Parquet-кэш)
├── indicators.py       # KAMA и ER (Numba JIT)
├── reporting.py        # Метрики, графики, отчёты
└── cli_utils.py        # Общие CLI-утилиты

scripts/
└── backtest_s2.py      # Бэктест с фиксированными параметрами
```
