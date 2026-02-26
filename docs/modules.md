# Описание модулей проекта optimal_portfolio

KAMA Momentum Strategy — фреймворк для бэктестинга адаптивной моментум-стратегии на кросс-активном ETF-юниверсе с walk-forward оптимизацией и анализом чувствительности параметров.

**Стек:** Python 3.14, pandas, NumPy, Numba, yfinance, Optuna, Streamlit, Plotly, matplotlib, structlog.

---

## Точки входа

### `run_custom_sim.py`
Standalone-скрипт для запуска единичного бэктеста с фиксированными параметрами. Загружает ETF-юниверс, скачивает 5 лет ценовых данных (с Parquet-кешем), запускает симуляцию с заданными параметрами (KAMA=20, lookback=60, top_n=20, buffer=0.02) и выводит equity-кривую, лог сделок и метрики (CAGR, Sharpe, max drawdown).

### `app.py`
Точка входа Streamlit multi-page приложения. Маршрутизирует на страницу Backtest (`dashboard.py`) и страницу правил стратегии (`pages/strategy_rules.py`).

### `dashboard.py`
Интерактивный Streamlit-дашборд. Предоставляет слайдеры параметров, equity-кривую (Plotly), таблицы помесячной и годовой доходности, rolling Sharpe, drawdown, визуализацию режимов рынка (бык/медведь), аллокацию по активам и журнал сделок.

### CLI: `src/portfolio_sim/__main__.py` + `cli.py`
Диспетчер CLI-подкоманд:
- `portfolio optimize` — анализ чувствительности
- `portfolio max-profit` — поиск макс. CAGR
- `portfolio walk-forward` — walk-forward оптимизация

`cli.py` парсит аргументы через `argparse` и делегирует выполнение в `commands/`.

---

## Ядро: `src/portfolio_sim/`

### `config.py`
Центральный конфиг с фиксированными (неоптимизируемыми) константами:
- **Торговые издержки:** начальный капитал (10 000), комиссия (2 bps), проскальзывание (5 bps), безрисковая ставка (4%).
- **Дефолтные параметры:** KAMA_PERIOD=20, LOOKBACK_PERIOD=60, TOP_N=10, KAMA_BUFFER=0.024.
- **ETF_UNIVERSE:** ~160 тикеров — US Equity, Intl Equity, EM Equity, Sector ETFs, Bonds, Metals, Real Estate, Crypto ETFs.
- **ASSET_CLASS_MAP:** маппинг тикер → класс актива для репортинга.
- **Пространства поиска:** `SENSITIVITY_SPACE` и `MAX_PROFIT_SPACE` — диапазоны параметров для Optuna.

### `params.py`
`StrategyParams` — frozen dataclass (иммутабельный, хешируемый) — контейнер параметров стратегии:
- `kama_period` (20) — окно KAMA.
- `lookback_period` (60) — окно моментума.
- `top_n` (10) — макс. позиций.
- `kama_buffer` (0.024) — буфер гистерезиса для тренд-фильтра.
- `use_risk_adjusted` — использовать Sharpe-like моментум (return/vol) или raw return.
- `enable_regime_filter` — глобальный SPY-фильтр режима рынка.
- `enable_correlation_filter` / `correlation_threshold` — жадный фильтр диверсификации.
- `sizing_mode` — `"equal_weight"` или `"risk_parity"`.
- `volatility_lookback`, `max_weight` — параметры sizing.
- Свойство `warmup` — минимум баров до начала торговли.

### `models.py`
Dataclass-контейнеры результатов:
- **`SimulationResult`**: equity-кривая, SPY benchmark, история позиций (DataFrame тикер×дата), история кеша, режим рынка, лог сделок.
- **`WFOStep`**: одна ступень walk-forward — даты IS/OOS, оптимизированные параметры, метрики, equity-кривые.
- **`WFOResult`**: все ступени WFO, сшитая OOS equity, агрегированные метрики, финальные параметры (для live).

### `data.py`
Загрузка и кеширование ценовых данных:
- **`fetch_etf_tickers()`** — возвращает отсортированный список тикеров из `ETF_UNIVERSE`.
- **`fetch_price_data(tickers, period, refresh, cache_suffix, min_rows)`** — возвращает `(close_prices, open_prices)` как DataFrame. Проверяет наличие Parquet-файлов в `output/cache/`, при наличии загружает с диска; иначе качает через yfinance и сохраняет. `refresh=True` — принудительная перезагрузка. `min_rows` — автообновление при недостатке строк.
- **`_download_from_yfinance()`** — скачивает OHLCV батчами по 100 тикеров, forward-fill пропусков, удаление timezone.

### `indicators.py`
Расчёт Kaufman's Adaptive Moving Average (KAMA):
- **`compute_kama(data, period)`** — Numba JIT-компилированный KAMA. Efficiency Ratio (ER) = изменение цены / волатильность адаптирует константу сглаживания (SC). Рекуррентное обновление: `kama[i] = kama[i-1] + SC² × (price[i] - kama[i-1])`. NaN-пэддинг первых `period` баров.
- **`compute_kama_series()`** — обёртка, возвращающая pd.Series с оригинальным индексом.
- **`_kama_recurrent_loop()`** — низкоуровневый Numba-цикл рекурренции.

### `alpha.py`
Генерация альфа-сигналов (отбор кандидатов на покупку):
- **`get_buy_candidates()`** — трёхэтапная фильтрация:
  1. **KAMA Trend Filter:** оставляет тикеры с `Close > KAMA × (1 + buffer)`.
  2. **Momentum Ranking:** ранжирование по risk-adjusted score (return / volatility) или raw return; топ N.
  3. **Correlation Filter (опц.):** жадный отбор — итерирует по рангу, пропускает тикеры с корреляцией > threshold к уже выбранным.
- **`_greedy_correlation_filter()`** — вычисляет корреляции дневных returns, последовательно добавляет некоррелированные тикеры.

### `engine.py`
Bar-by-bar симуляционный движок:
- **`run_simulation(close, open, tickers, params, kama_cache)`** — основной цикл:
  1. На каждом баре (день): вычисляет KAMA, определяет режим рынка (бык/медведь по SPY KAMA с гистерезисом).
  2. В медвежьем режиме — ликвидация всех позиций, кеш.
  3. В бычьем — вызывает `get_buy_candidates()`, определяет сигналы на продажу (KAMA stop-loss), исполняет сделки на Open(T+1).
  4. **Lazy Hold:** позиция продаётся только по собственному стоп-лоссу KAMA, а не из-за падения в ранге моментума.
  5. **Sizing:** equal weight (1/top_n) или risk parity (inverse volatility с cap + redistribute).
  6. Торговые издержки: `price × shares × (commission + slippage)`.
- **`_execute_trades()`** — исполнение продаж и покупок, обновление позиций и кеша.
- **`_compute_inverse_vol_weights()`** — веса обратной волатильности.
- **`_cap_and_redistribute()`** — ограничение макс. веса с перераспределением.
- Возвращает `SimulationResult`.

### `reporting.py`
Метрики и отчётность:
- **`compute_metrics(equity)`** — total return, CAGR, max drawdown, Sharpe, Calmar, annualized vol, win rate, n_days.
- **`compute_drawdown_series()`** — underwater-кривая.
- **`compute_monthly_returns()`** — таблица Год × Месяц.
- **`compute_yearly_returns()`** — годовая доходность.
- **`compute_rolling_sharpe()`** — rolling Sharpe (252-дневное окно).
- **`save_equity_png()`** — двухпанельный PNG: equity vs SPY + drawdown; matplotlib 14×8", 150 dpi.
- **`format_comparison_table()`** — сравнение Strategy vs S&P 500.
- **`format_asset_report()`** — per-asset отчёт: return, vol, дни удержания, кол-во сделок, средний вес, сектор.

### `optimizer.py`
Анализ чувствительности параметров (robustness check, НЕ оптимизация):
- **`run_sensitivity()`** — Optuna TPE-сэмплер:
  1. Предвычисление KAMA для всех значений kama_period (параллельно).
  2. Сэмплирование комбинаций параметров через Optuna.
  3. Параллельная оценка через ProcessPoolExecutor.
  4. Вычисление 1D marginal profiles (среднее значение objective по каждому значению параметра).
  5. Оценка робастности: плоский профиль = робастный параметр (score 0–1).
- **`compute_objective()`** — Calmar ratio (CAGR / MaxDD) с порогом отсечения.
- **`compute_marginal_profiles()`** — маргинальные профили по каждому параметру.
- **`compute_robustness_scores()`** — score 1.0 = идеально плоский профиль.
- **`format_sensitivity_report()`** — текстовый отчёт чувствительности.

### `max_profit.py`
Поиск параметров с максимальным CAGR:
- **`run_max_profit_search()`** — Optuna TPE с целевой функцией CAGR. Мягкий лимит drawdown (60% vs 30% в sensitivity).
- **`compute_cagr_objective()`** — raw CAGR с штрафом за превышение max_dd_limit.
- **`run_pareto_search()`** — мультиобъективная оптимизация NSGA-II: максимизация CAGR + минимизация MaxDD → Парето-фронт.

### `walk_forward.py`
Walk-forward оптимизация (защита от переобучения):
- **`generate_wfo_schedule()`** — генерация скользящих окон: IS = 756 дней (3 года), OOS = 252 дня (1 год), шаг = OOS_DAYS.
- **`run_walk_forward()`** — основной цикл WFO:
  1. Для каждого шага: оптимизация параметров на IS-выборке, тестирование на OOS.
  2. Сшивание OOS equity-кривых.
  3. Итоговые метрики на сшитой кривой.
  4. Возвращает оптимизированные параметры последнего шага (рекомендация для live).

### `parallel.py`
Инфраструктура параллельного исполнения:
- **`init_eval_worker()`** — инициализатор worker-процессов. Сохраняет DataFrames и KAMA-кеши в глобальный `_shared` dict один раз при старте воркера (избегает повторного pickling).
- **`evaluate_combo()`** — оценка одной комбинации параметров. Обращается к `_shared` за данными, запускает симуляцию, возвращает метрики.
- **`suggest_params()`** — конвертация Optuna trial в `StrategyParams`. Поддержка categorical, int, float параметров и fixed overrides.

### `cli_utils.py`
Утилиты для CLI-скриптов:
- **`setup_logging()`** — настройка structlog с console rendering.
- **`create_output_dir(prefix)`** — создание timestamped каталога `output/{prefix}_{YYYYMMDD_HHMMSS}`.
- **`filter_valid_tickers(close_prices, min_days)`** — фильтрация тикеров с минимумом non-NaN строк.

### `commands/`
CLI-подкоманды (делегируют в соответствующие модули):
- **`commands/optimize.py`** — подкоманда `optimize`: загрузка данных → `run_sensitivity()` → отчёт.
- **`commands/max_profit.py`** — подкоманда `max-profit`: загрузка данных → `run_max_profit_search()` → отчёт.
- **`commands/walk_forward.py`** — подкоманда `walk-forward`: загрузка данных → `run_walk_forward()` → отчёт + equity PNG.

---

## Streamlit UI

### `pages/strategy_rules.py`
Страница с описанием правил стратегии на украинском языке. Объясняет логику KAMA-фильтра, режима рынка, отбора кандидатов, lazy hold и sizing.

---

## Тесты: `tests/`

| Файл | Назначение |
|------|------------|
| `conftest.py` | Фикстуры: синтетические цены (500 дней, 20 тикеров + SPY), open prices с шумом |
| `test_config.py` | Проверка констант конфига |
| `test_data.py` | Загрузка данных и кеширование (мок yfinance) |
| `test_indicators.py` | Расчёт KAMA: корректность, edge cases |
| `test_alpha.py` | Генерация альфа-сигналов, корреляционный фильтр |
| `test_engine.py` | Симуляционный движок: equity, сделки, режимы |
| `test_reporting.py` | Метрики, визуализация, отчёты |
| `test_optimizer.py` | Sensitivity analysis, маргинальные профили |
| `test_max_profit.py` | Поиск макс. CAGR, Парето-фронт |
| `test_walk_forward.py` | WFO pipeline, генерация расписания |
| `test_dashboard.py` | Streamlit UI (мок Streamlit) |

---

## Поток данных

```
yfinance
   │
   ▼
fetch_price_data() ──► Parquet cache (output/cache/)
   │
   ▼
(close_prices, open_prices)  [DataFrame: дата × тикер]
   │
   ├──► compute_kama_series()  →  kama_cache  [dict: тикер → Series]
   │
   ▼
run_simulation()
   ├── regime filter (SPY KAMA + buffer)
   ├── get_buy_candidates() (trend + momentum + correlation)
   ├── lazy hold / stop-loss
   ├── sizing (equal_weight / risk_parity)
   └── execute trades (Open T+1, с комиссией)
   │
   ▼
SimulationResult
   │
   ▼
compute_metrics()  →  CAGR, Sharpe, MaxDD, Calmar, ...
   │
   ▼
save_equity_png() / format_comparison_table() / format_asset_report()
```

Оптимизация (Optuna) оборачивает `run_simulation()` в параллельный цикл через `parallel.py`, варьируя параметры из `StrategyParams`.
