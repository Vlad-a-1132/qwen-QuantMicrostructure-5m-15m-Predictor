# PROJECT SPEC: QuantMicrostructure 5m/15m Predictor

**Версия:** 1.0.0  
**Стек:** Python Core + Tauri 2.x + ZeroMQ  
**Горизонт:** 5m / 15m (параллельно)  
**Цель:** Статистическое опережение с WinRate ≤ 35% и 𝔼[R] > 0

---

## 1. 🔢 Математический фундамент

### 1.1. Целевая переменная и сигнал

Пусть Δ ∈ {5, 15} минут. Лог-доходность:

$$r_{t+\Delta} = \ln \frac{P_{t+\Delta}}{P_t}$$

Целевая метка для обучения:

$$Y_t = \text{sign}(r_{t+\Delta}) \cdot \log(1 + |r_{t+\Delta}|)$$

Скалярный сигнал на момент t (вычисляется строго из F_t):

$$S_t = f_\theta(x_t), \quad x_t \in \mathbb{R}^d$$

Правило входа: длинная если S_t > θ, короткая если S_t < -θ, иначе пропуск.

### 1.2. Микропорядковые признаки (Leading Invariants)

| Признак | Формула | Физический смысл |
|---------|---------|------------------|
| **Order Flow Imbalance (OFI)** | $\displaystyle \text{OFI}_t = \sum_{i \in \text{window}} v_i \cdot s_i, \quad s_i \in \{-1, +1\}$ | Чистый агрессивный поток, опережает сдвиг mid-price |
| **Microprice Drift** | $\displaystyle \mu_t = \frac{\sum P_i v_i}{\sum v_i} - \text{mid}_t$ | Смещение взвешенной цены сделок относительно центра стакана |
| **Liquidity Vacuum Index** | $\displaystyle \mathcal{V}_t = \frac{\max(q_{\text{top}}^b, q_{\text{top}}^a)}{\min(q_{\text{top}}^b, q_{\text{top}}^a)} \cdot \frac{\Delta \text{spread}}{\text{spread}}$ | Истончение защиты → высокая вероятность пробоя |
| **VPIN (Toxicity)** | $\displaystyle \text{VPIN}_t = \frac{1}{n}\sum_{k=1}^n \frac{|V_{buy,k} - V_{sell,k}|}{V_{buy,k} + V_{sell,k}}$ | Токсичность потока, индикатор информированных трейдеров |

### 1.3. Условие прибыльности при W ≤ 0.35

Пусть $W(\theta) = P(\text{верный прогноз} \mid |S_t| > \theta) \leq 0.35$.

Фиксированный стоп L, тейк K·L (K > 1), издержки C.

Матожидание одной сделки:

$$\mathbb{E}[R \mid \theta] = W(\theta)(KL - C) - (1 - W(\theta))(L + C)$$

Условие $\mathbb{E}[R] > 0$ эквивалентно:

$$K > \frac{1}{W(\theta)} - 1 + \frac{C}{W(\theta)L}$$

При W = 0.35, C = 0.05%, L = 0.2% ⇒ K ≳ 2.57.

### 1.4. Теорема существования порога θ*

**Утверждение:** При тяжёлых хвостах returns (α ∈ [2.5, 3.0]) и корреляции S_t с условной дисперсией, существует θ* такое, что:

1. W(θ*) ≤ 0.35
2. $\mathbb{E}[R \mid \theta^*] > 0$
3. $\frac{\partial K(\theta)}{\partial \theta} > 0$ при θ → ∞

**Доказательство (схема):**

g(θ) = $\mathbb{E}[R \mid \theta]$ непрерывна. g(0) ≈ -C < 0. По EVT и свойствам ликвидностных пробоев $\lim_{\theta \to \infty} K(\theta) = \infty$, а W(θ) монотонно убывает в область [0.25, 0.35]. По теореме о промежуточном значении ∃θ*: g(θ*) = 0 и ∀θ ∈ (θ*, θ_K]: g(θ) > 0, W(θ) ≤ 0.35. ■

### 1.5. Оптимизация порога

$$\theta^* = \arg\max_\theta \frac{\mathbb{E}[R \mid \theta]}{\sqrt{\text{Var}(R \mid \theta)}} \quad \text{при} \quad W(\theta) \leq 0.35$$

Калибруется rolling-окном (24–72ч) через scipy.optimize.minimize с ограничением на эмпирическую точность.

---

## 2. 🏗 Архитектура системы

```
┌─────────────────────────────────────────────────────────────┐
│                     Tauri 2.x Desktop UI                    │
│  [Управление 5m] [Управление 15m] [Текущий прогноз]         │
│  [График 5m]     [График 15m]     [Таблица последних 100]   │
│  [Полная история] [Экспорт]        [Метрики PnL]           │
└───────────────┬──────────────────────────────┬──────────────┘
                │ ZeroMQ (SUB/PUB, msgpack)    │
                ▼                              ▼
┌───────────────────────────┐  ┌──────────────────────────────┐
│   Python Worker: 5m       │  │   Python Worker: 15m         │
│  • Delta = 5 min          │  │  • Delta = 15 min            │
│  • Embargo = 7 min        │  │  • Embargo = 20 min          │
│  • K_min = 2.6            │  │  • K_min = 2.3               │
│  • Slippage = 0.03%       │  │  • Slippage = 0.02%          │
│                           │  │                              │
│  [Ingestion]→[Features]→[ML]→[θ*-Opt]→[Risk Sim]→[Publish] │
└───────────────────────────┘  └──────────────────────────────┘
         ▲                                  ▲
         └───────────── Binance WS ─────────┘
              L2 Orderbook + Trades (async)
```

### 2.1. Стек по слоям

| Слой | Технология | Назначение |
|------|------------|------------|
| **Data Ingestion** | aiohttp + websockets + polars (streaming) | Асинхронный парсинг L2, реконструкция стакана, агрегация в 1s окна |
| **Feature Engineering** | polars + numba + arch (EVT) | Векторизованный расчёт OFI, μ, 𝒱, VPIN без GIL-блоков |
| **ML & Inference** | lightgbm → ONNX → onnxruntime | Калиброванные вероятности, инференс ≤2ms, PurgedKFold валидация |
| **θ-Optimization** | scipy.optimize + empyrical | Rolling-поиск порога с ограничением W ≤ 0.35 |
| **IPC** | ZeroMQ (PUB/SUB) + msgpack | Низколатентная передача сигналов, буферизация при пиках |
| **UI** | Tauri 2.x + Svelte + Lightweight-Charts | Нативный десктоп <15MB RAM, real-time графики, экспорт |

### 2.2. Протокол обмена (ZeroMQ)

- **Топик signals_5m:** `{ts, dir, S_t, p_t, θ*, status}`
- **Топик signals_15m:** `{ts, dir, S_t, p_t, θ*, status}`
- **Топик metrics_{tf}:** `{win_rate, RR, E_R, drawdown, sharpe_rolling}`
- **Топик history:** `{ts, tf, dir, entry, exit, result, pnl}`

**Формат:** msgpack (бинарный, ≤1kb/msg).

---

## 3. 🖥 Спецификация интерфейса (Tauri UI)

### 3.1. Макет экрана

```
┌──────────────────────────────────────────────────────────────┐
│  🔘 [RUN 5m]  ⏸ [RUN 15m]  🎛 [Параметры: θ, K, L, Risk]    │
├──────────────────────┬───────────────────────────────────────┤
│  📊 ТЕКУЩИЙ ПРОГНОЗ  │          📈 ГРАФИК 5m / 15m           │
│  TF: 5m / 15m        │  Свечи + уровни θ + метки входа/выхода│
│  Dir: LONG / SHORT   │  Переключение вкладок / Split-view    │
│  S_t: 1.42  p_t: 0.81│                                       │
│  Status: WAITING     │                                       │
│  Next candle: 00:42  │                                       │
├──────────────────────┴───────────────────────────────────────┤
│  📜 ПОСЛЕДНИЕ 100 ПРОГНОЗОВ (Таблица)                        │
│  [Time] [TF] [Dir] [S_t] [p_t] [Result ✅/❌] [PnL] [Hold]   │
├──────────────────────────────────────────────────────────────┤
│  📚 ПОЛНАЯ ИСТОРИЯ (Фильтры, Экспорт CSV/JSON, Агрегация)    │
│  Итог: 5m W=31%, K=3.2, E[R]=+0.11% | 15m W=33%, K=2.8, E[R]=+0.09% │
└──────────────────────────────────────────────────────────────┘
```

### 3.2. Функциональные блоки

| Блок | Данные | Источники | Действия |
|------|--------|-----------|----------|
| **Текущий прогноз** | S_t, p_t, θ*, направление, статус, таймер | signals_{tf} (PUB) | Визуальная индикация готовности, подсветка при сигнале |
| **График 5m/15m** | OHLCV, сигналы, уровни θ, стоп/тейк | history + локальный кэш | Переключение TF, Split-mode, зум, экспорт PNG |
| **Последние 100** | Циклический буфер последних сделок | history (PUB) + локальный массив | Сортировка, фильтр по TF, подсветка ❌/✅ |
| **Полная история** | Все сделки с момента запуска/импорт | Локальный SQLite / CSV экспорт | Пагинация, агрегация метрик, экспорт, replay |
| **Управление** | θ, K, L, риск, запуск/остановка | UI → Core (ZeroMQ REQ/REP) | Динамическая калибровка, hot-reload без перезапуска |

### 3.3. Параллельный запуск 5m и 15m

- **Ядро:** Два независимых async-контура в Python. Каждый имеет свой delta, embargo, rolling_window, slippage.
- **IPC:** Раздельные ZeroMQ топики. UI подписывается на оба.
- **Синхронизация:** Время привязывается к UTC. В UI отображается независимый статус каждого потока. Конфликтов нет, так как ордера изолированы по горизонтам.
- **Риск-менеджмент:** Общий лимит капитала распределяется пропорционально $\mathbb{E}[R]$ каждого потока (настраивается в UI).

---

## 4. ⚙️ Конфигурация (config.yaml)

```yaml
# Глобальные
app:
  name: "QuantMicrostructure"
  version: "1.0.0"
  log_level: "INFO"

# Binance
exchange:
  ws_url: "wss://fstream.binance.com/ws"
  symbol: "BTCUSDT"
  max_l2_levels: 20
  max_depth_mb: 512

# Таймфреймы
timeframes:
  "5m":
    delta_minutes: 5
    embargo_minutes: 7
    theta_window_hours: 48
    min_K_ratio: 2.6
    slippage_bps: 0.3
    rolling_recal_minutes: 60
  "15m":
    delta_minutes: 15
    embargo_minutes: 20
    theta_window_hours: 72
    min_K_ratio: 2.3
    slippage_bps: 0.2
    rolling_recal_minutes: 120

# ML & Inference
ml:
  model_path: "models/lightgbm_calibrated.onnx"
  feature_cols: ["OFI", "microprice_drift", "liquidity_vacuum", "VPIN", "dOFI_dt", "dμ_dt"]
  inference_hz: 1  # раз в секунду

# Risk
risk:
  max_exposure_pct: 2.0
  position_sizing: "kelly_fraction"
  max_drawdown_stop: -5.0

# IPC
ipc:
  protocol: "zmq"
  addr_core: "tcp://127.0.0.1:5555"
  addr_ui:   "tcp://127.0.0.1:5556"
  format: "msgpack"
```

---

## 5. 🛡 Валидация и风险控制

| Этап | Метод | Критерий прохождения |
|------|-------|---------------------|
| **Backtest** | PurgedKFold + embargo | W ≤ 0.35, $\mathbb{E}[R]_{oos} > 0$, Sharpe > 0.8 |
| **Slippage Sim** | $C_{eff} = fee + 0.5 \cdot spread + \alpha \frac{V_{order}}{V_{book}}$ | $K_{net} > 2.57$ на 5m, > 2.3 на 15m |
| **Rolling Calibration** | Переобучение θ каждые 60–120 мин | Отсутствие дрейфа W за пределы [0.25, 0.38] |
| **Live Paper** | 7–14 дней на демо-счёте | Реальный $\mathbb{E}[R]$ отклоняется от backtest ≤15% |
| **Circuit Breaker** | Автоостановка при DD > 5% или задержке >200ms | Ручной сброс, аудит логов |

---

## 6. 📁 Структура проекта

```
quant-microstructure/
├── core/
│   ├── ingestion.py          # Async WS, L2 reconstruction
│   ├── features.py           # Polars pipelines, OFI/VPIN/μ/𝒱
│   ├── model.py              # LightGBM → ONNX export, inference
│   ├── theta_optimizer.py    # scipy.optimize, W≤0.35 constraint
│   ├── risk_sim.py           # Slippage, position sizing, PnL
│   └── workers.py            # Async workers per timeframe
├── ipc/
│   ├── pubsub.py             # ZeroMQ publishers
│   └── commands.py           # REQ/REP for UI control
├── tauri-app/
│   ├── src/
│   │   ├── main.js / Svelte
│   │   ├── charts/           # Lightweight-Charts wrappers
│   │   ├── tables/           # Virtualized history
│   │   └── store.ts          # Reactive state (ZMQ streams)
│   └── src-tauri/            # Rust backend, tray, system calls
├── models/                   # .onnx files, calibration weights
├── configs/                  # config.yaml, env files
├── tests/                    # PurgedKFold, EVT tail tests
├── docker-compose.yml        # Core + Redis/ZMQ (optional)
└── README.md                 # Эта документация
```

---

## 7. 🗺 Roadmap запуска

| Неделя | Задача | Результат |
|--------|--------|-----------|
| **1** | Ingestion + Polars features + ONNX export | Рабочий pipeline признаков, модель W≤0.35 на исторических данных |
| **2** | θ*-optimizer + PurgedKFold + slippage sim | Стабильный backtest с $\mathbb{E}[R] > 0$ на out-of-sample |
| **3** | ZeroMQ IPC + Tauri UI skeleton | Передача сигналов, графики, таблицы, параллельный 5m/15m |
| **4** | Rolling calibration + Paper trading | Автообновление θ, мониторинг метрик, готовность к live |

---

## 📎 Приложение: Как сохранить и запустить

1. Сохраните этот текст в файл `PROJECT_SPEC.md`.

2. Инициализация:

```bash
mkdir quant-microstructure && cd quant-microstructure
# Скопируйте структуру из п.6
pip install polars lightgbm onnxruntime scipy empyrical pyzmq msgpack websocket-client aiohttp
cargo create-tauri-app tauri-app --template vanilla
```

3. Запуск ядра:

```bash
python core/workers.py --config configs/config.yaml
```

4. Запуск UI:

```bash
cd tauri-app && npm install && cargo tauri dev
```

✅ **Готово.** Документация содержит полный математический аппарат, архитектуру, спецификацию интерфейса с параллельным 5m/15m запуском, конфигурацию и валидационные протоколы.
