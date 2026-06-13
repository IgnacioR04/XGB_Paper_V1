# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Que es este proyecto

Bot de paper trading para Bitcoin que ejecuta el modelo Approach B (XGBoost + calibracion isotonic + candidatos barriers_v2 + seleccion top_EV). Corre automatizado via GitHub Actions cada 5 minutos y publica un dashboard en GitHub Pages.

Este repo es el **sistema de despliegue**. Los modelos, datasets y notebooks de investigacion viven en un proyecto hermano en Google Drive (`Base de Datos BITCOIN/`). Aqui solo esta el codigo de inferencia, los artifacts exportados, y la maquinaria de ejecucion paper.

## Comandos esenciales

```bash
# Instalar dependencias
pip install -r requirements.txt

# Tests
pytest -q

# Verificar que todos los artifacts existen
python scripts/audit_artifacts.py

# Refrescar cache de fuentes externas (macro, onchain, sentiment)
python scripts/update_external_data.py

# Tick completo (dry-run, sin abrir operaciones)
python scripts/run_paper_tick.py --dry-run

# Tick real
python scripts/run_paper_tick.py

# Forzar timeframes especificos (ignora ventana del scheduler)
python scripts/run_paper_tick.py --force-tf 15m,1h,4h

# Override de timestamp (para testing)
python scripts/run_paper_tick.py --now 2026-06-12T12:00:00
```

## Arquitectura: flujo de un tick

```
GitHub Actions (cron */5)
  |
  v
run_paper_tick.py
  |
  |-- 1. Scheduler: detecta TFs due (15m/1h/4h) segun ventana de cierre
  |-- 2. Exit monitor: cierra posiciones abiertas por TP/SL/TIMEOUT
  |      (verifica velas cerradas del TF, velas 1m intrabar, y ticker actual)
  |-- 3. Macro cache: carga data/live_features/macro_cache.json
  |-- 4. Para cada TF due:
  |      a. build_live_features() -> DataFrame grid 15m (compartido entre TFs)
  |      b. EWMA vol -> vol_decile (usando bounds de entrenamiento)
  |      c. classify_regime(feats) -> "alcista"/"bajista"/"lateral"
  |      d. Lookup regime_rules[tf][regime] -> si no enabled, skip con REGIME_NOT_ENABLED
  |      e. signal_engine.evaluate_timeframe() con threshold del regimen:
  |         - Carga candidatos del decile desde la libreria parquet
  |         - Inferencia XGB -> calibrador isotonic -> p_win_calibrated
  |         - Filtra por threshold, elige top EV
  |      f. Si hay ganador: notional = cash * leverage, open_position(leverage, regime)
  |-- 5. export_dashboard_data() -> dashboard/data/*.json
  |
  v
Commit state+trades+dashboard al repo -> trigger pages.yml
```

**Dato clave**: los 3 modelos (15m, 1h, 4h) usan features del **grid de 15 minutos** (igual que en entrenamiento). El feature builder siempre descarga velas de 15m y computa una unica vez; la misma fila sirve para los tres TFs.

## Estructura del repo

```
XGB_Paper_V1/
├── artifacts/
│   ├── models/          # approach_B_xgb_{15m,1h,4h}.json (~1.3 MB c/u)
│   ├── calibration/     # calib_approach_B_compact.json (isotonic por TF)
│   ├── candidates/      # barrier_candidate_library_{tf}.parquet
│   └── schemas/         # feature_schema.json (615 features en orden), vol_decile_bounds.json
├── config/
│   ├── paper_trading.yaml   # Carteras, bandas, scheduler, costes, stop conditions
│   ├── strategy.yaml        # Modelo, candidatos, EV formula, volatilidad
│   └── data_sources.yaml    # URLs, refresh intervals, stale thresholds, rate limits
├── src/
│   ├── config.py            # Config dataclass con @lru_cache, carga 3 YAMLs
│   ├── data/                # Extractores: binance_spot, binance_futures, yahoo, fred, etc
│   ├── features/
│   │   ├── live_feature_builder.py  # Orquestador: descarga klines -> parity_features -> calendar -> macro
│   │   ├── parity_features.py       # Replica T02/T03/T06/T13 causales (310/318 features <1% err)
│   │   ├── technicals.py            # RSI, EMA, ATR, BB, MACD, true_range
│   │   ├── macro_cache.py           # Lee/escribe macro_cache.json
│   │   └── feature_schema.py        # Valida y reordena features al schema del modelo
│   ├── models/
│   │   ├── xgb_loader.py       # load_xgb_model() con LRU cache
│   │   ├── calibrator.py       # Reconstruye IsotonicRegression desde JSON compact
│   │   └── inference.py        # predict_p_win(): XGB -> p_raw -> isotonic -> p_calibrated
│   ├── volatility/
│   │   └── ewma_vol.py          # EWMA(20) del true_range normalizado; vol_to_decile()
│   ├── strategy/
│   │   ├── signal_engine.py     # evaluate_timeframe(): orquesta candidatos -> inferencia -> filtro -> top EV
│   │   ├── candidates.py        # load_library(), candidates_for(tf, decile, sides)
│   │   ├── ev.py                # expected_value(), p_break_even(), annotate_candidates()
│   │   └── probability_filters.py  # filter_by_band(), top_ev()
│   ├── execution/
│   │   ├── paper_broker.py      # open/close_position(), idempotencia via signal_id
│   │   ├── position_manager.py  # find_exit_in_klines(), check_timeout(), pnl_eur()
│   │   └── exits.py             # monitor_and_close_positions() (3 niveles: velas TF, 1m intrabar, ticker)
│   ├── portfolio/
│   │   └── wallet.py            # Una cartera por TF (100 EUR), equity_curve in-state
│   ├── dashboard/
│   │   └── export_json.py       # Genera summary.json, trades.json, equity_{tf}.json, signals.json
│   └── utils/
│       ├── time_utils.py        # Scheduler: get_due_timeframes(), make_signal_id()
│       ├── io.py                # write_json_atomic(), append_csv(), append_parquet()
│       ├── http.py              # get_json() con retry + backoff
│       ├── csv_logger.py        # append_decision(), append_features() (auditoria por vela)
│       └── errors.py            # ArtifactMissing, FeatureSchemaMismatch
├── scripts/
│   ├── run_paper_tick.py        # ENTRYPOINT PRINCIPAL del bot
│   ├── update_external_data.py  # Refresca macro/onchain/sentiment/funding -> cache JSON
│   ├── audit_artifacts.py       # Verifica que existen modelos, calibrador, schemas
│   └── (varios build_*.py, analyze_*.py, fix_*.py de investigacion)
├── tests/                       # pytest: EV, wallet, signal idempotency, feature schema, probability band
├── data/                        # Runtime state (commiteado desde Actions)
│   ├── state/                   # open_positions.json, processed_candles.json, wallet_{tf}.json
│   ├── paper_trades/            # trades.csv, trades.parquet
│   ├── logs/                    # paper_trader.log, last_tick.json, decisions/decisions_YYYY-MM.csv
│   ├── live_raw/                # klines cacheados
│   └── live_features/           # macro_cache.json
└── dashboard/                   # HTML + Chart.js (desplegado en GitHub Pages)
    └── data/                    # JSONs generados por export_json.py
```

## GitHub Actions workflows

| Workflow | Trigger | Que hace |
|----------|---------|----------|
| `paper_trader.yml` | cron `*/5 * * * *` + manual | Tick: decide y abre/cierra trades. Commitea state al repo. |
| `external_update.yml` | cron `5 */6 * * *` + manual | Refresca macro_cache.json (Yahoo, FRED, blockchain.com, CryptoCompare, F&G, funding) |
| `pages.yml` | tras paper_trader exitoso + manual | Despliega `dashboard/` a GitHub Pages |
| `tests.yml` | push + PR | `pytest -q` |

## Configuracion: 3 YAMLs

**`paper_trading.yaml`** - lo que cambia operativamente:
- `wallets`: capital por TF (100 EUR cada una). **v3: 15m esta `enabled: false`** (ningun filtro 15m supero el estudio). Solo operan 1h y 4h.
- `regime_rules`: reglas por TF y regimen (enabled, threshold, leverage). El regimen (alcista/bajista/lateral) se clasifica con EMA50 vs EMA200 en 1h via `src/features/regime.py`. **v3: solo 3 filtros activos** — 1h bajista (0.55, 2x), 1h lateral (0.55, 4x), 4h bajista (0.75, 5x).
- `allow_above_band: true` - banda abierta por arriba (threshold -> 1.0)
- `allowed_sides: [long, short]`
- `signal_mode: closed_candle_only`
- `scheduler`: delays post-cierre y ventana maxima de latencia
- `stop_conditions`: max consecutive losses, drawdown limits

**`strategy.yaml`** - parametros del modelo:
- Rutas a artifacts (modelos, calibrador, libreria, schemas)
- Formula EV: `p_win * TP - (1 - p_win) * SL - 0.0012`
- `intrabar_conflict_rule: SL_first` (conservador)
- Volatilidad: EWMA span=20

**`data_sources.yaml`** - fuentes de datos live:
- Binance spot/futures: hosts, symbols, timeframes, limites
- Yahoo, FRED, CryptoCompare, Fear&Greed, blockchain.com
- Stale thresholds y rate limits

## Modelo: 615 features

El XGBoost espera exactamente 615 features en un orden fijo (`feature_schema.json`):
- 10 **candidate-level**: `vol_pred`, `vol_decile`, `tp_mult`, `sl_mult`, `H`, `tp_pct`, `sl_pct`, `side_long`, `barrier_quality_score`, `p_break_even`
- 605 **market features**: del grid 15m (prefijos `ohlcv_`, `ret_`, `ta_`, `lag_`, `roll_`, `cal_`, `der_`, `vp_`, `ext_`, `macro_`, `cx_`, `oc_`, `sent_`, `reg_`, `inter_`)

Features faltantes se rellenan con NaN — XGBoost las maneja nativamente. El builder live cubre ~310/605 market features con paridad (<1% error vs master causal). Las features `vp_*`, `reg_*`, y algunas `ta_*` avanzadas quedan en NaN.

## Flujo de decision por vela (semantica t -> t+1)

```
vela t cierra -> features = vela t cerrada -> modelo decide -> entra al open de t+1
```

Cada evaluacion se registra en `data/logs/decisions/decisions_YYYY-MM.csv` con: `candle_close_time` (cierre de t = momento de la pregunta), `execution_candle_open` (apertura de t+1 = momento de ejecucion), `decision` YES/NO, y diagnosticos.

## Idempotencia

Cada senal tiene un `signal_id = "{symbol}|{tf}|{candle_close_iso}|{side}|{candidate_id}"`. Una vez procesado, se registra en `processed_candles.json` y no se re-ejecuta aunque el tick corra multiples veces sobre la misma vela (comun con GitHub Actions cron */5).

## Resiliencia de Binance

`binance_spot.py` prueba 5 hosts en orden (`api3`, `api2`, `api1`, `api4`, `api`) porque los runners de GitHub Actions suelen tener IP bloqueada. Si todos fallan, hay fallback a Coinbase para BTC. El primer host que responde se cachea en memoria para el resto del tick.

## Relacion con el proyecto de investigacion (Drive)

- Los **artifacts** (`artifacts/`) se generan en los notebooks de Drive (07_directional_modeling, 06_barriers_v2) y se copian manualmente a este repo.
- Las **formulas de features** (`parity_features.py`) replican los notebooks T02/T03/T06/T13 en sus versiones **causales** (post-fix de leakage 2026-06-12).
- Si se reentrena el modelo en Drive, hay que actualizar: los 3 `.json` de modelos, el calibrador, las librerias de candidatos, y posiblemente el feature_schema y vol_decile_bounds.
- Script para validar coherencia: `scripts/test_feature_parity.py` (compara builder live vs master causal del Drive).

## Estado actual del modelo (2026-06-13)

AUC honesto post-fix causal: ~0.55 (antes 0.69-0.77 inflado por leakage). **Estrategia v3: solo los 3 filtros ganadores** del estudio de contribucion por filtro (backtest causal 2025-2026 a 1x, notebook RS01 seccion 11 en Drive): 1h bajista (2x), 1h lateral (4x), 4h bajista (5x). 15m deshabilitado; 4h solo opera en bajista. Carteras reseteadas a 100 EUR. Caveat: el backtest que selecciono los filtros usa features completas del master (en vivo solo ~310/605), y son resultados de alta varianza (win-rate ~0.5-0.6 + leverage).

## Secrets de GitHub Actions

| Secret | Requerido | Uso |
|--------|-----------|-----|
| `FRED_API_KEY` | Opcional | Series FRED (DXY, M2, CPI, US10Y). Sin ella quedan NaN, el modelo lo tolera. |

Las APIs de Binance, Yahoo, CryptoCompare, Fear&Greed, y blockchain.com son publicas y no requieren keys.
