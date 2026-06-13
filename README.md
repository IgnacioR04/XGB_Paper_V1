# XGB Paper Trader v3 — Regime-Aware BTC Bot (top-3 filtros)

Paper-trading bot para Bitcoin usando XGBoost Approach B (calibracion isotonic + candidatos barriers_v2 + seleccion top_EV) con **reglas condicionales por regimen de mercado y apalancamiento variable**.

## Que hace

Cada 5 minutos (cron de GitHub Actions) el bot:

1. Detecta que timeframes (`1h`, `4h`) tienen una vela cerrada (`15m` deshabilitado).
2. Cierra posiciones abiertas por TP / SL / TIMEOUT.
3. Para cada TF due:
   - Construye features live (615 features del grid 15m).
   - Estima volatilidad EWMA y mapea a decile.
   - **Clasifica el regimen de mercado** (alcista/bajista/lateral).
   - Si el regimen NO esta habilitado para ese TF, salta.
   - Si esta habilitado: usa el threshold y leverage del regimen.
   - Inferencia XGB + calibrador isotonic -> filtra por threshold -> top EV.
   - Si pasa y la cartera esta libre -> abre con `notional = cash * leverage`.

Cada TF activo tiene su propia cartera de **100 EUR**.

## Clasificacion de regimen

Se usan EMA50 y EMA200 sobre el cierre de 1h (resampleado del grid 15m):

| Regimen | Condicion |
|---------|-----------|
| **Alcista** | close > EMA200 AND EMA50 > EMA200 |
| **Bajista** | close < EMA200 AND EMA50 < EMA200 |
| **Lateral** | todo lo demas |

Las EMAs son indicadores rezagados — solo usan datos hasta t. El regimen clasificado en t se aplica a la decision de t+1 sin leakage.

## Reglas por regimen y timeframe (v3 — solo los 3 filtros ganadores)

Tras estudiar la contribucion de cada filtro por separado a 1x (backtest causal
2025-2026, notebook RS01 seccion 11), solo se conservan los 3 con PnL y Sharpe
netos positivos. Cada uno mantiene el leverage que tenia asignado.

| TF | Regimen | Threshold p_win | Leverage | Opera? |
|----|---------|----------------|----------|--------|
| 1h | Bajista | 0.55 | 2x | Si |
| 1h | Lateral | 0.55 | 4x | Si |
| 4h | Bajista | 0.75 | 5x | Si |
| 15m | (todos) | — | — | No (TF deshabilitado) |
| 1h | Alcista | — | — | No |
| 4h | Alcista / Lateral | — | — | No |

Resultados del estudio por filtro (100 EUR, 1x, 2025-2026 causal):

| Filtro | Trades | Win-rate | PnL % | Max DD | Sharpe |
|--------|--------|----------|-------|--------|--------|
| 4h bajista | 29 | 0.62 | +15.8% | -4.0% | 1.68 |
| 1h bajista | 281 | 0.53 | +28.1% | -10.5% | 1.35 |
| 1h lateral | 67 | 0.61 | +10.0% | -5.0% | 1.27 |

Editables en [config/paper_trading.yaml](config/paper_trading.yaml).

## Leverage

El apalancamiento se aplica al notional: `notional_eur = cash_eur * leverage`. El PnL se amplifica automaticamente porque `pnl = notional * return`. Con leverage > 1x, una perdida puede superar el cash disponible (drawdown >100%).

## Semantica temporal: vela t -> ejecucion en t+1

```
|---- vela t ----|---- vela t+1 ----|
                 ^
                 cierre de t = apertura de t+1
                 AQUI se pregunta al modelo (features = vela t CERRADA)
                 y AQUI MISMO se ejecuta la entrada
```

**Nunca se decide con la vela en curso** (`signal_mode: closed_candle_only`).

## Coste round-trip

`0.06% entrada + 0.06% salida = 0.12% total`.

```
EV_pred = p_win * tp_pct - (1 - p_win) * sl_pct - 0.0012
```

## Cron jobs (GitHub Actions)

| Workflow | Cron | Que hace |
|----------|------|----------|
| `paper_trader.yml` | `*/5 * * * *` | Tick: clasifica regimen, decide, abre/cierra trades |
| `external_update.yml` | `5 */6 * * *` | Refresca macro/onchain/sentiment/funding cache |
| `pages.yml` | tras paper_trader ok | Despliega dashboard a GitHub Pages |
| `tests.yml` | push & PR | pytest |

## Dashboard

URL: `https://ignacior04.github.io/XGB_Paper_V1/`

Muestra: grafica BTC en tiempo real, graficas por TF con trades, equity por cartera, historial de senales, posiciones abiertas y trades cerrados.

## Como ejecutar localmente

```bash
pip install -r requirements.txt
python scripts/audit_artifacts.py          # verifica artifacts
python scripts/update_external_data.py     # cache externa
python scripts/run_paper_tick.py --dry-run # sin abrir posiciones
python scripts/run_paper_tick.py           # tick real
python scripts/run_paper_tick.py --force-tf 1h,4h      # forzar TFs activos
```

## Modelo: 615 features

- 10 candidate-level: `vol_pred`, `vol_decile`, `tp_mult`, `sl_mult`, `H`, `tp_pct`, `sl_pct`, `side_long`, `barrier_quality_score`, `p_break_even`.
- 605 market features del grid 15m (prefijos `ohlcv_`, `ret_`, `ta_`, `lag_`, `roll_`, `cal_`, `der_`, `vp_`, `ext_`, `macro_`, `cx_`, `oc_`, `sent_`, `reg_`, `inter_`).
- ~310/605 features se computan en live con paridad <1% vs master causal. El resto quedan NaN (XGBoost las tolera).

## Relacion con el proyecto de investigacion

Los modelos, datasets (8.8 anos, 308K velas, 646 columnas) y notebooks de investigacion viven en Google Drive (`Base de Datos BITCOIN/`). Este repo es el sistema de despliegue. Los artifacts se generan en Drive y se copian aqui.

## Estado del modelo (2026-06-13)

AUC honesto post-fix leakage causal: ~0.55 (antes 0.69-0.77 inflado). La estrategia v3 concentra las operaciones en los 3 filtros (TF, regimen) que mostraron PnL y Sharpe netos positivos en el backtest causal por filtro a 1x, con leverage variable para amplificar.

> **Caveats honestos del backtest que selecciono los filtros**: usa las features completas del master (en vivo el bot solo genera ~310/605, resto NaN), y con win-rates ~0.5-0.6 + leverage 2x-5x los resultados son de alta varianza. Los filtros se eligieron por contribucion neta positiva, no por edge garantizado.

## Historial de versiones

| Version | Fecha | Cambios |
|---------|-------|---------|
| v1 | 2026-06-08 | Bandas de probabilidad fijas por TF (0.55-0.67) |
| v1.1 | 2026-06-12 | Fix leakage causal (chikou, swings, cx_daily). AUC cae a 0.55 |
| v2 | 2026-06-12 | Regimen condicional + leverage variable (6 filtros). Reset a 100 EUR |
| **v3** | **2026-06-13** | **Solo los 3 filtros ganadores (1h bajista 2x, 1h lateral 4x, 4h bajista 5x). 15m deshabilitado. Reset de carteras** |
