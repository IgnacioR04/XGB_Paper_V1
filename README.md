# XGB Paper Trader (BTC v1)

Paper-trading bot para Bitcoin sobre el modelo Approach B (XGBoost sobre
`label_win`, calibracion isotonic, candidatos `barriers_v2`, seleccion
`top_EV_per_timeframe`).

## Que hace

Cada 5 minutos (cron de GitHub Actions) el bot:

1. Detecta que timeframes (`15m`, `1h`, `4h`) tienen una vela cerrada lista
   para procesar (`closed_candle_only`).
2. Monitorea las posiciones abiertas y cierra por TP / SL / TIMEOUT.
3. Para cada TF due:
   - Descarga klines spot + futures de Binance (publico, sin key).
   - Construye features live (OHLCV, returns, TA, lags, rolling, calendar).
   - Combina con cache externa (macro, on-chain, sentiment, cross-crypto)
     que se actualiza por otro workflow cada 6 horas.
   - Estima volatilidad (EWMA del true range) y mapea a decile.
   - Carga la libreria de candidatos de ese decile.
   - Inferencia XGBoost + calibrador isotonic -> `p_win_calibrated`.
   - Calcula EV usando `coste round-trip = 0.0012`.
   - Filtra por la banda de probabilidad del TF y elige el mejor por EV.
   - Si pasa el filtro y no hay posicion abierta en esa cartera -> abre.

Cada timeframe tiene su PROPIA cartera de **100 EUR**. Las carteras son
independientes y un trade en `15m` no afecta a `1h` ni `4h`.

## Bandas de probabilidad por timeframe

| TF  | Banda `p_win_calibrated` |
|-----|--------------------------|
| 15m | `0.65 <= p < 0.70`       |
| 1h  | `0.70 <= p < 0.75`       |
| 4h  | `0.65 <= p < 0.70`       |

Editables en [config/paper_trading.yaml](config/paper_trading.yaml). Hay un
flag `allow_above_band` (default `false`) que abre la banda por arriba si lo
activas.

## Coste round-trip

`0.06% entrada + 0.06% salida = 0.12% total`. Define la formula de EV:

```
EV_pred = p_win * tp_pct - (1 - p_win) * sl_pct - 0.0012
```

## Estructura del repo

```
XGB_Paper_V1/
  artifacts/        # modelos XGB, calibrador, libreria de candidatos, schemas
  config/           # 3 YAMLs editables
  src/              # codigo Python
    data/           # extractores Binance / Yahoo / FRED / etc
    features/       # builders en vivo
    models/         # loader + calibrador + inferencia
    strategy/       # candidates, EV, signal engine
    execution/      # paper broker, position manager, exits
    portfolio/      # carteras y accounting
    dashboard/      # exporter de JSON
    utils/          # tiempo, IO, http, logging
  scripts/          # entrypoints
  tests/            # pytest
  dashboard/        # HTML + Chart.js
  data/             # estado runtime (ignorado por git; se commitea desde Actions)
  .github/workflows/
```

## Cron jobs (GitHub Actions)

| Workflow | Cron | Que hace |
|----------|------|----------|
| `paper_trader.yml` | `*/5 * * * *` | Tick: decide y abre/cierra trades |
| `external_update.yml` | `5 */6 * * *` | Refresca macro/onchain/sentiment/funding |
| `pages.yml` | tras `paper_trader` ok | Despliega el dashboard a GitHub Pages |
| `tests.yml` | push & PR | Corre pytest |

**Por que un cron separado para datos externos?** Las APIs de macro (Yahoo,
FRED) y on-chain (Blockchain.com) no cambian cada 5 minutos. Refrescarlas
en cada tick desperdicia tiempo y rate limit. El cache se commitea al repo
y el tick lo lee.

> Si necesitas un cron diferente al de Actions (ej. una maquina propia
> 24/7 para evitar los retrasos tipicos de GitHub), [scripts/run_paper_tick.py](scripts/run_paper_tick.py)
> es el unico entrypoint y se puede llamar desde cualquier scheduler externo.

## Como ejecutar localmente

```bash
# 1. Crear entorno
python -m venv .venv
source .venv/bin/activate    # en Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. (opcional) Copiar .env.example -> .env y rellenar FRED_API_KEY

# 3. Verificar que estan todos los artifacts
python scripts/audit_artifacts.py

# 4. Refrescar cache de fuentes externas (una vez al dia minimo)
python scripts/update_external_data.py

# 5. Dry-run (no abre operaciones, solo loguea decisiones)
python scripts/run_paper_tick.py --dry-run

# 6. Tick real
python scripts/run_paper_tick.py

# 7. Forzar TFs concretos ignorando ventana de scheduler
python scripts/run_paper_tick.py --force-tf 15m,1h
```

## Ver el dashboard

Local: abrir `dashboard/index.html` en el navegador despues de correr al
menos un tick (genera los JSONs en `dashboard/data/`).

En produccion: el workflow `pages.yml` lo publica en GitHub Pages tras cada
tick exitoso. URL: `https://<usuario>.github.io/XGB_Paper_V1/`.

Para activar Pages: Settings -> Pages -> Source = "GitHub Actions".

## Modelo y artifacts

Los artifacts pequenos viven dentro del repo:

- `artifacts/models/approach_B_xgb_{15m,1h,4h}.json` (~1.3 MB cada uno)
- `artifacts/calibration/calib_approach_B_compact.json` (31 KB, isotonic
  reconstruido desde los `x_thresholds/y_thresholds` por TF)
- `artifacts/candidates/barrier_candidate_library_{tf}.parquet` (~50 KB cada uno)
- `artifacts/schemas/feature_schema.json` (615 features en orden)
- `artifacts/schemas/vol_decile_bounds.json` (rangos para mapear vol a decile)

El modelo final usa **615 features**:

- 10 candidate-level: `vol_pred`, `vol_decile`, `tp_mult`, `sl_mult`, `H`,
  `tp_pct`, `sl_pct`, `side_long`, `barrier_quality_score`, `p_break_even`.
- 605 market features: prefijos `ohlcv_`, `ret_`, `ta_`, `lag_`, `roll_`,
  `cal_`, `der_`, `vp_`, `ext_`, `macro_`, `cx_`, `oc_`, `sent_`, `reg_`,
  `inter_`.

## Cobertura honesta de features en live

El feature builder live (`src/features/live_feature_builder.py`) calcula
de forma fidedigna:

- `ohlcv_*`, `ret_*`, `ta_*` (subset: RSI, EMA, SMA, ATR, BB, MACD, StochRSI),
  `cal_*`, `lag_*`, `roll_*` (subset sobre las columnas mas importantes),
  `inter_*` (parcial), spread spot-perp, ratios cross-asset.
- `ext_*`, `macro_*`, `cx_*`, `oc_*`, `sent_*`, `der_funding_*` desde la
  cache externa (refrescada cada 6h).

Lo que aun queda en NaN (XGBoost lo maneja):

- `vp_*` (Volume Profile - requiere aggTrades streaming).
- Algunos `ta_*` avanzados (Ichimoku, Fibonacci completo).
- `reg_*` (regime flags - requeririan portar el modulo M11 del pipeline
  original).
- `roll_*` de columnas externas (macro/onchain).

El log incluye un contador `n_missing` por tick para que sea visible.

## Como activar el bot en GitHub Actions

1. Push del repo a GitHub.
2. Settings -> Secrets -> Actions -> New secret `FRED_API_KEY` (opcional;
   sin esta key las series FRED quedan NaN, el modelo tolera NaN).
3. El cron `*/5` arranca solo. Para forzar un tick: Actions -> paper_trader
   -> Run workflow.
4. Para evitar costes de minutos cuando estes lejos, deshabilita el cron
   comentandolo en `.github/workflows/paper_trader.yml`.

## Como pasar de paper a live (futuro)

`config/paper_trading.yaml -> execution_market`:

- `paper_synthetic` (actual): shorts simulados; sin Binance keys.
- `spot_long_only`: solo longs reales en Binance Spot. Pendiente de
  implementar `src/execution/binance_broker.py`.
- `futures`: longs y shorts en USDM Futures. Mismo TODO.

Cuando llegue ese momento, configurar Binance API keys como secrets y
sustituir `paper_broker` por `binance_broker` en `run_paper_tick.py`.

## Lo que falta o queda como TODO

Documentado en [TODO.md](TODO.md) (si existe) o como comentarios `# TODO:`
en el codigo. Los principales:

1. **Volume Profile (`vp_*`)** - requiere consumo de aggTrades de Binance,
   probablemente via WebSocket. Para v1 estos features quedan en NaN.
2. **Regime features (`reg_*`)** - portar el notebook `T11_regime.ipynb`
   al builder live.
3. **Volatility model real** - usamos EWMA del true range como proxy; el
   pipeline original usa `lgbm`/`xgb`. Si esos modelos estan exportados,
   reemplazar `src/volatility/ewma_vol.py`.
4. **Open Interest y Long/Short ratio** - endpoints existen en
   `binance_futures.py` pero no se integran al feature row aun.
