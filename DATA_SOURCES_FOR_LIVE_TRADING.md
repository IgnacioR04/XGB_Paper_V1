# Datos necesarios para Live Trading - BTC Prediction Pipeline

_Documento generado para que una IA u operador pueda replicar el sistema en produccion._

## 0. Resumen para impacientes

El modelo final (Approach B - XGBoost sobre `label_win`, calibracion isotonic, top_EV_per_ts, threshold p_win > 0.84, coste 0.0012) **solo lee del dataset master** (`Data/master/master_15m.parquet`), pero ese master se construye a partir de **8 fuentes distintas**. Para operar en vivo necesitas mantener actualizadas estas fuentes a cada cierre de vela de 15 min.

Si tuvieras que elegir solo **lo critico para que el sistema arranque** (orden de prioridad):

1. **Binance Spot klines** (BTC, ETH, ETHBTC, XRP, XRPBTC) — sin esto no hay nada.
2. **Binance Spot aggTrades BTCUSDT** — para VWAP y Volume Profile.
3. **Binance Futures klines + funding rate** — para `der_*` features.
4. **CryptoCompare histoday** (10 coins) — para dominance / TOTAL / cross-crypto.
5. **Yahoo Finance** (VIX, SPX, NDX, Gold, Oil) — para macro.
6. **Fear & Greed Index** (alternative.me) — para sentiment.
7. **Blockchain.com charts** — para on-chain basico.
8. **FRED** (DXY, M2, CPI, tipos) — para macro a baja frecuencia.

Lo NO necesario (en este pipeline esta vacio o desactivado):

- Order book historico (M04 vacio)
- CoinGlass (requiere plan de pago)
- Glassnode premium
- Twitter / X
- Santiment, LunarCrush, Reddit (opcionales)

---

## 1. Estructura de almacenamiento

```text
Data/raw/
|-- binance_spot/{SYMBOL}/klines_{tf}/{SYMBOL}_klines_{tf}_{YYYY}_{MM}.parquet
|-- binance_spot/BTCUSDT/aggTrades/BTCUSDT_aggTrades_{YYYY}_{MM}_{DD}.parquet
|-- binance_futures/{SYMBOL}/klines_{tf}/...
|-- binance_futures/{SYMBOL}/markPriceKlines_{tf}/...
|-- binance_futures/{SYMBOL}/indexPriceKlines_{tf}/...
|-- binance_futures/{SYMBOL}/fundingRate/{SYMBOL}_fundingRate_full.parquet
|-- binance_futures/{SYMBOL}/openInterest_{5m,1h}/...
|-- binance_futures/{SYMBOL}/longShortRatio/...
|-- onchain/blockchain_com/{metric}.parquet
|-- onchain/glassnode/{category}/{metric}.parquet
|-- sentiment/fear_greed/fear_greed_daily.parquet
|-- sentiment/google_trends/gtrends_*.parquet
|-- sentiment/santiment/santiment_*.parquet
|-- sentiment/lunarcrush/lc_btc_hourly.parquet
|-- sentiment/reddit/reddit_posts.parquet
|-- macro/fred/fred_{name}.parquet
|-- macro/yahoo/yahoo_{ticker}_{tf}.parquet
|-- cross_crypto/cryptocompare_history/{coin}.parquet
|-- cross_crypto/total_mcap/{total,total2,total3,dominance}.parquet
`-- orderbook/{symbol}/snapshots_{spot,futures}/...
```

Master final (lo que lee el modelo en live):

```text
Data/master/master_15m.parquet      # 308k filas x 646 columnas
Data/master/master_15m_latest.parquet  # ultimos 30 dias (para inferencia)
```

---

## 2. Fuente 1 - Binance Spot (E01)

**Critica. Sin esto no hay sistema.**

### Klines (OHLCV)

- Bulk publico: `https://data.binance.vision/data/spot/monthly/klines/`
- En live: `GET https://api.binance.com/api/v3/klines`
- Sin API key necesaria para bulk historico.

| Simbolo | Para que | Timeframes |
|---|---|---|
| BTCUSDT | activo principal | 1m, 5m, 15m, 1h, 4h, 1d |
| ETHUSDT | cross-asset reg | 15m, 1h, 4h |
| ETHBTC | dominance proxy | 15m, 1h, 4h |
| XRPUSDT | cross-asset reg | 15m, 1h, 4h |
| XRPBTC | cross-asset reg | 15m, 1h, 4h |

**Frecuencia live**: cada 15 min (cierre de vela 15m).
**Latencia aceptable**: <30 s.
**Endpoint sugerido en live**: REST `klines?symbol=BTCUSDT&interval=15m&limit=2`.

### aggTrades (solo BTCUSDT)

- Bulk diario: `https://data.binance.vision/data/spot/daily/aggTrades/`
- En live: `GET https://api.binance.com/api/v3/aggTrades`
- Necesario para Volume Profile (M06) y VWAP (M02).
- Solo BTCUSDT. Tamaño grande (10-40 MB/dia parquet).
- **Si quieres ahorrar**: en live se puede aproximar VWAP desde klines 1m. El sistema actual usa aggTrades reales.

---

## 3. Fuente 2 - Binance Futures (E02)

Necesaria para features `der_*` (funding, basis spot-perp, premium index).

### Klines futures + markPrice + indexPrice

- Bulk: `https://data.binance.vision/data/futures/um/monthly/klines/`
- Live: `GET https://fapi.binance.com/fapi/v1/klines`
- Simbolos: BTCUSDT, ETHUSDT.
- TFs: 15m minimo. 1m, 5m, 1h, 4h, 1d historicos.

### Funding rate

- Historico: REST paginado en `GET https://fapi.binance.com/fapi/v1/fundingRate`
- Frecuencia natural: cada 8h (00:00, 08:00, 16:00 UTC).
- En live: simplemente actualizar 3 veces al dia.

### Open Interest

- 5m ultimos 30 dias: `GET https://fapi.binance.com/futures/data/openInterestHist?period=5m`
- 1h historico: `period=1h`
- **Importante**: Binance solo expone OI desde 2020. En el dataset hay NaN antes de esa fecha.

### Long/Short Ratio

- 4 endpoints REST en `fapi.binance.com/futures/data/`:
  - `globalLongShortAccountRatio`
  - `topLongShortAccountRatio`
  - `topLongShortPositionRatio`
  - `takerlongshortRatio`
- TF nativo 1h.

### Rate limits Binance Futures REST

- IP weight: 2400 / minuto.
- 429 = backoff exponencial; 418 = IP baneada temporal.
- En live el consumo es minimo (1-2 calls/15min).

---

## 4. Fuente 3 - Order Book Snapshots (E03)

**Opcional. En el sistema actual M04 esta vacio.**

- No hay historico LOB gratis.
- Tardis.dev es de pago.
- Si en el futuro quieres incluir features LOB (imbalance, depth, whales), necesitas un servidor 24/7 capturando snapshots cada 15 min.

Endpoints:
- Spot: `GET https://api.binance.com/api/v3/depth?symbol=BTCUSDT&limit=1000`
- Futures: `GET https://fapi.binance.com/fapi/v1/depth?symbol=BTCUSDT&limit=1000`

---

## 5. Fuente 4 - CoinGlass (E04)

**Requiere API key de pago. Actualmente vacio en el pipeline.**

API: `https://open-api-v4.coinglass.com`

Endpoints usados:
- `/api/futures/funding-rate/oi-weight-ohlc-history` (funding agregado ponderado por OI)
- `/api/futures/open-interest/aggregated-history` (OI agregado de TODOS los exchanges)
- `/api/futures/liquidation/aggregated-history` (liquidaciones agregadas)
- `/api/futures/global-long-short-account-ratio` (LS ratio agregado)
- `/api/option/max-pain` (max pain - bonus)

Si lo activas, da features `ext_*_agg` ricas en informacion de derivados cross-exchange.

---

## 6. Fuente 5 - On-chain (E05)

Dos sub-fuentes:

### A) Blockchain.com Charts (GRATIS)

Sin key. Solo metricas BTC basicas:

- `hash-rate`
- `difficulty`
- `n-transactions` (tx count)
- `n-unique-addresses`
- `miners-revenue`

Endpoint: `https://api.blockchain.info/charts/{name}?timespan=all&format=json`
Granularidad: daily.

### B) Glassnode (REQUIERE KEY)

Plan free: solo 1 metrica gratis. Plan con key: 10 req/min.

Endpoints: `https://api.glassnode.com/v1/metrics/`
Categorias usadas:
- mining (SOPR, MVRV, hash ribbon)
- indicators (NVT, NUPL)
- transactions (volumes, fees)
- exchange flows
- stablecoins supply

**En el dataset actual solo hay 4 columnas on-chain** porque Blockchain.com cubre lo basico. Si quieres mas, plan Glassnode.

---

## 7. Fuente 6 - Sentiment (E06)

### Fear & Greed Index (alternative.me) - GRATIS - CRITICO

- Endpoint: `https://api.alternative.me/fng/?limit=0&format=json`
- Daily, 1 sola llamada baja todo el historico.
- En live: refrescar 1 vez al dia.

### Google Trends (pytrends) - GRATIS

- Keywords: `bitcoin`, `BTC`, `crypto`, `ethereum`.
- Rate limit oficioso: 1 req cada 30s.
- En el dataset actual NO esta descargado (no bloquea).

### Santiment - PREMIUM (opcional)

- `getMetric`: social_volume_total, sentiment_balance_total.
- Free: 1000 calls/mes.

### LunarCrush - PREMIUM (opcional)

- galaxy_score, social_volume, social_dominance.

### Reddit (PRAW + VADER) - GRATIS con cuenta

- r/bitcoin, r/cryptocurrency, r/CryptoMarkets.
- Sentiment con VADER local.

**Conclusion**: para live trading basico solo necesitas Fear & Greed. El resto son extras.

---

## 8. Fuente 7 - Macro (E07)

### FRED - GRATIS con key

API: `fredapi` Python wrapper. Key gratis en `https://fred.stlouisfed.org/docs/api/api_key.html`.

Series usadas:

| Variable local | Codigo FRED | Frecuencia |
|---|---|---|
| dxy | DTWEXBGS | weekly |
| us_10y | DGS10 | daily |
| us_2y | DGS2 | daily |
| fed_funds | FEDFUNDS | monthly |
| m2 | M2SL | monthly |
| cpi | CPIAUCSL | monthly |
| pce | PCEPI | monthly |

En live: refrescar 1 vez al dia.

### Yahoo Finance (yfinance) - GRATIS

Sin key. Libreria `yfinance`.

| Variable local | Ticker | TF |
|---|---|---|
| spx | ^GSPC | 1h, 1d |
| ndx | ^IXIC | 1h, 1d |
| rut | ^RUT | 1d |
| vix | ^VIX | 1h, 1d |
| move | ^MOVE | 1d |
| nikkei | ^N225 | 1d |
| hangseng | ^HSI | 1d |
| arkk | ARKK | 1d |
| soxx | SOXX | 1d |
| gold | GC=F | 1d |
| silver | SI=F | 1d |
| oil | CL=F | 1d |
| copper | HG=F | 1d |

**Importante en live**: estos datos solo se actualizan en horario de mercado US (lun-vie 14:30-21:00 UTC). Fuera de eso los features `macro_*` quedan en NaN — eso es esperado y el modelo XGBoost lo maneja nativamente.

---

## 9. Fuente 8 - Cross-crypto (E08)

### CryptoCompare histoday - GRATIS sin key

- Endpoint: `https://min-api.cryptocompare.com/data/v2/histoday`
- Rate limit: 50 calls/seg.
- Max 2000 dias por llamada.

10 coins descargadas:

| Coin | Ticker |
|---|---|
| bitcoin | BTC |
| ethereum | ETH |
| ripple | XRP |
| binancecoin | BNB |
| solana | SOL |
| dogecoin | DOGE |
| cardano | ADA |
| tether | USDT |
| usdcoin | USDC |
| dai | DAI |

### CoinGecko (1 llamada para supply actual)

- Endpoint: `https://api.coingecko.com/api/v3/coins/markets`
- Devuelve `circulating_supply` actual de cada coin.
- Se usa para aproximar mcap historico = precio x supply (la supply cambia pero la dominance relativa es razonable).

Features derivadas: TOTAL, TOTAL2, TOTAL3, BTC dominance, ETH dominance, stablecoin mcap, ratios.

---

## 10. Pipeline live - resumen del flujo

```text
1. esperar cierre de vela 15m UTC (xx:00, xx:15, xx:30, xx:45)
2. fetch klines nuevos
   - Binance Spot:    BTC/ETH/ETHBTC/XRP/XRPBTC en 15m (mas 1m/5m si quieres)
   - Binance Futures: BTC/ETH 15m + markPrice + indexPrice
3. fetch aggTrades BTCUSDT del periodo nuevo (~15 min)
4. recomputar features del master a partir del ultimo punto:
   - M02_returns, M03_technicals, M06_vpvr, M11_regime, M13_lags_rolling
5. cada 8h (00, 08, 16 UTC): fetch funding rate
6. cada 1h: fetch open_interest 1h, long_short_ratio 1h
7. cada 1d: fetch
   - blockchain.com (hash-rate, difficulty, tx-count, addresses)
   - alternative.me (fear & greed)
   - FRED (DXY, M2, CPI, yields, fed_funds)
   - Yahoo (VIX, SPX, NDX, Gold, Oil, etc.)
   - CryptoCompare (10 coins prices)
8. recomputar M07 a M10 (incluye macro y cross-crypto)
9. recomputar M14_targets NO (eso es solo entreno)
10. pasar el ultimo row del master por el modelo direccional Approach B
11. aplicar calibrador isotonic
12. calcular EV_pred con coste 0.0012
13. si p_win_isotonic > 0.84 (threshold actual), abrir orden paper
14. monitorizar TP/SL/timeout
```

---

## 11. Tabla resumen - prioridad y dependencia

| Fuente | Coste | Key | Critico para live | Frecuencia refresh | Latencia maxima |
|---|---|---|---|---|---|
| Binance Spot klines | gratis | no | SI (master) | 15 min | 30 s |
| Binance Spot aggTrades | gratis | no | SI (VWAP/VPVR) | 15 min | 5 min |
| Binance Futures klines | gratis | no | SI (basis) | 15 min | 30 s |
| Binance Futures funding | gratis | no | SI (der_*) | 8 h | 5 min |
| Binance Futures OI | gratis | no | recomendado | 1 h | 5 min |
| Binance Futures LS ratio | gratis | no | recomendado | 1 h | 5 min |
| Order book | gratis* | no | NO (M04 vacio) | - | - |
| CoinGlass | de pago | si | NO en el pipeline | - | - |
| Blockchain.com | gratis | no | SI (oc_*) | daily | 1 h |
| Glassnode | de pago | si | opcional | daily | 1 h |
| Fear & Greed | gratis | no | SI (sent_*) | daily | 1 h |
| Google Trends | gratis | no | opcional | daily | 1 h |
| FRED | gratis | si | SI (macro_*) | daily | 1 h |
| Yahoo Finance | gratis | no | SI (macro_*) | daily | 1 h |
| CryptoCompare | gratis | no | SI (cx_*) | daily | 1 h |
| CoinGecko (supply) | gratis | no | 1 vez | 1/mes | - |

*Order book es gratis vía Binance REST, pero requiere servidor 24/7.

---

## 12. API keys necesarias para live

Solo dos son **realmente necesarias** para el sistema actual:

```yaml
# repo/config/config.local.yaml
api_keys:
  fred: "TU_KEY_FRED_GRATIS"          # gratis, instantanea, https://fred.stlouisfed.org/docs/api/api_key.html
  # opcionales:
  coinglass: ""                        # solo si quieres datos cross-exchange
  glassnode: ""                        # solo si quieres on-chain premium
  santiment: ""                        # opcional sentiment premium
  lunarcrush: ""                       # opcional sentiment premium
```

Binance NO requiere API key para datos publicos (klines, aggTrades, funding, OI, LS ratio, order book). Solo necesitas key para **operar** (place orders, leer balances, etc.).

---

## 13. Que features del master alimentan al modelo

El modelo Approach B usa 605 features de mercado. Distribuidas por prefijo:

| Prefijo | Cols | Origen | Critico? |
|---|---|---|---|
| `ohlcv_` | 55 | Binance Spot + Futures | SI |
| `ext_` | 32 | Yahoo + cross-crypto | SI |
| `ret_` | 22 | derivado de OHLCV | SI |
| `ta_` | 95 | derivado de OHLCV (RSI, EMA, MACD, ATR...) | SI |
| `der_` | 14 | Binance Futures funding/basis | SI |
| `vp_` | 45 | aggTrades BTCUSDT | importante |
| `oc_` | 4 | Blockchain.com | importante |
| `sent_` | 5 | Fear & Greed | importante |
| `macro_` | 19 | FRED + Yahoo | importante |
| `cx_` | 20 | CryptoCompare + CoinGecko | importante |
| `reg_` | 25 | derivado (regimen) | SI |
| `cal_` | 22 | derivado (calendario) | SI |
| `lag_` | 56 | derivado (lags) | SI |
| `roll_` | 192 | derivado (rolling) | SI |
| `inter_` | 4 | derivado (interacciones) | SI |
| `cost_` | 11 | derivado | si solo informativo |
| `qa_` | 13 | derivado (quality flags) | si solo informativo |

**NO incluir nunca como feature** (son targets):

- `tgt_return_*`
- `tgt_direction_*`
- `tgt_triple_barrier`
- `tgt_tb_*`
- `tgt_realized_vol_*`
- `tgt_breakout_*`

---

## 14. Implementacion minima de live data fetcher

Pseudocodigo para un fetcher minimo (Python):

```python
import requests, time, pandas as pd
from pathlib import Path

BASE_SPOT = "https://api.binance.com"
BASE_FUTURES = "https://fapi.binance.com"


def fetch_klines(symbol, interval, limit=2, market="spot"):
    base = BASE_SPOT if market == "spot" else BASE_FUTURES
    path = "/api/v3/klines" if market == "spot" else "/fapi/v1/klines"
    r = requests.get(base + path,
                     params={"symbol": symbol, "interval": interval, "limit": limit},
                     timeout=10)
    r.raise_for_status()
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades",
            "taker_buy_vol", "taker_buy_qvol", "ignore"]
    df = pd.DataFrame(r.json(), columns=cols)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df


def fetch_funding_rate(symbol, limit=1):
    r = requests.get(BASE_FUTURES + "/fapi/v1/fundingRate",
                     params={"symbol": symbol, "limit": limit}, timeout=10)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def fetch_fear_greed():
    r = requests.get("https://api.alternative.me/fng/?limit=1&format=json", timeout=10)
    return r.json()["data"][0]


# Cada cierre de vela 15m
def on_candle_close():
    klines_spot = {sym: fetch_klines(sym, "15m") for sym in
                    ["BTCUSDT", "ETHUSDT", "ETHBTC", "XRPUSDT", "XRPBTC"]}
    klines_futures = {sym: fetch_klines(sym, "15m", market="futures") for sym in
                       ["BTCUSDT", "ETHUSDT"]}
    # ... actualizar master, recomputar features, predecir, decidir orden
```

---

## 15. Que NO necesita el sistema en live

Para no perder tiempo:

- **Order book snapshots**: M04 esta vacio, no se usa.
- **CoinGlass**: requiere plan de pago, no esta en el pipeline.
- **Glassnode premium**: en el dataset hay solo 4 cols on-chain de Blockchain.com.
- **Twitter / X**: no usado.
- **Santiment / LunarCrush / Reddit**: opcionales, el pipeline funciona sin ellos.
- **Klines 1m y 5m de timeframes operados**: el modelo solo opera 15m, 1h, 4h. Las velas 1m/5m solo se usan en barriers_v2 para replay intra-vela (entreno offline).

---

## 16. Riesgos y gotchas en live

- **Latencia datos macro**: Yahoo y FRED solo dan datos en horario US. El modelo tolera NaN aqui pero los features `macro_*` y `ext_*` estaran stale fuera de mercado. No es bug, es comportamiento esperado.
- **Funding rate cada 8h**: si tu fetcher corre cada 15 min, la mayoria de las veces `der_funding_rate` no cambia. El feature es de baja frecuencia.
- **Rate limits Binance**: en futures el limite es 2400 weight/min. Con 7 simbolos * 6 timeframes = 42 calls cada 15 min, estas muy por debajo del limite.
- **Re-descarga de aggTrades**: pesado. Idealmente usa WebSocket trade stream en live y construye VWAP/VPVR incremental, no fetch REST por dia.
- **Discrepancias spot vs futures**: el simbolo futures BTCUSDT no siempre coincide en timing con spot. El sistema usa BOTH y deriva el basis.

---

## 17. Validacion antes de poner live

Antes de operar real (incluso paper), confirma:

1. El master en live tiene las mismas 646 columnas que el master de entrenamiento.
2. Los nombres de columnas coinciden exactamente (case-sensitive).
3. `tgt_*` columnas estan AUSENTES o son NaN (no se calculan en live).
4. El modelo `approach_B_xgb_{tf}.json` carga sin warnings.
5. El calibrador isotonic se aplica antes del threshold.
6. `EV_pred = p_win * tp - (1 - p_win) * sl - 0.0012` se calcula con el coste correcto.
7. Solo abres orden si `p_win_isotonic > 0.84`.

---

## 18. Referencias rapidas

- **Master schema**: `Data/master/master_15m_metadata.json`
- **Feature registry**: `repo/config/feature_registry.yaml`
- **Availability rules** (delays por fuente): `repo/config/availability_rules.yaml`
- **Symbols / tickers**: `repo/config/symbols.yaml`
- **Config general**: `repo/config/config.yaml`
- **Extractores**: `repo/notebooks/01_extract/E0[1-8]_*.ipynb`
- **Transformers**: `repo/notebooks/02_transform/T0[1-9]_*.ipynb` y `T1[0-6]_*.ipynb`
- **Merge**: `repo/notebooks/03_merge/MERGE_master.ipynb`
- **Modelo final**: `Data/model_outputs/directional/models/approach_B_xgb_{15m,1h,4h}.json`
- **Calibrador**: `Data/model_outputs/directional/calibration/calib_approach_B.parquet`
- **Config estrategia final**: `Data/model_outputs/real_backtest/config/final_strategy_config.json`
- **Config paper trading**: `Data/model_outputs/paper_trading/config/paper_trading_config.json`
