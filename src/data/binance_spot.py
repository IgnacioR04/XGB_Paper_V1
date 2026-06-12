"""Cliente de Binance Spot para klines y ticker en vivo.

Binance bloquea `api.binance.com` desde IPs de Azure/cloud providers
(GitHub Actions runners). Probamos varios hosts en orden y reportamos
cual funciono. Si todos fallan, el caller decide si abortar.

Para BTC tambien hay un fallback a Coinbase via `coinbase_fallback`.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import requests

from ..utils.http import get_json
from ..utils.logging_utils import get_logger

log = get_logger("binance_spot")

# Orden de prueba. api.binance.com va el ultimo porque es el mas bloqueado.
HOST_CANDIDATES = [
    "https://api3.binance.com",
    "https://api2.binance.com",
    "https://api1.binance.com",
    "https://api4.binance.com",
    "https://api.binance.com",
]

# Cache del primer host que respondio en este proceso
_working_host: str | None = None


KLINE_COLS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "num_trades",
    "taker_buy_vol", "taker_buy_qvol", "_ignore",
]


def _parse_klines(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=KLINE_COLS)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ("open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_vol", "taker_buy_qvol"):
        df[c] = df[c].astype(float)
    df["num_trades"] = df["num_trades"].astype(int)
    return df.drop(columns=["_ignore"])


def _try_host(host: str, path: str, params: dict, timeout: int = 10):
    """Devuelve (response, error_str) sin levantar excepcion."""
    try:
        r = requests.get(host + path, params=params, timeout=timeout,
                          headers={"User-Agent": "Mozilla/5.0"})
        return r, None
    except Exception as e:
        return None, str(e)


def _resilient_get(path: str, params: dict, timeout: int = 10) -> dict | list:
    """Prueba HOST_CANDIDATES en orden. Devuelve el JSON del primero que
    responde 200. Si ninguno funciona, levanta RuntimeError con detalle."""
    global _working_host
    attempts = []

    hosts = ([_working_host] if _working_host else []) + \
            [h for h in HOST_CANDIDATES if h != _working_host]

    for host in hosts:
        r, err = _try_host(host, path, params, timeout)
        if err is not None:
            attempts.append(f"{host} -> ERR {err[:80]}")
            continue
        attempts.append(f"{host} -> HTTP {r.status_code}")
        if r.status_code == 200:
            _working_host = host
            log.info("Binance Spot OK via %s", host)
            return r.json()

    msg = "All Binance Spot hosts failed:\n  " + "\n  ".join(attempts)
    log.error(msg)
    raise RuntimeError(msg)


def fetch_klines(symbol: str, interval: str, limit: int = 1000,
                 end_time_ms: int | None = None,
                 timeout: int = 15) -> pd.DataFrame:
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    data = _resilient_get("/api/v3/klines", params, timeout=timeout)
    return _parse_klines(data)


def fetch_ticker_price(symbol: str, timeout: int = 10) -> float:
    data = _resilient_get("/api/v3/ticker/price", {"symbol": symbol},
                          timeout=timeout)
    return float(data["price"])


def coinbase_spot_price(symbol: str, timeout: int = 10) -> float | None:
    """Precio spot actual via Coinbase. Solo BTCUSDT/ETHUSDT."""
    pair_map = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
    if symbol not in pair_map:
        return None
    url = f"https://api.exchange.coinbase.com/products/{pair_map[symbol]}/ticker"
    try:
        r = requests.get(url, timeout=timeout,
                         headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            return None
        return float(r.json()["price"])
    except Exception as e:
        log.warning("Coinbase ticker failed for %s: %s", symbol, e)
        return None


def fetch_ticker_price_with_fallback(symbol: str,
                                      timeout: int = 10) -> tuple[float, str]:
    """Devuelve (price, source). source in {binance_spot, coinbase}.
    Levanta RuntimeError si ambas fallan."""
    try:
        return fetch_ticker_price(symbol, timeout=timeout), "binance_spot"
    except Exception as e:
        log.warning("Binance ticker failed for %s: %s", symbol, e)
    price = coinbase_spot_price(symbol, timeout=timeout)
    if price is not None:
        return price, "coinbase"
    raise RuntimeError(f"No ticker source available for {symbol}")


def fetch_last_n_closed(symbol: str, interval: str, n: int,
                        timeout: int = 15) -> pd.DataFrame:
    raw = fetch_klines(symbol, interval, limit=n + 1, timeout=timeout)
    now = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
    closed = raw[raw["close_time"] <= now].copy()
    return closed.tail(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fallback: Coinbase (solo BTC, ETH). Sus klines tienen otro formato.
# Limitaciones de Coinbase Exchange API:
#   - max 300 velas por request (paginamos con start/end)
#   - granularity solo en {60, 300, 900, 3600, 21600, 86400};
#     4h NO existe -> bajamos 1h y resampleamos a 4h alineado UTC.
# ---------------------------------------------------------------------------

COINBASE_GRANULARITY = {"1m": 60, "15m": 900, "1h": 3600}


def _coinbase_candles_paginated(pair: str, granularity: int, n: int) -> list:
    """Devuelve hasta n velas [time, low, high, open, close, vol] ASC."""
    url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
    collected: dict[int, list] = {}
    end = dt.datetime.now(dt.timezone.utc)
    for _ in range(10):  # max 10 paginas de 300
        if len(collected) >= n:
            break
        start = end - dt.timedelta(seconds=granularity * 300)
        try:
            r = requests.get(url, params={
                "granularity": granularity,
                "start": start.isoformat(),
                "end": end.isoformat(),
            }, timeout=15, headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                log.warning("Coinbase HTTP %d for %s gran=%d",
                            r.status_code, pair, granularity)
                break
            rows = r.json()
        except Exception as e:
            log.warning("Coinbase request failed for %s: %s", pair, e)
            break
        if not rows:
            break
        for row in rows:
            if row and len(row) >= 6:
                collected[int(row[0])] = row
        oldest = min(int(row[0]) for row in rows if row)
        end = dt.datetime.fromtimestamp(oldest, dt.timezone.utc)
    return [collected[t] for t in sorted(collected)][-n:]


def _coinbase_rows_to_df(rows: list, gran: int) -> pd.DataFrame:
    out = []
    for ts, low, high, open_, close, vol in rows:
        out.append({
            "open_time": pd.Timestamp(int(ts), unit="s", tz="UTC"),
            "open": float(open_), "high": float(high), "low": float(low),
            "close": float(close), "volume": float(vol),
            "close_time": pd.Timestamp(int(ts) + gran - 1, unit="s", tz="UTC"),
            "quote_volume": float(vol) * float(close),
            "num_trades": 0, "taker_buy_vol": 0.0, "taker_buy_qvol": 0.0,
        })
    return pd.DataFrame(out)


def _resample_1h_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega velas 1h en bloques 4h alineados a UTC 00/04/08/12/16/20."""
    if df.empty:
        return df
    df = df.set_index("open_time").sort_index()
    agg = df.resample("4h", origin="epoch").agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum", "quote_volume": "sum",
        "num_trades": "sum", "taker_buy_vol": "sum", "taker_buy_qvol": "sum",
    }).dropna(subset=["open"])
    agg["close_time"] = agg.index + pd.Timedelta(hours=4) - pd.Timedelta(seconds=1)
    # Descartar el bloque 4h aun en curso (incompleto: menos de 4 velas 1h)
    counts = df["close"].resample("4h", origin="epoch").count()
    agg = agg[counts >= 4]
    return agg.reset_index()


def coinbase_fallback_klines(symbol: str, interval: str, n: int) -> pd.DataFrame:
    """Convierte BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD.

    Mismo formato que fetch_klines (num_trades/taker_* en 0).
    """
    pair_map = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
    if symbol not in pair_map:
        return pd.DataFrame()
    pair = pair_map[symbol]

    if interval == "4h":
        rows = _coinbase_candles_paginated(pair, 3600, n * 4 + 8)
        df1h = _coinbase_rows_to_df(rows, 3600)
        return _resample_1h_to_4h(df1h).tail(n + 1).reset_index(drop=True)

    gran = COINBASE_GRANULARITY.get(interval)
    if gran is None:
        return pd.DataFrame()
    rows = _coinbase_candles_paginated(pair, gran, n + 2)
    df = _coinbase_rows_to_df(rows, gran)
    return df.tail(n + 1).reset_index(drop=True)


def fetch_last_n_closed_with_fallback(symbol: str, interval: str, n: int,
                                       timeout: int = 15) -> tuple[pd.DataFrame, str]:
    """Intenta Binance Spot. Si falla y el simbolo es BTCUSDT o ETHUSDT,
    cae a Coinbase.

    Devuelve (df, source) donde source ∈ {"binance_spot", "coinbase", "empty"}.
    """
    try:
        df = fetch_last_n_closed(symbol, interval, n, timeout=timeout)
        if not df.empty:
            return df, "binance_spot"
    except Exception as e:
        log.warning("Binance Spot failed for %s %s: %s", symbol, interval, e)

    # Fallback
    df = coinbase_fallback_klines(symbol, interval, n)
    if not df.empty:
        # pd.Timestamp.utcnow() en pandas >=3.0 ya devuelve tz-aware;
        # en versiones anteriores devuelve naive. Normalizamos:
        now = pd.Timestamp.utcnow()
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        df = df[df["close_time"] <= now].tail(n).reset_index(drop=True)
        if not df.empty:
            log.warning("Using Coinbase fallback for %s %s", symbol, interval)
            return df, "coinbase"
    return pd.DataFrame(), "empty"


def fetch_last_n_closed_paginated(symbol: str, interval: str, n: int,
                                   timeout: int = 15) -> tuple[pd.DataFrame, str]:
    """Como fetch_last_n_closed_with_fallback pero soporta n > 1000 en
    Binance paginando con endTime (Coinbase ya pagina internamente)."""
    if n <= 1000:
        return fetch_last_n_closed_with_fallback(symbol, interval, n, timeout)
    try:
        frames = []
        end_ms = None
        remaining = n + 1
        while remaining > 0:
            df = fetch_klines(symbol, interval, limit=min(1000, remaining),
                              end_time_ms=end_ms, timeout=timeout)
            if df.empty:
                break
            frames.append(df)
            remaining -= len(df)
            end_ms = int(df["open_time"].iloc[0].timestamp() * 1000) - 1
            if len(df) < 1000:
                break
        if frames:
            full = (pd.concat(frames)
                    .drop_duplicates("open_time")
                    .sort_values("open_time"))
            now = pd.Timestamp.utcnow()
            if now.tzinfo is None:
                now = now.tz_localize("UTC")
            closed = full[full["close_time"] <= now]
            if not closed.empty:
                return closed.tail(n).reset_index(drop=True), "binance_spot"
    except Exception as e:
        log.warning("Binance paginated failed for %s %s: %s", symbol, interval, e)
    return fetch_last_n_closed_with_fallback(symbol, interval, min(n, 8000), timeout)


def save_klines_parquet(df: pd.DataFrame, out_dir: Path, symbol: str,
                        interval: str) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"{symbol}_{interval}.parquet"
    if p.exists():
        old = pd.read_parquet(p)
        combined = pd.concat([old, df], ignore_index=True)
        combined = combined.drop_duplicates("open_time", keep="last")
        combined = combined.sort_values("open_time").reset_index(drop=True)
    else:
        combined = df.copy()
    combined.to_parquet(p, index=False)
    return p
