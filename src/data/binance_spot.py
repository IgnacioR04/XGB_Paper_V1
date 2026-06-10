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


def fetch_last_n_closed(symbol: str, interval: str, n: int,
                        timeout: int = 15) -> pd.DataFrame:
    raw = fetch_klines(symbol, interval, limit=n + 1, timeout=timeout)
    now = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
    closed = raw[raw["close_time"] <= now].copy()
    return closed.tail(n).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Fallback: Coinbase (solo BTC, ETH). Sus klines tienen otro formato.
# ---------------------------------------------------------------------------

COINBASE_INTERVAL_SEC = {"15m": 900, "1h": 3600, "4h": 14400}


def coinbase_fallback_klines(symbol: str, interval: str, n: int) -> pd.DataFrame:
    """Convierte BTCUSDT -> BTC-USD, ETHUSDT -> ETH-USD.

    Devuelve el mismo formato que fetch_klines (sin trades/taker). Las
    columnas que no existen en Coinbase quedan en 0/NaN.
    """
    pair_map = {"BTCUSDT": "BTC-USD", "ETHUSDT": "ETH-USD"}
    if symbol not in pair_map:
        return pd.DataFrame()
    pair = pair_map[symbol]
    gran = COINBASE_INTERVAL_SEC[interval]
    url = f"https://api.exchange.coinbase.com/products/{pair}/candles"
    try:
        r = requests.get(url, params={"granularity": gran},
                         timeout=15, headers={"User-Agent": "Mozilla/5.0"})
        if r.status_code != 200:
            log.warning("Coinbase fallback HTTP %d for %s", r.status_code, pair)
            return pd.DataFrame()
        data = r.json()
    except Exception as e:
        log.warning("Coinbase fallback failed for %s: %s", pair, e)
        return pd.DataFrame()

    # Coinbase: [time, low, high, open, close, volume]
    rows = []
    for entry in data:
        if not entry or len(entry) < 6:
            continue
        ts, low, high, open_, close, vol = entry
        rows.append({
            "open_time": pd.Timestamp(ts, unit="s", tz="UTC"),
            "open": float(open_),
            "high": float(high),
            "low": float(low),
            "close": float(close),
            "volume": float(vol),
            "close_time": pd.Timestamp(ts + gran - 1, unit="s", tz="UTC"),
            "quote_volume": float(vol) * float(close),
            "num_trades": 0,
            "taker_buy_vol": 0.0,
            "taker_buy_qvol": 0.0,
        })
    df = pd.DataFrame(rows).sort_values("open_time").reset_index(drop=True)
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
        now = pd.Timestamp.utcnow().tz_localize("UTC")
        df = df[df["close_time"] <= now].tail(n).reset_index(drop=True)
        if not df.empty:
            log.warning("Using Coinbase fallback for %s %s", symbol, interval)
            return df, "coinbase"
    return pd.DataFrame(), "empty"


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
