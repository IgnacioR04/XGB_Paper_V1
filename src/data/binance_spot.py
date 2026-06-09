"""Cliente de Binance Spot para klines y ticker en vivo.

API publica (sin key). Para paper trading es todo lo que necesitamos.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..utils.http import get_json
from ..utils.logging_utils import get_logger

log = get_logger("binance_spot")

BASE = "https://api.binance.com"

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


def fetch_klines(symbol: str, interval: str, limit: int = 1000,
                 end_time_ms: int | None = None,
                 timeout: int = 15) -> pd.DataFrame:
    """Devuelve hasta `limit` klines hasta `end_time_ms` (default: ahora)."""
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time_ms is not None:
        params["endTime"] = end_time_ms
    data = get_json(BASE + "/api/v3/klines", params=params, timeout=timeout)
    return _parse_klines(data)


def fetch_ticker_price(symbol: str, timeout: int = 10) -> float:
    """Precio actual (uso live para realistic_entry_price)."""
    data = get_json(BASE + "/api/v3/ticker/price",
                    params={"symbol": symbol}, timeout=timeout)
    return float(data["price"])


def fetch_last_n_closed(symbol: str, interval: str, n: int,
                        timeout: int = 15) -> pd.DataFrame:
    """Devuelve las ultimas n velas CERRADAS (descarta vela en curso).

    Binance devuelve la vela actual incompleta como ultimo elemento. La
    detectamos comparando close_time con now y la dropeamos si aplica.
    """
    raw = fetch_klines(symbol, interval, limit=n + 1, timeout=timeout)
    now = pd.Timestamp.utcnow().tz_localize(None).tz_localize("UTC")
    closed = raw[raw["close_time"] <= now].copy()
    return closed.tail(n).reset_index(drop=True)


def save_klines_parquet(df: pd.DataFrame, out_dir: Path, symbol: str,
                        interval: str) -> Path:
    """Guarda klines en data/live_raw/binance_spot/SYMBOL_INTERVAL.parquet,
    deduplicando por open_time si ya existia."""
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
    log.info("Saved %d rows -> %s", len(combined), p.name)
    return p
