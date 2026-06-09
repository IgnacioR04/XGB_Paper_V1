"""Yahoo Finance via yfinance: VIX, SPX, NDX, Gold, Oil, etc."""
from __future__ import annotations

import pandas as pd
import yfinance as yf

from ..utils.logging_utils import get_logger

log = get_logger("yahoo_macro")


def fetch_ticker_history(ticker: str, period: str = "60d",
                         interval: str = "1d") -> pd.DataFrame:
    try:
        df = yf.download(ticker, period=period, interval=interval,
                         progress=False, auto_adjust=False, threads=False)
    except Exception as e:
        log.warning("yfinance failed for %s: %s", ticker, e)
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.reset_index()
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    if "date" in df.columns:
        df["time"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    elif "datetime" in df.columns:
        df["time"] = pd.to_datetime(df["datetime"]).dt.tz_localize("UTC", nonexistent="shift_forward")
    return df


def fetch_all(tickers: dict[str, str]) -> dict[str, pd.DataFrame]:
    out = {}
    for name, sym in tickers.items():
        out[name] = fetch_ticker_history(sym)
    return out
