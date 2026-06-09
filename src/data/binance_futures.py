"""Cliente de Binance USDM Futures: klines, markPrice, indexPrice, funding."""
from __future__ import annotations

import pandas as pd

from ..utils.http import get_json
from ..utils.logging_utils import get_logger
from .binance_spot import KLINE_COLS, _parse_klines

log = get_logger("binance_futures")

BASE = "https://fapi.binance.com"


def fetch_klines(symbol: str, interval: str, limit: int = 1000,
                 timeout: int = 15) -> pd.DataFrame:
    data = get_json(BASE + "/fapi/v1/klines",
                    params={"symbol": symbol, "interval": interval, "limit": limit},
                    timeout=timeout)
    return _parse_klines(data)


def fetch_mark_price_klines(symbol: str, interval: str, limit: int = 1000,
                            timeout: int = 15) -> pd.DataFrame:
    data = get_json(BASE + "/fapi/v1/markPriceKlines",
                    params={"symbol": symbol, "interval": interval,
                            "limit": limit},
                    timeout=timeout)
    return _parse_klines(data)


def fetch_index_price_klines(symbol: str, interval: str, limit: int = 1000,
                             timeout: int = 15) -> pd.DataFrame:
    data = get_json(BASE + "/fapi/v1/indexPriceKlines",
                    params={"pair": symbol, "interval": interval, "limit": limit},
                    timeout=timeout)
    return _parse_klines(data)


def fetch_funding_rate(symbol: str, limit: int = 100,
                       timeout: int = 15) -> pd.DataFrame:
    data = get_json(BASE + "/fapi/v1/fundingRate",
                    params={"symbol": symbol, "limit": limit},
                    timeout=timeout)
    df = pd.DataFrame(data)
    if df.empty:
        return df
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    return df.rename(columns={"fundingTime": "time", "fundingRate": "funding_rate"})
