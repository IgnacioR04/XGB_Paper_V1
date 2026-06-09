"""CryptoCompare histoday - gratis sin key."""
from __future__ import annotations

import pandas as pd

from ..utils.http import get_json


URL = "https://min-api.cryptocompare.com/data/v2/histoday"


def fetch_histoday(fsym: str, tsym: str = "USD", limit: int = 365,
                   timeout: int = 15) -> pd.DataFrame:
    data = get_json(URL, params={"fsym": fsym, "tsym": tsym, "limit": limit},
                    timeout=timeout)
    rows = data.get("Data", {}).get("Data", [])
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close",
               "volumefrom", "volumeto"]].sort_values("time").reset_index(drop=True)


def fetch_all_coins(coins: list[str]) -> dict[str, pd.DataFrame]:
    return {c: fetch_histoday(c, "USD") for c in coins}
