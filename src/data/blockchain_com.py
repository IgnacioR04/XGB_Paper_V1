"""Blockchain.com charts API (gratis, sin key)."""
from __future__ import annotations

import pandas as pd

from ..utils.http import get_json


BASE = "https://api.blockchain.info/charts"


def fetch_chart(name: str, timespan: str = "180days",
                timeout: int = 20) -> pd.DataFrame:
    data = get_json(f"{BASE}/{name}",
                    params={"timespan": timespan, "format": "json"},
                    timeout=timeout)
    values = data.get("values", [])
    df = pd.DataFrame(values)
    if df.empty:
        return df
    df["time"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.rename(columns={"y": name.replace("-", "_")})
    return df[["time", name.replace("-", "_")]].sort_values("time").reset_index(drop=True)


def fetch_basic_onchain() -> dict[str, pd.DataFrame]:
    """Devuelve las 4 metricas que usa el modelo (oc_hashrate, oc_difficulty,
    oc_tx_count_24h, oc_addresses_24h)."""
    return {
        "hashrate":   fetch_chart("hash-rate"),
        "difficulty": fetch_chart("difficulty"),
        "tx_count":   fetch_chart("n-transactions"),
        "addresses":  fetch_chart("n-unique-addresses"),
    }
