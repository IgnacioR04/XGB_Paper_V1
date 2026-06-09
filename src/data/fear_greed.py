"""Fear & Greed Index daily (alternative.me)."""
from __future__ import annotations

import pandas as pd

from ..utils.http import get_json


URL = "https://api.alternative.me/fng/"


def fetch_fear_greed(limit: int = 180, timeout: int = 15) -> pd.DataFrame:
    data = get_json(URL, params={"limit": limit, "format": "json"}, timeout=timeout)
    rows = data.get("data", [])
    if not rows:
        return pd.DataFrame(columns=["time", "fear_greed", "classification"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["timestamp"].astype(int), unit="s", utc=True)
    df["fear_greed"] = df["value"].astype(int)
    df["classification"] = df["value_classification"]
    return df[["time", "fear_greed", "classification"]].sort_values("time").reset_index(drop=True)
