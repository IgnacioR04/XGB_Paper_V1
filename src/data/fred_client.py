"""FRED API (St. Louis Fed). Requiere FRED_API_KEY como env var."""
from __future__ import annotations

import os
import pandas as pd

from ..utils.http import get_json
from ..utils.logging_utils import get_logger

log = get_logger("fred")

URL = "https://api.stlouisfed.org/fred/series/observations"


def fetch_series(series_id: str, observation_start: str | None = None,
                 timeout: int = 20) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        log.warning("FRED_API_KEY missing, skipping series %s", series_id)
        return pd.DataFrame(columns=["time", "value"])
    params = {"series_id": series_id, "api_key": api_key, "file_type": "json"}
    if observation_start:
        params["observation_start"] = observation_start
    data = get_json(URL, params=params, timeout=timeout)
    rows = data.get("observations", [])
    if not rows:
        return pd.DataFrame(columns=["time", "value"])
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df[["time", "value"]].dropna(subset=["value"]).reset_index(drop=True)
