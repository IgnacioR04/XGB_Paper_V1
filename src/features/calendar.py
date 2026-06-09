"""Features de calendario (prefijo cal_). No requiere datos externos."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index).tz_convert("UTC") if index.tz is not None \
        else pd.DatetimeIndex(index).tz_localize("UTC")
    out = pd.DataFrame(index=index)
    out["cal_hour"] = idx.hour
    out["cal_day_of_week"] = idx.dayofweek
    out["cal_weekend"] = (idx.dayofweek >= 5).astype(int)
    out["cal_month"] = idx.month
    out["cal_quarter"] = idx.quarter
    out["cal_day_of_month"] = idx.day
    out["cal_is_us_session"] = ((idx.hour >= 14) & (idx.hour < 21)).astype(int)
    out["cal_is_asia_session"] = ((idx.hour >= 0) & (idx.hour < 8)).astype(int)
    out["cal_is_europe_session"] = ((idx.hour >= 7) & (idx.hour < 16)).astype(int)
    out["cal_hour_sin"] = np.sin(2 * np.pi * idx.hour / 24)
    out["cal_hour_cos"] = np.cos(2 * np.pi * idx.hour / 24)
    out["cal_dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["cal_dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)
    return out
