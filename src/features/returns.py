"""Features de returns y candle shape (prefijo ret_)."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_returns(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    c = df["close"]
    o = df["open"]
    h = df["high"]
    lo = df["low"]
    v = df["volume"]

    out["ret_log_return"] = np.log(c / c.shift(1))
    out["ret_arith_return"] = c.pct_change()

    for n in (3, 6, 12, 24, 48, 96):
        out[f"ret_return_{n}"] = c.pct_change(n)

    out["ret_candle_body"] = (c - o) / o
    out["ret_candle_upper_wick"] = (h - np.maximum(c, o)) / o
    out["ret_candle_lower_wick"] = (np.minimum(c, o) - lo) / o
    out["ret_candle_range_pct"] = (h - lo) / o
    out["ret_volume_log"] = np.log1p(v)
    out["ret_volume_ratio_20"] = v / v.rolling(20, min_periods=5).mean()

    # VWAP rolling sobre la propia vela aproximada con (H+L+C)/3
    typical = (h + lo + c) / 3
    out["ret_vwap_typical"] = (typical * v).rolling(20, min_periods=5).sum() \
        / v.rolling(20, min_periods=5).sum()
    return out
