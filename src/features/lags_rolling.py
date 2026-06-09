"""Lags y rolling stats (prefijos lag_ y roll_)."""
from __future__ import annotations

import numpy as np
import pandas as pd


LAG_PERIODS = [1, 2, 3, 4, 8, 16, 32]
ROLL_WINDOWS = [4, 12, 24, 96]


def compute_lags_rolling(base: pd.DataFrame) -> pd.DataFrame:
    """Recibe un DataFrame con columnas clave (close, log_return, volume, etc.)
    y devuelve lags + rolling de un conjunto reducido de columnas.

    En el master original esto se hace sobre decenas de columnas. Aqui hacemos
    lo mismo sobre las mas importantes:
        - ret_log_return
        - ret_arith_return
        - ret_volume_log
        - ta_rsi_14
        - ta_atr_14_pct
        - ta_bb_pctb_20
        - ta_macd_hist
    """
    targets = [
        "ret_log_return", "ret_arith_return", "ret_volume_log",
        "ta_rsi_14", "ta_atr_14_pct", "ta_bb_pctb_20", "ta_macd_hist",
    ]
    out = pd.DataFrame(index=base.index)
    for col in targets:
        if col not in base.columns:
            continue
        s = base[col]
        for lag in LAG_PERIODS:
            out[f"lag_{col}_{lag}"] = s.shift(lag)
        for w in ROLL_WINDOWS:
            roll = s.rolling(w, min_periods=max(2, w // 3))
            out[f"roll_{col}_mean_{w}"] = roll.mean()
            out[f"roll_{col}_std_{w}"]  = roll.std()
            out[f"roll_{col}_min_{w}"]  = roll.min()
            out[f"roll_{col}_max_{w}"]  = roll.max()
            # zscore y pctile son los costos extra que el master tenia
            mean = out[f"roll_{col}_mean_{w}"]
            std = out[f"roll_{col}_std_{w}"]
            out[f"roll_{col}_zscore_{w}"] = (s - mean) / std.replace(0, np.nan)
            out[f"roll_{col}_pctile_{w}"] = s.rolling(w, min_periods=max(2, w // 3)) \
                .apply(lambda x: (x <= x.iloc[-1]).mean(), raw=False)
    return out
