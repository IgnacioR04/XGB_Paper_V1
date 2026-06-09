"""Filtros de banda de probabilidad por timeframe."""
from __future__ import annotations

import pandas as pd


def in_band(p: float, lo: float, hi: float, allow_above_band: bool = False) -> bool:
    if allow_above_band:
        return p >= lo
    return (p >= lo) and (p < hi)


def filter_by_band(candidates: pd.DataFrame, lo: float, hi: float,
                   allow_above_band: bool = False,
                   prob_col: str = "p_win_calibrated") -> pd.DataFrame:
    if allow_above_band:
        mask = candidates[prob_col] >= lo
    else:
        mask = (candidates[prob_col] >= lo) & (candidates[prob_col] < hi)
    return candidates[mask].copy()


def top_ev(candidates: pd.DataFrame, n: int = 1) -> pd.DataFrame:
    return candidates.sort_values("EV_pred", ascending=False).head(n).reset_index(drop=True)
