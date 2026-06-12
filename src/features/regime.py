"""Clasificacion de regimen de mercado usando EMA50/EMA200 en 1h.

Reglas (iguales a T11_regime.ipynb y RB16):
  - alcista: close_1h > EMA200 AND EMA50 > EMA200
  - bajista: close_1h < EMA200 AND EMA50 < EMA200
  - lateral: todo lo demas

Solo usa informacion hasta t (causal). Los EMAs son indicadores rezagados
por definicion: el regimen en t aplica a la decision de t+1 sin leakage.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger("regime")

MIN_CANDLES_1H = 200  # EMA200 necesita al menos 200 velas para converger


def classify_regime(ohlc_15m: pd.DataFrame) -> str:
    """Clasifica el regimen actual a partir de velas de 15m.

    Resamplea a 1h, computa EMA50 y EMA200, y devuelve 'alcista',
    'bajista', o 'lateral'.
    """
    close_col = "ohlcv_btc_close" if "ohlcv_btc_close" in ohlc_15m.columns else "close"
    if close_col not in ohlc_15m.columns:
        log.warning("No close column found, defaulting to 'lateral'")
        return "lateral"

    h1 = ohlc_15m[close_col].resample("1h").last().dropna()

    if len(h1) < MIN_CANDLES_1H:
        log.warning("Only %d 1h candles (need %d for EMA200), defaulting to 'lateral'",
                     len(h1), MIN_CANDLES_1H)
        return "lateral"

    ema50 = h1.ewm(span=50, adjust=False).mean()
    ema200 = h1.ewm(span=200, adjust=False).mean()

    last_close = float(h1.iloc[-1])
    last_ema50 = float(ema50.iloc[-1])
    last_ema200 = float(ema200.iloc[-1])

    if last_close > last_ema200 and last_ema50 > last_ema200:
        regime = "alcista"
    elif last_close < last_ema200 and last_ema50 < last_ema200:
        regime = "bajista"
    else:
        regime = "lateral"

    log.info("Regime: %s (close=%.2f, ema50=%.2f, ema200=%.2f)",
             regime, last_close, last_ema50, last_ema200)
    return regime


def classify_regime_series(close_1h: pd.Series) -> pd.Series:
    """Vectorizado: devuelve una Serie con 'alcista'/'bajista'/'lateral' por vela."""
    ema50 = close_1h.ewm(span=50, adjust=False).mean()
    ema200 = close_1h.ewm(span=200, adjust=False).mean()

    regime = pd.Series("lateral", index=close_1h.index, name="regimen")
    regime[(close_1h > ema200) & (ema50 > ema200)] = "alcista"
    regime[(close_1h < ema200) & (ema50 < ema200)] = "bajista"
    return regime
