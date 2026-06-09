"""Calculo de features tecnicas estandar sobre OHLCV.

NO replica al 100% lo que hace M03_technicals.ipynb del pipeline original.
Calcula los indicadores mas comunes que el modelo XGB usa (RSI, EMA, SMA,
MACD, ATR, BB, ADX, Stoch). Si el modelo espera una feature ta_* que no
generamos aqui, quedara en NaN y XGB la maneja.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0)
    dn = (-delta).clip(lower=0)
    roll_up = up.ewm(alpha=1 / n, adjust=False).mean()
    roll_dn = dn.ewm(alpha=1 / n, adjust=False).mean()
    rs = roll_up / roll_dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def true_range(df: pd.DataFrame) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    return pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)


def bollinger(close: pd.Series, n: int = 20, k: float = 2.0):
    ma = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std()
    upper = ma + k * sd
    lower = ma - k * sd
    width = (upper - lower) / ma
    pctb = (close - lower) / (upper - lower).replace(0, np.nan)
    return ma, upper, lower, width, pctb


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    line = ema_fast - ema_slow
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def stoch_rsi(close: pd.Series, n: int = 14, k_smooth: int = 3, d_smooth: int = 3):
    r = rsi(close, n)
    min_r = r.rolling(n, min_periods=n).min()
    max_r = r.rolling(n, min_periods=n).max()
    raw = (r - min_r) / (max_r - min_r).replace(0, np.nan)
    k = raw.rolling(k_smooth, min_periods=k_smooth).mean()
    d = k.rolling(d_smooth, min_periods=d_smooth).mean()
    return k, d


def compute_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """Recibe OHLCV de BTC y devuelve un dict de columnas ta_*."""
    out = pd.DataFrame(index=df.index)
    c = df["close"]

    for n in (7, 14):
        out[f"ta_rsi_{n}"] = rsi(c, n)
    k, d = stoch_rsi(c)
    out["ta_stoch_rsi_k"] = k
    out["ta_stoch_rsi_d"] = d

    for n in (8, 21, 50, 100, 200):
        out[f"ta_ema_{n}"] = ema(c, n)
        out[f"ta_ema_{n}_dist"] = (c - out[f"ta_ema_{n}"]) / c

    for n in (20, 50, 200):
        out[f"ta_sma_{n}"] = sma(c, n)
        out[f"ta_sma_{n}_dist"] = (c - out[f"ta_sma_{n}"]) / c

    out["ta_atr_14"] = atr(df, 14)
    out["ta_atr_14_pct"] = out["ta_atr_14"] / c

    bb_ma, bb_up, bb_lo, bb_w, bb_pct = bollinger(c, 20, 2.0)
    out["ta_bb_mid_20"] = bb_ma
    out["ta_bb_upper_20"] = bb_up
    out["ta_bb_lower_20"] = bb_lo
    out["ta_bb_width_20"] = bb_w
    out["ta_bb_pctb_20"] = bb_pct

    macd_line, macd_sig, macd_hist = macd(c)
    out["ta_macd"] = macd_line
    out["ta_macd_signal"] = macd_sig
    out["ta_macd_hist"] = macd_hist

    # Tag: si el modelo espera mas ta_*, queda NaN.
    return out
