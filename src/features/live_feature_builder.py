"""Constructor de features en vivo - v2 con PARIDAD de entrenamiento.

CLAVE: en entrenamiento, las features de mercado de los TRES modelos
(15m/1h/4h) salen del GRID DE 15 MINUTOS (el master). Por eso este builder
SIEMPRE descarga velas de 15m y computa las features sobre ese grid; la
misma fila final sirve para los tres modelos (los cierres de vela de
1h/4h coinciden con cierres de 15m).

Las formulas replican T02/T03/T06/T13 (versiones causales) via
src/features/parity_features.py, validadas contra el master causal.

Sigue sin cubrir (NaN, XGB lo tolera):
- cx_* de mcap/dominance (requiere fuente cross-crypto, pendiente)
- ext_/macro_ series intradia (solo broadcast del ultimo valor de cache)
- der_* (funding geo-bloqueado en runners), reg_*, sent_ series
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import binance_spot as bspot
from ..data import binance_futures as bfut
from ..utils.logging_utils import get_logger
from . import calendar as cal_mod
from .parity_features import compute_parity_features

log = get_logger("feature_builder")

# Velas 15m necesarias: vp_7d (672) + warmup EMA200/rolling96 + apertura
# semanal (hasta 14 dias = 1344). 1600 da margen.
MIN_HISTORY_15M = 1600


def _kline_to_ohlcv_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    cols = {
        "open": f"ohlcv_{prefix}_open",
        "high": f"ohlcv_{prefix}_high",
        "low": f"ohlcv_{prefix}_low",
        "close": f"ohlcv_{prefix}_close",
        "volume": f"ohlcv_{prefix}_volume",
        "quote_volume": f"ohlcv_{prefix}_quote_volume",
        "num_trades": f"ohlcv_{prefix}_num_trades",
        "taker_buy_vol": f"ohlcv_{prefix}_taker_buy_vol",
        "taker_buy_qvol": f"ohlcv_{prefix}_taker_buy_qvol",
    }
    df = df.set_index("open_time").sort_index()
    keep = [v for k, v in cols.items() if k in df.columns]
    return df.rename(columns=cols)[keep]


def build_live_features(timeframe: str = "15m",
                        rolling_history: int = MIN_HISTORY_15M,
                        macro_cache: dict | None = None) -> pd.DataFrame:
    """Construye el frame de features sobre el GRID DE 15M.

    El argumento `timeframe` se conserva por compatibilidad pero las
    features siempre se computan en 15m (como en entrenamiento). La ultima
    fila es la vela 15m cerrada mas reciente y es la fila de inferencia
    para cualquier TF cuyo cierre coincida.
    """
    n = max(int(rolling_history), MIN_HISTORY_15M)
    spot_syms = ["BTCUSDT", "ETHUSDT", "ETHBTC", "XRPUSDT", "XRPBTC"]
    spot_prefixes = ["btc", "eth", "ethbtc", "xrp", "xrpbtc"]

    log.info("Fetching 15m klines (%d candles) for %d symbols...",
             n, len(spot_syms))
    spot_dfs: dict[str, pd.DataFrame] = {}
    sources: dict[str, str] = {}
    for sym, pfx in zip(spot_syms, spot_prefixes):
        try:
            kdf, src = bspot.fetch_last_n_closed_paginated(sym, "15m", n)
            sources[sym] = src
            spot_dfs[pfx] = (_kline_to_ohlcv_prefix(kdf, pfx)
                             if not kdf.empty else pd.DataFrame())
        except Exception as e:
            log.warning("Failed spot %s: %s", sym, e)
            sources[sym] = f"error: {e}"
            spot_dfs[pfx] = pd.DataFrame()
    log.info("Spot sources: %s", sources)

    btc = spot_dfs["btc"]
    if btc.empty:
        raise RuntimeError("BTC spot klines empty - cannot build features")
    if btc.index.tz is not None:
        for pfx in spot_dfs:
            if not spot_dfs[pfx].empty:
                spot_dfs[pfx].index = (spot_dfs[pfx].index
                                       .tz_convert("UTC").tz_localize(None))
        btc = spot_dfs["btc"]

    feats = pd.DataFrame(index=btc.index)
    feats = feats.join(pd.concat(
        [df for df in spot_dfs.values() if not df.empty], axis=1), how="left")

    # Futures / mark / index (15m; NaN si geo-bloqueado)
    for name, fn in [("ohlcv_btc_futures_close", bfut.fetch_klines),
                     ("ohlcv_btc_mark_price", bfut.fetch_mark_price_klines),
                     ("ohlcv_btc_index_price", bfut.fetch_index_price_klines)]:
        try:
            fdf = fn("BTCUSDT", "15m", limit=1000).set_index("open_time").sort_index()
            if fdf.index.tz is not None:
                fdf.index = fdf.index.tz_convert("UTC").tz_localize(None)
            feats[name] = fdf["close"].reindex(feats.index)
        except Exception as e:
            log.warning("Failed %s: %s", name, e)
            feats[name] = np.nan
    feats["ohlcv_btc_spot_perp_spread"] = (
        feats["ohlcv_btc_futures_close"] - feats["ohlcv_btc_close"]
    ) / feats["ohlcv_btc_close"]

    # Cross ratios (igual que M01)
    if "ohlcv_ethbtc_close" in feats:
        feats["ohlcv_ethbtc_ratio"] = feats["ohlcv_ethbtc_close"]
        feats["ohlcv_ethbtc_return"] = feats["ohlcv_ethbtc_close"].pct_change()
    if "ohlcv_eth_close" in feats:
        feats["ohlcv_eth_rel_strength"] = (
            feats["ohlcv_eth_close"].pct_change()
            - feats["ohlcv_btc_close"].pct_change())
    if "ohlcv_xrpbtc_close" in feats:
        feats["ohlcv_xrpbtc_ratio"] = feats["ohlcv_xrpbtc_close"]
    if "ohlcv_xrp_close" in feats:
        feats["ohlcv_xrp_rel_strength"] = (
            feats["ohlcv_xrp_close"].pct_change()
            - feats["ohlcv_btc_close"].pct_change())
        feats["ohlcv_xrp_volume_spike"] = (
            feats["ohlcv_xrp_volume"]
            / feats["ohlcv_xrp_volume"].rolling(20, min_periods=5).mean())

    # ======== Features con paridad de entrenamiento (T02/T03/T06/T13) ====
    parity = compute_parity_features(feats)
    feats = pd.concat([feats, parity], axis=1)
    feats = feats.loc[:, ~feats.columns.duplicated(keep="last")]

    # inter_ que dependen de columnas cruzadas
    if "ohlcv_eth_rel_strength" in feats:
        feats["inter_logret_minus_ethrelstr"] = (
            feats["ret_log_return"] - feats["ohlcv_eth_rel_strength"])

    # Calendario
    feats = feats.join(cal_mod.compute_calendar(feats.index))

    # Macro cache (broadcast del ultimo valor)
    if macro_cache:
        for k, v in macro_cache.items():
            feats[k] = v
        if "macro_ndx_return_1d" in feats:
            feats["inter_logret_minus_ndx"] = (
                feats["ret_log_return"] - feats["macro_ndx_return_1d"])

    log.info("Built %d features over %d candles (grid 15m)",
             feats.shape[1], len(feats))
    return feats
