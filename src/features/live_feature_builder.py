"""Constructor de features en vivo.

Para un timeframe y un timestamp objetivo (= ultimo close confirmado),
descarga datos de Binance, calcula returns/TA/lags/rolling/calendar y
combina con cache externa (macro/onchain/sentiment/cross-crypto/funding)
para producir UNA fila de features alineada al feature_schema.

Lo que NO genera (queda NaN, XGB lo tolera):
- vp_*  (Volume Profile - requiere aggTrades)
- reg_*  (regime flags - requeririan portar M11)
- Algunas ta_* avanzadas (Ichimoku, Fibonacci, ADX completo)

Esto es una version honesta de MVP. Cubre las ~400 features mas importantes
(OHLCV + ret + ta basicos + lags + rolling + calendar + macro/onchain/sentiment).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..data import binance_spot as bspot
from ..data import binance_futures as bfut
from ..utils.logging_utils import get_logger
from . import calendar as cal_mod
from . import lags_rolling as lr_mod
from . import returns as ret_mod
from . import technicals as ta_mod

log = get_logger("feature_builder")


def _kline_to_ohlcv_prefix(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
    """Renombra columnas de un kline DataFrame al esquema ohlcv_<prefix>_."""
    cols = {
        "open": f"ohlcv_{prefix}_open",
        "high": f"ohlcv_{prefix}_high",
        "low":  f"ohlcv_{prefix}_low",
        "close": f"ohlcv_{prefix}_close",
        "volume": f"ohlcv_{prefix}_volume",
        "quote_volume": f"ohlcv_{prefix}_quote_volume",
        "num_trades": f"ohlcv_{prefix}_num_trades",
        "taker_buy_vol": f"ohlcv_{prefix}_taker_buy_vol",
        "taker_buy_qvol": f"ohlcv_{prefix}_taker_buy_qvol",
    }
    df = df.set_index("open_time").sort_index()
    return df.rename(columns=cols)[[v for v in cols.values()]]


def build_live_features(timeframe: str,
                        rolling_history: int = 400,
                        macro_cache: dict | None = None) -> pd.DataFrame:
    """Construye un DataFrame de features. Devuelve TODAS las velas en historial;
    la ultima fila es la usada para inferencia.

    Args:
        timeframe: "15m", "1h" o "4h"
        rolling_history: cuantas velas pedir a Binance para calcular indicadores
        macro_cache: dict con valores externos por nombre {"sent_fear_greed": x,
                      "macro_spx_close": y, "oc_hashrate": z, ...}.
                     Estos valores se rellenan en la fila final (broadcasted).

    Returns:
        DataFrame indexed by candle open_time UTC. La ultima fila tiene los
        valores mas recientes (vela cerrada).
    """
    spot_syms = ["BTCUSDT", "ETHUSDT", "ETHBTC", "XRPUSDT", "XRPBTC"]
    spot_prefixes = ["btc", "eth", "ethbtc", "xrp", "xrpbtc"]

    log.info("Fetching spot klines (%s) for %d symbols...", timeframe, len(spot_syms))
    spot_dfs: dict[str, pd.DataFrame] = {}
    for sym, pfx in zip(spot_syms, spot_prefixes):
        try:
            kdf = bspot.fetch_last_n_closed(sym, timeframe, rolling_history)
            spot_dfs[pfx] = _kline_to_ohlcv_prefix(kdf, pfx)
        except Exception as e:
            log.warning("Failed spot %s %s: %s", sym, timeframe, e)
            spot_dfs[pfx] = pd.DataFrame()

    # base = BTC spot (es el indice maestro)
    btc = spot_dfs["btc"]
    if btc.empty:
        raise RuntimeError("BTC spot klines empty - cannot build features")

    feats = pd.DataFrame(index=btc.index)
    # Concatenar todos los OHLCV con prefijos
    ohlcv_parts = [df for df in spot_dfs.values() if not df.empty]
    feats = feats.join(pd.concat(ohlcv_parts, axis=1), how="left")

    # Futures
    log.info("Fetching futures klines (%s)...", timeframe)
    try:
        fut = bfut.fetch_klines("BTCUSDT", timeframe, limit=rolling_history)
        fut = fut.set_index("open_time").sort_index()
        feats["ohlcv_btc_futures_close"] = fut["close"].reindex(feats.index)
    except Exception as e:
        log.warning("Failed futures klines: %s", e)
        feats["ohlcv_btc_futures_close"] = np.nan

    try:
        mark = bfut.fetch_mark_price_klines("BTCUSDT", timeframe, limit=rolling_history)
        mark = mark.set_index("open_time").sort_index()
        feats["ohlcv_btc_mark_price"] = mark["close"].reindex(feats.index)
    except Exception as e:
        log.warning("Failed mark price: %s", e)
        feats["ohlcv_btc_mark_price"] = np.nan

    try:
        idx_df = bfut.fetch_index_price_klines("BTCUSDT", timeframe, limit=rolling_history)
        idx_df = idx_df.set_index("open_time").sort_index()
        feats["ohlcv_btc_index_price"] = idx_df["close"].reindex(feats.index)
    except Exception as e:
        log.warning("Failed index price: %s", e)
        feats["ohlcv_btc_index_price"] = np.nan

    # Spread spot-perp
    feats["ohlcv_btc_spot_perp_spread"] = (
        feats["ohlcv_btc_futures_close"] - feats["ohlcv_btc_close"]
    ) / feats["ohlcv_btc_close"]

    # Cross ratios
    if "ohlcv_ethbtc_close" in feats.columns:
        feats["ohlcv_ethbtc_ratio"] = feats["ohlcv_ethbtc_close"]
        feats["ohlcv_ethbtc_return"] = feats["ohlcv_ethbtc_close"].pct_change()
    if "ohlcv_eth_close" in feats.columns and "ohlcv_btc_close" in feats.columns:
        feats["ohlcv_eth_rel_strength"] = (
            feats["ohlcv_eth_close"].pct_change() - feats["ohlcv_btc_close"].pct_change()
        )
    if "ohlcv_xrpbtc_close" in feats.columns:
        feats["ohlcv_xrpbtc_ratio"] = feats["ohlcv_xrpbtc_close"]
    if "ohlcv_xrp_close" in feats.columns and "ohlcv_btc_close" in feats.columns:
        feats["ohlcv_xrp_rel_strength"] = (
            feats["ohlcv_xrp_close"].pct_change() - feats["ohlcv_btc_close"].pct_change()
        )
        feats["ohlcv_xrp_volume_spike"] = (
            feats.get("ohlcv_xrp_volume", pd.Series(np.nan, index=feats.index))
            / feats.get("ohlcv_xrp_volume", pd.Series(np.nan, index=feats.index))
                .rolling(20, min_periods=5).mean()
        )

    # Returns / TA
    btc_ohlc = pd.DataFrame({
        "open": feats["ohlcv_btc_open"],
        "high": feats["ohlcv_btc_high"],
        "low": feats["ohlcv_btc_low"],
        "close": feats["ohlcv_btc_close"],
        "volume": feats["ohlcv_btc_volume"],
    })
    feats = feats.join(ret_mod.compute_returns(btc_ohlc))
    feats = feats.join(ta_mod.compute_technicals(btc_ohlc))

    # Lags + Rolling
    feats = feats.join(lr_mod.compute_lags_rolling(feats))

    # Calendar
    feats = feats.join(cal_mod.compute_calendar(feats.index))

    # Macro cache (broadcast del ultimo valor a todas las filas; el modelo
    # ve la ultima fila como input)
    if macro_cache:
        for k, v in macro_cache.items():
            feats[k] = v

    log.info("Built %d features over %d candles", feats.shape[1], len(feats))
    return feats
