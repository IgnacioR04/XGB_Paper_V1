"""Derivadas de series externas: lag/roll (T13), interacciones, regimen (T11).

El builder live cubre las features de precio, pero las lag/roll de funding/VIX/NDX
y los flags de regimen no se computaban (quedaban NaN -> el modelo decidia con
datos incompletos). Este modulo replica T13 y T11 sobre las series base externas,
reindexadas al grid 15m.

Las series base (der_funding_rate 8h, macro_vix diario, NDX diario) llegan como
historia corta en el cache (`series`), no como escalar, porque lag/roll necesitan
la variacion temporal. Las ventanas T13 son <=24h (WINS<=96), asi que ~16 dias de
grid 15m bastan; las percentiles a 1 ano (reg_*_pctile_1y) quedan NaN (necesitan
1 ano de historia) y el modelo las tolera.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..utils.logging_utils import get_logger

log = get_logger("derivatives")

# Identicos a T13 / parity_features
LAGS = [1, 2, 3, 4, 8, 16, 32]
WINS = [4, 12, 24, 96]


def _series_to_grid(points: list, grid: pd.DatetimeIndex) -> pd.Series | None:
    """Convierte [(iso, valor), ...] en una Serie reindexada al grid 15m (ffill)."""
    if not points:
        return None
    try:
        ts = pd.to_datetime([p[0] for p in points], utc=True)
        vals = pd.to_numeric([p[1] for p in points], errors="coerce")
    except Exception as e:
        log.warning("series parse failed: %s", e)
        return None
    s = pd.Series(vals, index=ts).sort_index()
    s = s[~s.index.duplicated(keep="last")]
    grid_utc = grid.tz_localize("UTC") if grid.tz is None else grid.tz_convert("UTC")
    out = s.reindex(grid_utc, method="ffill")
    out.index = grid
    return out


def _add_lag_roll(out: pd.DataFrame, name: str, s: pd.Series) -> None:
    """Anade lag_{name}_{lag} y roll_{name}_{stat}_{w} (patron exacto T13)."""
    for lag in LAGS:
        out[f"lag_{name}_{lag}"] = s.shift(lag)
    for w in WINS:
        r = s.rolling(w)
        mean = r.mean()
        std = r.std()
        out[f"roll_{name}_mean_{w}"] = mean
        out[f"roll_{name}_std_{w}"] = std
        out[f"roll_{name}_min_{w}"] = r.min()
        out[f"roll_{name}_max_{w}"] = r.max()
        out[f"roll_{name}_zscore_{w}"] = (s - mean) / std
        out[f"roll_{name}_pctile_{w}"] = s.rolling(w).rank(pct=True)


def compute_derivatives(feats: pd.DataFrame, cache: dict) -> pd.DataFrame:
    """Devuelve un DataFrame (mismo indice que feats) con lag/roll/inter/reg.

    `feats` debe traer ya las features de precio (ohlcv_, ret_, ta_).
    `cache` es el external_features_cache (con `series` y `values`).
    """
    grid = feats.index
    out = pd.DataFrame(index=grid)
    series = cache.get("series", {})
    values = cache.get("values", {})

    # ---- Series base reindexadas al grid 15m ----
    funding = _series_to_grid(series.get("der_funding_rate", []), grid)
    vix = _series_to_grid(series.get("macro_vix", []), grid)
    ndx_price = _series_to_grid(series.get("macro_ndx_price", []), grid)
    spx_price = _series_to_grid(series.get("macro_spx_price", []), grid)

    # Niveles (coherentes con M05/M09; sustituyen al escalar broadcast)
    if funding is not None:
        out["der_funding_rate"] = funding
    if vix is not None:
        out["macro_vix"] = vix
        out["macro_vix_change_1d"] = vix.diff(96)
    ndx_ret = None
    if ndx_price is not None:
        ndx_ret = ndx_price.pct_change(96)        # T09: 24h return en grid 15m
        out["macro_ndx_return_1d"] = ndx_ret
    spx_ret = None
    if spx_price is not None:
        spx_ret = spx_price.pct_change(96)
        out["macro_spx_return_1d"] = spx_ret

    # ---- lag/roll T13 sobre las series base ----
    if funding is not None:
        _add_lag_roll(out, "der_funding_rate", funding)
    if vix is not None:
        _add_lag_roll(out, "macro_vix", vix)
    if ndx_ret is not None:
        _add_lag_roll(out, "macro_ndx_return_1d", ndx_ret)

    # ---- Interacciones (T13) ----
    logret = feats["ret_log_return"] if "ret_log_return" in feats else None
    if logret is not None and funding is not None:
        out["inter_logret_x_funding"] = logret * funding
    if logret is not None and ndx_ret is not None:
        out["inter_logret_minus_ndx"] = logret - ndx_ret
    if logret is not None and "ohlcv_eth_rel_strength" in feats:
        out["inter_logret_minus_ethrelstr"] = logret - feats["ohlcv_eth_rel_strength"]

    # ---- Regimen (T11) ----
    _add_regime(out, feats, cache, vix, spx_ret, funding, series)
    return out


def _add_regime(out, feats, cache, vix, spx_ret, funding, series) -> None:
    grid = feats.index
    values = cache.get("values", {})

    def col(name):
        return feats[name] if name in feats.columns else pd.Series(np.nan, index=grid)

    c = col("ohlcv_btc_close")
    # Trend (T11): EMA diaria 50/200 y 4h(=ema50/200 en grid intradia)
    ema50 = c.ewm(span=50).mean()
    ema200 = c.ewm(span=200).mean()
    daily = c.resample("1D").last()
    ema_d50 = daily.ewm(span=50).mean().reindex(grid, method="ffill")
    ema_d200 = daily.ewm(span=200).mean().reindex(grid, method="ffill")
    out["reg_trend_1d"] = np.sign(ema_d50 - ema_d200)
    out["reg_trend_4h"] = np.sign(ema50 - ema200)
    out["reg_bull_score"] = ((out["reg_trend_1d"] == 1).astype("int8")
                             + (out["reg_trend_4h"] == 1).astype("int8"))
    out["reg_bear_score"] = ((out["reg_trend_1d"] == -1).astype("int8")
                             + (out["reg_trend_4h"] == -1).astype("int8"))
    out["reg_trend_strength"] = col("ta_adx_14")

    # Volatilidad (las pctile_1y necesitan 1 ano -> NaN; el resto computable)
    atr_pct = col("ta_atr_pct")
    pct_1y = atr_pct.rolling(96 * 365).rank(pct=True)
    out["reg_atr_pctile_1y"] = pct_1y
    out["reg_realized_vol_pctile_1y"] = col("ta_realized_vol_1d").rolling(96 * 365).rank(pct=True)
    out["reg_vol_low"] = (pct_1y < 0.33).astype("float32")
    out["reg_vol_medium"] = ((pct_1y >= 0.33) & (pct_1y < 0.66)).astype("float32")
    out["reg_vol_high"] = (pct_1y >= 0.66).astype("float32")
    out["reg_vol_expansion"] = (atr_pct > atr_pct.rolling(96).quantile(0.80)).astype("float32")
    out["reg_vol_compression"] = (atr_pct < atr_pct.rolling(96).quantile(0.20)).astype("float32")

    # Liquidez
    out["reg_spread_regime"] = col("ta_bb_width").rolling(96).rank(pct=True)
    out["reg_volume_regime"] = col("ohlcv_btc_volume").rolling(96).rank(pct=True)
    out["reg_liq_high"] = (out["reg_volume_regime"] > 0.7).astype("float32")
    out["reg_liq_low"] = (out["reg_volume_regime"] < 0.3).astype("float32")

    # Macro regime (vix_zscore_1y viene del cache como escalar; spx_ret de la serie)
    vix_z = float(values.get("macro_vix_zscore_1y", np.nan))
    spx_r5 = spx_ret.rolling(1).mean() if spx_ret is not None else pd.Series(np.nan, index=grid)
    # T11 usa spx_return_5d; aproximamos con el retorno disponible
    spx5 = spx_ret.pct_change(96 * 5) if spx_ret is not None else None
    spx_pos = (spx_ret > 0) if spx_ret is not None else pd.Series(False, index=grid)
    out["reg_risk_on"] = (((vix_z < 0) if vix_z == vix_z else False) & spx_pos).astype("float32")
    out["reg_risk_off"] = (((vix_z > 1) if vix_z == vix_z else False) & (~spx_pos)).astype("float32")
    out["reg_equities_up"] = spx_pos.astype("float32")

    # Derivatives regime: funding extreme (zscore sobre la serie 8h nativa ~30d)
    fr_pts = series.get("der_funding_rate", [])
    fr_extreme = np.nan
    if fr_pts:
        fr = pd.to_numeric([p[1] for p in fr_pts], errors="coerce")
        fr = pd.Series(fr).dropna()
        if len(fr) > 5 and fr.std() > 0:
            z_last = (fr.iloc[-1] - fr.mean()) / fr.std()
            fr_extreme = 1.0 if abs(z_last) > 2 else 0.0
    out["reg_funding_extreme"] = fr_extreme
