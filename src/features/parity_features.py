"""Features con PARIDAD respecto al entrenamiento (T02/T03/T06/T13).

Todas las formulas replican las de los notebooks de transformacion del
proyecto de investigacion, en sus versiones CAUSALES (post fix de leakage
2026-06-11):
  - chikou = close de hace 26 velas 1h (no el futuro)
  - nube Ichimoku = la visible (calculada hace 26 velas 1h)
  - swings = pivote en t-10 confirmado con la ventana [t-20, t]
  - el resto de indicadores: mismas formulas que T03 (pandas_ta equivalente)

IMPORTANTE: en entrenamiento las features de mercado de los 3 modelos
(15m/1h/4h) salen del GRID DE 15 MINUTOS (master). Este modulo se computa
SIEMPRE sobre velas de 15m y la misma fila sirve para los tres modelos.

Entrada: DataFrame indexado por open_time (UTC naive) con columnas
ohlcv_btc_{open,high,low,close,volume,quote_volume,taker_buy_vol}.
Minimo recomendado: 1500 velas (15.6 dias) para convergencia de EMA200,
vp_7d (672) y apertura semanal.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VP_WINDOWS_H = [4, 12, 24, 72, 168]
VP_NBINS = 100
VP_VA = 0.7
LAGS = [1, 2, 3, 4, 8, 16, 32]
WINS = [4, 12, 24, 96]


# ----------------------------------------------------------------- helpers
def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rma(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def _rsi(c, n):
    d = c.diff()
    up = _rma(d.clip(lower=0), n)
    dn = _rma((-d).clip(lower=0), n)
    return 100 * up / (up + dn)


def _atr(h, l, c, n):
    tr = pd.concat([h - l, (h - c.shift(1)).abs(),
                    (l - c.shift(1)).abs()], axis=1).max(axis=1)
    return _rma(tr, n)


def _vpvr_last(typical, vol, n_candles):
    """Volume profile sobre la ventana [-n_candles:] (pasado, excluye la
    vela actual igual que T06: ventana [i-n, i))."""
    p = typical.iloc[-n_candles - 1:-1].values
    vv = vol.iloc[-n_candles - 1:-1].values
    if len(p) < n_candles or np.nanmax(p) == np.nanmin(p):
        return None
    bins = np.linspace(np.nanmin(p), np.nanmax(p), VP_NBINS + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, VP_NBINS - 1)
    prof = np.zeros(VP_NBINS)
    np.add.at(prof, idx, np.nan_to_num(vv))
    if prof.sum() == 0:
        return None
    cent = (bins[:-1] + bins[1:]) / 2
    poc = cent[prof.argmax()]
    order = np.argsort(prof)[::-1]
    target = prof.sum() * VP_VA
    cum, va_idx = 0.0, []
    for i in order:
        cum += prof[i]
        va_idx.append(i)
        if cum >= target:
            break
    va_idx = sorted(va_idx)
    hva_lower, hva_upper = cent[va_idx[0]], cent[va_idx[-1]]
    mid = (np.nanmax(p) + np.nanmin(p)) / 2
    return {
        "poc": poc, "hva_lower": hva_lower, "hva_upper": hva_upper,
        "va_width": (hva_upper - hva_lower) / poc,
        "skew": (poc - mid) / (np.nanmax(p) - np.nanmin(p)),
        "hi_node": cent[prof.argmax()], "lo_node": cent[prof.argmin()],
    }


# ----------------------------------------------------------------- main
def compute_parity_features(m: pd.DataFrame) -> pd.DataFrame:
    """Devuelve DataFrame (mismo indice 15m) con features ret_/ta_/vp_/
    lag_/roll_/inter_ identicas a las de entrenamiento (causales).
    Las vp_ solo se calculan en la ULTIMA fila (es la unica que usa el bot).
    """
    o = m["ohlcv_btc_open"].astype(float)
    h = m["ohlcv_btc_high"].astype(float)
    l = m["ohlcv_btc_low"].astype(float)
    c = m["ohlcv_btc_close"].astype(float)
    v = m["ohlcv_btc_volume"].astype(float)
    qv = m.get("ohlcv_btc_quote_volume",
               pd.Series(np.nan, index=m.index)).astype(float)
    tbv = m.get("ohlcv_btc_taker_buy_vol",
                pd.Series(np.nan, index=m.index)).astype(float)
    f = pd.DataFrame(index=m.index)

    # ============================================== ret_ (T02)
    f["ret_log_return"] = np.log(c).diff()
    f["ret_arith_return"] = c.pct_change(fill_method=None)
    for n in [3, 6, 12, 24, 96]:
        f[f"ret_return_{n}"] = np.log(c).diff(n)
    f["ret_candle_body"] = (c - o) / o
    f["ret_upper_shadow"] = (h - np.maximum(o, c)) / c
    f["ret_lower_shadow"] = (np.minimum(o, c) - l) / c
    f["ret_range_pct"] = (h - l) / c
    f["ret_close_in_range"] = (c - l) / (h - l).replace(0, np.nan)
    f["ret_body_to_range"] = (c - o).abs() / (h - l).replace(0, np.nan)
    f["ret_green"] = (c > o).astype("int8")
    f["ret_red"] = (c < o).astype("int8")
    typical = (h + l + c) / 3
    # vwap aprox typical (en training venia de aggTrades desde 2023; aprox)
    f["ret_vwap"] = typical
    f["ret_close_to_vwap"] = (c - typical) / c
    f["ret_vwap_slope"] = typical.diff(4)
    f["ret_log_volume"] = np.log1p(v)
    f["ret_log_quote_volume"] = np.log1p(qv)
    f["ret_volume_return"] = v.pct_change(fill_method=None)
    f["ret_vol_adj_return"] = f["ret_log_return"] / \
        f["ret_log_return"].rolling(12).std()

    # ============================================== ta_ momentum (T03)
    f["ta_rsi_14"] = _rsi(c, 14)
    f["ta_rsi_7"] = _rsi(c, 7)
    rsi14 = f["ta_rsi_14"]
    sr = (rsi14 - rsi14.rolling(14).min()) / \
        (rsi14.rolling(14).max() - rsi14.rolling(14).min()).replace(0, np.nan) * 100
    f["ta_stoch_rsi_k"] = sr.rolling(3).mean()
    f["ta_stoch_rsi_d"] = f["ta_stoch_rsi_k"].rolling(3).mean()
    macd_line = _ema(c, 12) - _ema(c, 26)
    macd_sig = _ema(macd_line, 9)
    f["ta_macd_line"] = macd_line
    f["ta_macd_hist"] = macd_line - macd_sig
    f["ta_macd_signal"] = macd_sig
    f["ta_roc_12"] = 100 * (c / c.shift(12) - 1)
    hh14, ll14 = h.rolling(14).max(), l.rolling(14).min()
    f["ta_williams_r_14"] = -100 * (hh14 - c) / (hh14 - ll14).replace(0, np.nan)
    tp20 = typical
    sma_tp = tp20.rolling(20).mean()
    mad = tp20.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    f["ta_cci_20"] = (tp20 - sma_tp) / (0.015 * mad.replace(0, np.nan))
    f["ta_momentum_10"] = c.diff(10)
    mdiff = c.diff()
    f["ta_tsi"] = 100 * _ema(_ema(mdiff, 25), 13) / \
        _ema(_ema(mdiff.abs(), 25), 13)

    # trend: EMAs/SMAs
    for n in [8, 21, 50, 100, 200]:
        f[f"ta_ema_{n}"] = _ema(c, n)
    for n in [20, 50, 200]:
        f[f"ta_sma_{n}"] = c.rolling(n).mean()
    f["ta_ema_cross_8_21"] = (f["ta_ema_8"] - f["ta_ema_21"]) / c
    f["ta_ema_cross_21_50"] = (f["ta_ema_21"] - f["ta_ema_50"]) / c
    f["ta_ema_cross_50_200"] = (f["ta_ema_50"] - f["ta_ema_200"]) / c
    f["ta_close_above_ema_21"] = (c > f["ta_ema_21"]).astype("int8")
    f["ta_close_above_ema_200"] = (c > f["ta_ema_200"]).astype("int8")

    # ADX (Wilder)
    up_m = h.diff()
    dn_m = -l.diff()
    plus_dm = pd.Series(np.where((up_m > dn_m) & (up_m > 0), up_m, 0.0), index=c.index)
    minus_dm = pd.Series(np.where((dn_m > up_m) & (dn_m > 0), dn_m, 0.0), index=c.index)
    atr14 = _atr(h, l, c, 14)
    plus_di = 100 * _rma(plus_dm, 14) / atr14
    minus_di = 100 * _rma(minus_dm, 14) / atr14
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    f["ta_adx_14"] = _rma(dx, 14)
    f["ta_plus_di_14"] = plus_di
    f["ta_minus_di_14"] = minus_di

    # Aroon 25
    n_ar = 25
    f["ta_aroon_up_25"] = h.rolling(n_ar + 1).apply(
        lambda x: 100 * x.argmax() / n_ar, raw=True)
    f["ta_aroon_down_25"] = l.rolling(n_ar + 1).apply(
        lambda x: 100 * x.argmin() / n_ar, raw=True)
    f["ta_aroon_osc_25"] = f["ta_aroon_up_25"] - f["ta_aroon_down_25"]

    # Supertrend (10, 3)
    atr10 = _atr(h, l, c, 10)
    hl2 = (h + l) / 2
    ub = (hl2 + 3.0 * atr10).values
    lb = (hl2 - 3.0 * atr10).values
    cv = c.values
    n_ = len(cv)
    st_line = np.full(n_, np.nan)
    st_dir = np.full(n_, 1.0)
    fub, flb = ub.copy(), lb.copy()
    for i in range(1, n_):
        if np.isnan(atr10.iloc[i]):
            continue
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or cv[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or cv[i - 1] < flb[i - 1]) else flb[i - 1]
        if st_dir[i - 1] == 1.0:
            st_dir[i] = -1.0 if cv[i] < flb[i] else 1.0
        else:
            st_dir[i] = 1.0 if cv[i] > fub[i] else -1.0
        st_line[i] = flb[i] if st_dir[i] == 1.0 else fub[i]
    f["ta_supertrend_dir"] = st_dir
    f["ta_supertrend_dist"] = (c - st_line) / c

    # ============================================== ta_ volatility
    f["ta_atr_14"] = atr14
    f["ta_atr_pct"] = atr14 / c
    logr = np.log(c).diff()
    for n in [12, 24, 96]:
        f[f"ta_rolling_vol_{n}"] = logr.rolling(n).std()
    bb_mid = c.rolling(20).mean()
    bb_std = c.rolling(20).std()
    f["ta_bb_lower"] = bb_mid - 2 * bb_std
    f["ta_bb_middle"] = bb_mid
    f["ta_bb_upper"] = bb_mid + 2 * bb_std
    f["ta_bb_width"] = (f["ta_bb_upper"] - f["ta_bb_lower"]) / bb_mid
    f["ta_bb_pctb"] = (c - f["ta_bb_lower"]) / \
        (f["ta_bb_upper"] - f["ta_bb_lower"]).replace(0, np.nan)
    f["ta_bb_squeeze"] = (f["ta_bb_width"] <
                          f["ta_bb_width"].rolling(96).quantile(0.20)).astype("int8")
    ema20 = _ema(c, 20)
    atr20 = _atr(h, l, c, 20)
    f["ta_keltner_upper"] = ema20 + 2 * atr20
    f["ta_keltner_lower"] = ema20 - 2 * atr20
    f["ta_keltner_width"] = (f["ta_keltner_upper"] - f["ta_keltner_lower"]) / ema20
    f["ta_donchian_high_20"] = h.rolling(20).max()
    f["ta_donchian_low_20"] = l.rolling(20).min()
    f["ta_donchian_width_20"] = (f["ta_donchian_high_20"] - f["ta_donchian_low_20"]) / c
    f["ta_realized_vol_1h"] = logr.rolling(4).std() * np.sqrt(4)
    f["ta_realized_vol_4h"] = logr.rolling(16).std() * np.sqrt(16)
    f["ta_realized_vol_1d"] = logr.rolling(96).std() * np.sqrt(96)

    # ============================================== ta_ volume
    obv = (np.sign(c.diff()).fillna(0) * v).cumsum()
    f["ta_obv"] = obv
    f["ta_obv_slope_5"] = obv.diff(5)
    # MFI 14
    tp_ = typical
    mf = tp_ * v
    pos_mf = mf.where(tp_ > tp_.shift(1), 0.0).rolling(14).sum()
    neg_mf = mf.where(tp_ < tp_.shift(1), 0.0).rolling(14).sum()
    f["ta_mfi_14"] = 100 * pos_mf / (pos_mf + neg_mf).replace(0, np.nan)
    clv = ((2 * c - h - l) / (h - l).replace(0, np.nan)).fillna(0)
    f["ta_ad_line"] = (clv * v).cumsum()
    f["ta_chaikin_mf_20"] = (clv * v).rolling(20).sum() / v.rolling(20).sum()
    f["ta_vwap_deviation"] = (c - (tp_ * v).rolling(20).sum()
                              / v.rolling(20).sum()) / c
    f["ta_vol_sma_ratio"] = v / v.rolling(20).mean()
    f["ta_vol_zscore"] = (v - v.rolling(96).mean()) / v.rolling(96).std()
    f["ta_vol_percentile_96"] = v.rolling(96).rank(pct=True)
    f["ta_taker_buy_ratio"] = tbv / v.replace(0, np.nan)
    f["ta_taker_sell_ratio"] = 1 - f["ta_taker_buy_ratio"]
    f["ta_buy_sell_delta"] = (2 * tbv - v) / v.replace(0, np.nan)

    # ============================================== ta_ fib + niveles
    f["ta_swing_high_96"] = h.rolling(96).max()
    f["ta_swing_low_96"] = l.rolling(96).min()
    rng = (f["ta_swing_high_96"] - f["ta_swing_low_96"]).replace(0, np.nan)
    for lvl, name in [(0.236, "236"), (0.382, "382"), (0.500, "500"),
                      (0.618, "618"), (0.786, "786")]:
        f[f"ta_fib_{name}_dist"] = (c - (f["ta_swing_low_96"] + lvl * rng)) / c
    daily = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}) \
        .resample("1D").agg({"open": "first", "high": "max",
                             "low": "min", "close": "last"}).shift(1)
    d15 = daily.reindex(c.index, method="ffill")
    f["ta_dist_prev_high_1d"] = (c - d15["high"]) / c
    f["ta_dist_prev_low_1d"] = (c - d15["low"]) / c
    f["ta_dist_prev_close_1d"] = (c - d15["close"]) / c
    weekly_open = o.resample("W").first().reindex(c.index, method="ffill")
    monthly_open = o.resample("ME").first().reindex(c.index, method="ffill")
    f["ta_dist_weekly_open"] = (c - weekly_open) / c
    f["ta_dist_monthly_open"] = (c - monthly_open) / c

    # ============================================== Ichimoku CAUSAL (fix)
    h1 = h.resample("1h").max()
    l1 = l.resample("1h").min()
    c1 = c.resample("1h").last()
    tenkan = (h1.rolling(9).max() + l1.rolling(9).min()) / 2
    kijun = (h1.rolling(26).max() + l1.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h1.rolling(52).max() + l1.rolling(52).min()) / 2).shift(26)
    chikou = c1.shift(26)
    to15 = lambda s: s.reindex(c.index, method="ffill")
    f["ta_tenkan_sen"] = to15(tenkan)
    f["ta_kijun_sen"] = to15(kijun)
    f["ta_senkou_a"] = to15(senkou_a)
    f["ta_senkou_b"] = to15(senkou_b)
    f["ta_chikou_span"] = to15(chikou)
    f["ta_chikou_dist"] = (c - f["ta_chikou_span"]) / c
    f["ta_cloud_width"] = (f["ta_senkou_a"] - f["ta_senkou_b"]).abs() / c
    f["ta_price_above_cloud"] = (
        c > pd.concat([f["ta_senkou_a"], f["ta_senkou_b"]], axis=1).max(axis=1)
    ).astype("int8")

    # market structure + swings CAUSALES (fix)
    f["ta_higher_high"] = (h > h.shift(1)).astype("int8")
    f["ta_lower_low"] = (l < l.shift(1)).astype("int8")
    f["ta_bos_up"] = (c > h.rolling(20).max().shift(1)).astype("int8")
    f["ta_bos_down"] = (c < l.rolling(20).min().shift(1)).astype("int8")
    piv_h = h.shift(10).where(h.shift(10) >= h.rolling(21).max() - 1e-9)
    piv_l = l.shift(10).where(l.shift(10) <= l.rolling(21).min() + 1e-9)
    f["ta_dist_last_swing_high"] = (c - piv_h.ffill()) / c
    f["ta_dist_last_swing_low"] = (c - piv_l.ffill()) / c
    round_lv = (c / 1000).round() * 1000
    f["ta_local_support_dist"] = (c - round_lv.where(round_lv < c).ffill()) / c
    f["ta_local_resistance_dist"] = (round_lv.where(round_lv > c).ffill() - c) / c
    xs = np.arange(20)
    f["ta_trend_slope_20"] = c.rolling(20).apply(
        lambda y: np.polyfit(xs, y, 1)[0], raw=True) / c
    xs50 = np.arange(50)
    f["ta_trend_slope_50"] = c.rolling(50).apply(
        lambda y: np.polyfit(xs50, y, 1)[0], raw=True) / c

    # ============================================== vp_ (T06) - ultima fila
    for hrs in VP_WINDOWS_H:
        n_c = hrs * 4
        label = f"{hrs}h" if hrs < 24 else f"{hrs // 24}d"
        cols = [f"vp_poc_price_{label}", f"vp_poc_dist_{label}",
                f"vp_skew_{label}", f"vp_hva_upper_dist_{label}",
                f"vp_hva_lower_dist_{label}", f"vp_value_area_width_{label}",
                f"vp_inside_value_area_{label}",
                f"vp_dist_high_vol_node_{label}",
                f"vp_dist_low_vol_node_{label}"]
        for col in cols:
            f[col] = np.nan
        if len(m) > n_c + 1:
            res = _vpvr_last(typical, v.fillna(0), n_c)
            if res:
                cur = c.iloc[-1]
                f.iloc[-1, f.columns.get_loc(f"vp_poc_price_{label}")] = res["poc"]
                f.iloc[-1, f.columns.get_loc(f"vp_poc_dist_{label}")] = (cur - res["poc"]) / cur
                f.iloc[-1, f.columns.get_loc(f"vp_skew_{label}")] = res["skew"]
                f.iloc[-1, f.columns.get_loc(f"vp_hva_upper_dist_{label}")] = \
                    (res["hva_upper"] - cur) / cur
                f.iloc[-1, f.columns.get_loc(f"vp_hva_lower_dist_{label}")] = \
                    (cur - res["hva_lower"]) / cur
                f.iloc[-1, f.columns.get_loc(f"vp_value_area_width_{label}")] = res["va_width"]
                f.iloc[-1, f.columns.get_loc(f"vp_inside_value_area_{label}")] = \
                    1 if res["hva_lower"] <= cur <= res["hva_upper"] else 0
                f.iloc[-1, f.columns.get_loc(f"vp_dist_high_vol_node_{label}")] = \
                    (cur - res["hi_node"]) / cur
                f.iloc[-1, f.columns.get_loc(f"vp_dist_low_vol_node_{label}")] = \
                    (cur - res["lo_node"]) / cur

    # ============================================== lag_/roll_ (T13)
    series = {"ret_log_return": f["ret_log_return"],
              "ret_range_pct": f["ret_range_pct"],
              "ohlcv_btc_volume": v,
              "ta_rsi_14": f["ta_rsi_14"],
              "ta_taker_buy_ratio": f["ta_taker_buy_ratio"]}
    lr_parts = {}
    for name, s in series.items():
        for lag in LAGS:
            lr_parts[f"lag_{name}_{lag}"] = s.shift(lag)
        for w in WINS:
            r = s.rolling(w)
            lr_parts[f"roll_{name}_mean_{w}"] = r.mean()
            lr_parts[f"roll_{name}_std_{w}"] = r.std()
            lr_parts[f"roll_{name}_min_{w}"] = r.min()
            lr_parts[f"roll_{name}_max_{w}"] = r.max()
            lr_parts[f"roll_{name}_zscore_{w}"] = (s - r.mean()) / r.std()
            lr_parts[f"roll_{name}_pctile_{w}"] = r.rank(pct=True)
    f = pd.concat([f, pd.DataFrame(lr_parts, index=f.index)], axis=1)

    # inter_ (las computables sin macro/funding)
    f["inter_logret_x_volume"] = f["ret_log_return"] * v

    # Features SIN paridad verificable -> NaN (XGB las trata como missing).
    # - ta_obv / ta_ad_line: acumulativas desde 2017, irreproducibles en
    #   ventana live (ta_obv_slope_5 SI es valida: el diff cancela el offset)
    # - ta_cci_20: el master usa una escala ~1600x (bug de pandas_ta)
    # - ta_plus/minus_di_14: escala pandas_ta ~1.3x (ta_adx_14 si tiene
    #   paridad porque el ratio cancela la escala)
    # - ta_dist_monthly_open: requiere ~60 dias de historia
    for col in ["ta_obv", "ta_ad_line", "ta_cci_20", "ta_plus_di_14",
                "ta_minus_di_14", "ta_dist_monthly_open"]:
        f[col] = np.nan

    return f
