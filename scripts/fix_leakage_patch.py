# -*- coding: utf-8 -*-
"""Parche de leakage: recalcula CAUSALMENTE las columnas contaminadas y
escribe datasets NUEVOS *_CAUSAL.parquet SIN tocar los originales.
El usuario decide promocionarlos (renombrar) tras revisar la verificacion.

Columnas afectadas y fix:

M03 (Ichimoku + swings, ~37% del gain del modelo 15m):
  - ta_chikou_span/ta_chikou_dist: pandas_ta usaba close.shift(-26) en 1h
    (CLOSE FUTURO, corr -0.99 con el retorno a 26h). Causal: close de hace
    26 horas -> chikou_dist = momentum 26h causal.
  - ta_tenkan/kijun/senkou_a/senkou_b/cloud_width/price_above_cloud:
    redefinidos con la nube VISIBLE (calculada hace 26h, como en el chart).
  - ta_dist_last_swing_high/low: rolling(20, center=True) usaba 10 velas
    futuras. Causal: pivote en t-10 confirmado con la ventana [t-20, t].

M10 (cross-crypto diario, hasta 10% del gain):
  - los datos diarios se asignaban al INICIO del dia (cada vela intradia
    veia el cierre del dia completo). Causal: shift de 1 dia (el dato del
    dia D esta disponible desde D+1 00:00).
"""
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")
PROCESSED = BASE / "Data/processed"
MASTER_P = BASE / "Data/master/master_15m.parquet"
MASTER_OUT = BASE / "Data/master/master_15m_CAUSAL.parquet"

m01 = pd.read_parquet(PROCESSED / "M01_ohlcv.parquet")
o, h, l, c = (m01["ohlcv_btc_open"], m01["ohlcv_btc_high"],
              m01["ohlcv_btc_low"], m01["ohlcv_btc_close"])

# ================================================================= M03 fix
print("\n=== M03: Ichimoku + swings causales ===")
m03 = pd.read_parquet(PROCESSED / "M03_technicals.parquet")

# Ichimoku sobre resample 1h (como el original), pero CAUSAL
h1 = h.resample("1h").max()
l1 = l.resample("1h").min()
c1 = c.resample("1h").last()
tenkan = (h1.rolling(9).max() + l1.rolling(9).min()) / 2
kijun = (h1.rolling(26).max() + l1.rolling(26).min()) / 2
# nube VISIBLE en t = calculada hace 26 velas 1h (como se dibuja en el chart)
senkou_a = ((tenkan + kijun) / 2).shift(26)
senkou_b = ((h1.rolling(52).max() + l1.rolling(52).min()) / 2).shift(26)
# chikou CAUSAL: el close de hace 26 horas (momentum estandar del chikou)
chikou = c1.shift(26)

def to15(s):
    return s.reindex(c.index, method="ffill")

m03["ta_tenkan_sen"] = to15(tenkan)
m03["ta_kijun_sen"] = to15(kijun)
m03["ta_senkou_a"] = to15(senkou_a)
m03["ta_senkou_b"] = to15(senkou_b)
m03["ta_chikou_span"] = to15(chikou)
m03["ta_chikou_dist"] = (c - m03["ta_chikou_span"]) / c
m03["ta_cloud_width"] = (m03["ta_senkou_a"] - m03["ta_senkou_b"]).abs() / c
m03["ta_price_above_cloud"] = (
    c > pd.concat([m03["ta_senkou_a"], m03["ta_senkou_b"]], axis=1).max(axis=1)
).astype("int8")

# Swings causales: pivote en t-10 confirmado con [t-20, t]
piv_h = h.shift(10).where(h.shift(10) >= h.rolling(21).max() - 1e-9)
piv_l = l.shift(10).where(l.shift(10) <= l.rolling(21).min() + 1e-9)
m03["ta_dist_last_swing_high"] = (c - piv_h.ffill()) / c
m03["ta_dist_last_swing_low"] = (c - piv_l.ffill()) / c

m03.to_parquet(PROCESSED / "M03_technicals_CAUSAL.parquet", compression="snappy")
print("M03 causal escrito:", m03.shape)

# ================================================================= M10 fix
print("\n=== M10: cross-crypto con lag de 1 dia ===")
raw = pd.read_parquet(BASE / "Data/raw/cross_crypto/total_mcap/cross_crypto_daily.parquet")
raw.index = pd.to_datetime(raw.index, utc=True)
# El dato del dia D pasa a estar disponible desde D+1 00:00
raw_lag = raw.copy()
raw_lag.index = raw_lag.index + pd.Timedelta(days=1)

grid = m01.index
cx = pd.DataFrame(index=grid)
df = raw_lag.reindex(grid, method="ffill")
for col in df.columns:
    cx[col] = df[col]
cx["cx_btc_dom_chg_1d"] = cx["cx_btc_dominance"].diff(96)
cx["cx_total_mcap_return_1d"] = cx["cx_total_mcap"].pct_change(96)
cx["cx_total2_return_1d"] = cx["cx_total2_mcap"].pct_change(96)
cx["cx_total3_return_1d"] = cx["cx_total3_mcap"].pct_change(96)

# Ratios intradia (causales, sin cambios)
cx["cx_eth_btc_ratio"] = m01["ohlcv_eth_close"] / m01["ohlcv_btc_close"]
cx["cx_eth_btc_return"] = np.log(cx["cx_eth_btc_ratio"]).diff()
cx["cx_xrp_btc_ratio"] = m01["ohlcv_xrp_close"] / m01["ohlcv_btc_close"]
cx["cx_xrp_btc_return"] = np.log(cx["cx_xrp_btc_ratio"]).diff()

cx["cx_altcoin_idx_return"] = cx["cx_total3_return_1d"]
cx["cx_altcoin_idx_vol"] = cx["cx_total3_return_1d"].rolling(96).std()
cx["cx_eth_outperf_btc"] = (cx["cx_eth_btc_return"] > 0).rolling(96).mean()
btc_ret = np.log(m01["ohlcv_btc_close"]).diff(96)
cx["cx_alt_outperf_btc"] = ((cx["cx_total3_return_1d"] - btc_ret) > 0).astype("int8")
cx["cx_btc_dom_breakdown"] = ((btc_ret < 0) & (cx["cx_total3_return_1d"] > 0)).astype("int8")

cx = cx[pd.read_parquet(PROCESSED / "M10_cross_crypto.parquet").columns]  # mismo orden
cx.to_parquet(PROCESSED / "M10_cross_crypto_CAUSAL.parquet", compression="snappy")
print("M10 causal escrito:", cx.shape)

# ============================================================== master patch
print("\n=== Parcheando master_15m.parquet ===")
master = pd.read_parquet(MASTER_P)
patch_cols_m03 = ["ta_tenkan_sen", "ta_kijun_sen", "ta_senkou_a", "ta_senkou_b",
                  "ta_chikou_span", "ta_chikou_dist", "ta_cloud_width",
                  "ta_price_above_cloud", "ta_dist_last_swing_high",
                  "ta_dist_last_swing_low"]
n_patched = 0
for col in patch_cols_m03:
    if col in master.columns:
        master[col] = m03[col].reindex(master.index).astype(master[col].dtype
                       if master[col].dtype != "int8" else "int8")
        n_patched += 1
for col in cx.columns:
    if col in master.columns:
        master[col] = cx[col].reindex(master.index)
        n_patched += 1
master.to_parquet(MASTER_OUT, compression="snappy")
print(f"master causal escrito ({n_patched} columnas corregidas) -> {MASTER_OUT.name}")

# ============================================================== verificacion
print("\n=== VERIFICACION: corr contra retorno futuro tras el parche ===")
fwd_26h = (c.shift(-104) / c - 1)
fwd_4h = (c.shift(-16) / c - 1)
for col in ["ta_chikou_dist", "ta_dist_last_swing_high", "ta_dist_last_swing_low",
            "cx_total_mcap_return_1d", "cx_btc_dom_chg_1d", "cx_total2_return_1d",
            "cx_total3_return_1d", "cx_altcoin_idx_return"]:
    s = master[col].iloc[::4]
    c26 = s.corr(fwd_26h.iloc[::4])
    c4 = s.corr(fwd_4h.iloc[::4])
    flag = "OK" if max(abs(c26 or 0), abs(c4 or 0)) < 0.2 else "**SIGUE ALTO**"
    print(f"  {col:32s} fwd4h={c4:+.3f}  fwd26h={c26:+.3f}  {flag}")
print("\nParche completado.")
