# -*- coding: utf-8 -*-
"""Test de paridad: compute_parity_features vs master CAUSAL.

Para N timestamps muestreados, computa las features con la ventana de
1600 velas 15m que veria el bot y compara contra el master causal.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.features.parity_features import compute_parity_features

CACHE = Path(os.environ.get("LOCALAPPDATA", "C:/temp")) / "Temp/btc_cache"
M01_P = CACHE / "M01_ohlcv.parquet"
MASTER_P = CACHE / "master_15m_CAUSAL_v4.parquet"

m01 = pd.read_parquet(M01_P)
if m01.index.tz is not None:
    m01.index = m01.index.tz_convert("UTC").tz_localize(None)
master = pd.read_parquet(MASTER_P)
if master.index.tz is not None:
    master.index = master.index.tz_convert("UTC").tz_localize(None)

N_SAMPLES = 40
WINDOW = 1600
rng = np.random.RandomState(7)
valid_idx = np.arange(WINDOW + 10, len(m01) - 1)
# muestrear sobre 2024+ (val/test region)
start_2024 = m01.index.searchsorted(pd.Timestamp("2024-01-01"))
valid_idx = valid_idx[valid_idx >= start_2024]
sample_pos = np.sort(rng.choice(valid_idx, N_SAMPLES, replace=False))

# columnas que el modulo produce (en una pasada de prueba)
probe = compute_parity_features(m01.iloc[:WINDOW])
candidate_cols = [c for c in probe.columns if c in master.columns]
print(f"Features a testear: {len(candidate_cols)}")

# vwap real venia de aggTrades: aproximacion conocida, tolerancia aparte
APPROX_OK = {"ret_vwap", "ret_close_to_vwap", "ret_vwap_slope"}

errs = {c: [] for c in candidate_cols}
for pos in sample_pos:
    win = m01.iloc[pos - WINDOW + 1: pos + 1]
    f = compute_parity_features(win)
    ts = win.index[-1]
    mrow = master.loc[ts]
    frow = f.iloc[-1]
    for c in candidate_cols:
        mv, fv = float(mrow.get(c, np.nan)), float(frow.get(c, np.nan))
        if np.isnan(mv) and np.isnan(fv):
            continue
        if np.isnan(mv) != np.isnan(fv):
            errs[c].append(np.inf)
            continue
        denom = max(abs(mv), 1e-8)
        errs[c].append(abs(fv - mv) / denom)

rows = []
for c, e in errs.items():
    if not e:
        rows.append({"feature": c, "n": 0, "med_rel_err": np.nan, "ok": "SIN DATOS"})
        continue
    e = np.array(e)
    med = np.median(e[np.isfinite(e)]) if np.isfinite(e).any() else np.inf
    frac_ok = np.mean(e < 0.01)
    status = ("OK" if frac_ok >= 0.9 else
              ("APROX" if c in APPROX_OK else "MAL"))
    rows.append({"feature": c, "n": len(e), "med_rel_err": med,
                 "frac_lt_1pct": round(frac_ok, 2), "ok": status})
res = pd.DataFrame(rows)
n_ok = (res.ok == "OK").sum()
n_bad = (res.ok == "MAL").sum()
print(f"\nOK: {n_ok} | APROX: {(res.ok == 'APROX').sum()} | MAL: {n_bad} "
      f"| sin datos: {(res.ok == 'SIN DATOS').sum()}")
if n_bad:
    print("\nFeatures con divergencia (>1% rel err en >10% de muestras):")
    bad = res[res.ok == "MAL"].sort_values("frac_lt_1pct")
    print(bad.to_string(index=False))
res.to_csv(REPO / ".audit/parity_results.csv", index=False)
print("\nDetalle -> .audit/parity_results.csv")
