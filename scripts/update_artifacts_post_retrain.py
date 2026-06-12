# -*- coding: utf-8 -*-
"""Post-reentrenamiento (modelos causales 2026-06-11):
1. Copia los modelos nuevos a artifacts/models/
2. Reconstruye calib_approach_B_compact.json desde calib_approach_B.parquet
3. Mide cobertura de gain del live builder con los modelos NUEVOS
"""
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

DRIVE = Path("G:/Mi unidad/Base de Datos BITCOIN")
REPO = Path(__file__).resolve().parents[1]
DIRM = DRIVE / "Data/model_outputs/directional"

# 1. modelos
for tf in ["15m", "1h", "4h"]:
    src = DIRM / f"models/approach_B_xgb_{tf}.json"
    dst = REPO / f"artifacts/models/approach_B_xgb_{tf}.json"
    shutil.copy2(src, dst)
    print(f"modelo {tf}: copiado ({dst.stat().st_size/1e6:.1f} MB)")

# 2. calibracion compacta por TF (mapa p_win -> p_win_isotonic como step fn)
calib = pd.read_parquet(DIRM / "calibration/calib_approach_B.parquet",
                        columns=["timeframe", "p_win", "p_win_isotonic"])
out = {}
for tf, g in calib.groupby("timeframe"):
    g = g.sort_values("p_win")
    # los tramos de la isotonic: cada cambio de y es un threshold
    y = g["p_win_isotonic"].values
    x = g["p_win"].values
    chg = np.concatenate([[True], np.diff(y) > 1e-12])
    out[tf] = {"x_thresholds": [round(float(v), 6) for v in x[chg]],
               "y_thresholds": [round(float(v), 6) for v in y[chg]],
               "n_fit": int(len(g))}
    print(f"calib {tf}: {chg.sum()} tramos (n={len(g)})")
dst = REPO / "artifacts/calibration/calib_approach_B_compact.json"
dst.write_text(json.dumps(out), encoding="utf-8")
print(f"compacto -> {dst} ({dst.stat().st_size/1024:.0f} KB)")

# 3. cobertura de gain con los modelos NUEVOS
schema = json.loads((REPO / "artifacts/schemas/feature_schema.json").read_text())
CAND = set(schema["candidate_features"])
fe = pd.read_csv(next((REPO / "data/logs/features").glob("features_*.csv")), nrows=2)
live_cols = set(fe.columns)
GEO_NAN = {c for c in live_cols if c.startswith((
    "ohlcv_ethbtc", "ohlcv_xrp", "ohlcv_btc_futures", "ohlcv_btc_mark",
    "ohlcv_btc_index", "ohlcv_btc_spot_perp"))}

print("\n=== Cobertura de gain (modelos NUEVOS sin leakage) ===")
union_missing = {}
for tf in ["15m", "1h", "4h"]:
    b = xgb.Booster()
    b.load_model(str(REPO / f"artifacts/models/approach_B_xgb_{tf}.json"))
    gain = b.get_score(importance_type="total_gain")
    tot = sum(gain.values())
    parts = {"candidato": 0, "disponible": 0, "geo_nan": 0, "falta": 0}
    for f, v in gain.items():
        if f in CAND:
            parts["candidato"] += v
        elif f in GEO_NAN:
            parts["geo_nan"] += v
        elif f in live_cols:
            parts["disponible"] += v
        else:
            parts["falta"] += v
            union_missing[f] = union_missing.get(f, 0) + v / tot
    print(f"{tf}: candidato {parts['candidato']/tot*100:.1f}% | "
          f"disponible-live {parts['disponible']/tot*100:.1f}% | "
          f"geo-NaN {parts['geo_nan']/tot*100:.1f}% | "
          f"FALTA {parts['falta']/tot*100:.1f}%")

s = pd.Series(union_missing).sort_values(ascending=False)
cum = s.cumsum() / s.sum()
n90 = int((cum <= 0.90).sum()) + 1
print(f"\nFaltantes con gain: {len(s)} | 90% del gain perdido en {n90} features")
print("\nTop 40 faltantes (share de gain sumado entre TFs, %):")
for f in s.head(40).index:
    print(f"  {f:42s} {s[f]*100:.2f}")
