# -*- coding: utf-8 -*-
"""Cuanto pesan las 422 features que el live builder NO calcula?

Usa la importancia total_gain de cada booster: % del gain total del modelo
que corresponde a features disponibles vs faltantes en vivo.
"""
import json
from pathlib import Path

import pandas as pd
import xgboost as xgb

BASE = Path(__file__).resolve().parents[1]

schema = json.loads((BASE / "artifacts/schemas/feature_schema.json").read_text())
CAND = set(schema["candidate_features"])

fe = pd.read_csv(next((BASE / "data/logs/features").glob("features_*.csv")),
                 nrows=2)
live_cols = set(fe.columns)

# columnas 100% NaN en runtime por geo-block (de analyze_day1)
ALWAYS_NAN = {c for c in live_cols if c.startswith((
    "ohlcv_ethbtc", "ohlcv_xrp", "ohlcv_btc_futures", "ohlcv_btc_mark",
    "ohlcv_btc_index", "ohlcv_btc_spot_perp"))}

for tf in ["15m", "1h", "4h"]:
    booster = xgb.Booster()
    booster.load_model(str(BASE / f"artifacts/models/approach_B_xgb_{tf}.json"))
    gain = booster.get_score(importance_type="total_gain")
    total = sum(gain.values())
    rows = []
    for f, g in gain.items():
        if f in CAND:
            status = "candidato (OK)"
        elif f in ALWAYS_NAN:
            status = "en builder pero NaN (geo-block)"
        elif f in live_cols:
            status = "disponible en vivo"
        else:
            status = "FALTA en builder"
        rows.append({"feature": f, "gain": g, "status": status})
    df = pd.DataFrame(rows)
    agg = df.groupby("status")["gain"].agg(["sum", "count"])
    agg["pct_gain"] = (agg["sum"] / total * 100).round(1)
    print(f"=== {tf} (gain total del modelo) ===")
    print(agg[["count", "pct_gain"]].to_string())
    missing = df[df.status.isin(["FALTA en builder",
                                 "en builder pero NaN (geo-block)"])]
    top = missing.sort_values("gain", ascending=False).head(15)
    top["pct"] = (top.gain / total * 100).round(2)
    print(f"  Top features faltantes por gain ({tf}):")
    for _, r in top.iterrows():
        print(f"    {r.feature:45s} {r.pct:5.2f}%")
    # cuantas features faltantes acumulan el 90% del gain perdido
    ms = missing.sort_values("gain", ascending=False)
    cum = ms.gain.cumsum() / ms.gain.sum()
    n90 = int((cum <= 0.90).sum()) + 1
    print(f"  Gain perdido total: {ms.gain.sum()/total*100:.1f}% | "
          f"el 90% de ese gain esta en solo {n90} features")
    print()
