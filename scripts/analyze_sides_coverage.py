"""Diagnostico 2: cobertura de features vs schema + p_win por side.

(a) Que % del schema de 615 features cubre el feature builder live.
(b) Scoring local de candidatos long y short sobre las filas de features
    registradas, para ver si los shorts puntuan sistematicamente bajo.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(BASE))

from src.strategy.signal_engine import evaluate_timeframe, expand_candidates_with_features
from src.strategy.candidates import load_library, candidates_for
from src.models.inference import predict_p_win

schema = json.loads((BASE / "artifacts" / "schemas" / "feature_schema.json").read_text())
feat_names = schema["features_in_order"]
print(f"Schema: {len(feat_names)} features")

CAND_FEATS = ["vol_pred", "vol_decile", "tp_mult", "sl_mult", "H",
              "tp_pct", "sl_pct", "side_long", "barrier_quality_score",
              "p_break_even"]
market_feats = [f for f in feat_names if f not in CAND_FEATS]
print(f"Market features en schema: {len(market_feats)}")

fe_files = sorted((BASE / "data" / "logs" / "features").glob("features_*.csv"))
fe = pd.concat([pd.read_csv(f) for f in fe_files], ignore_index=True)
live_cols = set(fe.columns) - {"timeframe", "tick_ts_utc", "candle_close_time"}
covered = [f for f in market_feats if f in live_cols]
missing = [f for f in market_feats if f not in live_cols]
print(f"Cubiertas por live builder: {len(covered)}")
print(f"FALTAN del builder (NaN en inferencia): {len(missing)}")

# Agrupar las que faltan por prefijo
pref = pd.Series([m.split("_")[0] + "_" for m in missing]).value_counts()
print("\nFeatures faltantes por prefijo:")
print(pref.to_string())

# Ademas: de las cubiertas, cuantas vienen 100% NaN en los logs
sub = fe[covered].apply(pd.to_numeric, errors="coerce")
allnan = [c for c in covered if sub[c].isna().all()]
print(f"\nCubiertas pero 100% NaN en runtime: {len(allnan)}")
total_eff_nan = len(missing) + len(allnan)
print(f"==> TOTAL features sin informacion en live: {total_eff_nan} de "
      f"{len(market_feats)} market ({total_eff_nan/len(market_feats)*100:.0f}%)")

# ---------------------------------------------------------------- sides
print("\n" + "=" * 70)
print("p_win por SIDE: scoring local de las filas de features registradas")
print("=" * 70)

for tf in ["15m", "1h", "4h"]:
    rows = fe[fe.timeframe == tf].drop_duplicates(
        subset=["ohlcv_btc_close", "vol_decile"], keep="last")
    if rows.empty:
        continue
    lib = load_library(str(BASE / "artifacts" / "candidates" /
                           f"barrier_candidate_library_{tf}.parquet"))
    model_p = str(BASE / "artifacts" / "models" / f"approach_B_xgb_{tf}.json")
    calib_p = str(BASE / "artifacts" / "calibration" / "calib_approach_B_compact.json")

    recs = []
    for ridx, frow in rows.iterrows():
        if pd.isna(frow.get("vol_decile")):
            continue
        vol_decile = int(frow["vol_decile"])
        vol_pred = float(frow.get("vol_pred", np.nan))
        cands = candidates_for(lib, tf, vol_decile)
        feature_row = frow.drop(labels=["timeframe", "tick_ts_utc",
                                        "candle_open_time", "vol_pred",
                                        "vol_decile"], errors="ignore")
        feature_row = pd.to_numeric(feature_row, errors="coerce")
        X = expand_candidates_with_features(cands, feature_row, vol_pred, vol_decile)
        pred = predict_p_win(X, model_p, calib_p, tf)
        for i, (_, c) in enumerate(cands.iterrows()):
            recs.append({"candle": ridx,
                         "side": c["side"],
                         "p_raw": pred["p_win_raw"].iloc[i],
                         "p_cal": pred["p_win_isotonic"].iloc[i]})
    if not recs:
        print(f"\n--- {tf}: sin filas matcheables ---")
        continue
    r = pd.DataFrame(recs)
    print(f"\n--- {tf}: {r.candle.nunique()} velas x 10 candidatos ---")
    g = r.groupby("side").agg(
        n=("p_cal", "size"),
        p_raw_med=("p_raw", "median"),
        p_cal_min=("p_cal", "min"), p_cal_med=("p_cal", "median"),
        p_cal_max=("p_cal", "max"),
        pct_ge_band=("p_cal", lambda s: (s >= 0.65).mean() * 100),
    )
    print(g.round(4).to_string())
