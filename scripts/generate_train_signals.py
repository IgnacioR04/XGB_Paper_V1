# -*- coding: utf-8 -*-
"""Genera signals_train.parquet para los 3 TFs usando los modelos causales
ya entrenados (estan en cache local). Se sube despues a Drive.

Esto es necesario porque el pipeline original solo guardo predicciones de
val/test (era lo que se usaba para entrenar/evaluar). Para el reporte
RB16 necesitamos tambien train.
"""
import gc
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

CACHE = Path(os.environ["LOCALAPPDATA"]) / "Temp/btc_cache"
DRIVE = Path("G:/Mi unidad/Base de Datos BITCOIN")
DIRM = DRIVE / "Data/model_outputs/directional"
DIRRB = DRIVE / "Data/model_outputs/real_backtest"

# 1) master_features causal en local (esta ya en cache)
master = pd.read_parquet(CACHE / "master_features.parquet")
master["timestamp"] = pd.to_datetime(master["timestamp"])
if getattr(master["timestamp"].dt, "tz", None) is not None:
    master["timestamp"] = master["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
master = master.set_index("timestamp")
print("master:", master.shape)

# 2) por TF: cargar labels train, predict, guardar
LEAK = {"exit_price", "exit_reason", "gross_return", "net_return",
        "time_to_exit", "label_win", "label_tp", "label_sl",
        "label_timeout", "ambiguous", "split"}


def fcols(df):
    drop = LEAK | {"timestamp", "timeframe", "side", "split", "entry_price",
                   "direction_label", "candidate_id"}
    return [c for c in df.columns if c not in drop
            and not (df[c].dtype == "O"
                     or str(df[c].dtype).startswith("string"))]


PRED_COLS = ["timestamp", "candidate_id", "timeframe", "side", "vol_decile",
             "tp_pct", "sl_pct", "p_break_even", "net_return", "label_win"]

# limites de train para no explotar RAM (igual que D04)
MAX_TRAIN = {"4h": None, "1h": 600_000, "15m": 1_000_000}

train_parts = []
for tf in ["4h", "1h", "15m"]:
    t0 = time.time()
    bst = xgb.Booster()
    bst.load_model(str(CACHE / f"approach_B_xgb_{tf}.json"))

    lbl = pd.read_parquet(CACHE / f"labels_compact_{tf}.parquet")
    lbl["timestamp"] = pd.to_datetime(lbl["timestamp"])
    if getattr(lbl["timestamp"].dt, "tz", None) is not None:
        lbl["timestamp"] = lbl["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    lbl = lbl[lbl["split"] == "train"].reset_index(drop=True)
    if MAX_TRAIN[tf] and len(lbl) > MAX_TRAIN[tf]:
        lbl = lbl.sample(n=MAX_TRAIN[tf], random_state=42)
    print(f"{tf} train labels: {len(lbl):,}")

    df = lbl.merge(master, left_on="timestamp", right_index=True, how="left")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    df = df.dropna(subset=["label_win"])
    fc = fcols(df)
    X = np.ascontiguousarray(df[fc].to_numpy(dtype=np.float32))
    p = bst.predict(xgb.DMatrix(X, feature_names=fc),
                    iteration_range=(0, bst.best_iteration + 1))
    meta = df[PRED_COLS].copy()
    meta["p_win"] = p
    meta["dataset"] = "train"
    train_parts.append(meta)
    print(f"{tf}: {len(meta):,} predicciones en {time.time()-t0:.0f}s")
    del df, X, lbl
    gc.collect()

train_preds = pd.concat(train_parts, ignore_index=True)
del train_parts; gc.collect()
print("train_preds:", train_preds.shape)

# 3) calibrar usando la calibracion que ya tenemos (calib_approach_B.parquet)
# La isotonic se ajusto en VAL; aqui la aplicamos a train para mantener
# coherencia con val/test (mismo mapeo p_win -> p_iso).
calib_ref = pd.read_parquet(CACHE / "calib_approach_B.parquet",
                            columns=["p_win", "p_win_isotonic", "p_win_sigmoid"])
calib_ref = calib_ref.drop_duplicates("p_win").sort_values("p_win")
# interpolar
train_preds["p_win_isotonic"] = np.interp(
    train_preds["p_win"].values,
    calib_ref["p_win"].values, calib_ref["p_win_isotonic"].values)
train_preds["p_win_sigmoid"] = np.interp(
    train_preds["p_win"].values,
    calib_ref["p_win"].values, calib_ref["p_win_sigmoid"].values)

# 4) reprice con COST 0.0012
COST = 0.0012
tp = train_preds["tp_pct"].astype(float)
sl = train_preds["sl_pct"].astype(float)
train_preds["cost"] = COST
train_preds["p_break_even_new"] = (sl + COST) / (tp + sl)
for col in ["p_win", "p_win_sigmoid", "p_win_isotonic"]:
    pw = train_preds[col].astype(float)
    train_preds[f"EV_pred_{col}"] = pw * tp - (1 - pw) * sl - COST
    train_preds[f"edge_over_be_{col}"] = pw - train_preds["p_break_even_new"]
train_preds["p_win_main"] = train_preds["p_win_isotonic"]
train_preds["EV_pred"] = train_preds["EV_pred_p_win_isotonic"]
train_preds["edge_over_be"] = train_preds["edge_over_be_p_win_isotonic"]

cols_keep = ["timestamp", "candidate_id", "timeframe", "side", "vol_decile",
             "tp_pct", "sl_pct", "cost", "p_win", "p_win_sigmoid",
             "p_win_isotonic", "p_break_even", "p_break_even_new",
             "p_win_main", "EV_pred", "edge_over_be",
             "EV_pred_p_win", "EV_pred_p_win_sigmoid", "EV_pred_p_win_isotonic",
             "edge_over_be_p_win", "edge_over_be_p_win_sigmoid",
             "edge_over_be_p_win_isotonic", "net_return", "label_win", "dataset"]
cols_keep = [c for c in cols_keep if c in train_preds.columns]

# 5) guardar en cache + copiar a Drive
local_out = CACHE / "signals_train.parquet"
train_preds[cols_keep].to_parquet(local_out, index=False)
print(f"local: {local_out} ({local_out.stat().st_size/1e6:.0f} MB)")

drive_out = DIRRB / "signals/signals_train.parquet"
for i in range(8):
    try:
        shutil.copyfile(local_out, drive_out)
        print(f"-> {drive_out.name}")
        break
    except Exception as e:
        print(f"  retry {i+1}: {e}"); time.sleep(20)
print("OK")
