# -*- coding: utf-8 -*-
"""Rehace la cadena de reentrenamiento SOBRE DATOS CAUSALES, en local.

Contexto: la promocion de datasets quedo con los nombres INVERTIDOS
(official=LEAKY, *_LEAKY_backup=CAUSAL), asi que D01/D04/D09/RB01/RB13
corrieron sobre datos contaminados y el "modelo nuevo" salio identico al
viejo (mismos datos + seed fijo). Este script:

1. Intercambia los nombres (verificando la firma de leakage antes/despues)
2. D01-equiv: regenera master_features.parquet desde el master causal
3. D04-equiv: reentrena approach B (identico al notebook: params, seed,
   caps, QuantileDMatrix, early stopping)
4. D09-equiv: calibracion sigmoid+isotonic para B
5. RB01-equiv: signals_val/test con coste 0.0012
6. Tabla de buckets por TF (honesta) para decidir el threshold
"""
import gc
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")
PROC = BASE / "Data/processed"
MAST = BASE / "Data/master"
DIRM = BASE / "Data/model_outputs/directional"
DIRRB = BASE / "Data/model_outputs/real_backtest"

import os
import shutil

CACHE = Path(os.environ.get("LOCALAPPDATA", "C:/temp")) / "Temp/btc_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def hydrate(path: Path) -> Path:
    """Copia el parquet de Drive a cache local (C:) con reintentos largos.
    Lecturas posteriores van contra la copia local (Drive mirror puede servir
    contenido parcial justo despues de renames)."""
    dst = CACHE / path.name
    if dst.exists() and dst.stat().st_size == path.stat().st_size:
        return dst
    last = None
    for i in range(20):
        try:
            shutil.copyfile(path, dst)
            # validar footer
            import pyarrow.parquet as pq
            pq.read_metadata(dst)
            return dst
        except Exception as e:
            last = e
            time.sleep(45)
    raise RuntimeError(f"hydrate fallo para {path.name}: {last}")


m01_close = pd.read_parquet(hydrate(PROC / "M01_ohlcv.parquet"),
                            columns=["ohlcv_btc_close"])["ohlcv_btc_close"]
FWD = (m01_close.shift(-104) / m01_close - 1).iloc[::8]


def sig(path, col):
    s = pd.read_parquet(hydrate(Path(path)), columns=[col])[col].iloc[::8]
    return float(s.corr(FWD))


def is_leaky(path):
    if "M10" in Path(path).name:
        return abs(sig(path, "cx_total_mcap_return_1d")) > 0.2
    return abs(sig(path, "ta_chikou_dist")) > 0.5


# ================================================== 1. SWAP de nombres
print("=== 1. Verificando y corrigiendo nombres ===", flush=True)
swaps = [
    (PROC / "M03_technicals.parquet", PROC / "M03_technicals_LEAKY_backup (1).parquet"),
    (PROC / "M10_cross_crypto.parquet", PROC / "M10_cross_crypto_LEAKY_backup.parquet"),
    (MAST / "master_15m.parquet", MAST / "master_15m_LEAKY_backup.parquet"),
]
CAUSAL_LOCAL = {}
for official, backup in swaps:
    if not is_leaky(official):
        print(f"{official.name}: official ya CAUSAL", flush=True)
        CAUSAL_LOCAL[official.name] = CACHE / official.name
        continue
    if is_leaky(backup):
        raise RuntimeError(f"Ambas versiones LEAKY en {official.name}!")
    print(f"{official.name}: official=LEAKY backup=CAUSAL -> swap", flush=True)
    tmp = official.with_name(official.stem + "_SWAPTMP.parquet")
    official.rename(tmp)
    backup.rename(official)
    bk_name = official.with_name(official.stem + "_LEAKY_backup.parquet")
    if bk_name.exists():
        bk_name = official.with_name(official.stem + "_LEAKY_backup2.parquet")
    tmp.rename(bk_name)
    # la copia local hidratada del backup ES el contenido causal
    CAUSAL_LOCAL[official.name] = CACHE / backup.name
    print(f"  -> intercambiado (verificado via cache local)", flush=True)

# ================================================== 2. D01-equiv
print("\n=== 2. Regenerando master_features desde master causal ===", flush=True)
t0 = time.time()
master = pd.read_parquet(CAUSAL_LOCAL.get("master_15m.parquet", hydrate(MAST / "master_15m.parquet")))
DROP_FULL_NAN = ["reg_depth_regime", "reg_dollar_strength",
                 "cost_spread_cost", "cost_max_position_liq"]
DROP_CONSTANT = ["reg_rates_up", "reg_leverage_high", "reg_liq_risk",
                 "cost_expected_return_net", "cost_trade_allowed",
                 "cost_holding_time_expected", "qa_bad_tick_flag",
                 "qa_dup_timestamp_flag", "qa_publication_delay",
                 "qa_api_delay_sec"]
drop_all = (set(DROP_FULL_NAN) | set(DROP_CONSTANT)
            | {c for c in master.columns if c.startswith(("tgt_", "qa_", "cost_"))})
master = master.drop(columns=[c for c in drop_all if c in master.columns])
for c in master.select_dtypes(include=["float64"]).columns:
    master[c] = master[c].astype("float32")
for c in master.select_dtypes(include=["int64"]).columns:
    master[c] = pd.to_numeric(master[c], downcast="integer")
if master.index.tz is not None:
    master.index = master.index.tz_convert("UTC").tz_localize(None)
master.index.name = "timestamp"
master.reset_index().to_parquet(DIRM / "datasets/master_features.parquet",
                                compression="snappy", index=False)
master[["ohlcv_btc_close"]].reset_index().to_parquet(
    DIRM / "datasets/master_close_only.parquet", compression="snappy", index=False)
print(f"master_features: {master.shape} en {time.time()-t0:.0f}s", flush=True)
# verificar firma causal en master_features
mf_sig = master["ta_chikou_dist"].iloc[::8].astype(float).corr(FWD)
print(f"firma chikou en master_features: {mf_sig:+.3f} (debe ser ~-0.05)", flush=True)
assert abs(mf_sig) < 0.5
master_idx = master  # mantener en RAM para el merge

# ================================================== 3. D04-equiv
print("\n=== 3. Reentrenando Approach B (causal) ===", flush=True)
LEAKAGE_COLS = {"exit_price", "exit_reason", "gross_return", "net_return",
                "time_to_exit", "label_win", "label_tp", "label_sl",
                "label_timeout", "ambiguous", "split"}
MAX_TRAIN = {"4h": None, "1h": 600_000, "15m": 1_000_000}
XGB_PARAMS = dict(max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.5, min_child_weight=20.0,
                  tree_method="hist", nthread=-1,
                  objective="binary:logistic", eval_metric="logloss", seed=42)
PRED_COLS = ["timestamp", "candidate_id", "timeframe", "side", "vol_decile",
             "tp_pct", "sl_pct", "p_break_even", "net_return", "label_win"]


def load_split(tf, split, max_rows=None):
    lbl = pd.read_parquet(hydrate(DIRM / f"datasets/labels_compact_{tf}.parquet"))
    lbl["timestamp"] = pd.to_datetime(lbl["timestamp"])
    if getattr(lbl["timestamp"].dt, "tz", None) is not None:
        lbl["timestamp"] = lbl["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    lbl = lbl[lbl["split"] == split].reset_index(drop=True)
    if max_rows is not None and len(lbl) > max_rows:
        lbl = lbl.sample(n=max_rows, random_state=42)
    out = lbl.merge(master_idx, left_on="timestamp", right_index=True, how="left")
    del lbl
    gc.collect()
    for c in out.select_dtypes(include=["float64"]).columns:
        out[c] = out[c].astype("float32")
    return out


def feat_cols_of(df):
    drop = LEAKAGE_COLS | {"timestamp", "timeframe", "side", "split",
                           "entry_price", "direction_label", "candidate_id"}
    return [c for c in df.columns
            if c not in drop and not (df[c].dtype == "O"
                                      or str(df[c].dtype).startswith("string"))]


schema = json.loads((Path(__file__).resolve().parents[1]
                     / "artifacts/schemas/feature_schema.json").read_text())

parts_p, parts_m = [], []
for tf in ["4h", "1h", "15m"]:
    t0 = time.time()
    train = load_split(tf, "train", MAX_TRAIN[tf]).dropna(subset=["label_win"])
    train.replace([np.inf, -np.inf], np.nan, inplace=True)
    fc = feat_cols_of(train)
    assert fc == schema["features_in_order"], \
        f"{tf}: orden de features difiere del schema!"
    y_tr = train["label_win"].astype(np.int8).values
    X_tr = np.ascontiguousarray(train[fc].to_numpy(dtype=np.float32))
    del train; gc.collect()

    val = load_split(tf, "val").dropna(subset=["label_win"])
    val.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_v = val["label_win"].astype(np.int8).values
    val_meta = val[PRED_COLS].copy()
    X_v = np.ascontiguousarray(val[fc].to_numpy(dtype=np.float32))
    del val; gc.collect()

    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=fc)
    del X_tr; gc.collect()
    dv = xgb.QuantileDMatrix(X_v, label=y_v, ref=dtr, feature_names=fc)
    del X_v; gc.collect()
    bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=200,
                    evals=[(dv, "val")], early_stopping_rounds=30,
                    verbose_eval=False)
    bi = bst.best_iteration
    p_v = bst.predict(dv, iteration_range=(0, bi + 1))
    val_meta["p_win"] = p_v
    val_meta["dataset"] = "val"
    del dtr, dv; gc.collect()

    test = load_split(tf, "test").dropna(subset=["label_win"])
    test.replace([np.inf, -np.inf], np.nan, inplace=True)
    y_t = test["label_win"].astype(np.int8).values
    test_meta = test[PRED_COLS].copy()
    X_t = np.ascontiguousarray(test[fc].to_numpy(dtype=np.float32))
    del test; gc.collect()
    dt_ = xgb.DMatrix(X_t, feature_names=fc)
    p_t = bst.predict(dt_, iteration_range=(0, bi + 1))
    test_meta["p_win"] = p_t
    test_meta["dataset"] = "test"
    del dt_, X_t; gc.collect()

    bst.save_model(str(DIRM / f"models/approach_B_xgb_{tf}.json"))
    pred_df = pd.concat([val_meta, test_meta], ignore_index=True)
    pred_df.to_parquet(DIRM / f"predictions/preds_approach_B_{tf}.parquet",
                       index=False)
    from sklearn.metrics import log_loss, roc_auc_score
    for ds, y_, p_ in [("val", y_v, p_v), ("test", y_t, p_t)]:
        parts_m.append({"approach": "B", "timeframe": tf, "dataset": ds,
                        "logloss": log_loss(y_, np.clip(p_, 1e-7, 1 - 1e-7)),
                        "auc": roc_auc_score(y_, p_), "n": len(y_)})
    parts_p.append(pred_df)
    print(f"{tf}: best_iter={bi} fit={time.time()-t0:.0f}s "
          f"AUCval={parts_m[-2]['auc']:.4f} AUCtest={parts_m[-1]['auc']:.4f}",
          flush=True)
    del bst; gc.collect()

del master_idx; gc.collect()
preds_B = pd.concat(parts_p, ignore_index=True)
preds_B.to_parquet(DIRM / "predictions/preds_approach_B.parquet", index=False)
pd.DataFrame(parts_m).to_csv(DIRM / "metrics/model_metrics_B.csv", index=False)
for tf in ["4h", "1h", "15m"]:
    sub = [r for r in parts_m if r["timeframe"] == tf]
    pd.DataFrame(sub).to_csv(DIRM / f"metrics/model_metrics_B_{tf}.csv", index=False)
print("metricas:\n", pd.DataFrame(parts_m).to_string(index=False), flush=True)

# ================================================== 4. D09-equiv (solo B)
print("\n=== 4. Calibracion (sigmoid + isotonic, fit en val) ===", flush=True)
val = preds_B[preds_B.dataset == "val"]
test = preds_B[preds_B.dataset == "test"]
y_v = val["label_win"].astype(int).values
p_v = val["p_win"].values
lr = LogisticRegression(C=1e10, solver="lbfgs")
lr.fit(p_v.reshape(-1, 1), y_v)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(p_v, y_v)
calib = preds_B.copy()
calib["p_win_sigmoid"] = lr.predict_proba(calib["p_win"].values.reshape(-1, 1))[:, 1]
calib["p_win_isotonic"] = iso.predict(calib["p_win"].values)
calib.to_parquet(DIRM / "calibration/calib_approach_B.parquet", index=False)
print(f"calib_approach_B.parquet: {calib.shape}", flush=True)

# ================================================== 5. RB01-equiv
print("\n=== 5. Signals con coste 0.0012 ===", flush=True)
COST = 0.0012
out = calib.copy()
tp = out["tp_pct"].astype(float)
sl = out["sl_pct"].astype(float)
out["cost"] = COST
out["p_break_even_new"] = (sl + COST) / (tp + sl)
for col in ["p_win", "p_win_sigmoid", "p_win_isotonic"]:
    pw = out[col].astype(float)
    out[f"EV_pred_{col}"] = pw * tp - (1 - pw) * sl - COST
    out[f"edge_over_be_{col}"] = pw - out["p_break_even_new"]
out["p_win_main"] = out["p_win_isotonic"]
out["EV_pred"] = out["EV_pred_p_win_isotonic"]
out["edge_over_be"] = out["edge_over_be_p_win_isotonic"]
cols_keep = ["timestamp", "candidate_id", "timeframe", "side", "vol_decile",
             "tp_pct", "sl_pct", "cost", "p_win", "p_win_sigmoid",
             "p_win_isotonic", "p_break_even", "p_break_even_new",
             "p_win_main", "EV_pred", "edge_over_be",
             "EV_pred_p_win", "EV_pred_p_win_sigmoid", "EV_pred_p_win_isotonic",
             "edge_over_be_p_win", "edge_over_be_p_win_sigmoid",
             "edge_over_be_p_win_isotonic", "net_return", "label_win", "dataset"]
cols_keep = [c for c in cols_keep if c in out.columns]
out[out.dataset == "val"][cols_keep].to_parquet(
    DIRRB / "signals/signals_val.parquet", index=False)
out[out.dataset == "test"][cols_keep].to_parquet(
    DIRRB / "signals/signals_test.parquet", index=False)
print("signals val/test escritos", flush=True)

# ================================================== 6. Buckets honestos
print("\n=== 6. Tabla de buckets por TF (modelo causal) ===", flush=True)
out["net_return_cost012"] = out["net_return"].astype(float) - (COST - 0.001)
BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001]
LBL = [f"{a:.2f}-{b:.2f}" for a, b in zip(BUCKETS[:-1], BUCKETS[1:])]
for ds in ["val", "test"]:
    sub = out[out.dataset == ds].copy()
    sub["bucket"] = pd.cut(sub["p_win_isotonic"], bins=BUCKETS, labels=LBL,
                           include_lowest=True)
    rows = []
    for (tf, bk), g in sub.groupby(["timeframe", "bucket"], observed=True):
        if len(g) < 30:
            continue
        idx = g.groupby("timestamp")["EV_pred"].idxmax()
        t = g.loc[idx]
        nr = t["net_return_cost012"].values
        pos, neg = nr[nr > 0], -nr[nr < 0]
        rows.append({"timeframe": tf, "bucket": str(bk), "n_trades": len(t),
                     "win_rate": round((nr > 0).mean(), 3),
                     "expectancy": round(nr.mean(), 6),
                     "profit_factor": round(pos.sum() / neg.sum(), 2)
                     if neg.sum() > 0 else np.inf,
                     "total_return": round(nr.sum(), 2)})
    df = pd.DataFrame(rows)
    df.to_csv(DIRRB / f"threshold_grid/buckets_causal_{ds}.csv", index=False)
    print(f"-- {ds} --", flush=True)
    print(df.to_string(index=False), flush=True)

print("\nCADENA COMPLETA OK", flush=True)
