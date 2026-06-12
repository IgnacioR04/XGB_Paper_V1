# -*- coding: utf-8 -*-
"""Cadena causal v4 - a prueba de Google Drive.

Estrategia: NO depender de los archivos renombrados (Drive los sirve
corruptos un rato tras el rename). Se reconstruye todo desde el master
LEAKY legible + M01 + raw cx, parcheando causalmente EN LOCAL (C:), y al
final se escriben a Drive: master causal, M03/M10 regenerados (ambas
versiones, reparando posibles corrupciones), master_features, modelos,
calibracion, signals y buckets.
"""
import gc
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")
PROC = BASE / "Data/processed"
MAST = BASE / "Data/master"
DIRM = BASE / "Data/model_outputs/directional"
DIRRB = BASE / "Data/model_outputs/real_backtest"
CACHE = Path(os.environ.get("LOCALAPPDATA", "C:/temp")) / "Temp/btc_cache"
CACHE.mkdir(parents=True, exist_ok=True)


def hydrate(path: Path, validate_col=None) -> Path:
    dst = CACHE / path.name
    for i in range(15):
        try:
            if not (dst.exists() and dst.stat().st_size == path.stat().st_size):
                shutil.copyfile(path, dst)
            pq.read_metadata(dst)
            if validate_col:
                pd.read_parquet(dst, columns=[validate_col])
            return dst
        except Exception as e:
            if dst.exists():
                dst.unlink()
            print(f"  hydrate retry {i+1} {path.name}: {str(e)[:60]}", flush=True)
            time.sleep(40)
    raise RuntimeError(f"hydrate fallo: {path.name}")


def copy_back(local: Path, drive: Path):
    for i in range(10):
        try:
            shutil.copyfile(local, drive)
            return
        except Exception as e:
            print(f"  copy_back retry {i+1} {drive.name}: {str(e)[:60]}", flush=True)
            time.sleep(30)
    raise RuntimeError(f"copy_back fallo: {drive.name}")


# ============================== 1. localizar un master LEAKY legible
print("=== 1. Localizando master LEAKY legible ===", flush=True)
m01_local = hydrate(PROC / "M01_ohlcv.parquet", "ohlcv_btc_close")
m01 = pd.read_parquet(m01_local)
c0 = m01["ohlcv_btc_close"]
FWD = (c0.shift(-104) / c0 - 1).iloc[::8]

leaky_local = None
for cand in [MAST / "master_15m.parquet", MAST / "master_15m_LEAKY_backup.parquet"]:
    try:
        loc = hydrate(cand, "ta_chikou_dist")
        s = pd.read_parquet(loc, columns=["ta_chikou_dist"])["ta_chikou_dist"].iloc[::8]
        corr = float(s.corr(FWD))
        print(f"{cand.name}: corr={corr:+.3f}", flush=True)
        if abs(corr) > 0.5:
            leaky_local = loc
            break
        else:
            causal_ready = loc  # ya tenemos un causal legible!
    except Exception as e:
        print(f"{cand.name}: ilegible ({str(e)[:50]})", flush=True)

# ============================== 2. construir master CAUSAL en local
if leaky_local is None and "causal_ready" in dir():
    print("Master causal ya legible; usandolo directamente", flush=True)
    master_causal_local = causal_ready
else:
    print("\n=== 2. Parcheando master LEAKY -> CAUSAL (local) ===", flush=True)
    master = pd.read_parquet(leaky_local)
    o, h, l, c = (m01["ohlcv_btc_open"], m01["ohlcv_btc_high"],
                  m01["ohlcv_btc_low"], m01["ohlcv_btc_close"])
    h1 = h.resample("1h").max()
    l1 = l.resample("1h").min()
    c1 = c.resample("1h").last()
    tenkan = (h1.rolling(9).max() + l1.rolling(9).min()) / 2
    kijun = (h1.rolling(26).max() + l1.rolling(26).min()) / 2
    senkou_a = ((tenkan + kijun) / 2).shift(26)
    senkou_b = ((h1.rolling(52).max() + l1.rolling(52).min()) / 2).shift(26)
    chikou = c1.shift(26)
    to15 = lambda s: s.reindex(c.index, method="ffill")
    patch = {
        "ta_tenkan_sen": to15(tenkan), "ta_kijun_sen": to15(kijun),
        "ta_senkou_a": to15(senkou_a), "ta_senkou_b": to15(senkou_b),
        "ta_chikou_span": to15(chikou),
    }
    patch["ta_chikou_dist"] = (c - patch["ta_chikou_span"]) / c
    patch["ta_cloud_width"] = (patch["ta_senkou_a"] - patch["ta_senkou_b"]).abs() / c
    patch["ta_price_above_cloud"] = (
        c > pd.concat([patch["ta_senkou_a"], patch["ta_senkou_b"]], axis=1)
        .max(axis=1)).astype("int8")
    piv_h = h.shift(10).where(h.shift(10) >= h.rolling(21).max() - 1e-9)
    piv_l = l.shift(10).where(l.shift(10) <= l.rolling(21).min() + 1e-9)
    patch["ta_dist_last_swing_high"] = (c - piv_h.ffill()) / c
    patch["ta_dist_last_swing_low"] = (c - piv_l.ffill()) / c

    raw_local = hydrate(BASE / "Data/raw/cross_crypto/total_mcap/cross_crypto_daily.parquet")
    raw = pd.read_parquet(raw_local)
    raw.index = pd.to_datetime(raw.index, utc=True) + pd.Timedelta(days=1)
    grid = m01.index
    dfx = raw.reindex(grid, method="ffill")
    for col in dfx.columns:
        patch[col] = dfx[col]
    patch["cx_btc_dom_chg_1d"] = dfx["cx_btc_dominance"].diff(96)
    patch["cx_total_mcap_return_1d"] = dfx["cx_total_mcap"].pct_change(96)
    patch["cx_total2_return_1d"] = dfx["cx_total2_mcap"].pct_change(96)
    patch["cx_total3_return_1d"] = dfx["cx_total3_mcap"].pct_change(96)
    patch["cx_altcoin_idx_return"] = patch["cx_total3_return_1d"]
    patch["cx_altcoin_idx_vol"] = patch["cx_total3_return_1d"].rolling(96).std()
    btc_ret = np.log(c).diff(96)
    patch["cx_alt_outperf_btc"] = ((patch["cx_total3_return_1d"] - btc_ret) > 0).astype("int8")
    patch["cx_btc_dom_breakdown"] = ((btc_ret < 0) & (patch["cx_total3_return_1d"] > 0)).astype("int8")

    for col, vals in patch.items():
        if col in master.columns:
            master[col] = pd.Series(vals).reindex(master.index)
    # verificar
    s = master["ta_chikou_dist"].iloc[::8].astype(float).corr(FWD)
    s2 = master["cx_total_mcap_return_1d"].iloc[::8].astype(float).corr(FWD)
    print(f"firmas post-patch: chikou {s:+.3f} | cx {s2:+.3f}", flush=True)
    assert abs(s) < 0.5 and abs(s2) < 0.2
    master_causal_local = CACHE / "master_15m_CAUSAL_v4.parquet"
    master.to_parquet(master_causal_local, compression="snappy")
    del master
    gc.collect()
    print("master causal local escrito", flush=True)

# ============================== 3. D01-equiv (local)
print("\n=== 3. master_features (local) ===", flush=True)
master = pd.read_parquet(master_causal_local)
DROP = {"reg_depth_regime", "reg_dollar_strength", "cost_spread_cost",
        "cost_max_position_liq", "reg_rates_up", "reg_leverage_high",
        "reg_liq_risk", "cost_expected_return_net", "cost_trade_allowed",
        "cost_holding_time_expected", "qa_bad_tick_flag",
        "qa_dup_timestamp_flag", "qa_publication_delay", "qa_api_delay_sec"}
DROP |= {col for col in master.columns if col.startswith(("tgt_", "qa_", "cost_"))}
master = master.drop(columns=[col for col in DROP if col in master.columns])
for col in master.select_dtypes(include=["float64"]).columns:
    master[col] = master[col].astype("float32")
for col in master.select_dtypes(include=["int64"]).columns:
    master[col] = pd.to_numeric(master[col], downcast="integer")
if master.index.tz is not None:
    master.index = master.index.tz_convert("UTC").tz_localize(None)
master.index.name = "timestamp"
mf_local = CACHE / "master_features.parquet"
master.reset_index().to_parquet(mf_local, compression="snappy", index=False)
mc_local = CACHE / "master_close_only.parquet"
master[["ohlcv_btc_close"]].reset_index().to_parquet(mc_local, compression="snappy",
                                                     index=False)
print(f"master_features: {master.shape}", flush=True)

# ============================== 4. D04-equiv
print("\n=== 4. Entrenamiento Approach B causal ===", flush=True)
LEAK = {"exit_price", "exit_reason", "gross_return", "net_return",
        "time_to_exit", "label_win", "label_tp", "label_sl", "label_timeout",
        "ambiguous", "split"}
MAXTR = {"4h": None, "1h": 600_000, "15m": 1_000_000}
PARAMS = dict(max_depth=6, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.5, min_child_weight=20.0, tree_method="hist",
              nthread=-1, objective="binary:logistic", eval_metric="logloss",
              seed=42)
PRED_COLS = ["timestamp", "candidate_id", "timeframe", "side", "vol_decile",
             "tp_pct", "sl_pct", "p_break_even", "net_return", "label_win"]
schema = json.loads((Path(__file__).resolve().parents[1]
                     / "artifacts/schemas/feature_schema.json").read_text())


def load_split(tf, split, max_rows=None):
    lbl_local = hydrate(DIRM / f"datasets/labels_compact_{tf}.parquet")
    lbl = pd.read_parquet(lbl_local)
    lbl["timestamp"] = pd.to_datetime(lbl["timestamp"])
    if getattr(lbl["timestamp"].dt, "tz", None) is not None:
        lbl["timestamp"] = lbl["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    lbl = lbl[lbl["split"] == split].reset_index(drop=True)
    if max_rows and len(lbl) > max_rows:
        lbl = lbl.sample(n=max_rows, random_state=42)
    out = lbl.merge(master, left_on="timestamp", right_index=True, how="left")
    del lbl
    gc.collect()
    for col in out.select_dtypes(include=["float64"]).columns:
        out[col] = out[col].astype("float32")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def fcols(df):
    drop = LEAK | {"timestamp", "timeframe", "side", "split", "entry_price",
                   "direction_label", "candidate_id"}
    return [col for col in df.columns if col not in drop
            and not (df[col].dtype == "O" or str(df[col].dtype).startswith("string"))]


parts_p, parts_m, model_locals = [], [], {}
for tf in ["4h", "1h", "15m"]:
    t0 = time.time()
    tr = load_split(tf, "train", MAXTR[tf]).dropna(subset=["label_win"])
    fc = fcols(tr)
    assert fc == schema["features_in_order"], f"{tf}: orden features != schema"
    y_tr = tr["label_win"].astype(np.int8).values
    X_tr = np.ascontiguousarray(tr[fc].to_numpy(dtype=np.float32))
    del tr; gc.collect()
    va = load_split(tf, "val").dropna(subset=["label_win"])
    y_v = va["label_win"].astype(np.int8).values
    vmeta = va[PRED_COLS].copy()
    X_v = np.ascontiguousarray(va[fc].to_numpy(dtype=np.float32))
    del va; gc.collect()
    dtr = xgb.QuantileDMatrix(X_tr, label=y_tr, feature_names=fc); del X_tr; gc.collect()
    dv = xgb.QuantileDMatrix(X_v, label=y_v, ref=dtr, feature_names=fc); del X_v; gc.collect()
    bst = xgb.train(PARAMS, dtr, num_boost_round=200, evals=[(dv, "val")],
                    early_stopping_rounds=30, verbose_eval=False)
    bi = bst.best_iteration
    p_v = bst.predict(dv, iteration_range=(0, bi + 1))
    vmeta["p_win"] = p_v; vmeta["dataset"] = "val"
    del dtr, dv; gc.collect()
    te = load_split(tf, "test").dropna(subset=["label_win"])
    y_t = te["label_win"].astype(np.int8).values
    tmeta = te[PRED_COLS].copy()
    X_t = np.ascontiguousarray(te[fc].to_numpy(dtype=np.float32))
    del te; gc.collect()
    p_t = bst.predict(xgb.DMatrix(X_t, feature_names=fc), iteration_range=(0, bi + 1))
    tmeta["p_win"] = p_t; tmeta["dataset"] = "test"
    del X_t; gc.collect()
    mdl_local = CACHE / f"approach_B_xgb_{tf}.json"
    bst.save_model(str(mdl_local))
    model_locals[tf] = mdl_local
    parts_p.append(pd.concat([vmeta, tmeta], ignore_index=True))
    for ds, y_, p_ in [("val", y_v, p_v), ("test", y_t, p_t)]:
        parts_m.append({"approach": "B", "timeframe": tf, "dataset": ds,
                        "logloss": log_loss(y_, np.clip(p_, 1e-7, 1 - 1e-7)),
                        "auc": roc_auc_score(y_, p_), "n": len(y_)})
    print(f"{tf}: it={bi} {time.time()-t0:.0f}s AUCval={parts_m[-2]['auc']:.4f} "
          f"AUCtest={parts_m[-1]['auc']:.4f}", flush=True)
    del bst; gc.collect()

del master; gc.collect()
preds_B = pd.concat(parts_p, ignore_index=True)
metrics = pd.DataFrame(parts_m)
print(metrics.to_string(index=False), flush=True)

# ============================== 5. calibracion + signals + buckets (local)
print("\n=== 5. Calibracion + signals + buckets ===", flush=True)
val = preds_B[preds_B.dataset == "val"]
lr = LogisticRegression(C=1e10, solver="lbfgs")
lr.fit(val["p_win"].values.reshape(-1, 1), val["label_win"].astype(int).values)
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(val["p_win"].values, val["label_win"].astype(int).values)
calib = preds_B.copy()
calib["p_win_sigmoid"] = lr.predict_proba(calib["p_win"].values.reshape(-1, 1))[:, 1]
calib["p_win_isotonic"] = iso.predict(calib["p_win"].values)

COST = 0.0012
out = calib
tp, sl = out["tp_pct"].astype(float), out["sl_pct"].astype(float)
out["cost"] = COST
out["p_break_even_new"] = (sl + COST) / (tp + sl)
for col in ["p_win", "p_win_sigmoid", "p_win_isotonic"]:
    pw = out[col].astype(float)
    out[f"EV_pred_{col}"] = pw * tp - (1 - pw) * sl - COST
    out[f"edge_over_be_{col}"] = pw - out["p_break_even_new"]
out["p_win_main"] = out["p_win_isotonic"]
out["EV_pred"] = out["EV_pred_p_win_isotonic"]
out["edge_over_be"] = out["edge_over_be_p_win_isotonic"]
out["net_return_cost012"] = out["net_return"].astype(float) - (COST - 0.001)

BUCKETS = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0001]
LBL = [f"{a:.2f}-{b:.2f}" for a, b in zip(BUCKETS[:-1], BUCKETS[1:])]
bucket_results = {}
for ds in ["val", "test"]:
    sub = out[out.dataset == ds].copy()
    sub["bucket"] = pd.cut(sub["p_win_isotonic"], bins=BUCKETS, labels=LBL,
                           include_lowest=True)
    rows = []
    for (tf, bk), g in sub.groupby(["timeframe", "bucket"], observed=True):
        if len(g) < 30:
            continue
        t = g.loc[g.groupby("timestamp")["EV_pred"].idxmax()]
        nr = t["net_return_cost012"].values
        pos, neg = nr[nr > 0], -nr[nr < 0]
        rows.append({"timeframe": tf, "bucket": str(bk), "n_trades": len(t),
                     "win_rate": round((nr > 0).mean(), 3),
                     "expectancy": round(nr.mean(), 6),
                     "pf": round(pos.sum() / neg.sum(), 2) if neg.sum() > 0 else np.inf,
                     "total_return": round(nr.sum(), 2)})
    bucket_results[ds] = pd.DataFrame(rows)
    print(f"-- buckets {ds} --", flush=True)
    print(bucket_results[ds].to_string(index=False), flush=True)

# proporcion long/short con threshold 0.67
print("\n-- senales con p>=0.67 (test): mix de sides --", flush=True)
sub = out[(out.dataset == "test") & (out.p_win_isotonic >= 0.67)]
sel = sub.loc[sub.groupby(["timestamp", "timeframe"])["EV_pred"].idxmax()]
print(sel.groupby("timeframe")["side"].value_counts().to_string(), flush=True)

# ============================== 6. escribir TODO a Drive
print("\n=== 6. Escribiendo resultados a Drive ===", flush=True)
cols_keep = ["timestamp", "candidate_id", "timeframe", "side", "vol_decile",
             "tp_pct", "sl_pct", "cost", "p_win", "p_win_sigmoid",
             "p_win_isotonic", "p_break_even", "p_break_even_new",
             "p_win_main", "EV_pred", "edge_over_be",
             "EV_pred_p_win", "EV_pred_p_win_sigmoid", "EV_pred_p_win_isotonic",
             "edge_over_be_p_win", "edge_over_be_p_win_sigmoid",
             "edge_over_be_p_win_isotonic", "net_return", "label_win", "dataset"]
cols_keep = [c for c in cols_keep if c in out.columns]

writes = []
tmpdir = CACHE
def wlocal(df, name, **kw):
    p = tmpdir / name
    df.to_parquet(p, index=False, **kw) if name.endswith(".parquet") else df.to_csv(p, index=False)
    return p

writes.append((wlocal(preds_B, "preds_approach_B.parquet"), DIRM / "predictions/preds_approach_B.parquet"))
for tf in ["4h", "1h", "15m"]:
    sub = preds_B[preds_B.timeframe == tf]
    writes.append((wlocal(sub, f"preds_approach_B_{tf}.parquet"), DIRM / f"predictions/preds_approach_B_{tf}.parquet"))
    writes.append((model_locals[tf], DIRM / f"models/approach_B_xgb_{tf}.json"))
    m_tf = metrics[metrics.timeframe == tf]
    writes.append((wlocal(m_tf, f"model_metrics_B_{tf}.csv"), DIRM / f"metrics/model_metrics_B_{tf}.csv"))
writes.append((wlocal(metrics, "model_metrics_B.csv"), DIRM / "metrics/model_metrics_B.csv"))
writes.append((wlocal(calib[["timestamp", "candidate_id", "timeframe", "side",
                             "vol_decile", "tp_pct", "sl_pct", "p_break_even",
                             "net_return", "label_win", "dataset", "p_win",
                             "p_win_sigmoid", "p_win_isotonic"]],
                      "calib_approach_B.parquet"), DIRM / "calibration/calib_approach_B.parquet"))
writes.append((wlocal(out[out.dataset == "val"][cols_keep], "signals_val.parquet"), DIRRB / "signals/signals_val.parquet"))
writes.append((wlocal(out[out.dataset == "test"][cols_keep], "signals_test.parquet"), DIRRB / "signals/signals_test.parquet"))
for ds, df in bucket_results.items():
    writes.append((wlocal(df, f"buckets_causal_{ds}.csv"), DIRRB / f"threshold_grid/buckets_causal_{ds}.csv"))
writes.append((mf_local, DIRM / "datasets/master_features.parquet"))
writes.append((mc_local, DIRM / "datasets/master_close_only.parquet"))
writes.append((master_causal_local, MAST / "master_15m.parquet"))

for local, drive in writes:
    copy_back(Path(local), drive)
    print("  ->", drive.name, flush=True)

print("\nCADENA v4 COMPLETA OK", flush=True)
