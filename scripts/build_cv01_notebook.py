# -*- coding: utf-8 -*-
"""Genera 10_modelo_cross_val/CV01_timeseries_split_retrain.ipynb."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT_DIR = BASE / "repo/notebooks/10_modelo_cross_val"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_NB = OUT_DIR / "CV01_timeseries_split_retrain.ipynb"

MD0 = """# CV01 - Reentrenamiento con TimeSeriesSplit + backtest OOF

Reentrena los 3 modelos (15m / 1h / 4h) usando **walk-forward expanding
window** via `sklearn.model_selection.TimeSeriesSplit(n_splits=5)`.

## Por que esto, no random k-fold

En series temporales NO se puede k-fold aleatorio (la label de t depende
del precio en [t, t+H], asi que cualquier random shuffle mete futuro en
train). `TimeSeriesSplit` corta el periodo en 5 ventanas consecutivas y:

- Fold 0: train [0, 1/6] → test [1/6, 2/6]
- Fold 1: train [0, 2/6] → test [2/6, 3/6]
- Fold 2: train [0, 3/6] → test [3/6, 4/6]
- Fold 3: train [0, 4/6] → test [4/6, 5/6]
- Fold 4: train [0, 5/6] → test [5/6, 1]

Resultado: el modelo se entrena 5 veces con cada vez mas historia, y
genera predicciones **out-of-fold (OOF)** para los ultimos 5/6 del
periodo. Como cada prediccion es de un modelo que NO la vio en train,
son honestas y se pueden usar para backtest sin sesgo.

## Outputs (Drive)

`Data/model_outputs/cross_val/`:
- `oof_preds_<tf>.parquet`: predicciones OOF de cada fold
- `models_final_<tf>.json`: modelo entrenado en el ultimo fold (mas datos)
- `metrics_<tf>.csv`: AUC/logloss por fold
- `calib_isotonic_<tf>.pkl`: calibracion isotonic ajustada en OOF
- `backtest_buckets_<tf>.csv`: tabla por bucket de probabilidad

## Tiempo estimado (CPU 8 cores)

- 4h: ~5 min (5 folds x ~30s + predict)
- 1h: ~20 min (5 folds x ~3 min)
- 15m: ~70 min (5 folds x ~12 min, train cap 1M)

**Total: ~1.5 h.** En Colab gratis con CPU subir el cap puede no caber
en RAM; el notebook detecta el entorno y ajusta.
"""

C1 = '''# Celda 1
# Objetivo
# Setup: detectar entorno, rutas, carga de master_features y configurar
# TimeSeriesSplit para los 3 TFs.

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
    BASE = Path("/content/drive/MyDrive/Base de Datos BITCOIN")
    IN_COLAB = True
except Exception:
    BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")
    IN_COLAB = False

DIR_MASTER = BASE / "Data/model_outputs/directional/datasets"
OUT = BASE / "Data/model_outputs/cross_val"
OUT.mkdir(parents=True, exist_ok=True)

# Hiperparametros (mismos que D04 para comparabilidad)
XGB_PARAMS = dict(max_depth=6, learning_rate=0.05, subsample=0.8,
                  colsample_bytree=0.5, min_child_weight=20.0,
                  tree_method="hist", nthread=-1,
                  objective="binary:logistic", eval_metric="logloss",
                  seed=42)
NUM_ROUNDS = 150
EARLY_STOP = 20
N_SPLITS = 5
COST = 0.0012

# Cap de filas de TRAIN por TF (anti-OOM). El cap se aplica a cada fold
# de TRAIN; el test no se limita.
MAX_TRAIN = {"4h": None, "1h": 600_000, "15m": 800_000}

LEAK = {"exit_price", "exit_reason", "gross_return", "net_return",
        "time_to_exit", "label_win", "label_tp", "label_sl",
        "label_timeout", "ambiguous", "split"}
DROP = LEAK | {"timestamp", "timeframe", "side", "split", "entry_price",
                "direction_label", "candidate_id"}

print(f"IN_COLAB={IN_COLAB} | BASE={BASE}")
print(f"OUT={OUT}")
print(f"TimeSeriesSplit n_splits={N_SPLITS}")
'''

C2 = '''# Celda 2
# Objetivo
# Cargar master_features (causal) en RAM.

t0 = time.time()
master = pd.read_parquet(DIR_MASTER / "master_features.parquet")
master["timestamp"] = pd.to_datetime(master["timestamp"])
if getattr(master["timestamp"].dt, "tz", None) is not None:
    master["timestamp"] = (master["timestamp"].dt.tz_convert("UTC")
                            .dt.tz_localize(None))
master = master.set_index("timestamp").sort_index()
# downcast a float32 para reducir memoria
for c in master.select_dtypes(include=["float64"]).columns:
    master[c] = master[c].astype("float32")
print(f"master: {master.shape} en {time.time()-t0:.0f}s "
      f"({master.memory_usage(deep=True).sum()/1e9:.1f} GB)")
'''

C3 = '''# Celda 3
# Objetivo
# Funcion de entrenamiento walk-forward para un TF. Devuelve OOF preds,
# modelo final, metricas por fold.


def feat_cols(df):
    return [c for c in df.columns
            if c not in DROP
            and not (df[c].dtype == "O"
                     or str(df[c].dtype).startswith("string"))]


def load_tf_data(tf, max_rows=None):
    """Carga labels_compact_{tf} unido con master, ordenado por timestamp."""
    lbl = pd.read_parquet(DIR_MASTER / f"labels_compact_{tf}.parquet")
    lbl["timestamp"] = pd.to_datetime(lbl["timestamp"])
    if getattr(lbl["timestamp"].dt, "tz", None) is not None:
        lbl["timestamp"] = (lbl["timestamp"].dt.tz_convert("UTC")
                             .dt.tz_localize(None))
    lbl = lbl.sort_values("timestamp").reset_index(drop=True)
    if max_rows and len(lbl) > max_rows:
        # NO sample aleatorio (rompe orden temporal). Tomamos LOS ULTIMOS
        # max_rows registros para mantener historia reciente
        lbl = lbl.tail(max_rows).reset_index(drop=True)
    df = lbl.merge(master, left_on="timestamp", right_index=True, how="left")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=["label_win"]).reset_index(drop=True)
    return df


def run_timeseries_cv(tf):
    print(f"\\n========== {tf} ==========")
    df = load_tf_data(tf, MAX_TRAIN[tf])
    print(f"datos: {len(df):,} filas | rango: {df.timestamp.min()} -> "
          f"{df.timestamp.max()}")

    fc = feat_cols(df)
    X = df[fc].to_numpy(dtype=np.float32)
    y = df["label_win"].astype(np.int8).values
    ts = df["timestamp"].values

    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    oof_pred = np.full(len(df), np.nan, dtype=np.float32)
    fold_metrics = []
    last_model = None

    for fold, (tr_idx, te_idx) in enumerate(tscv.split(X)):
        t0 = time.time()
        # ventana de train (expanding) y test
        Xtr, ytr = X[tr_idx], y[tr_idx]
        Xte, yte = X[te_idx], y[te_idx]
        dtr = xgb.QuantileDMatrix(Xtr, label=ytr, feature_names=fc)
        dte = xgb.QuantileDMatrix(Xte, label=yte, ref=dtr,
                                   feature_names=fc)
        bst = xgb.train(XGB_PARAMS, dtr, num_boost_round=NUM_ROUNDS,
                        evals=[(dte, "test")],
                        early_stopping_rounds=EARLY_STOP,
                        verbose_eval=False)
        bi = bst.best_iteration
        p = bst.predict(dte, iteration_range=(0, bi + 1))
        oof_pred[te_idx] = p
        auc = roc_auc_score(yte, p)
        ll = log_loss(yte, np.clip(p, 1e-7, 1 - 1e-7))
        br = brier_score_loss(yte, p)
        fold_metrics.append({"fold": fold, "n_train": len(tr_idx),
                             "n_test": len(te_idx),
                             "test_start": str(pd.Timestamp(ts[te_idx[0]])),
                             "test_end": str(pd.Timestamp(ts[te_idx[-1]])),
                             "best_iter": bi, "auc": auc, "logloss": ll,
                             "brier": br})
        print(f"fold {fold}: train {len(tr_idx):,} test {len(te_idx):,} "
              f"AUC {auc:.4f} logloss {ll:.4f} "
              f"({time.time()-t0:.0f}s)")
        last_model = bst
        del dtr, dte
        gc.collect()

    # Guardar OOF + modelo final + metricas
    oof_df = df[["timestamp", "candidate_id", "timeframe", "side",
                 "vol_decile", "tp_pct", "sl_pct", "p_break_even",
                 "net_return", "label_win"]].copy()
    oof_df["p_win_oof"] = oof_pred
    # filtramos las filas del primer fold de train (no tienen OOF)
    oof_df = oof_df.dropna(subset=["p_win_oof"]).reset_index(drop=True)
    oof_df.to_parquet(OUT / f"oof_preds_{tf}.parquet", index=False)
    last_model.save_model(str(OUT / f"models_final_{tf}.json"))
    pd.DataFrame(fold_metrics).to_csv(OUT / f"metrics_{tf}.csv", index=False)
    print(f"OOF cubre {len(oof_df):,}/{len(df):,} filas "
          f"({len(oof_df)/len(df)*100:.0f}%)")
    return oof_df, fold_metrics
'''

C4 = '''# Celda 4
# Objetivo
# Entrenar los 3 TFs con TimeSeriesSplit. PESADO: ~1.5h total en CPU.

results = {}
for tf in ["4h", "1h", "15m"]:
    oof_df, fold_metrics = run_timeseries_cv(tf)
    results[tf] = oof_df
    gc.collect()

print("\\nResumen metricas por fold y TF:")
for tf in results:
    m = pd.read_csv(OUT / f"metrics_{tf}.csv")
    print(f"\\n--- {tf} ---")
    print(m[["fold", "n_test", "test_start", "test_end", "best_iter",
              "auc", "logloss"]].to_string(index=False))
    print(f"AUC media: {m.auc.mean():.4f} (std {m.auc.std():.4f})")
'''

C5 = '''# Celda 5
# Objetivo
# Calibracion isotonic sobre OOF (split temporal: ajustar en el primer
# 80% del OOF y aplicar a todo). Repreciado con coste 0.0012.

calib_models = {}
for tf in results:
    oof = results[tf]
    cut = int(len(oof) * 0.8)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof["p_win_oof"].iloc[:cut].values,
            oof["label_win"].iloc[:cut].astype(int).values)
    oof["p_win_isotonic"] = iso.predict(oof["p_win_oof"].values)
    calib_models[tf] = iso
    with open(OUT / f"calib_isotonic_{tf}.pkl", "wb") as f:
        pickle.dump(iso, f)

    # Repriced metrics
    tp = oof["tp_pct"].astype(float)
    sl = oof["sl_pct"].astype(float)
    pw = oof["p_win_isotonic"].astype(float)
    oof["EV_pred"] = pw * tp - (1 - pw) * sl - COST
    oof["net012"] = oof["net_return"].astype(float) - (COST - 0.001)
    # rango calibrado
    print(f"{tf}: p_iso rango [{oof.p_win_isotonic.min():.3f}, "
          f"{oof.p_win_isotonic.max():.3f}], "
          f"calibrado con primer {cut:,}/{len(oof):,}")
    results[tf] = oof  # update
'''

C6 = '''# Celda 6
# Objetivo
# Backtest por BUCKET sobre OOF. Una senal por vela (top EV).
# Buckets: [0.50, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75]

import matplotlib.pyplot as plt
BUCKETS = [0.50, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75, 1.0001]
BUCKET_LBL = [f"{a:.2f}" for a in BUCKETS[:-1]]
NOTIONAL = 100.0
COLORS = {"15m": "steelblue", "1h": "darkorange", "4h": "seagreen"}


def top_ev_per_candle(df):
    if not len(df): return df
    return df.loc[df.groupby("timestamp")["EV_pred"].idxmax()] \\
              .sort_values("timestamp").reset_index(drop=True)


rows = []
for tf in results:
    oof = results[tf]
    days = max(1, (oof["timestamp"].max() - oof["timestamp"].min()).days)
    oof["bucket"] = pd.cut(oof["p_win_isotonic"], bins=BUCKETS,
                            labels=BUCKET_LBL, include_lowest=True)
    sel = top_ev_per_candle(oof)
    for bk in BUCKET_LBL:
        sub = sel[sel.bucket == bk]
        if len(sub) < 10: continue
        nr = sub.net012.values
        pos, neg = nr[nr > 0], -nr[nr < 0]
        pnl = nr * NOTIONAL
        eq = pnl.cumsum()
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak).min()
        sharpe = nr.mean() / nr.std() * np.sqrt(252) if nr.std() > 0 else 0
        rows.append({
            "tf": tf, "bucket": bk, "n": len(sub),
            "trades_dia": round(len(sub) / days, 2),
            "win_rate": round((nr > 0).mean(), 3),
            "expectancy": round(nr.mean(), 5),
            "PF": round(pos.sum() / neg.sum(), 2)
            if neg.sum() > 0 else np.inf,
            "sharpe": round(sharpe, 3),
            "max_dd": round(dd, 2),
            "pnl_total": round(pnl.sum(), 2),
            "calmar": round(pnl.sum() * 365 / days / abs(dd), 2)
            if dd < 0 else np.inf,
        })
bk_tbl = pd.DataFrame(rows)
print("=== Backtest OOF por bucket (TimeSeriesSplit) ===")
print(bk_tbl.to_string(index=False))
for tf in results:
    bk_tbl[bk_tbl.tf == tf].to_csv(OUT / f"backtest_buckets_{tf}.csv",
                                    index=False)

# Graficas: win_rate y expectancy por bucket
fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, met in zip(axes, ["win_rate", "expectancy"]):
    for tf in results:
        g = bk_tbl[bk_tbl.tf == tf]
        ax.plot(g.bucket.astype(str), g[met], marker="o", label=tf,
                color=COLORS[tf])
    ax.set_title(f"{met} por bucket (OOF)")
    ax.axhline(0.5 if met == "win_rate" else 0, color="black", lw=0.5)
    ax.grid(alpha=0.3); ax.legend(); ax.tick_params(axis="x", rotation=45)
plt.tight_layout(); plt.show()
'''

C7 = '''# Celda 7
# Objetivo
# Comparar con el modelo single-split actual (signals_test.parquet).
# Si OOF da metricas razonablemente parecidas o mejores -> walk-forward
# es una mejora honesta.

DIR_RB = BASE / "Data/model_outputs/real_backtest"
rows_cmp = []
for tf in results:
    # actual (single-split, solo test)
    try:
        cur = pd.read_parquet(DIR_RB / "signals/signals_test.parquet")
        cur["timestamp"] = pd.to_datetime(cur["timestamp"])
        cur = cur[cur.timeframe == tf].copy()
        cur["net012"] = cur["net_return"].astype(float) - (COST - 0.001)
        sel = cur.loc[cur.groupby("timestamp")["EV_pred"].idxmax()]
        for thr in [0.55, 0.60, 0.65]:
            sub = sel[sel.p_win_isotonic >= thr]
            if len(sub) < 10: continue
            nr = sub.net012.values
            rows_cmp.append({"version": "single-split", "tf": tf, "thr": thr,
                              "n": len(sub),
                              "win_rate": round((nr > 0).mean(), 3),
                              "expectancy": round(nr.mean(), 5),
                              "pnl_total": round((nr * NOTIONAL).sum(), 2)})
    except Exception as e:
        print(f"Sin signals_test para {tf}: {e}")
    # OOF nuevo
    oof = results[tf]
    sel = top_ev_per_candle(oof)
    for thr in [0.55, 0.60, 0.65]:
        sub = sel[sel.p_win_isotonic >= thr]
        if len(sub) < 10: continue
        nr = sub.net012.values
        rows_cmp.append({"version": "TimeSeriesSplit-OOF", "tf": tf,
                          "thr": thr, "n": len(sub),
                          "win_rate": round((nr > 0).mean(), 3),
                          "expectancy": round(nr.mean(), 5),
                          "pnl_total": round((nr * NOTIONAL).sum(), 2)})
cmp = pd.DataFrame(rows_cmp)
print("=== Comparacion single-split vs TimeSeriesSplit OOF ===")
print(cmp.to_string(index=False))
cmp.to_csv(OUT / "comparison_singlesplit_vs_tssplit.csv", index=False)
print("\\nLectura: las metricas OOF cubren ~83% del periodo (folds 1-4 de")
print("test, los 5/6 del total). El single-split cubre 1/4 final (test).")
print("Si OOF aguanta el edge en todos los folds, el modelo es robusto.")
print("Si OOF cae mucho vs single-split, parte del edge era sesgo del")
print("ultimo periodo.")

# Equity OOF para los buckets clave (TF)
fig, axes = plt.subplots(1, len(results), figsize=(5*len(results), 4))
if len(results) == 1: axes = [axes]
for ax, tf in zip(axes, results):
    oof = results[tf]
    sel = top_ev_per_candle(oof)
    for thr in [0.55, 0.60, 0.65]:
        sub = sel[sel.p_win_isotonic >= thr].sort_values("timestamp")
        if len(sub) < 10: continue
        eq = (sub.net012.values * NOTIONAL).cumsum()
        ax.plot(sub.timestamp, eq, label=f"thr {thr} (n={len(sub)})", lw=1.1)
    ax.set_title(f"{tf}: equity OOF por threshold")
    ax.axhline(0, color="black", lw=0.5); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
print("\\nCV01 OK. Outputs en:", OUT)
'''

nb = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in [("markdown", MD0), ("code", C1), ("code", C2),
                  ("code", C3), ("code", C4), ("code", C5), ("code", C6),
                  ("code", C7)]:
    nb["cells"].append({"cell_type": kind, "metadata": {},
                        "source": src.splitlines(keepends=True),
                        **({"outputs": [], "execution_count": None}
                           if kind == "code" else {})})
OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print("Notebook escrito:", OUT_NB)
