# -*- coding: utf-8 -*-
"""Genera RB16_reporte_modelo_causal.ipynb (Drive, Colab/local).

Reporte completo por BUCKET DE PROBABILIDAD del modelo causal sin leakage.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT_NB = BASE / "repo/notebooks/08_real_backtest/RB16_reporte_modelo_causal.ipynb"

MD0 = """# RB16 - Reporte del modelo CAUSAL (sin leakage)

Reporte completo del modelo Approach B reentrenado el 2026-06-12 sin
leakage. Todo desglosado por **bucket de probabilidad calibrada** y por
**timeframe**, en train / val / test.

Semantica: 1 senal por vela (top EV), coste 0.0012, notional fijo 100 EUR.
AUC test ~0.547 en los 3 TFs. p_iso maximo: 15m 0.613 | 1h 0.678 | 4h 1.0.
"""

C1 = '''# Celda 1
# Objetivo
# Setup + carga de signals (train/val/test) + duraciones reales.

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
    BASE = Path("/content/drive/MyDrive/Base de Datos BITCOIN")
except Exception:
    BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")

DIR_RB = BASE / "Data/model_outputs/real_backtest"
DIR_LBL = BASE / "Data/model_outputs/directional/datasets"
COST = 0.0012
NOTIONAL = 100.0
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}
TFS = ["15m", "1h", "4h"]
SPLITS = ["train", "val", "test"]
BUCKETS = [0.50, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75, 0.80, 1.0001]
BUCKET_LBL = [f"{a:.2f}" for a in BUCKETS[:-1]]
COLORS = {"15m": "steelblue", "1h": "darkorange", "4h": "seagreen"}

sig = {}
for ds in SPLITS:
    p = DIR_RB / f"signals/signals_{ds}.parquet"
    if not p.exists():
        print(f"AVISO: falta {p.name}"); continue
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sig[ds] = df

lk = []
for tf in TFS:
    h = pd.read_parquet(DIR_LBL / f"labels_compact_{tf}.parquet",
                        columns=["timestamp", "candidate_id", "timeframe",
                                 "H", "time_to_exit", "exit_reason"])
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    if getattr(h["timestamp"].dt, "tz", None) is not None:
        h["timestamp"] = h["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    lk.append(h)
lk = pd.concat(lk, ignore_index=True)
for ds in sig:
    sig[ds] = sig[ds].merge(lk, on=["timestamp", "candidate_id", "timeframe"],
                            how="left")
    sig[ds]["net012"] = sig[ds]["net_return"].astype(float) - (COST - 0.001)
    sig[ds]["dur_min"] = (sig[ds]["timeframe"].map(TF_MIN)
                          * sig[ds]["time_to_exit"].fillna(sig[ds]["H"]))
    sig[ds]["bucket"] = pd.cut(sig[ds]["p_win_isotonic"], bins=BUCKETS,
                                labels=BUCKET_LBL, include_lowest=True)


def top_ev_per_candle(df):
    """1 senal por (timestamp, tf): la de mayor EV. Asi es como opera el bot."""
    if not len(df):
        return df
    return (df.loc[df.groupby(["timestamp", "timeframe"])["EV_pred"].idxmax()]
              .sort_values("timestamp").reset_index(drop=True))


def metrics(nr, dur_min, days):
    """Win rate, expectancy, PF, Sharpe (por trade), Calmar (anual/|maxDD|),
    maxDD, PnL/dia (EUR sobre 100 notional), trades_dia/mes/anyo."""
    if len(nr) < 5:
        return None
    pnl = nr * NOTIONAL
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    max_dd = float(dd.min())
    pos = pnl[pnl > 0]; neg = -pnl[pnl < 0]
    pf = pos.sum() / neg.sum() if neg.sum() > 0 else np.inf
    sharpe = nr.mean() / nr.std() * np.sqrt(252) if nr.std() > 0 else 0
    n_per_day = len(nr) / max(days, 1)
    pnl_per_day = pnl.sum() / max(days, 1)
    anual = pnl_per_day * 365
    calmar = anual / abs(max_dd) if max_dd < 0 else np.inf
    return {
        "n": int(len(nr)),
        "win_rate": float((nr > 0).mean()),
        "expectancy": float(nr.mean()),
        "PF": float(pf),
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "max_dd_eur": max_dd,
        "pnl_total": float(pnl.sum()),
        "pnl_dia": float(pnl_per_day),
        "trades_dia": float(n_per_day),
        "trades_mes": float(n_per_day * 30),
        "trades_anyo": float(n_per_day * 365),
    }


print("Splits cargados:", list(sig.keys()))
for ds in sig:
    print(f"  {ds}: {len(sig[ds]):,} senales raw")
'''

C2 = '''# Celda 2
# Objetivo
# Tabla maestra: metricas por (split, tf, bucket). UNA senal por vela.

rows = []
for ds in sig:
    days = max(1, (sig[ds]["timestamp"].max() - sig[ds]["timestamp"].min()).days)
    sel = top_ev_per_candle(sig[ds])
    for tf in TFS:
        for bk in BUCKET_LBL:
            sub = sel[(sel.timeframe == tf) & (sel.bucket == bk)]
            m = metrics(sub.net012.values, sub.dur_min.values, days)
            if m is None: continue
            m.update({"split": ds, "tf": tf, "bucket": bk})
            rows.append(m)
master_tbl = pd.DataFrame(rows)
# orden + redondeo
ord_cols = ["split", "tf", "bucket", "n", "trades_dia", "trades_mes",
            "trades_anyo", "win_rate", "expectancy", "PF", "sharpe",
            "calmar", "max_dd_eur", "pnl_total", "pnl_dia"]
master_tbl = master_tbl[ord_cols]
for c in ["win_rate", "expectancy", "PF", "sharpe", "calmar"]:
    master_tbl[c] = master_tbl[c].round(3)
for c in ["max_dd_eur", "pnl_total", "pnl_dia"]:
    master_tbl[c] = master_tbl[c].round(2)
for c in ["trades_dia", "trades_mes", "trades_anyo"]:
    master_tbl[c] = master_tbl[c].round(1)

for ds in sig:
    print(f"\\n=== {ds.upper()} ===")
    print(master_tbl[master_tbl.split == ds].to_string(index=False))
'''

C3 = '''# Celda 3
# Objetivo
# Win rate por BUCKET (subplots TF x split). La linea horizontal 0.5
# es el umbral de paridad.

fig, axes = plt.subplots(len(TFS), len(sig), figsize=(4.5*len(sig), 3*len(TFS)),
                         sharey=True)
if len(TFS) == 1: axes = np.array([axes])
if len(sig) == 1: axes = axes.reshape(-1, 1)
for i, tf in enumerate(TFS):
    for j, ds in enumerate(sig.keys()):
        ax = axes[i][j]
        g = master_tbl[(master_tbl.tf == tf) & (master_tbl.split == ds)]
        ax.bar(g.bucket.astype(str), g.win_rate, color=COLORS[tf])
        ax.axhline(0.5, color="black", lw=0.5)
        ax.set_title(f"{tf} | {ds}")
        ax.set_xlabel("bucket p_iso"); ax.tick_params(axis="x", rotation=45)
        if j == 0: ax.set_ylabel("win rate")
        ax.set_ylim(0.3, 1.0)
        ax.grid(alpha=0.3)
plt.suptitle("Win rate por bucket de probabilidad", y=1.01)
plt.tight_layout(); plt.show()
'''

C4 = '''# Celda 4
# Objetivo
# Win rate por DECIL DE VOLATILIDAD (subplots TF x split). Sin filtro de
# threshold (todas las senales).

fig, axes = plt.subplots(len(TFS), len(sig), figsize=(4.5*len(sig), 3*len(TFS)),
                         sharey=True)
if len(TFS) == 1: axes = np.array([axes])
if len(sig) == 1: axes = axes.reshape(-1, 1)
for i, tf in enumerate(TFS):
    for j, ds in enumerate(sig.keys()):
        ax = axes[i][j]
        sel = top_ev_per_candle(sig[ds])
        sub = sel[sel.timeframe == tf]
        rows_d = []
        for dec, g in sub.groupby("vol_decile"):
            if len(g) < 30: continue
            rows_d.append({"decil": int(dec),
                           "win_rate": (g.net012 > 0).mean(),
                           "n": len(g)})
        d = pd.DataFrame(rows_d)
        if len(d):
            ax.bar(d.decil, d.win_rate, color=COLORS[tf])
        ax.axhline(0.5, color="black", lw=0.5)
        ax.set_title(f"{tf} | {ds}")
        ax.set_xlabel("decil vol"); ax.set_ylim(0.3, 1.0)
        if j == 0: ax.set_ylabel("win rate")
        ax.grid(alpha=0.3)
plt.suptitle("Win rate por decil de volatilidad", y=1.01)
plt.tight_layout(); plt.show()
'''

C5 = '''# Celda 5
# Objetivo
# Metricas clave por bucket (TEST): expectancy, Sharpe, Calmar, max DD
# en 4 paneles, una linea por TF.

fig, axes = plt.subplots(2, 2, figsize=(13, 7))
panels = [("expectancy", "Expectancy por trade"),
          ("sharpe", "Sharpe (por trade, anualizado)"),
          ("calmar", "Calmar (retorno anual / |max DD|)"),
          ("max_dd_eur", "Max drawdown EUR (notional 100)")]
for ax, (col, title) in zip(axes.flat, panels):
    for tf in TFS:
        g = master_tbl[(master_tbl.split == "test") & (master_tbl.tf == tf)]
        v = g[col].replace([np.inf, -np.inf], np.nan)
        ax.plot(g.bucket.astype(str), v, marker="o", label=tf,
                color=COLORS[tf])
    ax.set_title(title); ax.set_xlabel("bucket p_iso")
    ax.tick_params(axis="x", rotation=45)
    ax.axhline(0, color="black", lw=0.5); ax.grid(alpha=0.3); ax.legend()
plt.suptitle("Metricas por bucket (TEST)", y=1.00)
plt.tight_layout(); plt.show()

# Tabla TEST resumen
print("\\nTabla TEST resumen (para anotar):")
print(master_tbl[master_tbl.split == "test"][[
    "tf", "bucket", "n", "trades_dia", "trades_anyo", "win_rate",
    "expectancy", "PF", "sharpe", "calmar", "max_dd_eur", "pnl_total"
]].to_string(index=False))
'''

C6 = '''# Celda 6
# Objetivo
# Trades por dia / mes / anyo por bucket (TEST). Util para ver actividad.

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, met in zip(axes, ["trades_dia", "trades_mes", "trades_anyo"]):
    for tf in TFS:
        g = master_tbl[(master_tbl.split == "test") & (master_tbl.tf == tf)]
        ax.plot(g.bucket.astype(str), g[met], marker="o", label=tf,
                color=COLORS[tf])
    ax.set_title(met); ax.set_xlabel("bucket p_iso")
    ax.tick_params(axis="x", rotation=45); ax.set_yscale("log")
    ax.grid(alpha=0.3); ax.legend()
plt.suptitle("Frecuencia de trades por bucket (TEST, escala log)", y=1.01)
plt.tight_layout(); plt.show()
'''

C7 = '''# Celda 7
# Objetivo
# Apalancamiento: cuanto aumenta el PnL y el drawdown con L = 1..10.
# Con notional fijo es LINEAL en L, asi que sale a partir de las cifras
# base. Tambien graficas equity y drawdown para L = 1, 3, 5, 10.

L_VALUES = list(range(1, 11))
FUNDING_8H = 0.0001  # ~0.01% cada 8h en BTC perpetual

# tabla escalada por bucket TEST mejor de cada TF
print("=== Apalancamiento (TEST) - PnL anual y max DD por L ===")
print(f"{'TF':>4} {'bucket':>8} {'L':>3}  PnL/anyo  funding/anyo  netPnL/anyo  maxDD_EUR  Calmar")
sel_test = top_ev_per_candle(sig["test"]) if "test" in sig else None
days_test = max(1, (sig["test"]["timestamp"].max()
                    - sig["test"]["timestamp"].min()).days)

def stats_with_L(sub, days, L):
    pnl_unit = sub.net012.values * NOTIONAL
    # funding por trade: notional * L * (dur_h / 8) * funding_8h
    fund_unit = sub.dur_min.values / 60 / 8 * FUNDING_8H * NOTIONAL
    pnl = pnl_unit * L - fund_unit * L
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return {
        "pnl_total": pnl.sum(),
        "pnl_anyo": pnl.sum() * 365 / days,
        "fund_anyo": -fund_unit.sum() * L * 365 / days,
        "max_dd": dd.min(),
        "calmar": (pnl.sum() * 365 / days) / abs(dd.min()) if dd.min() < 0 else np.inf,
        "eq": eq, "dd": dd,
    }

# elegir mejor bucket por TF (mayor Calmar test)
best = {}
for tf in TFS:
    g = master_tbl[(master_tbl.split == "test") & (master_tbl.tf == tf)
                   & (master_tbl.n >= 20)]
    if not len(g): continue
    g = g[~g.calmar.isin([np.inf])]
    if not len(g): continue
    best[tf] = g.sort_values("calmar", ascending=False).iloc[0].bucket

print()
for tf, bk in best.items():
    sub = sel_test[(sel_test.timeframe == tf) & (sel_test.bucket == bk)]
    for L in L_VALUES:
        s = stats_with_L(sub, days_test, L)
        print(f"{tf:>4} {str(bk):>8} {L:>3}  {s['pnl_anyo']:+8.1f}  "
              f"{s['fund_anyo']:+8.2f}  {s['pnl_anyo']+s['fund_anyo']:+8.1f}  "
              f"{s['max_dd']:+8.1f}  {s['calmar']:.2f}")
    print()

# graficas equity + drawdown apalancado (mejor bucket por TF)
fig, axes = plt.subplots(2, len(best), figsize=(5*len(best), 7), sharex="col")
for k, (tf, bk) in enumerate(best.items()):
    sub = sel_test[(sel_test.timeframe == tf) & (sel_test.bucket == bk)]
    for L in [1, 3, 5, 10]:
        s = stats_with_L(sub, days_test, L)
        axes[0][k].plot(sub.timestamp, s["eq"], label=f"L={L}", lw=1.0)
        axes[1][k].plot(sub.timestamp, s["dd"], label=f"L={L}", lw=1.0)
    axes[0][k].set_title(f"{tf} bucket {bk}: Equity")
    axes[1][k].set_title(f"{tf} bucket {bk}: Drawdown")
    axes[0][k].grid(alpha=0.3); axes[1][k].grid(alpha=0.3)
    axes[0][k].axhline(0, color="black", lw=0.5)
    axes[0][k].legend(fontsize=8); axes[1][k].legend(fontsize=8)
plt.suptitle("Equity y Drawdown apalancados (TEST, sobre 100 EUR base)",
             y=1.00)
plt.tight_layout(); plt.show()

# Resumen: ratio retorno/riesgo NO cambia con L (escalan lineal). Es UN
# numero por (tf, bucket).
print("\\nRatio retorno/riesgo (anual/|maxDD|) - constante en L, util para")
print("comparar configuraciones:")
print(f"{'TF':>4} {'bucket':>8}  ratio")
for tf, bk in best.items():
    sub = sel_test[(sel_test.timeframe == tf) & (sel_test.bucket == bk)]
    s = stats_with_L(sub, days_test, 1)
    print(f"{tf:>4} {str(bk):>8}  {s['calmar']:.2f}")
'''

C8 = '''# Celda 8
# Objetivo
# Win rate / PnL / Sharpe / Drawdown por REGIMEN DE TENDENCIA, por bucket.
# Regimen calculado con EMA50 vs EMA200 sobre close 1h del master 15m
# (alcista / bajista / lateral).

mc = pd.read_parquet(DIR_LBL / "master_close_only.parquet")
mc["timestamp"] = pd.to_datetime(mc["timestamp"])
if getattr(mc["timestamp"].dt, "tz", None) is not None:
    mc["timestamp"] = mc["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
mc = mc.set_index("timestamp").sort_index()
h1 = mc["ohlcv_btc_close"].resample("1h").last().dropna()
e50 = h1.ewm(span=50, adjust=False).mean()
e200 = h1.ewm(span=200, adjust=False).mean()
regime = pd.Series("lateral", index=h1.index, name="regimen")
regime[(h1 > e200) & (e50 > e200)] = "alcista"
regime[(h1 < e200) & (e50 < e200)] = "bajista"
reg_df = regime.reset_index()

def annotate_regime(df):
    df = df.sort_values("timestamp")
    return pd.merge_asof(df, reg_df, on="timestamp", direction="backward")

# Tabla por (tf, regimen, bucket) en TEST
sel = annotate_regime(top_ev_per_candle(sig["test"]))
days_test = max(1, (sig["test"]["timestamp"].max()
                    - sig["test"]["timestamp"].min()).days)
rows = []
for tf in TFS:
    for reg in ["alcista", "bajista", "lateral"]:
        for bk in BUCKET_LBL:
            sub = sel[(sel.timeframe == tf) & (sel.regimen == reg)
                       & (sel.bucket == bk)]
            if len(sub) < 10: continue
            m = metrics(sub.net012.values, sub.dur_min.values, days_test)
            if m is None: continue
            m.update({"tf": tf, "regimen": reg, "bucket": bk})
            rows.append(m)
trend_tbl = pd.DataFrame(rows)
print("=== Por regimen de tendencia (TEST) ===")
print(trend_tbl[["tf", "regimen", "bucket", "n", "win_rate", "expectancy",
                  "sharpe", "max_dd_eur", "pnl_total"]].round(3).to_string(index=False))

# Heatmap win_rate (filas TF, columnas regimen, valor: media ponderada por bucket)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, met, title in zip(axes,
                          ["win_rate", "expectancy", "pnl_total"],
                          ["Win rate", "Expectancy", "PnL total EUR"]):
    for tf in TFS:
        for j, reg in enumerate(["alcista", "lateral", "bajista"]):
            g = trend_tbl[(trend_tbl.tf == tf) & (trend_tbl.regimen == reg)]
            v = (g[met] * g.n).sum() / g.n.sum() if g.n.sum() else np.nan
            ax.bar(f"{reg}\\n{tf}", v, color=COLORS[tf])
    ax.set_title(title); ax.tick_params(axis="x", rotation=45)
    if met == "win_rate": ax.axhline(0.5, color="black", lw=0.5)
    else: ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3)
plt.suptitle("Por regimen (medias ponderadas por bucket, TEST)", y=1.01)
plt.tight_layout(); plt.show()

# Equity por regimen y TF (mejor bucket de cada TF)
fig, axes = plt.subplots(1, len(best), figsize=(5*len(best), 4))
if len(best) == 1: axes = [axes]
for ax, (tf, bk) in zip(axes, best.items()):
    sub_tf = sel[(sel.timeframe == tf) & (sel.bucket == bk)]
    for reg, color in [("alcista", "seagreen"), ("lateral", "grey"),
                        ("bajista", "firebrick")]:
        g = sub_tf[sub_tf.regimen == reg].sort_values("timestamp")
        if len(g) < 5: continue
        eq = (g.net012.values * NOTIONAL).cumsum()
        ax.plot(g.timestamp, eq, label=f"{reg} (n={len(g)})", color=color,
                lw=1.1)
    ax.set_title(f"{tf} bucket {bk}: PnL acumulado por regimen")
    ax.axhline(0, color="black", lw=0.5); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()

print("\\nRB16 OK")
'''

nb = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in [("markdown", MD0), ("code", C1), ("code", C2),
                  ("code", C3), ("code", C4), ("code", C5), ("code", C6),
                  ("code", C7), ("code", C8)]:
    nb["cells"].append({"cell_type": kind, "metadata": {},
                        "source": src.splitlines(keepends=True),
                        **({"outputs": [], "execution_count": None}
                           if kind == "code" else {})})
OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print("Notebook escrito:", OUT_NB)
