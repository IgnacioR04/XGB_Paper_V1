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

## Semantica importante (leer antes de los numeros)

**Buckets de probabilidad**: cada bucket es un **intervalo cerrado-abierto**:
- `0.50` = `[0.50, 0.55)` (incluye 0.50, no incluye 0.55)
- `0.55` = `[0.55, 0.57)`, `0.57` = `[0.57, 0.60)`, etc.

Vista **acumulativa** (lo que opera el bot con threshold X): se enseña en
una celda aparte como `p_iso >= X`.

**Notas criticas a tener en cuenta al leer**:

1. El TEST (abr 2025 - jun 2026) tiene mas regimen bajista que alcista
   (BTC en bajada desde hace >8 meses). Lo que parece "edge en bajista"
   puede ser sesgo de muestra. Por eso comparamos **train + val + test**
   y al final hay walk-forward CV.
2. Bucket `0.80` quitado: solo agrupaba ~19-60 trades en 4h, demasiado
   ruidoso para sacar conclusiones (el "Calmar 13.9" era inestadistico).
   El bucket maximo que mostramos es `0.75`.
3. 15m parece volverse rentable en **lateral con p_iso >= 0.55** (87 trades,
   win 60.9%). Hay una celda dedicada a verificarlo y discutir si conviene
   un modelo separado por regimen.
4. En 1h, el edge parece concentrarse en **lateral y bajista**. Como TEST
   es muy bajista, hay que validar con train + val que esto no es sesgo.
5. Apalancamiento: revisamos x3-x4 en 1h y rangos seguros en 4h con
   buckets sensatos (no el 0.80 antiguo).

Coste 0.0012, notional fijo 100 EUR, 1 senal por vela (top EV).
AUC test ~0.547. p_iso maximo: 15m 0.613 | 1h 0.678 | 4h 1.0.
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
BUCKETS = [0.50, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75, 1.0001]
BUCKET_LBL = [f"{a:.2f}" for a in BUCKETS[:-1]]
# CUM_THRESHOLDS = vista acumulativa (p_iso >= X) = como opera el bot
CUM_THRESHOLDS = [0.50, 0.55, 0.57, 0.60, 0.62, 0.65, 0.67, 0.70]
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

# elegir mejor bucket por TF (mayor Calmar test, n>=50 para estabilidad)
# n>=50 evita el clasico "Calmar gigante con 19 trades" que vimos en 4h 0.80
best = {}
for tf in TFS:
    g = master_tbl[(master_tbl.split == "test") & (master_tbl.tf == tf)
                   & (master_tbl.n >= 50)]
    if not len(g):
        # fallback: relajar a n>=20 si no hay buckets con 50+
        g = master_tbl[(master_tbl.split == "test") & (master_tbl.tf == tf)
                       & (master_tbl.n >= 20)]
    if not len(g): continue
    g = g[~g.calmar.isin([np.inf])]
    if not len(g): continue
    best[tf] = g.sort_values("calmar", ascending=False).iloc[0].bucket
print(f"Mejor bucket por TF (Calmar, n>=50): {best}")

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

print("\\nRB16 OK celdas 1-8")
'''

C9 = '''# Celda 9
# Objetivo
# Vista ACUMULATIVA (p_iso >= X) - es como opera el bot con threshold X.
# Aclara la pregunta "los buckets son >=0.55 o solo [0.55, 0.60)?".

rows_cum = []
for ds in sig:
    days = max(1, (sig[ds]["timestamp"].max()
                    - sig[ds]["timestamp"].min()).days)
    sel = top_ev_per_candle(sig[ds])
    for tf in TFS:
        for thr in CUM_THRESHOLDS:
            sub = sel[(sel.timeframe == tf) & (sel.p_win_isotonic >= thr)]
            m = metrics(sub.net012.values, sub.dur_min.values, days)
            if m is None: continue
            m.update({"split": ds, "tf": tf, "thr": thr})
            rows_cum.append(m)
cum_tbl = pd.DataFrame(rows_cum)
for c in ["win_rate","expectancy","PF","sharpe","calmar"]:
    cum_tbl[c] = cum_tbl[c].round(3)
for c in ["max_dd_eur","pnl_total","pnl_dia"]:
    cum_tbl[c] = cum_tbl[c].round(2)

print("=== Vista ACUMULATIVA (p_iso >= X) - como opera el bot ===")
for ds in sig:
    print(f"\\n--- {ds.upper()} ---")
    print(cum_tbl[cum_tbl.split == ds][[
        "tf", "thr", "n", "trades_dia", "win_rate", "expectancy",
        "sharpe", "calmar", "max_dd_eur", "pnl_total"
    ]].to_string(index=False))

# Comparativa visual win rate acumulado por TF para los 3 splits
fig, axes = plt.subplots(1, len(TFS), figsize=(5*len(TFS), 4), sharey=True)
for ax, tf in zip(axes, TFS):
    for ds in sig:
        g = cum_tbl[(cum_tbl.tf == tf) & (cum_tbl.split == ds)]
        ax.plot(g.thr, g.win_rate, marker="o", label=ds, lw=1.3)
    ax.axhline(0.5, color="black", lw=0.5)
    ax.set_title(f"{tf}: win rate vs threshold (acumulativo)")
    ax.set_xlabel("p_iso >= X"); ax.grid(alpha=0.3); ax.legend()
axes[0].set_ylabel("win rate")
plt.tight_layout(); plt.show()
'''

C10 = '''# Celda 10
# Objetivo
# Regimen de tendencia comparando train + val + test. Si la "ventaja en
# bajista de 1h" se mantiene en train (mucha mas variedad de regimenes),
# es real; si solo aparece en test, es sesgo del periodo bajista actual.

rows_all = []
for ds in sig:
    days_s = max(1, (sig[ds]["timestamp"].max()
                      - sig[ds]["timestamp"].min()).days)
    sel = annotate_regime(top_ev_per_candle(sig[ds]))
    for tf in TFS:
        for reg in ["alcista", "bajista", "lateral"]:
            for thr in [0.55, 0.60, 0.65]:
                sub = sel[(sel.timeframe == tf) & (sel.regimen == reg)
                           & (sel.p_win_isotonic >= thr)]
                m = metrics(sub.net012.values, sub.dur_min.values, days_s)
                if m is None: continue
                m.update({"split": ds, "tf": tf, "regimen": reg, "thr": thr})
                rows_all.append(m)
reg_tbl = pd.DataFrame(rows_all)
for c in ["win_rate", "expectancy", "sharpe"]:
    reg_tbl[c] = reg_tbl[c].round(3)
for c in ["max_dd_eur", "pnl_total"]:
    reg_tbl[c] = reg_tbl[c].round(2)

print("=== Regimen TF/threshold en TRAIN/VAL/TEST ===")
print("(busca: si VAL+TRAIN no confirman lo que VES en TEST, es sesgo)\\n")
print(reg_tbl[["split","tf","regimen","thr","n","win_rate","expectancy",
                "sharpe","pnl_total","max_dd_eur"]].to_string(index=False))

# Heatmap simple: win_rate por (tf+regimen) en cada split, threshold 0.55
fig, axes = plt.subplots(1, len(sig), figsize=(4.5*len(sig), 4), sharey=True)
for ax, ds in zip(axes, sig.keys()):
    g = reg_tbl[(reg_tbl.split == ds) & (reg_tbl.thr == 0.55)]
    if not len(g):
        ax.set_title(f"{ds}: sin datos thr 0.55"); ax.axis("off"); continue
    piv = g.pivot_table(index="tf", columns="regimen", values="win_rate")
    piv = piv.reindex(index=TFS, columns=["alcista","lateral","bajista"])
    im = ax.imshow(piv.values, cmap="RdYlGn", vmin=0.45, vmax=0.7, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(TFS))); ax.set_yticklabels(piv.index)
    ax.set_title(f"{ds}: win rate (thr 0.55)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black", fontsize=10)
    plt.colorbar(im, ax=ax, shrink=0.7)
plt.suptitle("Win rate por TF x regimen (heatmap, thr 0.55, p_iso >= 0.55)",
             y=1.02)
plt.tight_layout(); plt.show()

print("\\nLectura: si una celda esta verde en TEST pero amarilla/roja")
print("en TRAIN y VAL -> sesgo. Si la celda esta verde en los 3 splits ->")
print("efecto real.")
'''

C11 = '''# Celda 11
# Objetivo
# Foco en el caso 15m lateral. Investigar si el edge observado en test
# (87 trades, +11 EUR, win 60.9%) se sostiene en train y val.

print("=== Foco: 15m EN LATERAL, thr 0.55 (p_iso >= 0.55) ===\\n")
for ds in sig:
    days_s = max(1, (sig[ds]["timestamp"].max()
                      - sig[ds]["timestamp"].min()).days)
    sel = annotate_regime(top_ev_per_candle(sig[ds]))
    sub = sel[(sel.timeframe == "15m") & (sel.regimen == "lateral")
               & (sel.p_win_isotonic >= 0.55)]
    if len(sub) < 5:
        print(f"{ds}: sin datos"); continue
    m = metrics(sub.net012.values, sub.dur_min.values, days_s)
    print(f"{ds}: n={m['n']:4d} | win {m['win_rate']:.3f} | "
          f"exp {m['expectancy']:+.4f} | sharpe {m['sharpe']:+.2f} | "
          f"PnL {m['pnl_total']:+.1f} EUR | maxDD {m['max_dd_eur']:+.1f}")

# Distribucion: cuanto tiempo pasamos en cada regimen, por split
# (sirve para entender el sesgo)
print("\\n=== Distribucion de regimen por split (horas) ===")
sel_reg = annotate_regime(top_ev_per_candle(sig.get("test")))
for ds in sig:
    sel = annotate_regime(top_ev_per_candle(sig[ds]))
    dist = sel.groupby("regimen")["timestamp"].count() / len(sel) * 100
    print(f"{ds}: {dist.round(1).to_dict()}")

print()
print("Si en TRAIN+VAL el edge 15m lateral se mantiene -> entrenar un")
print("clasificador de regimen + 3 modelos (uno por regimen) puede valer.")
print("Si solo aparece en TEST con 87 trades -> NO concluyente todavia.")

# Equity 15m thr 0.55 por regimen, los 3 splits
fig, axes = plt.subplots(1, len(sig), figsize=(5*len(sig), 4))
if len(sig) == 1: axes = [axes]
for ax, ds in zip(axes, sig.keys()):
    sel = annotate_regime(top_ev_per_candle(sig[ds]))
    sub = sel[(sel.timeframe == "15m") & (sel.p_win_isotonic >= 0.55)]
    for reg, color in [("alcista", "seagreen"), ("lateral", "grey"),
                        ("bajista", "firebrick")]:
        g = sub[sub.regimen == reg].sort_values("timestamp")
        if len(g) < 5: continue
        eq = (g.net012.values * NOTIONAL).cumsum()
        ax.plot(g.timestamp, eq, label=f"{reg} (n={len(g)})", color=color,
                lw=1.1)
    ax.set_title(f"15m thr 0.55 - {ds}")
    ax.axhline(0, color="black", lw=0.5); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout(); plt.show()
'''

C12 = '''# Celda 12
# Objetivo
# Walk-forward CV (series temporales). Particiona TODO el periodo
# (train+val+test) en N ventanas consecutivas y mide el edge por TF y
# threshold en cada ventana. Si la ventaja se mantiene en MAS ventanas
# que la moneda al aire, es real. Si solo aparece en 1-2 ventanas
# concretas (las recientes), es sesgo del periodo.

# Unir los tres splits
union = pd.concat([sig["train"], sig["val"], sig["test"]],
                   ignore_index=True)
union["timestamp"] = pd.to_datetime(union["timestamp"])
union = union.sort_values("timestamp")
N_FOLDS = 10
fold_edges = pd.qcut(union["timestamp"].astype("int64"), N_FOLDS,
                      labels=False, duplicates="drop")
union["fold"] = fold_edges

# Para cada (tf, threshold), expectancy por fold
print(f"=== Walk-forward {N_FOLDS} ventanas (sobre union train+val+test) ===")
print(f"Cada fold cubre ~{(union.timestamp.max()-union.timestamp.min()).days // N_FOLDS} dias")
print()
fold_dates = (union.groupby("fold")["timestamp"]
              .agg(["min", "max"]).round("D"))
print("Rangos de cada fold:")
print(fold_dates.to_string())
print()

rows_w = []
for tf in TFS:
    for thr in [0.55, 0.60, 0.65]:
        for fold in range(N_FOLDS):
            sub_fold = union[union.fold == fold]
            sel = top_ev_per_candle(sub_fold)
            sub = sel[(sel.timeframe == tf) & (sel.p_win_isotonic >= thr)]
            if len(sub) < 10: continue
            nr = sub.net012.values
            rows_w.append({"tf": tf, "thr": thr, "fold": int(fold),
                           "n": len(sub),
                           "win_rate": round((nr > 0).mean(), 3),
                           "expectancy": round(nr.mean(), 5),
                           "pnl_total": round((nr * NOTIONAL).sum(), 2)})
wf = pd.DataFrame(rows_w)

# Resumen: % de folds rentables por (tf, thr)
print("=== % folds con expectancy > 0 (estabilidad del edge) ===")
summary = []
for tf in TFS:
    for thr in [0.55, 0.60, 0.65]:
        g = wf[(wf.tf == tf) & (wf.thr == thr)]
        if not len(g): continue
        summary.append({"tf": tf, "thr": thr, "n_folds": len(g),
                         "pct_rentable": round((g.expectancy > 0).mean()*100, 0),
                         "exp_media": round(g.expectancy.mean(), 5),
                         "exp_std": round(g.expectancy.std(), 5)})
sm = pd.DataFrame(summary)
print(sm.to_string(index=False))

# Grafica: expectancy por fold en cada (tf, thr)
fig, axes = plt.subplots(len(TFS), 1, figsize=(13, 3.5*len(TFS)),
                         sharex=True)
for ax, tf in zip(axes, TFS):
    for thr in [0.55, 0.60, 0.65]:
        g = wf[(wf.tf == tf) & (wf.thr == thr)].sort_values("fold")
        ax.plot(g.fold, g.expectancy, marker="o",
                label=f"thr {thr}", lw=1.3)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{tf}: expectancy por fold (walk-forward)")
    ax.set_ylabel("expectancy"); ax.grid(alpha=0.3); ax.legend()
axes[-1].set_xlabel("fold (cronologico)")
plt.tight_layout(); plt.show()

print("\\nLectura: una linea por encima de 0 en la MAYORIA de folds")
print("indica edge estable. Una linea que cae a negativo en folds")
print("recientes pero positiva antes -> el modelo se ha degradado en")
print("el ultimo periodo (o el regimen reciente le sienta mal).")
'''

C13 = '''# Celda 13
# Objetivo
# Apalancamiento concreto para casos sensatos:
#   - 1h con thr 0.55-0.60, L = 3 y L = 4
#   - 4h con thr 0.55-0.65, L = 2, 3, 5
# Con notional fijo todo escala lineal en L; mostramos riesgos absolutos.

FUNDING_8H = 0.0001
casos = [("1h", 0.55, [1, 2, 3, 4]),
         ("1h", 0.60, [1, 2, 3, 4]),
         ("4h", 0.55, [1, 2, 3, 5]),
         ("4h", 0.60, [1, 2, 3, 5]),
         ("4h", 0.65, [1, 2, 3, 5])]

sel_test = top_ev_per_candle(sig["test"])
days_test = max(1, (sig["test"]["timestamp"].max()
                    - sig["test"]["timestamp"].min()).days)
print(f"=== Apalancamiento por threshold (TEST, {days_test} dias) ===")
print(f"{'tf':>4} {'thr':>5} {'L':>3} {'trades':>7} "
      f"{'PnL anyo':>10} {'fund anyo':>10} {'maxDD EUR':>10} "
      f"{'Calmar':>7} {'L safe':>7}")
for tf, thr, Ls in casos:
    sub = sel_test[(sel_test.timeframe == tf)
                    & (sel_test.p_win_isotonic >= thr)]
    if len(sub) < 10:
        print(f"{tf:>4} {thr:>5.2f}: pocos trades ({len(sub)})"); continue
    pnl_unit = sub.net012.values * NOTIONAL
    fund_unit = sub.dur_min.values / 60 / 8 * FUNDING_8H * NOTIONAL
    max_sl = sub.sl_pct.max()
    L_safe = round(1.0 / (3 * max_sl + 0.005), 1)
    for L in Ls:
        pnl = pnl_unit * L - fund_unit * L
        eq = np.cumsum(pnl)
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak).min()
        pnl_anyo = pnl.sum() * 365 / days_test
        fund_anyo = -fund_unit.sum() * L * 365 / days_test
        calmar = pnl_anyo / abs(dd) if dd < 0 else np.inf
        print(f"{tf:>4} {thr:>5.2f} {L:>3} {len(sub):>7} "
              f"{pnl_anyo:>+10.1f} {fund_anyo:>+10.2f} {dd:>+10.1f} "
              f"{calmar:>7.2f} {L_safe:>7.1f}")
    print()

# Grafica equity de los casos clave
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
configs = [("1h", 0.55), ("1h", 0.60), ("4h", 0.55), ("4h", 0.65)]
for ax, (tf, thr) in zip(axes.flat, configs):
    sub = sel_test[(sel_test.timeframe == tf)
                    & (sel_test.p_win_isotonic >= thr)].sort_values("timestamp")
    if len(sub) < 10:
        ax.set_title(f"{tf} thr {thr}: pocos trades"); continue
    pnl_unit = sub.net012.values * NOTIONAL
    fund_unit = sub.dur_min.values / 60 / 8 * FUNDING_8H * NOTIONAL
    for L in [1, 2, 3, 5]:
        eq = np.cumsum(pnl_unit * L - fund_unit * L)
        ax.plot(sub.timestamp, eq, label=f"L={L}", lw=1.1)
    ax.set_title(f"{tf} thr {thr} (n={len(sub)}): equity apalancada")
    ax.axhline(0, color="black", lw=0.5)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print("\\nRB16 OK celdas 9-13")
'''

nb = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in [("markdown", MD0), ("code", C1), ("code", C2),
                  ("code", C3), ("code", C4), ("code", C5), ("code", C6),
                  ("code", C7), ("code", C8), ("code", C9), ("code", C10),
                  ("code", C11), ("code", C12), ("code", C13)]:
    nb["cells"].append({"cell_type": kind, "metadata": {},
                        "source": src.splitlines(keepends=True),
                        **({"outputs": [], "execution_count": None}
                           if kind == "code" else {})})
OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print("Notebook escrito:", OUT_NB)
