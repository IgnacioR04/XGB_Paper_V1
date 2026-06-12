# -*- coding: utf-8 -*-
"""Genera RB16_reporte_modelo_causal.ipynb (Drive, Colab/local)."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT_NB = BASE / "repo/notebooks/08_real_backtest/RB16_reporte_modelo_causal.ipynb"

MD0 = """# RB16 - Reporte del modelo CAUSAL (sin leakage)

Todo lo que antes veias del backtest, en un solo notebook, para el modelo
reentrenado sin leakage (2026-06-12): operaciones por threshold y por
temporalidad, trades/dia, PnL diario (sin compuesto), apalancamiento,
mix long/short, curvas de equity.

Semantica identica al bot: `p_win_isotonic >= threshold` (sin tope),
1 senal max por vela (top EV), coste 0.0012, cartera de 1 posicion con
notional fijo 100 EUR. AUC test ~0.547 en los 3 TFs.
p_iso maximo alcanzable: 15m 0.613 | 1h 0.678 | 4h 1.000.
"""

C1 = """# Celda 1
# Objetivo
# Setup + carga de signals (val/test) y duraciones reales desde labels.

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
THRESHOLDS = [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75]
COLORS = {"15m": "steelblue", "1h": "darkorange", "4h": "seagreen"}

sig = {}
for ds in ["val", "test"]:
    df = pd.read_parquet(DIR_RB / f"signals/signals_{ds}.parquet")
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
print("val:", sig["val"].shape, "| test:", sig["test"].shape)


def bot_stream(df, tf, thr):
    sub = df[(df.timeframe == tf) & (df.p_win_isotonic >= thr)]
    if not len(sub):
        return sub
    return (sub.loc[sub.groupby("timestamp")["EV_pred"].idxmax()]
               .sort_values("timestamp").reset_index(drop=True))


def wallet_fixed(s):
    open_until, rows = None, []
    for _, r in s.iterrows():
        if open_until is not None and r.timestamp < open_until:
            continue
        if not np.isfinite(r.dur_min):
            continue
        open_until = r.timestamp + pd.Timedelta(minutes=float(r.dur_min))
        rows.append(r)
    return pd.DataFrame(rows)
"""

C2 = """# Celda 2
# Objetivo
# Grid de thresholds por TF (todas las senales, sin restriccion de cartera).
# Tablas val y test + graficas n_trades / expectancy / retorno total.

grids = {}
for ds in ["val", "test"]:
    rows = []
    for tf in TFS:
        for thr in THRESHOLDS:
            s = bot_stream(sig[ds], tf, thr)
            if len(s) < 5:
                continue
            days = max(1, (s.timestamp.max() - s.timestamp.min()).days)
            nr = s.net012.values
            pos, neg = nr[nr > 0], -nr[nr < 0]
            rows.append({
                "tf": tf, "thr": thr, "n": len(s),
                "por_dia": round(len(s) / days, 2),
                "pct_long": round((s.side == "long").mean() * 100, 0),
                "win_rate": round((nr > 0).mean(), 3),
                "expectancy": round(nr.mean(), 5),
                "PF": round(pos.sum() / neg.sum(), 2) if neg.sum() > 0 else np.inf,
                "ret_total": round(nr.sum(), 2)})
    grids[ds] = pd.DataFrame(rows)
    print(f"=== {ds.upper()} ===")
    print(grids[ds].to_string(index=False))
    print()

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, met in zip(axes, ["n", "expectancy", "ret_total"]):
    for tf in TFS:
        g = grids["test"][grids["test"].tf == tf]
        ax.plot(g.thr, g[met], marker="o", label=tf, color=COLORS[tf])
    ax.axvline(0.67, color="grey", ls="--", lw=0.8, label="thr bot 0.67")
    ax.set_title(f"{met} vs threshold (TEST)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
    if met != "n":
        ax.axhline(0, color="black", lw=0.5)
plt.tight_layout(); plt.show()
"""

C3 = """# Celda 3
# Objetivo
# Regla del bot (1 posicion, notional fijo 100 EUR): trades/dia, PnL diario,
# rachas. TEST. Sin interes compuesto.


def maxstreak(nr):
    s = m = 0
    for x in nr:
        s = s + 1 if x <= 0 else 0
        m = max(m, s)
    return m


detail = {}
rows = []
for tf in TFS:
    for thr in THRESHOLDS:
        s = bot_stream(sig["test"], tf, thr)
        if len(s) < 5:
            continue
        t = wallet_fixed(s)
        if len(t) < 5:
            continue
        days = max(1, (t.timestamp.max() - t.timestamp.min()).days)
        pnl = t.net012.values * NOTIONAL
        daily = pd.Series(pnl, index=t.timestamp).resample("1D").sum()
        detail[(tf, thr)] = (t, daily)
        rows.append({"tf": tf, "thr": thr, "trades": len(t),
                     "trades_dia": round(len(t) / days, 2),
                     "PnL_total": round(pnl.sum(), 1),
                     "PnL_dia": round(daily.mean(), 3),
                     "peor_dia": round(daily.min(), 2),
                     "mejor_dia": round(daily.max(), 2),
                     "racha_perd": maxstreak(pnl),
                     "pct_dias_neg": round((daily < 0).mean() * 100, 0)})
bot_tbl = pd.DataFrame(rows)
print("=== Regla del bot (TEST, 100 EUR fijos) ===")
print(bot_tbl.to_string(index=False))
"""

C4 = """# Celda 4
# Objetivo
# Equity acumulada (sin compuesto, EUR sobre 100 fijos) para thresholds
# clave por TF + PnL diario en barras para la config del bot (0.67).

fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))
for ax, tf in zip(axes, TFS):
    for thr in [0.55, 0.60, 0.65, 0.67, 0.70]:
        if (tf, thr) not in detail:
            continue
        t, _ = detail[(tf, thr)]
        eq = (t.net012 * NOTIONAL).cumsum()
        ax.plot(t.timestamp, eq, label=f"thr {thr}", lw=1.1)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title(f"{tf}: PnL acumulado EUR (TEST, sin compuesto)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()

# PnL diario de la config desplegada (la que tenga senales con thr 0.67)
fig, axes = plt.subplots(len(TFS), 1, figsize=(13, 7), sharex=False)
for ax, tf in zip(axes, TFS):
    key = (tf, 0.67)
    if key not in detail:
        ax.set_title(f"{tf}: sin senales con thr 0.67 (p_iso max insuficiente)")
        ax.axis("off")
        continue
    _, daily = detail[key]
    ax.bar(daily.index, daily.values, width=0.9,
           color=np.where(daily.values >= 0, "#3fb950", "#f85149"))
    ax.set_title(f"{tf} thr=0.67: PnL diario (media {daily.mean():+.3f} EUR)")
    ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""

C5 = """# Celda 5
# Objetivo
# Apalancamiento: PnL/dia x L con funding, peor dia/racha, seguridad de
# liquidacion (aislada aprox: 1/L - mm; seguro si dist >= 3x SL max).

MM = 0.005
FUNDING_8H = 0.0001
LEVELS = [1, 2, 3, 5, 10]

rows = []
for (tf, thr), (t, daily) in detail.items():
    if thr not in (0.58, 0.60, 0.65, 0.67, 0.70):
        continue
    nr = t.net012.values * NOTIONAL
    max_sl = t.sl_pct.max()
    days = max(1, (t.timestamp.max() - t.timestamp.min()).days)
    fund1 = (t.dur_min / 60 / 8 * FUNDING_8H * NOTIONAL).mean() * len(t) / days
    cur = worst = 0.0
    for x in nr:
        cur = min(0, cur) + x
        worst = min(worst, cur)
    L_safe = 1.0 / (3 * max_sl + MM)
    for lv in LEVELS:
        rows.append({"tf": tf, "thr": thr, "L": lv,
                     "PnL_dia": round(daily.mean() * lv - fund1 * lv, 3),
                     "funding_dia": round(-fund1 * lv, 3),
                     "peor_dia": round(daily.min() * lv, 2),
                     "peor_racha": round(worst * lv, 2),
                     "SL_max_pct": round(max_sl * 100, 2),
                     "dist_liq_pct": round((1 / lv - MM) * 100, 1),
                     "seguro": "SI" if lv <= L_safe else
                               ("JUSTO" if lv <= 1 / (2 * max_sl + MM) else "NO")})
lev = pd.DataFrame(rows)
print(lev.to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 4))
for (tf, thr), g in lev.groupby(["tf", "thr"]):
    if thr != 0.67 and not (tf == "1h" and thr == 0.58):
        continue
    ax.plot(g.L, g.PnL_dia, marker="o", label=f"{tf} thr {thr}")
ax.set_xlabel("apalancamiento L"); ax.set_ylabel("PnL/dia EUR (100 fijos)")
ax.grid(alpha=0.3); ax.legend(); ax.set_title("PnL diario vs apalancamiento (TEST)")
plt.tight_layout(); plt.show()
"""

C6 = """# Celda 6
# Objetivo
# Mix long/short por TF y threshold + mix de salidas + compuesto informativo.

print("=== Mix long/short y salidas (TEST) ===")
for tf in TFS:
    for thr in [0.55, 0.60, 0.67]:
        s = bot_stream(sig["test"], tf, thr)
        if len(s) < 5:
            continue
        mix = s.side.value_counts(normalize=True).round(2).to_dict()
        exits = s.exit_reason.value_counts(normalize=True).round(2).to_dict()
        print(f"{tf} thr {thr}: n={len(s)} sides={mix} salidas={exits}")

print()
print("=== Compuesto informativo (TEST, reinvirtiendo todo, 1 pos) ===")
for (tf, thr), (t, _) in detail.items():
    if thr not in (0.58, 0.60, 0.65, 0.67, 0.70):
        continue
    eq = NOTIONAL * np.cumprod(1 + t.net012.values)
    print(f"{tf} thr {thr}: 100 EUR -> {eq[-1]:,.1f} EUR ({len(t)} trades)")
print()
print("RB16 OK")
"""

C7 = """# Celda 7
# Objetivo
# Acierto y rentabilidad por DECIL DE VOLATILIDAD (TEST). Dos vistas:
# todas las senales (thr 0.50) y las senales del threshold desplegado
# (15m 0.55, 1h/4h 0.67).

DEPLOYED = {"15m": 0.55, "1h": 0.67, "4h": 0.67}

rows = []
for tf in TFS:
    for vista, thr in [("todas", 0.50), ("bot", DEPLOYED[tf])]:
        s = bot_stream(sig["test"], tf, thr)
        if not len(s):
            continue
        for dec, g in s.groupby("vol_decile"):
            if len(g) < 10:
                continue
            nr = g.net012.values
            pos, neg = nr[nr > 0], -nr[nr < 0]
            rows.append({"tf": tf, "vista": vista, "decil": int(dec),
                         "n": len(g),
                         "win_rate": round((nr > 0).mean(), 3),
                         "expectancy": round(nr.mean(), 5),
                         "PF": round(pos.sum() / neg.sum(), 2)
                         if neg.sum() > 0 else np.inf,
                         "ret_total": round(nr.sum(), 2)})
vol_tbl = pd.DataFrame(rows)
print(vol_tbl.to_string(index=False))

fig, axes = plt.subplots(2, 3, figsize=(16, 7.5))
for j, tf in enumerate(TFS):
    for i, met in enumerate(["win_rate", "expectancy"]):
        ax = axes[i][j]
        for vista, color in [("todas", "lightsteelblue"), ("bot", COLORS[tf])]:
            g = vol_tbl[(vol_tbl.tf == tf) & (vol_tbl.vista == vista)]
            ax.bar(g.decil + (0.2 if vista == "bot" else -0.2), g[met],
                   width=0.4, color=color, label=vista)
        ax.set_title(f"{tf}: {met} por decil de vol (TEST)")
        ax.set_xlabel("vol_decile"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
        if met == "win_rate":
            ax.axhline(0.5, color="black", lw=0.5)
        else:
            ax.axhline(0, color="black", lw=0.5)
plt.tight_layout(); plt.show()
"""

C8 = """# Celda 8
# Objetivo
# Retorno/riesgo de APALANCARSE, por threshold y TF (TEST, regla del bot,
# notional fijo). Clave: retorno y riesgo escalan LINEAL con L, asi que el
# RATIO retorno/riesgo no depende de L - lo que cambia con L es el nivel
# absoluto y la distancia a liquidacion. Por eso se muestra:
#   - PnL/dia y peor racha a L=1
#   - ratio retorno/riesgo = PnL anual / |peor racha|  (independiente de L)
#   - L maximo seguro (dist. liquidacion >= 3x SL max) y PnL/dia a ese L

MM = 0.005
FUNDING_8H = 0.0001

rows = []
for tf in TFS:
    for thr in THRESHOLDS:
        if (tf, thr) not in detail:
            continue
        t, daily = detail[(tf, thr)]
        if len(t) < 10:
            continue
        nr = t.net012.values * NOTIONAL
        days = max(1, (t.timestamp.max() - t.timestamp.min()).days)
        fund1 = (t.dur_min / 60 / 8 * FUNDING_8H * NOTIONAL).mean() * len(t) / days
        cur = worst = 0.0
        for x in nr:
            cur = min(0, cur) + x
            worst = min(worst, cur)
        max_sl = t.sl_pct.max()
        L_safe = 1.0 / (3 * max_sl + MM)
        pnl_dia = daily.mean() - fund1
        anual = pnl_dia * 365
        ratio = anual / abs(worst) if worst < 0 else np.inf
        rows.append({"tf": tf, "thr": thr, "trades": len(t),
                     "PnL_dia_L1": round(pnl_dia, 3),
                     "peor_racha_L1": round(worst, 2),
                     "ret_anual_L1": round(anual, 1),
                     "ratio_ret_riesgo": round(ratio, 2),
                     "L_max_seguro": round(L_safe, 1),
                     "PnL_dia_Lmax": round(pnl_dia * L_safe, 2),
                     "peor_racha_Lmax": round(worst * L_safe, 1)})
lev_thr = pd.DataFrame(rows)
print(lev_thr.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.5))
for tf in TFS:
    g = lev_thr[lev_thr.tf == tf]
    axes[0].plot(g.thr, g.ratio_ret_riesgo, marker="o", label=tf, color=COLORS[tf])
    axes[1].plot(g.thr, g.PnL_dia_Lmax, marker="o", label=tf, color=COLORS[tf])
axes[0].set_title("Ratio retorno anual / peor racha (independiente de L)")
axes[0].set_xlabel("threshold"); axes[0].axhline(0, color="black", lw=0.5)
axes[1].set_title("PnL/dia EUR al L maximo seguro (100 EUR base)")
axes[1].set_xlabel("threshold"); axes[1].axhline(0, color="black", lw=0.5)
for ax in axes:
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
plt.tight_layout(); plt.show()
print()
print("Lectura: el ratio dice QUE config apalancar (mas alto = mas retorno")
print("por unidad de dolor); el L_max_seguro dice CUANTO; el PnL_dia_Lmax")
print("combina ambos. Racha y peor dia escalan lineal con el L elegido.")
"""


nb = {"cells": [], "metadata": {
    "kernelspec": {"display_name": "Python 3", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python", "version": "3.11"},
    "colab": {"provenance": []}}, "nbformat": 4, "nbformat_minor": 5}
for kind, src in [("markdown", MD0), ("code", C1), ("code", C2),
                  ("code", C3), ("code", C4), ("code", C5), ("code", C6), ("code", C7), ("code", C8)]:
    nb["cells"].append({"cell_type": kind, "metadata": {},
                        "source": src.splitlines(keepends=True),
                        **({"outputs": [], "execution_count": None}
                           if kind == "code" else {})})
OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print("Notebook escrito:", OUT_NB)
