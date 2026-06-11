# -*- coding: utf-8 -*-
"""Genera RB15_daily_pnl_and_leverage.ipynb en el repo de Drive.

Pregunta central: por que 1h/4h ganan ~0.5 EUR/dia cuando 15m gana ~1.3?
Es por perder mas? Por TPs mal puestos? O por frecuencia?
+ estudio de apalancamiento con seguridad de liquidacion y coste de funding.
TODO a notional FIJO de 100 EUR (sin interes compuesto); el compuesto solo
aparece como dato final informativo.
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]
OUT_NB = BASE / "repo/notebooks/08_real_backtest/RB15_daily_pnl_and_leverage.ipynb"

MD0 = """# RB15 - PnL diario por timeframe y estudio de apalancamiento

**Preguntas**:

1. Por que la cartera 1h gana ~0.6 EUR/dia y la 4h ~0.5, frente a ~1.3 de 15m
   (todo a notional fijo 100 EUR)? Es porque pierden mas veces? Porque los TP
   estan mal calibrados? O simplemente porque hay menos oportunidades?
2. Descomposicion exacta: `PnL/dia = trades/dia x PnL medio por trade`, y a su
   vez `trades/dia = velas/dia x %velas en banda x %capturadas por la cartera`.
3. Calidad de las salidas: mix TP/SL/TIMEOUT por TF, expectancy por tipo de
   salida, cuanto diluyen los timeouts.
4. Apalancamiento: si el PnL es lineal con L (notional fijo), que L es seguro
   frente a la liquidacion? Cuanto come el funding de un perpetuo? Tabla de
   PnL/dia, peor dia y peor racha por L.

**Regla de oro de este notebook: SIN interes compuesto.** Todas las cifras
diarias son con notional fijo de 100 EUR para poder seguir el proceso. El
compuesto aparece solo al final como referencia.
"""

C1 = """# Celda 1
# Objetivo
# Imports, BASE, carga de signals test + labels (igual que RB14).

from pathlib import Path
import json
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
OUT_DIR = DIR_RB / "daily_pnl_leverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST = 0.0012
NOTIONAL = 100.0
BANDS = {"15m": (0.65, 0.70), "1h": (0.70, 0.75), "4h": (0.65, 0.70)}
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}
CANDLES_DAY = {"15m": 96, "1h": 24, "4h": 6}
TFS = ["15m", "1h", "4h"]

test = pd.read_parquet(DIR_RB / "signals/signals_test.parquet")
test["timestamp"] = pd.to_datetime(test["timestamp"])
lk = []
for tf in TFS:
    h = pd.read_parquet(DIR_LBL / f"labels_compact_{tf}.parquet",
                        columns=["timestamp", "candidate_id", "timeframe", "H",
                                 "gross_return", "exit_reason", "time_to_exit"])
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    lk.append(h)
test = test.merge(pd.concat(lk, ignore_index=True),
                  on=["timestamp", "candidate_id", "timeframe"], how="left")
test["net_return"] = test["gross_return"] - COST
test["dur_min"] = test["timeframe"].map(TF_MIN) * \
    test["time_to_exit"].fillna(test["H"])
print("test:", test.shape)
"""

C2 = """# Celda 2
# Objetivo
# Flujo de senales del bot (banda [lo,hi) + top EV por vela) y trades tomados
# con la regla real (1 posicion, cartera se libera al exit). NOTIONAL FIJO.


def bot_signals(df, tf):
    lo, hi = BANDS[tf]
    sub = df[(df.timeframe == tf) & (df.p_win_isotonic >= lo)
             & (df.p_win_isotonic < hi)]
    idx = sub.groupby("timestamp")["EV_pred"].idxmax()
    return sub.loc[idx].sort_values("timestamp").reset_index(drop=True)


def taken_fixed(sig):
    open_until = None
    rows = []
    for _, r in sig.iterrows():
        ts = r["timestamp"]
        if open_until is not None and ts < open_until:
            continue
        if not np.isfinite(r["dur_min"]):
            continue
        open_until = ts + pd.Timedelta(minutes=float(r["dur_min"]))
        rows.append(r)
    return pd.DataFrame(rows).reset_index(drop=True)


signals, taken = {}, {}
for tf in TFS:
    signals[tf] = bot_signals(test, tf)
    taken[tf] = taken_fixed(signals[tf])
    print(f"{tf}: {len(signals[tf])} senales en banda -> "
          f"{len(taken[tf])} trades tomados (1 pos)")
"""

C3 = """# Celda 3
# Objetivo
# LA DESCOMPOSICION. PnL/dia = trades/dia x PnL/trade. Y trades/dia =
# velas/dia x %en_banda x %capturadas. Aqui se ve de donde sale la diferencia
# 15m vs 1h vs 4h.

rows = []
for tf in TFS:
    sig, t = signals[tf], taken[tf]
    days = max(1, (t.timestamp.max() - t.timestamp.min()).days)
    n_candles = days * CANDLES_DAY[tf]
    pnl_eur = t.net_return * NOTIONAL
    wins = pnl_eur[pnl_eur > 0]
    losses = pnl_eur[pnl_eur <= 0]
    rows.append({
        "TF": tf,
        "velas/dia": CANDLES_DAY[tf],
        "% velas en banda": round(len(sig) / n_candles * 100, 1),
        "senales/dia": round(len(sig) / days, 2),
        "% capturadas (1pos)": round(len(t) / len(sig) * 100, 1),
        "trades/dia": round(len(t) / days, 2),
        "win rate": round((t.net_return > 0).mean(), 3),
        "media gana EUR": round(wins.mean(), 3),
        "media pierde EUR": round(losses.mean(), 3),
        "PnL/trade EUR": round(pnl_eur.mean(), 3),
        "PnL/dia EUR": round(pnl_eur.sum() / days, 3),
        "tp_pct medio %": round(t.tp_pct.mean() * 100, 2),
        "sl_pct medio %": round(t.sl_pct.mean() * 100, 2),
        "% tiempo en mercado": round(t.dur_min.sum() / (days * 1440) * 100, 1),
    })
decomp = pd.DataFrame(rows).set_index("TF")
print(decomp.T.to_string())
decomp.to_csv(OUT_DIR / "decomposicion_pnl_diario.csv")
print()
print("LECTURA: si 'PnL/trade EUR' de 1h/4h es IGUAL O MAYOR que el de 15m,")
print("el deficit diario viene SOLO de la frecuencia (trades/dia), no de")
print("perder mas ni de TPs mal puestos.")
"""

C4 = """# Celda 4
# Objetivo
# Calidad de las salidas por TF: mix TP/SL/TIMEOUT, expectancy de cada tipo,
# y cuanto diluyen los timeouts. Si los TP estuvieran mal puestos, veriamos
# SL rate alto o TP rate bajo respecto al p_win predicho.

for tf in TFS:
    t = taken[tf]
    print(f"=== {tf} ===")
    mix = t.exit_reason.value_counts(normalize=True).round(3)
    print("  mix de salidas:", dict(mix))
    by_exit = t.groupby("exit_reason").agg(
        n=("net_return", "size"),
        ret_medio=("net_return", "mean"),
        eur_medio=("net_return", lambda s: (s * NOTIONAL).mean()),
        velas_hasta_exit=("time_to_exit", "mean"),
    ).round(4)
    print(by_exit.to_string())
    p_win_med = t.p_win_isotonic.mean()
    win_real = (t.net_return > 0).mean()
    print(f"  p_win predicho medio: {p_win_med:.3f} | win rate real: "
          f"{win_real:.3f} | gap: {win_real - p_win_med:+.3f}")
    sin_timeout = t[t.exit_reason.isin(['TP', 'SL'])]
    print(f"  expectancy sin timeouts: "
          f"{(sin_timeout.net_return * NOTIONAL).mean():+.3f} EUR/trade vs "
          f"con todo: {(t.net_return * NOTIONAL).mean():+.3f}")
    print()
"""

C5 = """# Celda 5
# Objetivo
# Series diarias a notional fijo (las cifras 'de seguimiento del proceso').

daily = {}
fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=False)
for ax, tf in zip(axes, TFS):
    t = taken[tf]
    d = (t.set_index("timestamp")["net_return"]
          .mul(NOTIONAL).resample("1D").sum())
    daily[tf] = d
    ax.bar(d.index, d.values, width=0.9,
           color=np.where(d.values >= 0, "#3fb950", "#f85149"))
    ax.set_title(f"{tf}: PnL diario a notional fijo {NOTIONAL:.0f} EUR "
                 f"(media {d.mean():+.2f}, mediana {d.median():+.2f})")
    ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_DIR / "pnl_diario_fijo.png", dpi=110)
plt.show()

for tf in TFS:
    d = daily[tf]
    neg_days = (d < 0).mean()
    print(f"{tf}: media {d.mean():+.3f} EUR/dia | % dias negativos "
          f"{neg_days*100:.0f}% | p10 {d.quantile(.1):+.2f} | "
          f"p90 {d.quantile(.9):+.2f} | peor {d.min():+.2f} | "
          f"mejor {d.max():+.2f}")
"""

C6 = """# Celda 6
# Objetivo
# Apalancamiento. Con notional fijo, el PnL escala LINEAL con L; el riesgo
# tambien. Tres preguntas:
#   a) hasta que L el SL sigue protegiendo antes de la liquidacion?
#      liq distance (long, aislado) ~ 1/L - mm  (mm = margen mantenimiento)
#      regla de seguridad: liq_distance >= 3 x SL mas grande del TF
#   b) cuanto come el funding (perpetuo, ~0.01%/8h sobre exposicion L x 100)
#   c) tabla PnL/dia, peor dia, peor racha en EUR por L

MM = 0.005            # maintenance margin tier bajo BTC
FUNDING_8H = 0.0001   # 0.01% cada 8h (medio historico aprox)
LEVELS = [1, 2, 3, 5, 10]

rows = []
for tf in TFS:
    t = taken[tf]
    d = daily[tf]
    max_sl = t.sl_pct.max()
    L_safe = 1.0 / (3 * max_sl + MM)
    # racha perdedora maxima en EUR (consecutivos)
    pnl = (t.net_return * NOTIONAL).values
    cur = worst = 0.0
    for x in pnl:
        cur = min(0, cur) + x
        worst = min(worst, cur)
    # funding por trade: exposicion L*100 durante dur_min
    fund_per_trade_L1 = (t.dur_min / 60 / 8 * FUNDING_8H * NOTIONAL).mean()
    days = max(1, (t.timestamp.max() - t.timestamp.min()).days)
    trades_day = len(t) / days
    for L in LEVELS:
        rows.append({
            "TF": tf, "L": L,
            "PnL/dia EUR": round(d.mean() * L
                                 - fund_per_trade_L1 * L * trades_day, 3),
            "funding/dia EUR": round(-fund_per_trade_L1 * L * trades_day, 3),
            "peor dia EUR": round(d.min() * L, 2),
            "peor racha EUR": round(worst * L, 2),
            "SL max %": round(max_sl * 100, 2),
            "dist. liquidacion %": round((1 / L - MM) * 100, 1),
            "seguro?": "SI" if L <= L_safe else
                       ("JUSTO" if L <= 1 / (2 * max_sl + MM) else "NO"),
        })
lev = pd.DataFrame(rows)
print(lev.to_string(index=False))
lev.to_csv(OUT_DIR / "leverage_grid.csv", index=False)
print()
for tf in TFS:
    t = taken[tf]
    max_sl = t.sl_pct.max()
    print(f"{tf}: SL max {max_sl*100:.2f}% -> L seguro (3x margen) = "
          f"{1.0/(3*max_sl+MM):.1f}x")
print()
print("OJO: 'peor dia' y 'peor racha' escalan lineal con L. Antes de subir L")
print("comprueba que la racha x L no supere lo que toleras perder.")
"""

C7 = """# Celda 7
# Objetivo
# Dato final informativo: equity compuesta (reinvirtiendo) a L=1, por TF.
# Solo como referencia - el seguimiento diario debe hacerse con la tabla
# de notional fijo de las celdas anteriores.

for tf in TFS:
    t = taken[tf]
    eq = NOTIONAL * np.cumprod(1 + t.net_return.values)
    days = max(1, (t.timestamp.max() - t.timestamp.min()).days)
    print(f"{tf}: compuesto final {eq[-1]:,.0f} EUR en {days} dias "
          f"({len(t)} trades) | sin componer: "
          f"{NOTIONAL + (t.net_return * NOTIONAL).sum():,.1f} EUR")
print()
print("RB15 OK -> outputs en", OUT_DIR)
"""

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def add_cell(kind, src):
    nb["cells"].append({
        "cell_type": kind,
        "metadata": {},
        "source": src.splitlines(keepends=True),
        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
    })


add_cell("markdown", MD0)
for c in [C1, C2, C3, C4, C5, C6, C7]:
    add_cell("code", c)

OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print(f"Notebook escrito: {OUT_NB}")
