# -*- coding: utf-8 -*-
"""Genera el reporte completo del modelo CAUSAL en un solo archivo MD.

Contenido: grid de thresholds por TF (n trades, trades/dia, win rate,
expectancy, PF, retorno), PnL diario a notional fijo (regla del bot:
1 posicion), apalancamiento con seguridad de liquidacion y funding,
mix long/short, compuesto informativo.
"""
import os
from pathlib import Path

import numpy as np
import pandas as pd

DRIVE = Path("G:/Mi unidad/Base de Datos BITCOIN")
CACHE = Path(os.environ["LOCALAPPDATA"]) / "Temp/btc_cache"
OUT_DIR = DRIVE / "Data/model_outputs/real_backtest/reporte_causal"
OUT_DIR.mkdir(parents=True, exist_ok=True)

COST = 0.0012
NOTIONAL = 100.0
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}
TFS = ["15m", "1h", "4h"]
THRESHOLDS = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.67, 0.70, 0.75, 0.80]
MM = 0.005
FUNDING_8H = 0.0001
LEVELS = [1, 2, 3, 5, 10]

sig = {}
for ds in ["val", "test"]:
    df = pd.read_parquet(CACHE / f"signals_{ds}.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    sig[ds] = df

# time_to_exit desde labels (cache local)
lk = []
for tf in TFS:
    h = pd.read_parquet(CACHE / f"labels_compact_{tf}.parquet",
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


def bot_stream(df, tf, thr):
    """Senales como las ve el bot: p>=thr (sin tope), top EV por vela."""
    sub = df[(df.timeframe == tf) & (df.p_win_isotonic >= thr)]
    if not len(sub):
        return sub
    return sub.loc[sub.groupby("timestamp")["EV_pred"].idxmax()] \
              .sort_values("timestamp").reset_index(drop=True)


def wallet_fixed(s):
    """1 posicion, notional fijo 100. Devuelve trades tomados."""
    open_until, rows = None, []
    for _, r in s.iterrows():
        if open_until is not None and r.timestamp < open_until:
            continue
        if not np.isfinite(r.dur_min):
            continue
        open_until = r.timestamp + pd.Timedelta(minutes=float(r.dur_min))
        rows.append(r)
    return pd.DataFrame(rows)


def maxstreak(nr):
    s = m = 0
    for x in nr:
        s = s + 1 if x <= 0 else 0
        m = max(m, s)
    return m


L = []
L.append("# Reporte del modelo CAUSAL (sin leakage) - 2026-06-12\n")
L.append("Modelos Approach B reentrenados sobre el master causal. "
         "AUC test: 15m 0.548 | 1h 0.549 | 4h 0.546. Coste 0.0012. "
         "Semantica de threshold: p_win_isotonic >= thr (sin tope), "
         "1 senal max por vela (top EV), igual que el bot.\n")
L.append("**p_win maximo alcanzable (calibracion isotonic):** "
         "15m 0.613 | 1h 0.678 | 4h 1.000 - por eso con thr 0.67 "
         "solo opera 4h.\n")

# ============================ 1. grid de thresholds
for ds in ["val", "test"]:
    L.append(f"\n## 1. Grid de thresholds ({ds.upper()}) - todas las senales\n")
    L.append("| TF | thr | n señales | señales/día | %long | win rate | "
             "expectancy | PF | retorno total |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for tf in TFS:
        for thr in THRESHOLDS:
            s = bot_stream(sig[ds], tf, thr)
            if len(s) < 5:
                continue
            days = max(1, (s.timestamp.max() - s.timestamp.min()).days)
            nr = s.net012.values
            pos, neg = nr[nr > 0], -nr[nr < 0]
            pf = pos.sum() / neg.sum() if neg.sum() > 0 else np.inf
            L.append(f"| {tf} | {thr:.2f} | {len(s)} | {len(s)/days:.2f} | "
                     f"{(s.side=='long').mean()*100:.0f}% | "
                     f"{(nr>0).mean():.3f} | {nr.mean():+.5f} | {pf:.2f} | "
                     f"{nr.sum():+.2f} |")

# ============================ 2. regla del bot (1 pos, notional fijo)
L.append("\n## 2. Regla del bot (1 posicion, notional fijo 100 EUR) - TEST\n")
L.append("PnL en EUR sobre 100 fijos, sin interes compuesto.\n")
L.append("| TF | thr | trades | trades/día | PnL total | PnL/día | "
         "peor día | mejor día | racha pérd. max | %días neg |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
detail = {}
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
        daily = (pd.Series(pnl, index=t.timestamp).resample("1D").sum())
        detail[(tf, thr)] = (t, daily)
        L.append(f"| {tf} | {thr:.2f} | {len(t)} | {len(t)/days:.2f} | "
                 f"{pnl.sum():+.1f} | {daily.mean():+.3f} | {daily.min():+.2f} | "
                 f"{daily.max():+.2f} | {maxstreak(pnl)} | "
                 f"{(daily<0).mean()*100:.0f}% |")

# ============================ 3. apalancamiento (mejores configs)
L.append("\n## 3. Apalancamiento (TEST, lineal con L, funding incluido)\n")
L.append("Liquidacion aislada aprox: 1/L - 0.5% margen mantenimiento. "
         "Seguro = distancia liq >= 3x SL max.\n")
L.append("| TF | thr | L | PnL/día EUR | funding/día | peor día | "
         "peor racha EUR | SL max | dist.liq | seguro |")
L.append("|---|---|---|---|---|---|---|---|---|---|")
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
        ok = "SI" if lv <= L_safe else ("JUSTO" if lv <= 1/(2*max_sl+MM) else "NO")
        L.append(f"| {tf} | {thr:.2f} | {lv} | "
                 f"{daily.mean()*lv - fund1*lv:+.3f} | {-fund1*lv:.3f} | "
                 f"{daily.min()*lv:+.2f} | {worst*lv:+.2f} | "
                 f"{max_sl*100:.2f}% | {(1/lv-MM)*100:.1f}% | {ok} |")

# ============================ 4. compuesto informativo
L.append("\n## 4. Compuesto (informativo, reinvirtiendo todo) - TEST\n")
L.append("| TF | thr | capital final desde 100 EUR | dias |")
L.append("|---|---|---|---|")
for (tf, thr), (t, _) in detail.items():
    if thr not in (0.58, 0.60, 0.65, 0.67, 0.70):
        continue
    eq = NOTIONAL * np.cumprod(1 + t.net012.values)
    days = max(1, (t.timestamp.max() - t.timestamp.min()).days)
    L.append(f"| {tf} | {thr:.2f} | {eq[-1]:,.1f} EUR | {days} |")

L.append("\n## 5. Notas\n")
L.append("- 15m: ningun threshold es rentable (expectancy <= 0); con 0.67 "
         "nunca opera (p_iso max 0.613) y eso es proteccion, no fallo.")
L.append("- 1h: edge marginal en 0.55-0.62; con 0.67 opera casi nunca "
         "(p_iso max 0.678).")
L.append("- 4h: el unico TF solido; senales escasas (paciencia).")
L.append("- Comparar el comportamiento del bot en vivo contra la seccion 2 "
         "(misma regla).")

(OUT_DIR / "REPORTE_MODELO_CAUSAL.md").write_text("\n".join(L), encoding="utf-8")
print("OK ->", OUT_DIR / "REPORTE_MODELO_CAUSAL.md")
