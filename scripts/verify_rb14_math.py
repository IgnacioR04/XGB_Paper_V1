# -*- coding: utf-8 -*-
"""Verificacion de la matematica de RB14:

1) Por que 100%/1pos gana a 25%/4pos (descomposicion: captura vs sizing).
2) El +62 EUR/dia es efecto compuesto: recalcular con notional FIJO de 100 EUR
   (sin reinversion) para comparar con el bot real del dia 1.
"""
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")
DIR_RB = BASE / "Data/model_outputs/real_backtest"
DIR_LBL = BASE / "Data/model_outputs/directional/datasets"

COST = 0.0012
BANDS = {"15m": (0.65, 0.70), "1h": (0.70, 0.75), "4h": (0.65, 0.70)}
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}

test = pd.read_parquet(DIR_RB / "signals/signals_test.parquet")
test["timestamp"] = pd.to_datetime(test["timestamp"])
lk = []
for tf in BANDS:
    h = pd.read_parquet(DIR_LBL / f"labels_compact_{tf}.parquet",
                        columns=["timestamp", "candidate_id", "timeframe",
                                 "H", "gross_return", "time_to_exit"])
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    lk.append(h)
lk = pd.concat(lk, ignore_index=True)
test = test.merge(lk, on=["timestamp", "candidate_id", "timeframe"], how="left")
test["net_return"] = test["gross_return"] - COST
test["dur_min"] = test["timeframe"].map(TF_MIN) * test["time_to_exit"].fillna(test["H"])


def bot_signals(df, tf):
    lo, hi = BANDS[tf]
    sub = df[(df.timeframe == tf) & (df.p_win_isotonic >= lo)
             & (df.p_win_isotonic < hi)]
    idx = sub.groupby("timestamp")["EV_pred"].idxmax()
    return sub.loc[idx].sort_values("timestamp").reset_index(drop=True)


def simulate(sig, frac, max_pos, capital0=100.0, compound=True):
    cash = capital0
    open_pos, taken = [], []
    fixed_pnl = 0.0
    for _, row in sig.iterrows():
        ts = row["timestamp"]
        open_pos = [p for p in open_pos
                    if not (p["close_ts"] <= ts and
                            (cash := cash + p["notional"] * (1 + p["net_return"])) is None)] \
            if False else open_pos
        still = []
        for p in open_pos:
            if p["close_ts"] <= ts:
                cash += p["notional"] * (1 + p["net_return"])
            else:
                still.append(p)
        open_pos = still
        if len(open_pos) >= max_pos:
            continue
        equity_now = cash + sum(p["notional"] for p in open_pos)
        notional = frac * (equity_now if compound else capital0)
        if compound and notional > cash + 1e-9:
            continue
        if not np.isfinite(row["dur_min"]):
            continue
        cash -= notional
        nr = float(row["net_return"])
        open_pos.append({"close_ts": ts + pd.Timedelta(minutes=float(row["dur_min"])),
                         "notional": notional, "net_return": nr})
        taken.append({"timestamp": ts, "net_return": nr,
                      "pnl_eur": notional * nr})
        fixed_pnl += notional * nr
    for p in open_pos:
        cash += p["notional"] * (1 + p["net_return"])
    return pd.DataFrame(taken), cash


for tf in ["15m"]:
    sig = bot_signals(test, tf)
    days = max(1, (sig.timestamp.max() - sig.timestamp.min()).days)
    print(f"=== {tf}: {len(sig)} senales en banda, {days} dias ===\n")

    # --- descomposicion 100%/1pos vs 25%/4pos ---
    for frac, mp in [(1.0, 1), (0.25, 4)]:
        t, final = simulate(sig, frac, mp, compound=True)
        exp = t.net_return.mean()
        # crecimiento log por trade aproximado = frac * net_return
        log_growth = np.log1p(frac * t.net_return).sum()
        print(f"frac={frac} max_pos={mp}: n_taken={len(t)} "
              f"expectancy/trade={exp:.5f}")
        print(f"  crecimiento = n x frac x expectancy ~ "
              f"{len(t)} x {frac} x {exp:.5f} = "
              f"{len(t)*frac*exp:.3f} (log-aprox {log_growth:.3f})")
        print(f"  capital final compuesto: {final:,.0f} EUR")
    print()
    t1, _ = simulate(sig, 1.0, 1, compound=True)
    t4, _ = simulate(sig, 0.25, 4, compound=True)
    print(f"Ratio de captura 4pos/1pos: {len(t4)}/{len(t1)} = {len(t4)/len(t1):.2f}x")
    print(f"Pero sizing por trade: 0.25x  ->  efecto neto {len(t4)/len(t1)*0.25:.2f}x")
    print()

    # --- notional fijo 100 EUR (sin reinversion) ---
    tfix, _ = simulate(sig, 1.0, 1, compound=False)
    daily_fix = tfix.set_index("timestamp")["pnl_eur"].resample("1D").sum()
    print("=== 100%/1pos con notional FIJO 100 EUR (sin componer) ===")
    print(f"  n trades: {len(tfix)} | PnL total: {tfix.pnl_eur.sum():+.1f} EUR "
          f"en {days} dias")
    print(f"  PnL diario medio: {daily_fix.mean():+.3f} EUR | mediana: "
          f"{daily_fix.median():+.3f}")
    print(f"  p10/p90: {daily_fix.quantile(.1):+.2f} / {daily_fix.quantile(.9):+.2f}")
    print(f"  mejor dia: {daily_fix.max():+.2f} | peor dia: {daily_fix.min():+.2f}")
    print(f"  PnL medio por trade: {tfix.pnl_eur.mean():+.3f} EUR")
    print()

    # --- el dia de +1342 EUR en compuesto: que paso ---
    tcomp, _ = simulate(sig, 1.0, 1, compound=True)
    tcomp["equity_before"] = 100 + tcomp.pnl_eur.cumsum() - tcomp.pnl_eur
    daily_comp = tcomp.set_index("timestamp")["pnl_eur"].resample("1D").sum()
    worst_best = daily_comp.sort_values()
    best_day = worst_best.index[-1]
    eq_that_day = tcomp[tcomp.timestamp.dt.date == best_day.date()]["equity_before"]
    print(f"=== Dia de mayor PnL compuesto: {best_day.date()} "
          f"({worst_best.iloc[-1]:+.0f} EUR) ===")
    print(f"  equity de la cartera ese dia: {eq_that_day.min():,.0f} - "
          f"{eq_that_day.max():,.0f} EUR")
    print(f"  es decir, ese dia el wallet ya valia ~{eq_that_day.mean():,.0f} EUR; "
          f"+{worst_best.iloc[-1]/eq_that_day.mean()*100:.1f}% del equity")
