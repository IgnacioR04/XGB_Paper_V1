"""Analisis del primer dia de paper trading.

Examina decisions, trades y features logs para responder:
1. Distribucion de p_win por TF (vs backtest)
2. Cuantas senales rechazadas por wallet ocupada
3. Proporcion long/short de senales
4. Calidad de features en vivo (NaN por TF)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)

# ---------------------------------------------------------------- decisions
dec_files = sorted((BASE / "data" / "logs" / "decisions").glob("decisions_*.csv"))
dec = pd.concat([pd.read_csv(f) for f in dec_files], ignore_index=True)
print(f"=== DECISIONS: {len(dec)} filas brutas ===")

# Dedupe: cada vela se evalua en varios ticks -> quedarnos con 1 fila por vela
dec_u = dec.sort_values("tick_ts_utc").drop_duplicates(
    subset=["timeframe", "candle_close_time"], keep="last")
# pero si alguna evaluacion de esa vela fue YES, prevalece
yes_keys = set(map(tuple, dec.loc[dec.decision == "YES",
                                  ["timeframe", "candle_close_time"]].values))
dec_u["was_yes"] = dec_u.apply(
    lambda r: (r.timeframe, r.candle_close_time) in yes_keys, axis=1)

print(f"Velas unicas evaluadas: {len(dec_u)}")
print(dec_u.groupby("timeframe").size().rename("velas_evaluadas"))
print()

for tf in ["15m", "1h", "4h"]:
    d = dec_u[dec_u.timeframe == tf]
    if d.empty:
        continue
    p = pd.to_numeric(d["p_win_max"], errors="coerce").dropna()
    print(f"--- {tf}: p_win_max sobre {len(p)} velas ---")
    print(f"  min={p.min():.4f}  p25={p.quantile(.25):.4f}  med={p.median():.4f}"
          f"  p75={p.quantile(.75):.4f}  max={p.max():.4f}")
    print(f"  % velas con p>=0.65: {(p >= 0.65).mean()*100:.1f}%"
          f"  | p>=0.70: {(p >= 0.70).mean()*100:.1f}%")
    reasons = d["reason_no_signal"].fillna("(YES)").value_counts()
    print("  motivos:", dict(reasons))
    sides = d.loc[d.winner_side.notna(), "winner_side"].value_counts()
    print("  side del ganador (cuando hay candidato en banda):", dict(sides))
    print()

# Senales perdidas por wallet ocupada
lost = dec_u[dec_u.reason_no_signal == "WALLET_AT_MAX_POSITIONS"]
print(f"=== Senales validas RECHAZADAS por wallet ocupada: {len(lost)} ===")
if not lost.empty:
    print(lost.groupby("timeframe").size())
    print(lost[["candle_close_time", "timeframe", "winner_side",
                "winner_p_win_calibrated", "EV_max"]].to_string(index=False))
print()

# ---------------------------------------------------------------- trades
trades_f = BASE / "data" / "paper_trades" / "trades.parquet"
if not trades_f.exists():
    cands = list((BASE / "data" / "paper_trades").glob("*.parquet")) + \
            list((BASE / "data" / "paper_trades").glob("*.csv"))
    print("trades candidates:", cands)
    trades_f = cands[0] if cands else None
if trades_f and trades_f.exists():
    tr = (pd.read_parquet(trades_f) if trades_f.suffix == ".parquet"
          else pd.read_csv(trades_f))
    print(f"=== TRADES CERRADOS: {len(tr)} ===")
    cols = [c for c in ["timeframe", "side", "entry_time", "exit_time",
                        "entry_price", "exit_price", "exit_reason", "p_win",
                        "pnl_eur", "pnl_pct"] if c in tr.columns]
    print(tr[cols].to_string(index=False))
    print()
    print("PnL por TF:")
    print(tr.groupby("timeframe").agg(
        n=("pnl_eur", "size"), pnl_total=("pnl_eur", "sum"),
        win_rate=("pnl_eur", lambda s: (s > 0).mean())))
    print()
    print("Por motivo de salida:", dict(tr.exit_reason.value_counts()))
    print("Por side:", dict(tr.side.value_counts()))
print()

# Posiciones abiertas
for tf in ["15m", "1h", "4h"]:
    pos_f = BASE / "data" / "state" / f"wallet_{tf}.json"
    if pos_f.exists():
        w = json.loads(pos_f.read_text())
        eq = w.get("equity_eur")
        npos = len(w.get("open_positions", []))
        print(f"wallet {tf}: equity={eq} abiertas={npos}")
print()

# ---------------------------------------------------------------- features
feat_files = sorted((BASE / "data" / "logs" / "features").glob("features_*.csv"))
if feat_files:
    fe = pd.concat([pd.read_csv(f) for f in feat_files], ignore_index=True)
    meta_cols = [c for c in fe.columns if not c[0].islower() or c in (
        "timeframe", "tick_ts_utc", "candle_close_time")]
    feat_cols = [c for c in fe.columns
                 if c not in ("timeframe", "tick_ts_utc", "candle_close_time")]
    print(f"=== FEATURES LOG: {len(fe)} filas, {len(feat_cols)} features ===")
    for tf in ["15m", "1h", "4h"]:
        f = fe[fe.timeframe == tf]
        if f.empty:
            continue
        sub = f[feat_cols].apply(pd.to_numeric, errors="coerce")
        nan_frac = sub.isna().mean()
        n_allnan = (nan_frac == 1.0).sum()
        print(f"--- {tf}: {len(f)} filas | features 100% NaN: {n_allnan} "
              f"| NaN medio: {nan_frac.mean()*100:.1f}% ---")
        worst = nan_frac[nan_frac > 0.5].sort_values(ascending=False)
        print(f"  features con >50% NaN ({len(worst)}):")
        for name, v in worst.head(40).items():
            print(f"    {name}: {v*100:.0f}%")
        if len(worst) > 40:
            print(f"    ... y {len(worst)-40} mas")
    print()
