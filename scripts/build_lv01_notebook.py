# -*- coding: utf-8 -*-
"""Genera notebooks/09_live_validation/LV01_replay_2026_06_10.ipynb.

Notebook de PRUEBAS REALES: replica un dia real del paper trader con la misma
logica del backtest (entrada en open de t+1, TP/SL/H con SL-first sobre velas
del TF, coste 0.0012, cartera 100%/1 posicion) y lo compara con lo que el bot
hizo de verdad ese dia.
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT_NB = REPO / "notebooks/09_live_validation/LV01_replay_2026_06_10.ipynb"

MD0 = """# LV01 - Replay del dia real 2026-06-10 (backtest vs paper trader)

**Objetivo**: coger las senales REALES que genero el bot el 2026-06-10 (las
que ejecuto y las que rechazo por cartera ocupada), simular cada una con la
**misma logica del backtest** (entrada en el open de la vela t+1, TP/SL/H
sobre velas del timeframe, conflicto TP+SL en la misma vela -> SL gana,
coste 0.0012) y comparar:

1. Lo que el motor de backtest dice que habria pasado ese dia.
2. Lo que el paper trader hizo de verdad (trades.parquet del repo).

Si ambos coinciden razonablemente, la ejecucion del bot es fiel al backtest
y las diferencias de rendimiento vienen del MODELO (features), no de la
ejecucion.

Funciona en Colab (descarga los datos del repo de GitHub y las velas de
Binance/Coinbase) y en local (lee los archivos del propio repo).
"""

C1 = """# Celda 1
# Objetivo
# Imports y configuracion. Deteccion local vs Colab.

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests

DAY = "2026-06-10"            # dia a replicar (UTC)
COST = 0.0012                 # round-trip, igual que backtest y bot
CAPITAL_INI = 100.0
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}
GH_RAW = "https://raw.githubusercontent.com/IgnacioR04/XGB_Paper_V1/main"

# Si corremos dentro del repo (local), usar archivos locales
_here = Path.cwd()
LOCAL_REPO = None
for cand in [_here, _here.parent, _here.parent.parent]:
    if (cand / "data" / "logs" / "decisions").exists():
        LOCAL_REPO = cand
        break
print("Modo:", "LOCAL" if LOCAL_REPO else "GitHub raw (Colab)")

DAY_START = pd.Timestamp(DAY)
DAY_END = DAY_START + pd.Timedelta(days=1)
print(f"Ventana: {DAY_START} -> {DAY_END} UTC")
"""

C2 = """# Celda 2
# Objetivo
# Cargar el historial de decisiones del bot y quedarnos con las senales en
# banda del dia (decision YES o rechazadas por cartera ocupada). Cada vela se
# evalua en varios ticks de 5 min -> dedupe por vela priorizando YES.


def load_decisions():
    name = f"decisions_{DAY[:7]}.csv"
    if LOCAL_REPO:
        return pd.read_csv(LOCAL_REPO / "data/logs/decisions" / name)
    r = requests.get(f"{GH_RAW}/data/logs/decisions/{name}", timeout=30)
    r.raise_for_status()
    return pd.read_csv(io.StringIO(r.text))


dec = load_decisions()
dec["candle_close_time"] = (pd.to_datetime(dec["candle_close_time"], utc=True)
                              .dt.tz_localize(None))
dec["_is_yes"] = (dec["decision"] == "YES").astype(int)
dec = (dec.sort_values(["_is_yes", "tick_ts_utc"])
          .drop_duplicates(subset=["timeframe", "candle_close_time"],
                           keep="last"))

day = dec[(dec["candle_close_time"] >= DAY_START)
          & (dec["candle_close_time"] < DAY_END)]
signals = day[day["winner_side"].notna()].sort_values("candle_close_time")
print(f"Velas evaluadas el {DAY}: {len(day)} | senales en banda: {len(signals)}")
print(signals.groupby(["timeframe", "decision"]).size().to_string())
print()
cols = ["candle_close_time", "timeframe", "winner_side",
        "winner_p_win_calibrated", "winner_tp_pct", "winner_sl_pct",
        "winner_H", "decision", "reason_no_signal"]
print(signals[cols].to_string(index=False))
"""

C3 = """# Celda 3
# Objetivo
# Descargar velas reales del dia (+horizonte H) por timeframe.
# Binance primero; si esta geo-bloqueado (Colab US), fallback a Coinbase.


def fetch_binance(interval, start, end):
    url = ("https://api.binance.com/api/v3/klines?symbol=BTCUSDT"
           f"&interval={interval}&startTime={int(start.timestamp()*1000)}"
           f"&endTime={int(end.timestamp()*1000)}&limit=1000")
    r = requests.get(url, timeout=20)
    r.raise_for_status()
    rows = [{"open_time": pd.Timestamp(k[0], unit="ms"),
             "open": float(k[1]), "high": float(k[2]),
             "low": float(k[3]), "close": float(k[4])} for k in r.json()]
    return pd.DataFrame(rows)


def fetch_coinbase(granularity, start, end):
    out = []
    cur = start
    step = pd.Timedelta(seconds=granularity * 290)
    while cur < end:
        chunk_end = min(cur + step, end)
        url = ("https://api.exchange.coinbase.com/products/BTC-USD/candles"
               f"?granularity={granularity}&start={cur.isoformat()}"
               f"&end={chunk_end.isoformat()}")
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        for k in r.json():
            out.append({"open_time": pd.Timestamp(k[0], unit="s"),
                        "low": float(k[1]), "high": float(k[2]),
                        "open": float(k[3]), "close": float(k[4])})
        cur = chunk_end
    df = pd.DataFrame(out).drop_duplicates("open_time")
    return df.sort_values("open_time").reset_index(drop=True)


def resample_4h(df1h):
    df = df1h.set_index("open_time")
    agg = df.resample("4h").agg({"open": "first", "high": "max",
                                  "low": "min", "close": "last"})
    return agg.dropna().reset_index()


def fetch_klines(tf, start, end):
    try:
        df = fetch_binance(tf, start, end)
        if len(df):
            return df, "binance"
    except Exception as e:
        print(f"  binance fallo ({e}); usando coinbase")
    if tf == "4h":
        df = resample_4h(fetch_coinbase(3600, start, end))
    else:
        df = fetch_coinbase({"15m": 900, "1h": 3600}[tf], start, end)
    return df, "coinbase"


klines = {}
for tf in signals["timeframe"].unique():
    horizon = pd.Timedelta(minutes=TF_MIN[tf] * 20)
    df, src = fetch_klines(tf, DAY_START - pd.Timedelta(hours=1),
                           DAY_END + horizon)
    klines[tf] = df
    print(f"{tf}: {len(df)} velas ({src}) "
          f"{df.open_time.min()} -> {df.open_time.max()}")
"""

C4 = """# Celda 4
# Objetivo
# Replay con la logica EXACTA del backtest (BV2_04):
#   - senal en cierre de t -> entrada en el OPEN de la vela t+1
#   - TP/SL chequeados intrabar desde la propia vela de entrada
#   - si TP y SL caen en la misma vela -> SL (conservador)
#   - timeout en el close de la vela H
#   - net = gross - 0.0012


def replay_signal(row, kl):
    tf = row["timeframe"]
    entry_open_time = row["candle_close_time"]   # open de t+1 == cierre de t
    sub = kl[kl["open_time"] >= entry_open_time].reset_index(drop=True)
    if len(sub) == 0:
        return None
    H = int(row["winner_H"])
    entry = float(sub.iloc[0]["open"])
    tp_pct = float(row["winner_tp_pct"])
    sl_pct = float(row["winner_sl_pct"])
    side = row["winner_side"]
    if side == "long":
        tp_price = entry * (1 + tp_pct)
        sl_price = entry * (1 - sl_pct)
    else:
        tp_price = entry * (1 - tp_pct)
        sl_price = entry * (1 + sl_pct)
    n = min(H, len(sub))
    for i in range(n):
        hi, lo = float(sub.iloc[i]["high"]), float(sub.iloc[i]["low"])
        if side == "long":
            hit_tp, hit_sl = hi >= tp_price, lo <= sl_price
        else:
            hit_tp, hit_sl = lo <= tp_price, hi >= sl_price
        if hit_sl:                      # SL-first si ambos
            exit_p, reason, bars = sl_price, "SL", i + 1
            break
        if hit_tp:
            exit_p, reason, bars = tp_price, "TP", i + 1
            break
    else:
        exit_p, reason, bars = float(sub.iloc[n - 1]["close"]), "TIMEOUT", n
    gross = (exit_p / entry - 1) if side == "long" else (entry / exit_p - 1)
    return {"candle_close_time": row["candle_close_time"], "timeframe": tf,
            "side": side, "p_win": row["winner_p_win_calibrated"],
            "entry": entry, "exit": exit_p, "exit_reason": reason,
            "bars_to_exit": bars,
            "dur_min": bars * TF_MIN[tf],
            "net_return": gross - COST,
            "decision_real": row["decision"]}


replays = []
for _, row in signals.iterrows():
    res = replay_signal(row, klines[row["timeframe"]])
    if res:
        replays.append(res)
rep = pd.DataFrame(replays)
print(f"Senales replicadas: {len(rep)}")
print(rep.to_string(index=False))
"""

C5 = """# Celda 5
# Objetivo
# Cartera 100%/1 posicion sobre el replay -> "lo que el backtest dice que
# habria pasado el dia 2026-06-10". La cartera se libera cuando la senal
# replicada toca TP/SL/timeout (duracion real del replay).

rep_sim = {}
for tf in rep["timeframe"].unique():
    sub = rep[rep["timeframe"] == tf].sort_values("candle_close_time")
    cash, open_until = CAPITAL_INI, None
    taken = []
    for _, r in sub.iterrows():
        ts = r["candle_close_time"]
        if open_until is not None and ts < open_until:
            continue
        open_until = ts + pd.Timedelta(minutes=float(r["dur_min"]))
        pnl = cash * r["net_return"]
        cash += pnl
        taken.append({**r.to_dict(), "pnl_eur": pnl, "equity": cash})
    t = pd.DataFrame(taken)
    rep_sim[tf] = t
    print(f"=== {tf}: backtest-replay del {DAY} ===")
    print(f"  senales: {len(sub)} | trades tomados: {len(t)} | "
          f"equity final: {cash:.2f} EUR ({cash-CAPITAL_INI:+.2f})")
    if len(t):
        print(t[["candle_close_time", "side", "entry", "exit_reason",
                 "pnl_eur", "equity"]].to_string(index=False))
    print()
"""

C6 = """# Celda 6
# Objetivo
# Cargar los trades REALES del paper trader (trades.parquet del repo) y
# compararlos con el replay.


def load_trades():
    if LOCAL_REPO:
        return pd.read_parquet(LOCAL_REPO / "data/paper_trades/trades.parquet")
    r = requests.get(f"{GH_RAW}/data/paper_trades/trades.parquet", timeout=30)
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content))


real = load_trades()
real["entry_time"] = (pd.to_datetime(real["entry_time"], utc=True, format="ISO8601")
                        .dt.tz_localize(None))
real["exit_time"] = (pd.to_datetime(real["exit_time"], utc=True, format="ISO8601")
                       .dt.tz_localize(None))
real_day = real[(real["entry_time"] >= DAY_START)
                & (real["entry_time"] < DAY_END)].copy()
print(f"Trades reales del bot con entrada el {DAY}: {len(real_day)}")
print(real_day[["timeframe", "side", "entry_time", "entry_price",
                "exit_time", "exit_reason", "pnl_eur",
                "wallet_equity_after"]].to_string(index=False))
"""

C7 = """# Celda 7
# Objetivo
# Comparacion final replay vs real, trade a trade (matching por hora de
# entrada redondeada a la vela) y en totales del dia.

rows = []
for tf, t in rep_sim.items():
    if len(t) == 0:
        continue
    t = t.copy()
    t["entry_candle"] = pd.to_datetime(t["candle_close_time"])
    r = real_day[real_day["timeframe"] == tf].copy()
    tfm = TF_MIN[tf]
    r["entry_candle"] = r["entry_time"].dt.floor(f"{tfm}min")
    merged = t.merge(r, on="entry_candle", how="outer",
                     suffixes=("_replay", "_real"), indicator=True)
    for _, m in merged.iterrows():
        rows.append({
            "tf": tf,
            "vela_entrada": m["entry_candle"],
            "en_replay": m["_merge"] != "right_only",
            "en_real": m["_merge"] != "left_only",
            "exit_replay": m.get("exit_reason_replay"),
            "exit_real": m.get("exit_reason_real"),
            "pnl_replay": round(m.get("pnl_eur_replay", np.nan), 3)
                if pd.notna(m.get("pnl_eur_replay")) else None,
            "pnl_real": round(m.get("pnl_eur_real", np.nan), 3)
                if pd.notna(m.get("pnl_eur_real")) else None,
        })
comp = pd.DataFrame(rows).sort_values(["tf", "vela_entrada"])
print(comp.to_string(index=False))
print()
for tf in rep_sim:
    t = rep_sim[tf]
    r = real_day[real_day["timeframe"] == tf]
    pnl_rep = t["pnl_eur"].sum() if len(t) else 0.0
    pnl_real = r["pnl_eur"].sum() if len(r) else 0.0
    print(f"{tf}: PnL replay-backtest {pnl_rep:+.2f} EUR "
          f"({len(t)} trades) | PnL real bot {pnl_real:+.2f} EUR "
          f"({len(r)} trades) | diferencia {pnl_real-pnl_rep:+.2f}")
print()
print("Notas sobre diferencias esperables:")
print(" - el bot entra unos segundos despues del cierre con precio ticker")
print("   (Coinbase si Binance esta bloqueado); el replay entra en el open")
print("   exacto de t+1 de las velas descargadas")
print(" - el bot detecta TP/SL con velas 1m + ticker entre ticks (cada 5")
print("   min); el replay resuelve dentro de la vela del TF con SL-first")
print(" - si el dia tuvo ticks perdidos de GitHub Actions, el bot pudo")
print("   abrir/cerrar mas tarde que el replay")
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

OUT_NB.parent.mkdir(parents=True, exist_ok=True)
OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print(f"Notebook escrito: {OUT_NB}")
