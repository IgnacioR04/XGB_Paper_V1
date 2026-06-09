"""Logica de PnL y deteccion de cierre para posiciones paper.

Soporta long y short (este ultimo sintetico).
"""
from __future__ import annotations

import datetime as dt
from typing import Any

import pandas as pd


def compute_return(side: str, entry_price: float, exit_price: float,
                   cost: float = 0.0012) -> float:
    """Return neto incluyendo coste round-trip."""
    if side == "long":
        gross = exit_price / entry_price - 1.0
    elif side == "short":
        gross = entry_price / exit_price - 1.0
    else:
        raise ValueError(f"side desconocido: {side}")
    return gross - cost


def pnl_eur(side: str, entry_price: float, exit_price: float,
            notional_eur: float, cost: float = 0.0012) -> float:
    return notional_eur * compute_return(side, entry_price, exit_price, cost)


def check_exit_intrabar(side: str, entry_price: float,
                        tp_price: float, sl_price: float,
                        candle_high: float, candle_low: float,
                        intrabar_rule: str = "SL_first") -> tuple[str, float] | None:
    """Devuelve (exit_reason, exit_price) si la vela toca TP o SL. Else None.

    Para long: tp si high>=tp, sl si low<=sl.
    Para short: tp si low<=tp, sl si high>=sl.

    Si la vela toca AMBOS, usa la regla conservadora intrabar_rule.
    """
    if side == "long":
        hit_tp = candle_high >= tp_price
        hit_sl = candle_low <= sl_price
    else:
        hit_tp = candle_low <= tp_price
        hit_sl = candle_high >= sl_price

    if hit_tp and hit_sl:
        if intrabar_rule == "SL_first":
            return ("SL", sl_price)
        if intrabar_rule == "TP_first":
            return ("TP", tp_price)
        return ("SL", sl_price)
    if hit_tp:
        return ("TP", tp_price)
    if hit_sl:
        return ("SL", sl_price)
    return None


def check_timeout(timeout_time_iso: str, now: dt.datetime) -> bool:
    if not timeout_time_iso:
        return False
    t = dt.datetime.fromisoformat(timeout_time_iso)
    if t.tzinfo is None:
        t = t.replace(tzinfo=dt.timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=dt.timezone.utc)
    return now >= t


def find_exit_in_klines(position: dict, klines: pd.DataFrame,
                        intrabar_rule: str = "SL_first") -> dict | None:
    """Revisa las velas desde el entry hacia adelante y devuelve dict con
    {exit_reason, exit_price, exit_time} si alguna vela cierra la posicion.
    """
    entry_time = pd.Timestamp(position["entry_time"])
    if entry_time.tzinfo is None:
        entry_time = entry_time.tz_localize("UTC")
    klines = klines.copy()
    if klines["open_time"].dt.tz is None:
        klines["open_time"] = klines["open_time"].dt.tz_localize("UTC")

    # Solo velas tras la entry
    sub = klines[klines["open_time"] > entry_time].sort_values("open_time")
    for _, row in sub.iterrows():
        res = check_exit_intrabar(
            position["side"], position["entry_price"],
            position["tp_price"], position["sl_price"],
            float(row["high"]), float(row["low"]),
            intrabar_rule=intrabar_rule,
        )
        if res is not None:
            reason, price = res
            return {"exit_reason": reason, "exit_price": price,
                    "exit_time": row["close_time"]}
    return None
