"""Calculos de Valor Esperado y derivados."""
from __future__ import annotations

import numpy as np


def expected_value(p_win: float, tp_pct: float, sl_pct: float,
                   cost: float = 0.0012) -> float:
    """EV = p_win * TP - (1 - p_win) * SL - cost.

    Por convencion tp_pct y sl_pct son positivos (tamanos brutos del movimiento).
    """
    return p_win * tp_pct - (1 - p_win) * sl_pct - cost


def p_break_even(tp_pct: float, sl_pct: float, cost: float = 0.0012) -> float:
    """Probabilidad de win necesaria para EV = 0."""
    denom = tp_pct + sl_pct
    if denom <= 0:
        return float("nan")
    return (sl_pct + cost) / denom


def edge_over_be(p_win: float, tp_pct: float, sl_pct: float,
                 cost: float = 0.0012) -> float:
    return p_win - p_break_even(tp_pct, sl_pct, cost)


def annotate_candidates(df, cost: float = 0.0012):
    """Anade columnas EV_pred, p_break_even_calc, edge_over_be al DataFrame.

    Requiere p_win_calibrated, tp_pct, sl_pct presentes.
    """
    df = df.copy()
    df["EV_pred"] = df["p_win_calibrated"] * df["tp_pct"] \
        - (1 - df["p_win_calibrated"]) * df["sl_pct"] - cost
    denom = df["tp_pct"] + df["sl_pct"]
    df["p_break_even_calc"] = (df["sl_pct"] + cost) / denom.replace(0, np.nan)
    df["edge_over_be"] = df["p_win_calibrated"] - df["p_break_even_calc"]
    return df
