"""Estimacion de volatilidad y mapeo a decile.

En el master original `vol_pred` se construye con modelos especificos por TF
(ewma_20 para 1m/5m/15m, lgbm para 15m/1h, xgb para 4h). Aqui usamos una
aproximacion EWMA del True Range como proxy. Es lo bastante razonable para
mapear a decile en live.

vol_decile_bounds.json fue construido del set de entrenamiento. Cada decile
1..10 tiene rango [vol_pred_min, vol_pred_max].
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.technicals import true_range


def predict_vol_ewma(ohlcv: pd.DataFrame, span: int = 20) -> float:
    """vol_pred = ultima EWMA del true_range, normalizado por close."""
    tr = true_range(ohlcv)
    tr_pct = tr / ohlcv["close"]
    ewma = tr_pct.ewm(span=span, adjust=False).mean()
    return float(ewma.iloc[-1])


def load_decile_bounds(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def vol_to_decile(vol_pred: float, bounds: list[dict]) -> int:
    """Mapea vol_pred a 1..10 segun rangos guardados.

    Si esta fuera de rango: clipea al extremo (decile 1 o 10).
    Si NaN: devuelve 5 (medio) como fallback razonable.
    """
    if vol_pred is None or np.isnan(vol_pred):
        return 5
    # bounds esta ordenado por decile 1..10 (low vol -> high vol)
    for b in bounds:
        if vol_pred <= b["vol_pred_max"]:
            return int(b["decile"])
    return int(bounds[-1]["decile"])
