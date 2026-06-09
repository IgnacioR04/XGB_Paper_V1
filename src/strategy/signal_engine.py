"""Genera la senal de operacion (si la hay) para un timeframe.

Flujo:
    1. Construir features live (last closed candle).
    2. Estimar vol_pred y vol_decile.
    3. Cargar candidatos de la libreria para (tf, decile, sides).
    4. Para cada candidato, expandir la fila de features con sus campos
       candidate-level (vol_pred, vol_decile, tp_mult, sl_mult, H, tp_pct,
       sl_pct, side_long, barrier_quality_score, p_break_even).
    5. Inferencia XGB + calibracion -> p_win_calibrated.
    6. Calcular EV.
    7. Filtrar por banda de probabilidad.
    8. Elegir top_EV (1 candidato).
    9. Devolver el candidato seleccionado o None.
"""
from __future__ import annotations

import pandas as pd

from ..models.inference import predict_p_win
from ..utils.logging_utils import get_logger
from .candidates import candidates_for, load_library
from .ev import annotate_candidates
from .probability_filters import filter_by_band, top_ev

log = get_logger("signal_engine")


def expand_candidates_with_features(candidates: pd.DataFrame,
                                    feature_row: pd.Series,
                                    vol_pred: float,
                                    vol_decile: int) -> pd.DataFrame:
    """Para cada candidato crea una fila con todas las features del schema:
        - 10 candidate-level vienen del candidato
        - 605 market features vienen de feature_row (mismas para todos)
    """
    n = len(candidates)
    # Broadcast feature_row a n filas
    market = pd.DataFrame([feature_row.to_dict()] * n)
    # Sobreescribir candidate-level
    market["vol_pred"] = vol_pred
    market["vol_decile"] = vol_decile
    market["tp_mult"] = candidates["tp_mult"].values
    market["sl_mult"] = candidates["sl_mult"].values
    market["H"] = candidates["H"].values
    market["tp_pct"] = candidates["tp_pct_mean"].values
    market["sl_pct"] = candidates["sl_pct_mean"].values
    market["side_long"] = (candidates["side"] == "long").astype(int).values
    market["barrier_quality_score"] = candidates["barrier_quality_score"].values
    market["p_break_even"] = candidates["p_break_even"].values
    return market


def evaluate_timeframe(timeframe: str,
                       feature_row: pd.Series,
                       vol_pred: float,
                       vol_decile: int,
                       library_path: str,
                       model_path: str,
                       calibrator_path: str,
                       prob_band_min: float,
                       prob_band_max: float,
                       allowed_sides: list[str],
                       allow_above_band: bool = False,
                       require_positive_ev: bool = False,
                       cost: float = 0.0012,
                       top_k_per_decile_side: int = 5) -> dict:
    """Devuelve dict con `winner` (Series o None) y `diagnostics`."""
    lib = load_library(library_path)
    cands = candidates_for(lib, timeframe, vol_decile,
                            sides=tuple(allowed_sides),
                            top_k=top_k_per_decile_side)

    diag = {
        "timeframe": timeframe,
        "vol_pred": vol_pred,
        "vol_decile": vol_decile,
        "n_candidates_initial": int(len(cands)),
    }

    if cands.empty:
        diag["reason_no_signal"] = "NO_CANDIDATES_IN_LIBRARY"
        return {"winner": None, "diagnostics": diag, "all_scored": pd.DataFrame()}

    X = expand_candidates_with_features(cands, feature_row, vol_pred, vol_decile)
    pred = predict_p_win(X, model_path, calibrator_path, timeframe)
    scored = cands.copy()
    scored["p_win_raw"] = pred["p_win_raw"].values
    scored["p_win_calibrated"] = pred["p_win_isotonic"].values
    scored["tp_pct"] = X["tp_pct"].values
    scored["sl_pct"] = X["sl_pct"].values
    scored = annotate_candidates(scored, cost=cost)

    diag["p_win_max"] = float(scored["p_win_calibrated"].max())
    diag["p_win_min"] = float(scored["p_win_calibrated"].min())
    diag["EV_max"] = float(scored["EV_pred"].max())

    in_band = filter_by_band(scored, prob_band_min, prob_band_max,
                              allow_above_band=allow_above_band)
    diag["n_in_band"] = int(len(in_band))

    if in_band.empty:
        diag["reason_no_signal"] = "NO_CANDIDATE_IN_PROB_BAND"
        return {"winner": None, "diagnostics": diag, "all_scored": scored}

    if require_positive_ev:
        in_band = in_band[in_band["EV_pred"] > 0]
        diag["n_positive_ev"] = int(len(in_band))
        if in_band.empty:
            diag["reason_no_signal"] = "NO_POSITIVE_EV"
            return {"winner": None, "diagnostics": diag, "all_scored": scored}

    winner = top_ev(in_band, n=1).iloc[0]
    return {"winner": winner, "diagnostics": diag, "all_scored": scored}
