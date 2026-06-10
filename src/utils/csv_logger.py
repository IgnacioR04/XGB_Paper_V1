"""Logs CSV auditables. Dos archivos rotados por mes:

- data/logs/decisions/decisions_YYYY-MM.csv
    Una fila por (tick, timeframe) con la decision y motivos.
    Columna `decision` = "YES" si abrio operacion, "NO" si no.
    Si NO: motivo + estadisticas del tick (p_win_max, EV_max, etc).
    Si YES: ademas los datos del candidato ganador.

- data/logs/features/features_YYYY-MM.csv
    Una fila por (tick, timeframe) con la fila de features que se paso al
    modelo (615 columnas + ts + timeframe). Sirve para auditar exactamente
    que vio el modelo.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

import pandas as pd


# --------------------------------------------------------------------------
# Decisions CSV
# --------------------------------------------------------------------------

DECISION_COLUMNS = [
    "tick_ts_utc",            # cuando corrio el tick
    "timeframe",
    "candle_open_time",       # apertura de la vela t (la CERRADA usada para inferencia)
    "candle_close_time",      # cierre de t = momento de la pregunta al modelo
    "execution_candle_open",  # apertura de t+1 = cuando se ejecuta la operacion
                              # (= candle_close_time; t+1 abre cuando t cierra)
    "btc_close",              # precio BTC en esa vela
    "vol_pred",
    "vol_decile",
    "n_candidates_initial",
    "n_in_band",
    "p_win_max",              # mejor p_win calibrado de los candidatos
    "p_win_min",
    "EV_max",
    "prob_band_min",
    "prob_band_max",
    "decision",               # "YES" o "NO"
    "reason_no_signal",       # solo si NO
    # Si YES, datos del candidato ganador:
    "winner_side",
    "winner_candidate_id",
    "winner_p_win_calibrated",
    "winner_EV_pred",
    "winner_tp_pct",
    "winner_sl_pct",
    "winner_tp_mult",
    "winner_sl_mult",
    "winner_H",
    "winner_p_break_even",
    "winner_edge_over_be",
    "signal_id",
    "position_id",            # si se abrio efectivamente
    "entry_price",            # precio real de entrada (NaN si dry-run)
    "entry_price_quality",    # ticker | fallback_last_close
    "dry_run",
    "mode",                   # closed_candle_only | preclose_preview
]


def _decision_csv_path(logs_dir: Path, when: dt.datetime) -> Path:
    return Path(logs_dir) / "decisions" / f"decisions_{when:%Y-%m}.csv"


def append_decision(logs_dir: Path, row: dict[str, Any]) -> Path:
    """Anade una fila al CSV de decisiones del mes correspondiente."""
    when = dt.datetime.fromisoformat(row["tick_ts_utc"]) \
        if isinstance(row.get("tick_ts_utc"), str) else (row.get("tick_ts_utc") or dt.datetime.utcnow())
    if when.tzinfo is None:
        when = when.replace(tzinfo=dt.timezone.utc)
    p = _decision_csv_path(logs_dir, when)
    p.parent.mkdir(parents=True, exist_ok=True)

    # Garantizar todas las columnas
    full = {c: row.get(c, "") for c in DECISION_COLUMNS}
    df = pd.DataFrame([full], columns=DECISION_COLUMNS)
    header = not p.exists()
    df.to_csv(p, mode="a", header=header, index=False)
    return p


# --------------------------------------------------------------------------
# Features CSV
# --------------------------------------------------------------------------

def _features_csv_path(logs_dir: Path, when: dt.datetime) -> Path:
    return Path(logs_dir) / "features" / f"features_{when:%Y-%m}.csv"


def append_features(logs_dir: Path, feature_row: pd.Series,
                     timeframe: str, tick_ts: dt.datetime,
                     candle_open: dt.datetime,
                     vol_pred: float, vol_decile: int) -> Path:
    """Persiste la fila de features que se paso al modelo (los 615 valores +
    metadata)."""
    if tick_ts.tzinfo is None:
        tick_ts = tick_ts.replace(tzinfo=dt.timezone.utc)
    p = _features_csv_path(logs_dir, tick_ts)
    p.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "tick_ts_utc": tick_ts.isoformat(),
        "timeframe": timeframe,
        "candle_open_time": candle_open.isoformat() if isinstance(candle_open, dt.datetime)
            else str(candle_open),
        "vol_pred": vol_pred,
        "vol_decile": vol_decile,
    }
    # feature_row es una pd.Series con los 605 market features
    for k, v in feature_row.items():
        record[str(k)] = v

    df = pd.DataFrame([record])
    if p.exists():
        # Garantizar mismas columnas (union)
        existing_cols = pd.read_csv(p, nrows=1).columns.tolist()
        new_cols = [c for c in df.columns if c not in existing_cols]
        if new_cols:
            # Re-leer y reescribir con union de columnas si llegaron nuevas
            old = pd.read_csv(p)
            for c in new_cols:
                old[c] = pd.NA
            combined = pd.concat([old, df], ignore_index=True, sort=False)
            combined.to_csv(p, index=False)
            return p
        df = df.reindex(columns=existing_cols)
        df.to_csv(p, mode="a", header=False, index=False)
    else:
        df.to_csv(p, index=False)
    return p
