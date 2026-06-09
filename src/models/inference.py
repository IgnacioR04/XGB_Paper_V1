"""Predice p_win (raw + calibrated) para un batch de candidatos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from ..utils.logging_utils import get_logger
from .calibrator import calibrate
from .xgb_loader import load_xgb_model

log = get_logger("inference")


def predict_p_win(features_df: pd.DataFrame,
                  model_path: str,
                  calibrator_path: str,
                  tf: str) -> pd.DataFrame:
    """features_df debe tener columnas exactamente en el orden del schema
    (incluyendo las 10 candidate-level + 605 market). Una fila por candidato.

    Returns: DataFrame con columnas ['p_win_raw', 'p_win_isotonic'].
    """
    model = load_xgb_model(model_path)
    expected = model.feature_names
    missing = [c for c in expected if c not in features_df.columns]
    if missing:
        log.warning("Missing %d features (filling NaN). First: %s",
                    len(missing), missing[:5])
        # Anadir todas las columnas faltantes de una vez (evita fragmentation)
        nan_block = pd.DataFrame(
            np.nan, index=features_df.index, columns=missing, dtype="float64",
        )
        features_df = pd.concat([features_df, nan_block], axis=1)
    X = features_df[expected].astype("float32").values
    dmat = xgb.DMatrix(X, feature_names=expected)
    p_raw = model.predict(dmat)
    p_cal = calibrate(p_raw, calibrator_path, tf)
    return pd.DataFrame({"p_win_raw": p_raw, "p_win_isotonic": p_cal},
                        index=features_df.index)
