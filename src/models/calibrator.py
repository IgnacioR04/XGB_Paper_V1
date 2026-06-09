"""Calibrador isotonic reconstruido desde el JSON compact.

El JSON compact contiene los `x_thresholds` y `y_thresholds` por timeframe.
Reconstruimos un IsotonicRegression de sklearn al vuelo.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
from sklearn.isotonic import IsotonicRegression

from ..utils.errors import ArtifactMissing


@lru_cache(maxsize=1)
def _load_compact_json(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        raise ArtifactMissing(f"Calibrator JSON not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def get_calibrator(compact_path: str, tf: str) -> IsotonicRegression:
    data = _load_compact_json(compact_path)
    if tf not in data:
        raise ArtifactMissing(f"No calibrator for tf={tf} in {compact_path}")
    block = data[tf]
    xs = np.asarray(block["x_thresholds"], dtype=float)
    ys = np.asarray(block["y_thresholds"], dtype=float)
    iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)
    iso.X_thresholds_ = xs
    iso.y_thresholds_ = ys
    iso.X_min_ = float(xs.min())
    iso.X_max_ = float(xs.max())
    iso.f_ = None  # se reconstruye internamente al llamar predict
    # IsotonicRegression usa una interp1d en _build_f. Forzamos el rebuild:
    from scipy.interpolate import interp1d
    iso.f_ = interp1d(xs, ys, kind="linear", bounds_error=False,
                       fill_value=(ys[0], ys[-1]))
    iso.increasing_ = True
    return iso


def calibrate(p_raw: np.ndarray, compact_path: str, tf: str) -> np.ndarray:
    iso = get_calibrator(compact_path, tf)
    return iso.predict(np.clip(p_raw, 1e-6, 1 - 1e-6))
