"""Carga y validacion del feature_schema.json.

El schema fija el ORDEN exacto de las 615 features que espera el XGB.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..utils.errors import FeatureSchemaMismatch
from ..utils.logging_utils import get_logger

log = get_logger("feature_schema")


def load_schema(path: Path | str) -> dict:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_row(row: pd.Series, schema: dict, strict: bool = False) -> dict:
    """Verifica que la fila tenga TODAS las features del schema.

    Si strict=True y falta alguna, levanta FeatureSchemaMismatch.
    Si strict=False, devuelve dict con counts de missing.
    """
    expected = schema["features_in_order"]
    missing = [c for c in expected if c not in row.index]
    nan_count = sum(1 for c in expected if c in row.index and pd.isna(row[c]))
    if missing and strict:
        raise FeatureSchemaMismatch(
            f"Missing {len(missing)} features. First 10: {missing[:10]}"
        )
    return {
        "n_expected": len(expected),
        "n_missing": len(missing),
        "n_nan": nan_count,
        "missing_first10": missing[:10],
    }


def reorder_to_schema(row: pd.Series, schema: dict) -> pd.Series:
    """Devuelve la fila con columnas exactamente en el orden del schema.
    Features faltantes se rellenan con NaN."""
    expected = schema["features_in_order"]
    out = pd.Series(index=expected, dtype="float64")
    for c in expected:
        if c in row.index:
            out[c] = row[c]
    return out
