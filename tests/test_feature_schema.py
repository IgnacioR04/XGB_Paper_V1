import json
from pathlib import Path

import pandas as pd
import pytest

from src.features.feature_schema import (
    load_schema, reorder_to_schema, validate_row,
)


REPO = Path(__file__).resolve().parent.parent
SCHEMA_PATH = REPO / "artifacts" / "schemas" / "feature_schema.json"


@pytest.fixture(scope="module")
def schema():
    return load_schema(SCHEMA_PATH)


def test_schema_has_615_features(schema):
    assert schema["n_features"] == 615
    assert len(schema["features_in_order"]) == 615


def test_schema_first_10_are_candidate_level(schema):
    first = schema["features_in_order"][:10]
    expected = ["vol_pred", "vol_decile", "tp_mult", "sl_mult", "H",
                "tp_pct", "sl_pct", "side_long", "barrier_quality_score",
                "p_break_even"]
    assert first == expected


def test_reorder_fills_missing_with_nan(schema):
    row = pd.Series({"vol_pred": 0.005, "tp_pct": 0.02, "sl_pct": 0.01})
    out = reorder_to_schema(row, schema)
    assert len(out) == 615
    assert out["vol_pred"] == 0.005
    assert pd.isna(out["barrier_quality_score"])


def test_validate_row_counts_missing(schema):
    row = pd.Series({"vol_pred": 0.005})
    diag = validate_row(row, schema, strict=False)
    assert diag["n_expected"] == 615
    assert diag["n_missing"] == 614
