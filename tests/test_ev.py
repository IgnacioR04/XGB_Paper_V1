import math
import pandas as pd
import pytest

from src.strategy.ev import (
    annotate_candidates, edge_over_be, expected_value, p_break_even,
)


def test_ev_signs():
    # 60% prob, tp=2%, sl=1%, cost=0.12% -> EV > 0
    assert expected_value(0.6, 0.02, 0.01, 0.0012) > 0


def test_ev_negative_with_high_cost():
    assert expected_value(0.5, 0.005, 0.005, 0.02) < 0


def test_p_break_even_formula():
    pbe = p_break_even(0.02, 0.01, 0.0012)
    # EV at pbe = 0
    assert math.isclose(expected_value(pbe, 0.02, 0.01, 0.0012), 0.0, abs_tol=1e-9)


def test_p_break_even_zero_size():
    assert math.isnan(p_break_even(0.0, 0.0, 0.0012))


def test_edge_positive_when_pwin_above_be():
    assert edge_over_be(0.7, 0.02, 0.01, 0.0012) > 0


def test_annotate_candidates_adds_cols():
    df = pd.DataFrame({
        "p_win_calibrated": [0.5, 0.7],
        "tp_pct": [0.02, 0.015],
        "sl_pct": [0.01, 0.01],
    })
    out = annotate_candidates(df, cost=0.0012)
    assert "EV_pred" in out.columns
    assert "p_break_even_calc" in out.columns
    assert "edge_over_be" in out.columns
    assert (out["EV_pred"].iloc[1] > out["EV_pred"].iloc[0])
