import pandas as pd
from src.strategy.probability_filters import filter_by_band, in_band, top_ev


def test_in_band_closed_below_open_above():
    assert in_band(0.65, 0.65, 0.70) is True
    assert in_band(0.699, 0.65, 0.70) is True
    assert in_band(0.70, 0.65, 0.70) is False   # abierta arriba
    assert in_band(0.5, 0.65, 0.70) is False


def test_in_band_allow_above():
    assert in_band(0.80, 0.65, 0.70, allow_above_band=True) is True
    assert in_band(0.50, 0.65, 0.70, allow_above_band=True) is False


def test_filter_by_band_default():
    df = pd.DataFrame({"p_win_calibrated": [0.5, 0.65, 0.68, 0.70, 0.80]})
    out = filter_by_band(df, 0.65, 0.70)
    assert list(out["p_win_calibrated"]) == [0.65, 0.68]


def test_filter_by_band_above():
    df = pd.DataFrame({"p_win_calibrated": [0.5, 0.65, 0.80]})
    out = filter_by_band(df, 0.65, 0.70, allow_above_band=True)
    assert list(out["p_win_calibrated"]) == [0.65, 0.80]


def test_top_ev_picks_highest():
    df = pd.DataFrame({"EV_pred": [0.001, 0.005, 0.003], "side": ["long", "short", "long"]})
    out = top_ev(df, 1)
    assert out.iloc[0]["EV_pred"] == 0.005
