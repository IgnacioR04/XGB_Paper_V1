import datetime as dt
import pandas as pd

from src.execution.position_manager import (
    check_exit_intrabar, check_timeout, compute_return,
    find_exit_in_klines, pnl_eur,
)


def test_long_tp_hit():
    res = check_exit_intrabar("long", entry_price=100.0,
                               tp_price=102.0, sl_price=98.0,
                               candle_high=102.5, candle_low=99.5)
    assert res == ("TP", 102.0)


def test_long_sl_hit():
    res = check_exit_intrabar("long", 100.0, 102.0, 98.0,
                               candle_high=101.0, candle_low=97.5)
    assert res == ("SL", 98.0)


def test_short_tp_hit():
    res = check_exit_intrabar("short", 100.0, 98.0, 102.0,
                               candle_high=100.5, candle_low=97.0)
    assert res == ("TP", 98.0)


def test_short_sl_hit():
    res = check_exit_intrabar("short", 100.0, 98.0, 102.0,
                               candle_high=103.0, candle_low=99.0)
    assert res == ("SL", 102.0)


def test_intrabar_conflict_sl_first():
    res = check_exit_intrabar("long", 100.0, 102.0, 98.0,
                               candle_high=103.0, candle_low=97.0,
                               intrabar_rule="SL_first")
    assert res == ("SL", 98.0)


def test_no_hit():
    res = check_exit_intrabar("long", 100.0, 102.0, 98.0,
                               candle_high=101.0, candle_low=99.0)
    assert res is None


def test_pnl_long_winning():
    # +2% bruto, -0.12% coste = +1.88%
    pnl = pnl_eur("long", 100, 102, notional_eur=100, cost=0.0012)
    assert abs(pnl - (100 * (0.02 - 0.0012))) < 1e-9


def test_pnl_short_winning():
    # short con precio bajando de 100 a 98
    pnl = pnl_eur("short", 100, 98, notional_eur=100, cost=0.0012)
    expected = 100 * ((100 / 98 - 1) - 0.0012)
    assert abs(pnl - expected) < 1e-9


def test_compute_return_cost_applied():
    r = compute_return("long", 100, 100, cost=0.0012)
    assert r == -0.0012


def test_check_timeout():
    now = dt.datetime(2026, 6, 9, 14, 0, tzinfo=dt.timezone.utc)
    assert check_timeout("2026-06-09T13:00:00+00:00", now) is True
    assert check_timeout("2026-06-09T15:00:00+00:00", now) is False


def test_find_exit_in_klines():
    pos = {
        "side": "long", "entry_price": 100.0,
        "tp_price": 102.0, "sl_price": 98.0,
        "entry_time": "2026-06-09T14:00:00+00:00",
    }
    base = dt.datetime(2026, 6, 9, 14, 0, tzinfo=dt.timezone.utc)
    rows = [
        {"open_time": base + dt.timedelta(minutes=15), "close_time": base + dt.timedelta(minutes=30),
         "high": 100.5, "low": 99.5},
        {"open_time": base + dt.timedelta(minutes=30), "close_time": base + dt.timedelta(minutes=45),
         "high": 101.2, "low": 99.8},
        {"open_time": base + dt.timedelta(minutes=45), "close_time": base + dt.timedelta(minutes=60),
         "high": 103.0, "low": 100.5},   # toca TP=102
        {"open_time": base + dt.timedelta(minutes=60), "close_time": base + dt.timedelta(minutes=75),
         "high": 104.0, "low": 102.0},
    ]
    df = pd.DataFrame(rows)
    out = find_exit_in_klines(pos, df)
    assert out["exit_reason"] == "TP"
    assert out["exit_price"] == 102.0
