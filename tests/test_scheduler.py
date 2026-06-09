import datetime as dt
from src.utils.time_utils import (
    candle_close_time, get_due_timeframes, last_closed_candle_start,
    next_candle_close, to_utc,
)


def test_last_closed_15m():
    now = dt.datetime(2026, 6, 9, 14, 17, 0, tzinfo=dt.timezone.utc)
    open_t = last_closed_candle_start("15m", now)
    # ultima vela CERRADA: la 14:00-14:15
    assert open_t == dt.datetime(2026, 6, 9, 14, 0, 0, tzinfo=dt.timezone.utc)
    assert candle_close_time(open_t, "15m") == dt.datetime(2026, 6, 9, 14, 15, 0,
                                                            tzinfo=dt.timezone.utc)


def test_last_closed_1h():
    now = dt.datetime(2026, 6, 9, 15, 5, 0, tzinfo=dt.timezone.utc)
    open_t = last_closed_candle_start("1h", now)
    assert open_t == dt.datetime(2026, 6, 9, 14, 0, 0, tzinfo=dt.timezone.utc)


def test_next_candle_close_4h():
    now = dt.datetime(2026, 6, 9, 14, 5, 0, tzinfo=dt.timezone.utc)
    nxt = next_candle_close("4h", now)
    assert nxt == dt.datetime(2026, 6, 9, 16, 0, 0, tzinfo=dt.timezone.utc)


def test_due_window_within():
    # justo despues del cierre 14:15 con delay 20s
    now = dt.datetime(2026, 6, 9, 14, 15, 30, tzinfo=dt.timezone.utc)
    due = get_due_timeframes(now, {"15m": 20, "1h": 30, "4h": 60},
                              {"15m": 4, "1h": 8, "4h": 15})
    by_tf = {d["timeframe"]: d for d in due}
    assert by_tf["15m"]["is_due"] is True


def test_due_window_too_late():
    # 14:15 cierre + 4 min max_late => >= 14:19:21
    now = dt.datetime(2026, 6, 9, 14, 25, 0, tzinfo=dt.timezone.utc)
    due = get_due_timeframes(now, {"15m": 20, "1h": 30, "4h": 60},
                              {"15m": 4, "1h": 8, "4h": 15})
    by_tf = {d["timeframe"]: d for d in due}
    # Para 15m la ultima vela cerrada en 14:25 es la de 14:00-14:15. cierre+delay+late = 14:15+0:20+4min = 14:19:20
    assert by_tf["15m"]["is_late"] is True
    assert by_tf["15m"]["is_due"] is False


def test_due_window_with_realistic_late_threshold():
    # Con max_late 15m=12, GitHub Actions arrancando a 14:25 (10 min tarde)
    # SI debe procesarse la vela 14:00-14:15
    now = dt.datetime(2026, 6, 9, 14, 25, 0, tzinfo=dt.timezone.utc)
    due = get_due_timeframes(now, {"15m": 20, "1h": 30, "4h": 60},
                              {"15m": 12, "1h": 45, "4h": 180})
    by_tf = {d["timeframe"]: d for d in due}
    assert by_tf["15m"]["is_due"] is True
    assert by_tf["15m"]["is_late"] is False
