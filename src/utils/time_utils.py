"""Utilidades de tiempo: detectar timeframes due, mapeo TF -> minutos.

El scheduler funciona asi: GitHub Actions corre cada 5 minutos. En cada tick
preguntamos `get_due_timeframes(now)` que devuelve la lista de timeframes
cuyo CIERRE de vela paso recientemente y aun no fue procesado.
"""
from __future__ import annotations

import datetime as dt
from typing import Iterable


TF_MINUTES: dict[str, int] = {"15m": 15, "1h": 60, "4h": 240}


def to_utc(now: dt.datetime | None = None) -> dt.datetime:
    if now is None:
        now = dt.datetime.utcnow()
    if now.tzinfo is None:
        return now.replace(tzinfo=dt.timezone.utc)
    return now.astimezone(dt.timezone.utc)


def last_closed_candle_start(tf: str, now: dt.datetime | None = None) -> dt.datetime:
    """Devuelve el inicio (open_time) de la ultima vela cerrada del timeframe."""
    minutes = TF_MINUTES[tf]
    now = to_utc(now)
    # epoch minutes
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    elapsed_minutes = int((now - epoch).total_seconds() // 60)
    # ultima vela CERRADA: la que empieza en floor((elapsed - 1)/minutes) * minutes
    last_open_min = (elapsed_minutes // minutes) * minutes - minutes
    return epoch + dt.timedelta(minutes=last_open_min)


def candle_close_time(open_time: dt.datetime, tf: str) -> dt.datetime:
    return open_time + dt.timedelta(minutes=TF_MINUTES[tf])


def next_candle_close(tf: str, now: dt.datetime | None = None) -> dt.datetime:
    """Inicio de la PROXIMA vela del timeframe (= close de la actual)."""
    minutes = TF_MINUTES[tf]
    now = to_utc(now)
    epoch = dt.datetime(1970, 1, 1, tzinfo=dt.timezone.utc)
    elapsed_minutes = int((now - epoch).total_seconds() // 60)
    next_open_min = ((elapsed_minutes // minutes) + 1) * minutes
    return epoch + dt.timedelta(minutes=next_open_min)


def get_due_timeframes(
    now: dt.datetime,
    candle_close_delay_seconds: dict[str, int],
    max_late_minutes: dict[str, int],
    timeframes: Iterable[str] = ("15m", "1h", "4h"),
) -> list[dict]:
    """Devuelve lista de timeframes elegibles para procesar en este tick.

    Para cada TF, calcula el `candle_close_time` de la ultima vela cerrada y
    decide si estamos dentro de la ventana:

        close_time + delay  <=  now  <=  close_time + delay + max_late

    Si SI -> incluye con metadata. Si NO -> se salta.
    """
    now = to_utc(now)
    out: list[dict] = []
    for tf in timeframes:
        open_t = last_closed_candle_start(tf, now)
        close_t = candle_close_time(open_t, tf)
        delay = dt.timedelta(seconds=candle_close_delay_seconds.get(tf, 30))
        window_end = close_t + delay + dt.timedelta(minutes=max_late_minutes.get(tf, 5))
        earliest_run = close_t + delay
        is_due = earliest_run <= now <= window_end
        seconds_into_window = (now - earliest_run).total_seconds()
        out.append({
            "timeframe": tf,
            "candle_open_time": open_t,
            "candle_close_time": close_t,
            "is_due": is_due,
            "seconds_since_eligible": seconds_into_window,
            "is_late": now > window_end,
        })
    return out


def make_signal_id(symbol: str, timeframe: str, candle_close_time_utc: dt.datetime,
                   side: str, candidate_id: int) -> str:
    """Clave estable para idempotencia."""
    ts = candle_close_time_utc.strftime("%Y%m%dT%H%M%SZ")
    return f"{symbol}|{timeframe}|{ts}|{side}|{candidate_id}"
