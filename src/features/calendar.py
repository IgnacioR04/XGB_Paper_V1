"""Features de calendario (prefijo cal_). No requiere datos externos.

Replica T12_calendar del proyecto de investigacion (formulas exactas) ademas
de las ciclicas seno/coseno que usa el bot. Cubre las 22 cal_ del schema.
"""
from __future__ import annotations

import calendar as pycal
from datetime import date

import numpy as np
import pandas as pd


def _nth_dow_of_month(year: int, month: int, n: int, dow: int):
    """n-esimo dia-de-semana del mes. dow: 0=lunes..6=domingo. n: 1..5."""
    c = pycal.Calendar()
    days = [d for d in c.itermonthdates(year, month)
            if d.month == month and d.weekday() == dow]
    return days[n - 1] if len(days) >= n else None


def _is_fomc_day(ts) -> bool:
    if ts.month not in (1, 3, 5, 6, 7, 9, 11, 12):
        return False
    target = _nth_dow_of_month(ts.year, ts.month, 3, 2)  # 3er miercoles
    return target is not None and ts.date() == target


def _is_nfp_day(ts) -> bool:
    target = _nth_dow_of_month(ts.year, ts.month, 1, 4)  # 1er viernes
    return target is not None and ts.date() == target


def _last_friday(ts) -> bool:
    last_day = pycal.monthrange(ts.year, ts.month)[1]
    for d in range(last_day, last_day - 7, -1):
        if date(ts.year, ts.month, d).weekday() == 4:
            return ts.date() == date(ts.year, ts.month, d)
    return False


def _is_futures_expiry(ts) -> bool:
    target = _nth_dow_of_month(ts.year, ts.month, 3, 4)  # 3er viernes
    return target is not None and ts.date() == target


def compute_calendar(index: pd.DatetimeIndex) -> pd.DataFrame:
    idx = pd.DatetimeIndex(index).tz_convert("UTC") if index.tz is not None \
        else pd.DatetimeIndex(index).tz_localize("UTC")
    out = pd.DataFrame(index=index)
    hour = idx.hour

    # --- Basicas ---
    out["cal_hour"] = hour
    out["cal_day_of_week"] = idx.dayofweek
    out["cal_weekend"] = (idx.dayofweek >= 5).astype("int8")
    out["cal_month"] = idx.month
    out["cal_quarter"] = idx.quarter
    out["cal_day_of_month"] = idx.day

    # --- Sesiones (UTC, identico a T12) ---
    out["cal_is_asia_session"] = ((hour >= 0) & (hour < 8)).astype("int8")
    out["cal_is_europe_session"] = ((hour >= 7) & (hour < 16)).astype("int8")
    out["cal_is_us_session"] = ((hour >= 13) & (hour < 21)).astype("int8")
    out["cal_overlap_eu_us"] = ((hour >= 13) & (hour < 16)).astype("int8")

    # --- Ciclicas seno/coseno (las usa el bot) ---
    out["cal_hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["cal_hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["cal_dow_sin"] = np.sin(2 * np.pi * idx.dayofweek / 7)
    out["cal_dow_cos"] = np.cos(2 * np.pi * idx.dayofweek / 7)

    # --- Eventos macro (aproximacion por calendario fijo, identico a T12) ---
    out["cal_fomc_day"] = pd.Series([_is_fomc_day(t) for t in idx],
                                    index=index).astype("int8")
    out["cal_cpi_day"] = pd.Series(idx.day, index=index).isin((10, 11, 12, 13)).astype("int8")
    out["cal_nfp_day"] = pd.Series([_is_nfp_day(t) for t in idx],
                                   index=index).astype("int8")
    out["cal_pce_day"] = pd.Series([_last_friday(t) for t in idx],
                                   index=index).astype("int8")

    # cal_fomc_week: misma semana ISO que el ultimo fomc_day
    ts_ser = pd.Series(idx, index=index)
    fomc_weeks = ts_ser.where(out["cal_fomc_day"] == 1).ffill()
    out["cal_fomc_week"] = (
        pd.Index(idx).isocalendar().week.to_numpy()
        == fomc_weeks.dt.isocalendar().week.to_numpy()
    ).astype("int8")

    # min hasta / horas tras el proximo/anterior evento macro
    event_mask = (out["cal_fomc_day"] | out["cal_cpi_day"]
                  | out["cal_nfp_day"] | out["cal_pce_day"]).astype(bool)
    event_ts_next = ts_ser.where(event_mask).bfill()
    out["cal_min_until_macro_event"] = (
        (event_ts_next - ts_ser).dt.total_seconds() / 60).clip(lower=0)
    event_ts_past = ts_ser.where(event_mask).ffill()
    out["cal_hours_after_macro_event"] = (
        (ts_ser - event_ts_past).dt.total_seconds() / 3600)

    # --- Eventos crypto ---
    mins_to_funding = (8 - hour % 8) * 60 - idx.minute
    out["cal_funding_near"] = (mins_to_funding <= 60).astype("int8")
    out["cal_options_expiry"] = pd.Series([_last_friday(t) for t in idx],
                                          index=index).astype("int8")
    out["cal_futures_expiry"] = pd.Series([_is_futures_expiry(t) for t in idx],
                                          index=index).astype("int8")

    # --- Cierres ---
    minute = idx.minute
    days_in_month = pd.Series(idx, index=index).dt.days_in_month.to_numpy()
    out["cal_daily_close_near"] = ((hour == 23) & (minute >= 30)).astype("int8")
    out["cal_weekly_close_near"] = (
        (idx.dayofweek == 6) & (hour == 23) & (minute >= 30)).astype("int8")
    out["cal_monthly_close_near"] = (
        (idx.day == days_in_month) & (hour == 23) & (minute >= 30)).astype("int8")

    return out
