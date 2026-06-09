"""Paper broker: ejecuta operaciones simuladas y gestiona open_positions.json."""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Any

from ..utils.io import read_json, write_json_atomic
from ..utils.logging_utils import get_logger

log = get_logger("paper_broker")


def positions_path(state_dir: Path) -> Path:
    return Path(state_dir) / "open_positions.json"


def processed_path(state_dir: Path) -> Path:
    return Path(state_dir) / "processed_candles.json"


def load_open_positions(state_dir: Path) -> dict[str, dict]:
    return read_json(positions_path(state_dir), default={})


def save_open_positions(state_dir: Path, positions: dict[str, dict]) -> None:
    write_json_atomic(positions_path(state_dir), positions)


def load_processed(state_dir: Path) -> dict[str, list[str]]:
    return read_json(processed_path(state_dir), default={})


def save_processed(state_dir: Path, processed: dict[str, list[str]]) -> None:
    write_json_atomic(processed_path(state_dir), processed)


def mark_signal_processed(state_dir: Path, signal_id: str) -> None:
    p = load_processed(state_dir)
    seen = set(p.get("signal_ids", []))
    if signal_id in seen:
        return
    seen.add(signal_id)
    p["signal_ids"] = sorted(seen)
    # Cap a 50k para evitar crecimiento sin limite
    if len(p["signal_ids"]) > 50000:
        p["signal_ids"] = p["signal_ids"][-50000:]
    save_processed(state_dir, p)


def is_signal_processed(state_dir: Path, signal_id: str) -> bool:
    p = load_processed(state_dir)
    return signal_id in set(p.get("signal_ids", []))


def open_position(state_dir: Path,
                  signal_id: str,
                  timeframe: str,
                  symbol: str,
                  side: str,
                  entry_price: float,
                  tp_price: float,
                  sl_price: float,
                  timeout_time: dt.datetime,
                  notional_eur: float,
                  p_win: float,
                  EV_pred: float,
                  candidate_id: int,
                  entry_price_quality: str = "ticker",
                  ideal_entry_price: float | None = None,
                  vol_decile: int | None = None) -> dict[str, Any]:
    pos_id = uuid.uuid4().hex[:12]
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    pos: dict[str, Any] = {
        "position_id": pos_id,
        "signal_id": signal_id,
        "timeframe": timeframe,
        "symbol": symbol,
        "side": side,
        "entry_time": now.isoformat(),
        "entry_price": float(entry_price),
        "ideal_entry_price": float(ideal_entry_price) if ideal_entry_price is not None else None,
        "entry_price_quality": entry_price_quality,
        "tp_price": float(tp_price),
        "sl_price": float(sl_price),
        "timeout_time": timeout_time.isoformat() if isinstance(timeout_time, dt.datetime) else timeout_time,
        "notional_eur": float(notional_eur),
        "p_win": float(p_win),
        "EV_pred": float(EV_pred),
        "candidate_id": int(candidate_id),
        "vol_decile": int(vol_decile) if vol_decile is not None else None,
        "status": "open",
    }
    positions = load_open_positions(state_dir)
    positions[pos_id] = pos
    save_open_positions(state_dir, positions)
    mark_signal_processed(state_dir, signal_id)
    log.info("Opened paper position %s: %s %s %s @ %.2f tp=%.2f sl=%.2f",
             pos_id, timeframe, side, symbol, entry_price, tp_price, sl_price)
    return pos


def close_position(state_dir: Path, position_id: str,
                   exit_price: float, exit_reason: str,
                   exit_time: dt.datetime | None = None) -> dict[str, Any]:
    positions = load_open_positions(state_dir)
    if position_id not in positions:
        raise KeyError(f"Position {position_id} not found")
    pos = positions[position_id]
    if exit_time is None:
        exit_time = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    pos["exit_time"] = exit_time.isoformat()
    pos["exit_price"] = float(exit_price)
    pos["exit_reason"] = exit_reason
    pos["status"] = "closed"
    del positions[position_id]
    save_open_positions(state_dir, positions)
    return pos
