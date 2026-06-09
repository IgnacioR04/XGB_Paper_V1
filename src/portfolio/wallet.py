"""Gestion de carteras paper. Una por timeframe.

El estado vive en data/state/wallet_<tf>.json y se carga/guarda atomicamente.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ..utils.io import read_json, write_json_atomic


def _empty_wallet(tf: str, initial_capital_eur: float) -> dict[str, Any]:
    return {
        "timeframe": tf,
        "initial_capital_eur": float(initial_capital_eur),
        "cash_eur": float(initial_capital_eur),
        "equity_eur": float(initial_capital_eur),
        "realized_pnl_eur": 0.0,
        "unrealized_pnl_eur": 0.0,
        "n_trades": 0,
        "n_wins": 0,
        "n_losses": 0,
        "open_position_id": None,
        "equity_curve": [],          # lista de {ts, equity}
    }


def wallet_path(state_dir: Path, tf: str) -> Path:
    return Path(state_dir) / f"wallet_{tf}.json"


def load_or_init(state_dir: Path, tf: str, initial_capital_eur: float) -> dict:
    p = wallet_path(state_dir, tf)
    if p.exists():
        return read_json(p)
    w = _empty_wallet(tf, initial_capital_eur)
    write_json_atomic(p, w)
    return w


def save(state_dir: Path, wallet: dict) -> None:
    write_json_atomic(wallet_path(state_dir, wallet["timeframe"]), wallet)


def apply_trade_close(wallet: dict, pnl_eur: float, exit_ts: str) -> None:
    wallet["cash_eur"] = float(wallet["cash_eur"]) + float(pnl_eur)
    wallet["equity_eur"] = wallet["cash_eur"]
    wallet["realized_pnl_eur"] = float(wallet["realized_pnl_eur"]) + float(pnl_eur)
    wallet["unrealized_pnl_eur"] = 0.0
    wallet["n_trades"] = int(wallet["n_trades"]) + 1
    if pnl_eur >= 0:
        wallet["n_wins"] = int(wallet["n_wins"]) + 1
    else:
        wallet["n_losses"] = int(wallet["n_losses"]) + 1
    wallet["open_position_id"] = None
    curve = wallet.setdefault("equity_curve", [])
    curve.append({"ts": exit_ts, "equity": wallet["equity_eur"]})
    # Limitar el tamano del curve in-state (los puntos persisten en CSV trades igualmente)
    if len(curve) > 5000:
        wallet["equity_curve"] = curve[-5000:]


def apply_position_open(wallet: dict, position_id: str) -> None:
    wallet["open_position_id"] = position_id
