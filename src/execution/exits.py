"""Monitor de posiciones abiertas: cierra TP/SL/TIMEOUT en cada tick."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

from ..data import binance_spot as bspot
from ..portfolio import wallet as wallet_mod
from ..utils.io import append_csv, append_parquet
from ..utils.logging_utils import get_logger
from . import paper_broker
from . import position_manager as pm

log = get_logger("exits")


def monitor_and_close_positions(state_dir: Path, trades_dir: Path,
                                wallets_initial_capital: dict[str, float],
                                cost: float = 0.0012,
                                intrabar_rule: str = "SL_first") -> list[dict]:
    """Recorre todas las posiciones abiertas y las cierra si procede.

    Devuelve la lista de trades cerrados en este tick.
    """
    positions = paper_broker.load_open_positions(state_dir)
    if not positions:
        return []

    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    closed_trades = []
    klines_cache: dict[tuple[str, str], object] = {}

    for pos_id, pos in list(positions.items()):
        tf = pos["timeframe"]
        symbol = pos["symbol"]

        # Cache de klines por (symbol, tf) para no llamar 3 veces si hay 3 posiciones
        key = (symbol, tf)
        if key not in klines_cache:
            try:
                klines_cache[key] = bspot.fetch_last_n_closed(symbol, tf, n=64)
            except Exception as e:
                log.warning("Could not fetch klines for exit check %s %s: %s",
                            symbol, tf, e)
                klines_cache[key] = None
        klines = klines_cache[key]

        exit_info = None
        if klines is not None and not klines.empty:
            exit_info = pm.find_exit_in_klines(pos, klines, intrabar_rule=intrabar_rule)

        if exit_info is None and pm.check_timeout(pos.get("timeout_time", ""), now):
            # Salir a precio actual via ticker
            try:
                price = bspot.fetch_ticker_price(symbol)
            except Exception as e:
                log.error("Timeout reached for %s but ticker failed: %s", pos_id, e)
                continue
            exit_info = {"exit_reason": "TIMEOUT", "exit_price": price, "exit_time": now}

        if exit_info is None:
            continue

        pnl = pm.pnl_eur(pos["side"], pos["entry_price"],
                          exit_info["exit_price"], pos["notional_eur"], cost=cost)
        closed = paper_broker.close_position(state_dir, pos_id,
                                              exit_info["exit_price"],
                                              exit_info["exit_reason"],
                                              exit_time=exit_info["exit_time"]
                                              if isinstance(exit_info["exit_time"], dt.datetime)
                                              else None)
        closed["pnl_eur"] = pnl
        closed["pnl_pct"] = pnl / pos["notional_eur"] if pos["notional_eur"] else 0.0

        # Actualizar wallet
        w = wallet_mod.load_or_init(state_dir, tf,
                                     wallets_initial_capital.get(tf, 100.0))
        exit_ts = closed["exit_time"]
        wallet_mod.apply_trade_close(w, pnl, exit_ts)
        wallet_mod.save(state_dir, w)
        closed["wallet_equity_after"] = w["equity_eur"]

        # Persistir el trade cerrado
        trades_dir = Path(trades_dir)
        trades_dir.mkdir(parents=True, exist_ok=True)
        append_csv(trades_dir / "trades.csv", closed)
        append_parquet(trades_dir / "trades.parquet", closed)
        closed_trades.append(closed)
        log.info("Closed position %s reason=%s pnl_eur=%.4f equity_after=%.2f",
                 pos_id, exit_info["exit_reason"], pnl, w["equity_eur"])

    return closed_trades
