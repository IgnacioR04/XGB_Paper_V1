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
                kdf, source = bspot.fetch_last_n_closed_with_fallback(symbol, tf, n=64)
                if kdf.empty:
                    log.warning("Empty klines for exit check %s %s (source=%s)",
                                symbol, tf, source)
                    klines_cache[key] = None
                else:
                    klines_cache[key] = kdf
            except Exception as e:
                log.warning("Could not fetch klines for exit check %s %s: %s",
                            symbol, tf, e)
                klines_cache[key] = None
        klines = klines_cache[key]

        exit_info = None
        # 1) Velas CERRADAS del timeframe de la posicion (igual que el backtest)
        if klines is not None and not klines.empty:
            exit_info = pm.find_exit_in_klines(pos, klines, intrabar_rule=intrabar_rule)

        # 2) Toque intra-vela: las TP/SL son ordenes en reposo en un exchange
        #    real, se ejecutan al tocar el precio aunque la vela del TF siga
        #    en curso. Detectamos el toque con velas de 1m cerradas (cubre la
        #    vela parcial actual con ~1 min de retraso). barriers_v2 etiqueto
        #    con replay 1m, asi que esto es fiel al entrenamiento.
        if exit_info is None:
            key_1m = (symbol, "1m")
            if key_1m not in klines_cache:
                try:
                    m1, _src = bspot.fetch_last_n_closed_with_fallback(symbol, "1m", n=20)
                    klines_cache[key_1m] = m1 if not m1.empty else None
                except Exception as e:
                    log.warning("Could not fetch 1m klines for intrabar check: %s", e)
                    klines_cache[key_1m] = None
            m1 = klines_cache[key_1m]
            if m1 is not None:
                exit_info = pm.find_exit_in_klines(pos, m1, intrabar_rule=intrabar_rule)
                if exit_info is not None:
                    log.info("Intrabar (1m) exit detected for %s: %s",
                             pos_id, exit_info["exit_reason"])

        # 3) Ticker actual: cubre el ultimo minuto no cerrado
        if exit_info is None:
            try:
                price, _src = bspot.fetch_ticker_price_with_fallback(symbol)
                res = pm.check_exit_intrabar(
                    pos["side"], pos["entry_price"],
                    pos["tp_price"], pos["sl_price"],
                    candle_high=price, candle_low=price,
                    intrabar_rule=intrabar_rule,
                )
                if res is not None:
                    reason, barrier_price = res
                    exit_info = {"exit_reason": reason,
                                 "exit_price": barrier_price,
                                 "exit_time": now}
                    log.info("Ticker exit detected for %s: %s @ %.2f",
                             pos_id, reason, barrier_price)
            except Exception as e:
                log.warning("Ticker intrabar check failed for %s: %s", pos_id, e)

        if exit_info is None and pm.check_timeout(pos.get("timeout_time", ""), now):
            # Salir a precio actual via ticker (con fallback Coinbase)
            try:
                price, _src = bspot.fetch_ticker_price_with_fallback(symbol)
            except Exception as e:
                # Ultimo recurso: close de la ultima vela disponible
                if klines is not None and not klines.empty:
                    price = float(klines.iloc[-1]["close"])
                    log.warning("Ticker failed for %s, using last close %.2f",
                                pos_id, price)
                else:
                    log.error("Timeout reached for %s but no price source: %s",
                              pos_id, e)
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

        # Actualizar wallet
        w = wallet_mod.load_or_init(state_dir, tf,
                                     wallets_initial_capital.get(tf, 100.0))
        cash_before = float(w.get("cash_eur", 0.0))   # capital de la cartera ANTES del cierre
        closed["pnl_eur"] = pnl
        # pnl_pct = cambio REAL de la cartera (incluye leverage), coherente con
        # la equity. notional = cash_before * leverage, asi que esto = leverage *
        # retorno de precio. Antes se dividia por notional (retorno sin apalancar)
        # y no cuadraba con la equity mostrada.
        closed["pnl_pct"] = pnl / cash_before if cash_before else 0.0
        # tambien guardamos el retorno sin apalancar por trazabilidad
        closed["pnl_pct_notional"] = (
            pnl / pos["notional_eur"] if pos.get("notional_eur") else 0.0)
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
