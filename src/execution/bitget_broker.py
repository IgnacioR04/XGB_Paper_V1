"""Broker de operativa REAL en Bitget USDT-M Futures (BTCUSDT).

Mismo interfaz que `paper_broker.open_position` / `close_position`, asi el
tick principal no necesita cambios — basta seleccionar el broker via perfil.

Diferencias clave vs paper:
- Lee el equity real de la cuenta antes de abrir (notional = equity * leverage).
- Coloca la entrada como ORDEN MARKET en Bitget.
- Coloca TP y SL como ORDENES CONDICIONALES en el exchange (reduceOnly) para
  que se respeten aunque el bot caiga.
- Al cerrar (TP/SL/TIMEOUT detectados por exits.py): cancela las ordenes
  condicionales y cierra a market.

Kill switch: si la equity de la cuenta cae por debajo del minimo configurado,
NO se abre ninguna posicion y se marca un fichero KILL_SWITCH_TRIGGERED.
"""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path
from typing import Any

from ..data.bitget_client import BitgetClient, BitgetError
from ..utils.io import read_json, write_json_atomic
from ..utils.logging_utils import get_logger

log = get_logger("bitget_broker")

SYMBOL = "BTCUSDT"
MIN_SIZE_BTC = 0.001   # tamano minimo de Bitget para BTC perpetuo


def positions_path(state_dir: Path) -> Path:
    return Path(state_dir) / "open_positions.json"


def processed_path(state_dir: Path) -> Path:
    return Path(state_dir) / "processed_candles.json"


def kill_switch_path(state_dir: Path) -> Path:
    return Path(state_dir) / "KILL_SWITCH_TRIGGERED"


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
    if signal_id in seen: return
    seen.add(signal_id)
    p["signal_ids"] = sorted(seen)
    if len(p["signal_ids"]) > 50000: p["signal_ids"] = p["signal_ids"][-50000:]
    save_processed(state_dir, p)


def is_signal_processed(state_dir: Path, signal_id: str) -> bool:
    p = load_processed(state_dir)
    return signal_id in set(p.get("signal_ids", []))


def check_kill_switch(state_dir: Path, min_equity_usdt: float) -> tuple[bool, float]:
    """Devuelve (killed, equity_actual). killed=True -> NO operar mas."""
    p = kill_switch_path(state_dir)
    if p.exists():
        return True, 0.0
    try:
        c = BitgetClient()
        if not c.authenticated:
            log.error("Bitget sin credenciales; bloqueando operativa")
            return True, 0.0
        eq = c.equity_usdt(SYMBOL)
    except Exception as e:
        log.error("No se pudo leer equity Bitget: %s; bloqueando por seguridad", e)
        return True, 0.0
    if eq <= min_equity_usdt:
        p.write_text(f"triggered_at={dt.datetime.utcnow().isoformat()}Z equity={eq}")
        log.error("KILL SWITCH disparado: equity %.2f <= min %.2f USDT", eq, min_equity_usdt)
        return True, eq
    return False, eq


def _round_size(size_btc: float) -> float:
    # Bitget BTCUSDT perpetuo: 3 decimales en size.
    # FLOOR (no round) para no exceder NUNCA el presupuesto de margen:
    # round() puede redondear hacia arriba (0.00156 -> 0.002) e inflar el
    # notional por encima del balance disponible -> error 40762.
    import math
    s = math.floor(size_btc * 1000) / 1000.0
    return s if s > 0 else 0.0


# Fraccion del equity que arriesgamos como margen. El resto (5%) es colchon
# para la comision de entrada (0.06% taker), el movimiento de precio entre la
# lectura del equity y la orden, y el redondeo de tamano. Sin este colchon
# Bitget rechaza con "The order amount exceeds the balance" (40762).
EQUITY_SAFETY_FACTOR = 0.95


def open_position(state_dir: Path, signal_id: str, timeframe: str, symbol: str,
                  side: str, entry_price: float, tp_price: float, sl_price: float,
                  timeout_time: dt.datetime, notional_eur: float,
                  p_win: float, EV_pred: float, candidate_id: int,
                  entry_price_quality: str = "bitget_market",
                  ideal_entry_price: float | None = None,
                  vol_decile: int | None = None,
                  leverage: float = 1.0, regime: str = "") -> dict[str, Any]:
    """Abre una posicion REAL en Bitget. Devuelve dict similar a paper_broker."""
    c = BitgetClient()
    if not c.authenticated:
        raise BitgetError("Bitget sin credenciales; no se puede abrir posicion")

    # 1) ajustar leverage en el exchange para ambos lados (crossed)
    try:
        c.set_margin_mode(symbol, "crossed")
    except Exception as e:
        log.warning("set_margin_mode (puede ser idempotente): %s", e)
    for hs in ("long", "short"):
        try: c.set_leverage(symbol, int(round(leverage)), hold_side=hs)
        except Exception as e: log.warning("set_leverage %s: %s", hs, e)

    # 2) calcular size en BTC a partir del EQUITY REAL de la cuenta (no del
    #    wallet de config, que puede estar desincronizado). margen disponible =
    #    equity * SAFETY_FACTOR; notional = margen * leverage. Asi el margen
    #    consumido nunca supera el balance y queda colchon para comisiones.
    #    (notional_eur que llega de run_paper_tick se ignora para el sizing real
    #    porque se calcula del wallet de config, que aqui esta desacoplado del
    #    equity real de Bitget.)
    try:
        equity_usdt = c.equity_usdt(symbol)
    except Exception as e:
        log.warning("No se pudo leer equity para sizing (%s); uso notional_eur/leverage", e)
        equity_usdt = float(notional_eur) / max(float(leverage), 1.0)
    margin_usdt = equity_usdt * EQUITY_SAFETY_FACTOR
    notional_usdt = margin_usdt * float(leverage)
    last_price = c.ticker_price(symbol)
    size_btc = _round_size(notional_usdt / last_price)
    log.info("Sizing: equity=%.2f margin=%.2f lev=%.1fx notional=%.2f price=%.1f -> size=%.4f BTC",
             equity_usdt, margin_usdt, leverage, notional_usdt, last_price, size_btc)
    if size_btc < MIN_SIZE_BTC:
        raise BitgetError(
            f"size {size_btc} < min {MIN_SIZE_BTC} BTC; equity={equity_usdt:.2f} "
            f"lev={leverage}x notional={notional_usdt:.2f}. Capital insuficiente "
            f"para el tamano minimo de Bitget (necesitas ~{MIN_SIZE_BTC*last_price/leverage:.0f} USDT de margen).")

    # 3) entrada market
    bg_side = "buy" if side == "long" else "sell"
    pos_id = uuid.uuid4().hex[:12]
    client_oid = f"e{pos_id}"
    resp = c.place_market_order(symbol, bg_side, size_btc, reduce_only=False,
                                 client_oid=client_oid)
    log.info("Bitget market %s %s size=%.4f -> %s", bg_side, symbol, size_btc, resp)

    # 4) TP/SL como ordenes condicionales en el exchange
    hold = "long" if side == "long" else "short"
    try:
        c.place_tp_sl(symbol, hold, tp_price, sl_price, size_btc)
    except Exception as e:
        log.warning("place_tp_sl fallo (la posicion sigue abierta; el bot cerrara por logica): %s", e)

    # 5) persistir en state
    now = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    pos = {
        "position_id": pos_id, "signal_id": signal_id, "timeframe": timeframe,
        "symbol": symbol, "side": side,
        "entry_time": now.isoformat(),
        "entry_price": float(last_price),
        "ideal_entry_price": float(ideal_entry_price) if ideal_entry_price is not None else None,
        "entry_price_quality": entry_price_quality,
        "tp_price": float(tp_price), "sl_price": float(sl_price),
        "timeout_time": timeout_time.isoformat() if isinstance(timeout_time, dt.datetime) else timeout_time,
        # notional REAL ejecutado = size redondeado * precio (no el target, que
        # es mayor por el floor del size). Asi el PnL cuadra con la posicion real.
        "notional_eur": float(size_btc * last_price),    # tratamos como USDT
        "size_btc": float(size_btc),
        "leverage": float(leverage), "regime": regime,
        "p_win": float(p_win), "EV_pred": float(EV_pred),
        "candidate_id": int(candidate_id),
        "vol_decile": int(vol_decile) if vol_decile is not None else None,
        "status": "open",
        "exchange": "bitget",
        "bitget_order": resp,
    }
    positions = load_open_positions(state_dir)
    positions[pos_id] = pos
    save_open_positions(state_dir, positions)
    mark_signal_processed(state_dir, signal_id)
    return pos


def close_position(state_dir: Path, position_id: str,
                   exit_price: float, exit_reason: str,
                   exit_time: dt.datetime | None = None) -> dict[str, Any]:
    """Cierra una posicion: cancela TP/SL pendientes y manda market reduceOnly."""
    positions = load_open_positions(state_dir)
    if position_id not in positions:
        raise KeyError(f"Position {position_id} not found")
    pos = positions[position_id]
    c = BitgetClient()
    # cancelar condicionales
    try: c.cancel_all_plan_orders(pos["symbol"])
    except Exception as e: log.warning("cancel_all_plan_orders: %s", e)
    # cerrar market reduceOnly (lado opuesto)
    close_side = "sell" if pos["side"] == "long" else "buy"
    try:
        c.place_market_order(pos["symbol"], close_side, float(pos["size_btc"]),
                              reduce_only=True, client_oid=f"x{position_id}")
    except Exception as e:
        log.error("close market failed (%s); posicion puede quedar abierta en exchange: %s",
                  position_id, e)
    if exit_time is None:
        exit_time = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    pos["exit_time"] = exit_time.isoformat()
    pos["exit_price"] = float(exit_price)
    pos["exit_reason"] = exit_reason
    pos["status"] = "closed"
    del positions[position_id]
    save_open_positions(state_dir, positions)
    return pos
