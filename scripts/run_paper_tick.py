"""Tick del paper trader. Entrypoint llamado por GitHub Actions / cron / manual.

Uso:
    python scripts/run_paper_tick.py [--dry-run] [--force-tf 15m,1h,4h]
                                     [--now ISO_UTC]

Flujo:
    1. Setup logger + ensure dirs.
    2. Detectar que timeframes estan due.
    3. Monitorizar y cerrar posiciones abiertas (antes de abrir nuevas).
    4. Para cada TF due:
        a. Build features live.
        b. Estimar vol y decile.
        c. Senal via signal_engine.
        d. Si gana candidato y no hay posicion abierta en la cartera, abrir.
        e. Sino, loguear razon.
    5. Persistir estado + exportar dashboard JSON.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import sys
from pathlib import Path

# Permitir `python scripts/run_paper_tick.py` desde root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import binance_spot as bspot
from src.execution import exits as exits_mod
from src.execution import paper_broker
from src.features.live_feature_builder import build_live_features
from src.features.macro_cache import load_cache, to_feature_dict
from src.features.regime import classify_regime
from src.portfolio import wallet as wallet_mod
from src.strategy.signal_engine import evaluate_timeframe
from src.utils.csv_logger import append_decision, append_features
from src.utils.logging_utils import setup_logger
from src.utils.time_utils import (
    candle_close_time, get_due_timeframes, make_signal_id, to_utc,
)
from src.volatility.ewma_vol import (
    load_decile_bounds, predict_vol_ewma, vol_to_decile,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true",
                   help="No abre/cierra operaciones; solo loguea decisiones.")
    p.add_argument("--force-tf", type=str, default=None,
                   help="Lista CSV de TFs a forzar (ignora ventana de scheduler).")
    p.add_argument("--now", type=str, default=None,
                   help="ISO UTC para override. Util en tests.")
    p.add_argument("--symbol", type=str, default="BTCUSDT")
    p.add_argument("--profile", type=str, default="",
                   help="'' = bot principal; 'oof' = bot del modelo OOF "
                        "(usa config/state/dashboard separados).")
    return p.parse_args()


def export_dashboard_data(cfg) -> None:
    """Exporta JSONs minimos para el dashboard."""
    from src.dashboard.export_json import export_all
    export_all(cfg)


def run_tick(cfg, args) -> int:
    log = setup_logger(cfg.logs_dir / "paper_trader.log",
                       level=logging.INFO)
    log.info("=" * 60)
    log.info("Paper tick start (dry_run=%s)", args.dry_run)
    cfg.ensure_runtime_dirs()

    # Diagnostics que SIEMPRE se escriben (incluso en error) para que el
    # repo refleje cada tick aunque no haya nada que commitear de trades.
    from src.utils.io import write_json_atomic
    diag_payload = {
        "tick_ts_utc": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
        "dry_run": bool(args.dry_run),
        "force_tf": args.force_tf,
        "phases": [],
        "errors": [],
        "due_timeframes": [],
        "tf_results": {},
    }
    def _save_diag():
        try:
            write_json_atomic(cfg.logs_dir / "last_tick.json", diag_payload)
        except Exception:
            pass

    now = to_utc(dt.datetime.fromisoformat(args.now)) if args.now else to_utc()
    paper = cfg.paper
    strat = cfg.strategy

    # 1. Detectar timeframes due
    sched = paper["scheduler"]
    if args.force_tf:
        forced = [t.strip() for t in args.force_tf.split(",")]
        due = [{"timeframe": tf,
                "candle_close_time": candle_close_time(
                    __import__("src.utils.time_utils", fromlist=["last_closed_candle_start"])
                    .last_closed_candle_start(tf, now), tf),
                "is_due": True, "is_late": False}
               for tf in forced]
    else:
        due = get_due_timeframes(
            now,
            candle_close_delay_seconds={k: int(v) for k, v in sched["candle_close_delay_seconds"].items()},
            max_late_minutes={k: int(v) for k, v in sched["max_late_minutes"].items()},
            timeframes=[tf for tf, w in paper["wallets"].items() if w.get("enabled", True)],
        )

    log.info("Timeframes due: %s",
             [(d["timeframe"], d["is_due"], d.get("is_late"))
              for d in due])
    diag_payload["due_timeframes"] = [
        {k: (v.isoformat() if hasattr(v, "isoformat") else v)
         for k, v in d.items()} for d in due
    ]
    diag_payload["phases"].append("scheduler_evaluated")
    _save_diag()

    # 2. Monitor de posiciones abiertas (cierra TP/SL/TIMEOUT)
    if not args.dry_run:
        wallets_initial = {tf: w["initial_capital_eur"]
                           for tf, w in paper["wallets"].items()}
        closed = exits_mod.monitor_and_close_positions(
            cfg.state_dir, cfg.trades_dir, wallets_initial,
            cost=paper["costs"]["total_round_trip"],
            intrabar_rule=strat["intrabar_conflict_rule"],
        )
        log.info("Closed positions this tick: %d", len(closed))

    # 3. Macro cache
    cache = load_cache(cfg.data_dir)
    macro_features = to_feature_dict(cache)
    log.info("Macro cache has %d features cached", len(macro_features))

    # 4. Procesar cada TF due
    vol_bounds = load_decile_bounds(cfg.schemas_dir / "vol_decile_bounds.json")

    # Las features se construyen UNA vez por tick sobre el grid de 15m
    # (en entrenamiento los 3 modelos usan features del grid 15m) y se
    # comparten entre los TFs que esten due.
    feats15 = None

    for d in due:
        if not d.get("is_due"):
            if d.get("is_late"):
                log.warning("SKIPPED_LATE tf=%s close=%s",
                            d["timeframe"], d.get("candle_close_time"))
            else:
                log.info("Not yet due tf=%s", d["timeframe"])
            continue

        tf = d["timeframe"]
        close_t = d["candle_close_time"]

        # 4a. Build features (compartidas, grid 15m)
        if feats15 is None:
            log.info("Building live features (grid 15m, compartido)...")
            try:
                feats15 = build_live_features(
                    "15m",
                    rolling_history=max(
                        cfg.data_sources["binance_spot"]
                        ["rolling_history_candles"], 1600),
                    macro_cache=macro_features)
            except Exception as e:
                log.error("Feature build failed: %s", e)
                diag_payload["tf_results"][tf] = {
                    "phase": "feature_build_failed", "error": str(e)}
                diag_payload["errors"].append(f"feature_build_failed: {e}")
                _save_diag()
                break
        feats = feats15

        last_row = feats.iloc[-1]
        candle_open = feats.index[-1].to_pydatetime() if hasattr(feats.index[-1], "to_pydatetime") else feats.index[-1]
        log.info("[%s] Last 15m candle: %s, close=%.2f",
                 tf, feats.index[-1], last_row.get("ohlcv_btc_close", float("nan")))

        # 4b. Vol prediction & decile (sobre velas del TF, resampleadas
        # del grid 15m; el decil usa los bounds de entrenamiento del TF)
        ohlc15 = feats[["ohlcv_btc_open", "ohlcv_btc_high",
                        "ohlcv_btc_low", "ohlcv_btc_close"]].rename(columns={
                            "ohlcv_btc_open": "open", "ohlcv_btc_high": "high",
                            "ohlcv_btc_low": "low", "ohlcv_btc_close": "close",
                        })
        if tf == "15m":
            ohlc = ohlc15
        else:
            rule = {"1h": "1h", "4h": "4h"}[tf]
            ohlc = ohlc15.resample(rule).agg({"open": "first", "high": "max",
                                              "low": "min", "close": "last"}).dropna()
        vol_pred = predict_vol_ewma(ohlc, span=strat["volatility"]["ewma_span"])
        decile = vol_to_decile(vol_pred, vol_bounds[tf])
        log.info("[%s] vol_pred=%.6f -> decile=%d", tf, vol_pred, decile)

        # Log la fila de features que se va a pasar al modelo (auditoria)
        try:
            append_features(cfg.logs_dir, last_row, tf, now, candle_open,
                             vol_pred, decile)
        except Exception as e:
            log.warning("[%s] Could not append features CSV: %s", tf, e)

        # 4c. Clasificar regimen (EMA50 vs EMA200 en 1h, causal)
        regime = classify_regime(feats)
        log.info("[%s] Regime: %s", tf, regime)

        # 4d. Verificar si el regimen esta habilitado para este TF
        regime_rules = paper.get("regime_rules", {})
        tf_rules = regime_rules.get(tf, {})
        regime_config = tf_rules.get(regime)

        if regime_config is None or not regime_config.get("enabled", False):
            diag_payload["tf_results"][tf] = {
                "phase": "regime_filtered", "regime": regime}
            _save_diag()
            decision_row_skip = {
                "tick_ts_utc": now.isoformat(),
                "timeframe": tf,
                "candle_close_time": close_t.isoformat() if hasattr(close_t, "isoformat") else str(close_t),
                "btc_close": float(last_row.get("ohlcv_btc_close", float("nan"))),
                "vol_pred": vol_pred, "vol_decile": decile,
                "regime": regime, "decision": "NO",
                "reason_no_signal": f"REGIME_{regime.upper()}_NOT_ENABLED",
                "dry_run": bool(args.dry_run),
            }
            try:
                append_decision(cfg.logs_dir, decision_row_skip)
            except Exception as e:
                log.warning("[%s] CSV decision fail: %s", tf, e)
            log.info("[%s] Regime '%s' not enabled for this TF. Skipping.", tf, regime)
            continue

        threshold = float(regime_config["threshold"])
        leverage = float(regime_config["leverage"])
        log.info("[%s] Regime '%s' enabled: threshold=%.2f, leverage=%.1fx",
                 tf, regime, threshold, leverage)

        # 4e. Senal
        model_path = cfg.models_dir / strat["model"]["model_filename"].format(tf=tf)
        calib_path = cfg.calibration_dir / strat["model"]["calibration_filename"]
        lib_path = cfg.candidates_dir / strat["candidates"]["library_filename"].format(tf=tf)

        result = evaluate_timeframe(
            timeframe=tf,
            feature_row=last_row,
            vol_pred=vol_pred,
            vol_decile=decile,
            library_path=str(lib_path),
            model_path=str(model_path),
            calibrator_path=str(calib_path),
            prob_band_min=threshold,
            prob_band_max=1.0,
            allowed_sides=paper["allowed_sides"],
            allow_above_band=paper["allow_above_band"],
            require_positive_ev=paper["require_positive_ev"],
            cost=paper["costs"]["total_round_trip"],
            top_k_per_decile_side=strat["candidates"]["top_k_per_decile_side"],
        )
        diag = result["diagnostics"]
        log.info("[%s] Diagnostics: %s", tf,
                 {k: v for k, v in diag.items() if k not in ("all_scored",)})

        winner = result["winner"]

        # Construir base del registro de decision (los campos comunes)
        decision_row = {
            "tick_ts_utc": now.isoformat(),
            "timeframe": tf,
            "candle_open_time": candle_open.isoformat() if hasattr(candle_open, "isoformat") else str(candle_open),
            "candle_close_time": close_t.isoformat() if hasattr(close_t, "isoformat") else str(close_t),
            "execution_candle_open": close_t.isoformat() if hasattr(close_t, "isoformat") else str(close_t),
            "btc_close": float(last_row.get("ohlcv_btc_close", float("nan"))),
            "vol_pred": vol_pred,
            "vol_decile": decile,
            "regime": regime,
            "leverage": leverage,
            "n_candidates_initial": diag.get("n_candidates_initial", 0),
            "n_in_band": diag.get("n_in_band", 0),
            "p_win_max": diag.get("p_win_max", ""),
            "p_win_min": diag.get("p_win_min", ""),
            "EV_max": diag.get("EV_max", ""),
            "prob_band_min": threshold,
            "prob_band_max": 1.0,
            "dry_run": bool(args.dry_run),
            "mode": paper["signal_mode"],
        }

        if winner is None:
            decision_row["decision"] = "NO"
            decision_row["reason_no_signal"] = diag.get("reason_no_signal", "UNKNOWN")
            try:
                append_decision(cfg.logs_dir, decision_row)
            except Exception as e:
                log.warning("[%s] Could not append decision CSV: %s", tf, e)
            log.info("[%s] NO_SIGNAL: %s", tf, diag.get("reason_no_signal", "?"))
            continue

        # Hay ganador. Rellenar campos del candidato en decision_row
        decision_row["winner_side"] = winner["side"]
        decision_row["winner_candidate_id"] = int(winner["candidate_id"])
        decision_row["winner_p_win_calibrated"] = float(winner["p_win_calibrated"])
        decision_row["winner_EV_pred"] = float(winner["EV_pred"])
        decision_row["winner_tp_pct"] = float(winner["tp_pct"])
        decision_row["winner_sl_pct"] = float(winner["sl_pct"])
        decision_row["winner_tp_mult"] = float(winner["tp_mult"])
        decision_row["winner_sl_mult"] = float(winner["sl_mult"])
        decision_row["winner_H"] = int(winner["H"])
        decision_row["winner_p_break_even"] = float(winner.get("p_break_even_calc", winner.get("p_break_even", float("nan"))))
        decision_row["winner_edge_over_be"] = float(winner.get("edge_over_be", float("nan")))

        # 4d. Check no hay posicion abierta en esta cartera
        positions = paper_broker.load_open_positions(cfg.state_dir)
        open_in_tf = [p for p in positions.values() if p.get("timeframe") == tf]
        if open_in_tf and not paper["allow_cross_timeframe_positions"]:
            decision_row["decision"] = "NO"
            decision_row["reason_no_signal"] = "WALLET_BLOCKED_BY_OPEN_POSITION_CROSS_TF"
            try: append_decision(cfg.logs_dir, decision_row)
            except Exception as e: log.warning("CSV decision fail: %s", e)
            log.info("[%s] Already has %d open position(s). Skipping.",
                     tf, len(open_in_tf))
            continue
        if len(open_in_tf) >= paper.get("max_positions_per_wallet", 1):
            decision_row["decision"] = "NO"
            decision_row["reason_no_signal"] = "WALLET_AT_MAX_POSITIONS"
            try: append_decision(cfg.logs_dir, decision_row)
            except Exception as e: log.warning("CSV decision fail: %s", e)
            log.info("[%s] Wallet at max positions (%d). Skipping.",
                     tf, len(open_in_tf))
            continue

        # 4e. Idempotencia
        signal_id = make_signal_id(args.symbol, tf, close_t,
                                     winner["side"], int(winner["candidate_id"]))
        decision_row["signal_id"] = signal_id
        if paper_broker.is_signal_processed(cfg.state_dir, signal_id):
            decision_row["decision"] = "NO"
            decision_row["reason_no_signal"] = "SIGNAL_ALREADY_PROCESSED"
            try: append_decision(cfg.logs_dir, decision_row)
            except Exception as e: log.warning("CSV decision fail: %s", e)
            log.info("[%s] Signal already processed (signal_id=%s). Skipping.",
                     tf, signal_id)
            continue

        if paper["signal_mode"] == "preclose_preview":
            decision_row["decision"] = "NO"
            decision_row["reason_no_signal"] = "PREVIEW_MODE_NO_ENTRY"
            try: append_decision(cfg.logs_dir, decision_row)
            except Exception as e: log.warning("CSV decision fail: %s", e)
            log.info("[%s] PREVIEW MODE - signal saved but not opened: %s",
                     tf, signal_id)
            paper_broker.mark_signal_processed(cfg.state_dir, signal_id)
            continue

        # 4f. Abrir operacion (paper)
        if args.dry_run:
            decision_row["decision"] = "YES"  # se habria abierto
            decision_row["reason_no_signal"] = "DRY_RUN_WOULD_OPEN"
            try: append_decision(cfg.logs_dir, decision_row)
            except Exception as e: log.warning("CSV decision fail: %s", e)
            log.info("[%s] DRY-RUN would open: %s p_win=%.4f EV=%.6f tp_pct=%.4f sl_pct=%.4f",
                     tf, winner["side"], winner["p_win_calibrated"],
                     winner["EV_pred"], winner["tp_pct"], winner["sl_pct"])
            continue

        ideal_entry = float(last_row["ohlcv_btc_close"])
        try:
            entry_price, ticker_src = bspot.fetch_ticker_price_with_fallback(args.symbol)
            entry_price_quality = f"ticker_{ticker_src}"
        except Exception as e:
            log.warning("[%s] Ticker failed, using last close: %s", tf, e)
            entry_price = ideal_entry
            entry_price_quality = "fallback_last_close"

        side_sign = 1 if winner["side"] == "long" else -1
        tp_price = entry_price * (1 + side_sign * float(winner["tp_pct"]))
        sl_price = entry_price * (1 - side_sign * float(winner["sl_pct"]))
        H_candles = int(winner["H"])
        tf_minutes = strat["timeframe_minutes"][tf]
        timeout_time = now + dt.timedelta(minutes=tf_minutes * H_candles)

        init_cap = paper["wallets"][tf]["initial_capital_eur"]
        wallet = wallet_mod.load_or_init(cfg.state_dir, tf, init_cap)
        notional_eur = float(wallet["cash_eur"]) * leverage

        pos = paper_broker.open_position(
            cfg.state_dir, signal_id=signal_id, timeframe=tf, symbol=args.symbol,
            side=winner["side"], entry_price=entry_price,
            tp_price=tp_price, sl_price=sl_price,
            timeout_time=timeout_time, notional_eur=notional_eur,
            p_win=float(winner["p_win_calibrated"]),
            EV_pred=float(winner["EV_pred"]),
            candidate_id=int(winner["candidate_id"]),
            entry_price_quality=entry_price_quality,
            ideal_entry_price=ideal_entry,
            vol_decile=decile,
            leverage=leverage,
            regime=regime,
        )
        wallet_mod.apply_position_open(wallet, pos["position_id"])
        wallet_mod.save(cfg.state_dir, wallet)

        decision_row["decision"] = "YES"
        decision_row["position_id"] = pos["position_id"]
        decision_row["entry_price"] = entry_price
        decision_row["entry_price_quality"] = entry_price_quality
        try: append_decision(cfg.logs_dir, decision_row)
        except Exception as e: log.warning("CSV decision fail: %s", e)

    # 5. Dashboard
    try:
        export_dashboard_data(cfg)
    except Exception as e:
        log.warning("Dashboard export failed: %s", e)

    diag_payload["phases"].append("tick_completed")
    _save_diag()
    log.info("Paper tick end.")
    return 0


def main() -> int:
    args = parse_args()
    cfg = load_config(profile=args.profile)
    return run_tick(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
