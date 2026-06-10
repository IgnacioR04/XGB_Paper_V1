"""Exporta dashboard/data/*.json: equity, trades, posiciones abiertas, resumen."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd

from ..execution import paper_broker
from ..portfolio import wallet as wallet_mod
from ..utils.io import read_json, write_json_atomic


def _safe_trades_df(trades_dir: Path) -> pd.DataFrame:
    p = Path(trades_dir) / "trades.parquet"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_parquet(p)


def _signals_history_df(logs_dir: Path, max_rows: int = 2000) -> pd.DataFrame:
    """Concatena todos los decisions_YYYY-MM.csv (historial de preguntas al
    modelo con semantica t -> t+1)."""
    decisions_dir = Path(logs_dir) / "decisions"
    if not decisions_dir.exists():
        return pd.DataFrame()
    frames = []
    for p in sorted(decisions_dir.glob("decisions_*.csv")):
        try:
            frames.append(pd.read_csv(p, on_bad_lines="skip"))
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True, sort=False)
    return df.tail(max_rows)


def export_all(cfg) -> None:
    out = Path(cfg.dashboard_data_dir)
    out.mkdir(parents=True, exist_ok=True)

    paper = cfg.paper

    # Equity per timeframe
    summary_by_tf = {}
    for tf, wcfg in paper["wallets"].items():
        w = read_json(wallet_mod.wallet_path(cfg.state_dir, tf))
        if w is None:
            continue
        curve = w.get("equity_curve", [])
        write_json_atomic(out / f"equity_{tf}.json", {
            "timeframe": tf,
            "initial_capital_eur": w["initial_capital_eur"],
            "curve": curve,
        })
        summary_by_tf[tf] = {
            "initial_capital_eur": w["initial_capital_eur"],
            "cash_eur": w["cash_eur"],
            "equity_eur": w["equity_eur"],
            "realized_pnl_eur": w["realized_pnl_eur"],
            "n_trades": w["n_trades"],
            "n_wins": w["n_wins"],
            "n_losses": w["n_losses"],
            "win_rate": (w["n_wins"] / w["n_trades"]) if w["n_trades"] else 0.0,
            "open_position_id": w["open_position_id"],
        }

    # Trades
    tdf = _safe_trades_df(cfg.trades_dir)
    trades = []
    if not tdf.empty:
        keep_cols = [c for c in (
            "position_id", "signal_id", "timeframe", "symbol", "side",
            "entry_time", "exit_time", "entry_price", "exit_price",
            "tp_price", "sl_price", "exit_reason", "p_win", "EV_pred",
            "candidate_id", "vol_decile", "pnl_eur", "pnl_pct",
            "wallet_equity_after",
        ) if c in tdf.columns]
        trades = tdf[keep_cols].sort_values("exit_time", ascending=False) \
            .head(500).to_dict(orient="records")
    write_json_atomic(out / "trades.json", {"trades": trades})

    # Open positions
    open_pos = list(paper_broker.load_open_positions(cfg.state_dir).values())
    write_json_atomic(out / "open_positions.json", {"open_positions": open_pos})

    # Signals history (cada pregunta al modelo: vela t -> ejecucion en t+1)
    sig_df = _signals_history_df(cfg.logs_dir)
    signals = []
    if not sig_df.empty:
        keep = [c for c in (
            "tick_ts_utc", "timeframe", "candle_open_time", "candle_close_time",
            "execution_candle_open", "btc_close", "vol_pred", "vol_decile",
            "n_candidates_initial", "n_in_band", "p_win_max", "EV_max",
            "decision", "reason_no_signal", "winner_side",
            "winner_p_win_calibrated", "winner_EV_pred", "winner_tp_pct",
            "winner_sl_pct", "winner_H", "entry_price", "signal_id", "dry_run",
        ) if c in sig_df.columns]
        sub = sig_df[keep].copy()
        # NaN -> None para JSON limpio
        sub = sub.where(pd.notna(sub), None)
        signals = sub.to_dict(orient="records")
    write_json_atomic(out / "signals.json", {"signals": signals})

    # Summary
    summary = {
        "generated_at": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
        "wallets": summary_by_tf,
        "n_open_positions": len(open_pos),
    }
    write_json_atomic(out / "summary.json", summary)
