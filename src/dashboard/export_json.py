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

    # Summary
    summary = {
        "generated_at": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
        "wallets": summary_by_tf,
        "n_open_positions": len(open_pos),
    }
    write_json_atomic(out / "summary.json", summary)
