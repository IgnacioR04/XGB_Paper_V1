import tempfile
from pathlib import Path

from src.portfolio import wallet as W


def test_init_creates_file():
    with tempfile.TemporaryDirectory() as d:
        w = W.load_or_init(Path(d), "15m", 100.0)
        assert w["cash_eur"] == 100.0
        assert w["n_trades"] == 0
        assert W.wallet_path(Path(d), "15m").exists()


def test_apply_trade_close_winning():
    w = W._empty_wallet("15m", 100.0)
    W.apply_trade_close(w, pnl_eur=5.0, exit_ts="2026-06-09T14:00:00Z")
    assert w["cash_eur"] == 105.0
    assert w["equity_eur"] == 105.0
    assert w["realized_pnl_eur"] == 5.0
    assert w["n_trades"] == 1
    assert w["n_wins"] == 1
    assert w["n_losses"] == 0
    assert w["open_position_id"] is None
    assert len(w["equity_curve"]) == 1


def test_apply_trade_close_losing():
    w = W._empty_wallet("1h", 100.0)
    W.apply_trade_close(w, pnl_eur=-3.0, exit_ts="2026-06-09T14:00:00Z")
    assert w["cash_eur"] == 97.0
    assert w["n_losses"] == 1


def test_position_open_sets_id():
    w = W._empty_wallet("4h", 100.0)
    W.apply_position_open(w, "pos123")
    assert w["open_position_id"] == "pos123"
