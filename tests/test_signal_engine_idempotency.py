import datetime as dt
import tempfile
from pathlib import Path

from src.execution import paper_broker as PB
from src.utils.time_utils import make_signal_id


def test_signal_id_stable():
    t = dt.datetime(2026, 6, 9, 14, 15, 0, tzinfo=dt.timezone.utc)
    a = make_signal_id("BTCUSDT", "15m", t, "long", 160)
    b = make_signal_id("BTCUSDT", "15m", t, "long", 160)
    assert a == b
    # cambiar candidate_id genera otro signal_id
    c = make_signal_id("BTCUSDT", "15m", t, "long", 161)
    assert a != c


def test_processed_signal_prevents_duplicate():
    with tempfile.TemporaryDirectory() as d:
        sid = "BTCUSDT|15m|20260609T141500Z|long|160"
        assert PB.is_signal_processed(Path(d), sid) is False
        PB.mark_signal_processed(Path(d), sid)
        assert PB.is_signal_processed(Path(d), sid) is True
        # idempotente
        PB.mark_signal_processed(Path(d), sid)
        assert PB.is_signal_processed(Path(d), sid) is True
