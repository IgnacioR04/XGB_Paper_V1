"""Logger central. Escribe a stdout y a data/logs/paper_trader.log."""
from __future__ import annotations

import logging
import sys
from pathlib import Path


_FMT = "%(asctime)s | %(levelname)-7s | %(name)-22s | %(message)s"


def setup_logger(log_path: Path | str | None = None,
                 level: int = logging.INFO,
                 name: str = "paper_trader") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if logger.handlers:
        return logger
    fmt = logging.Formatter(_FMT)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_path is not None:
        p = Path(log_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(p, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"paper_trader.{name}")
