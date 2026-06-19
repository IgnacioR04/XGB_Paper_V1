"""Cache de features externas (macro, onchain, sentiment, cross-crypto, der).

Estas features cambian a baja frecuencia (horas o dias). El bot las refresca
en su propio cron (update_external_data.py) y guarda un JSON con el ultimo
valor + timestamp. El feature builder lo lee y broadcasta el valor a la fila
de inferencia.

Si una fuente falla, conservamos el ultimo valor valido y marcamos `stale=True`.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from ..utils.io import read_json, write_json_atomic
from ..utils.logging_utils import get_logger

log = get_logger("macro_cache")

CACHE_FILENAME = "external_features_cache.json"


def cache_path(data_dir: Path) -> Path:
    return Path(data_dir) / "live_features" / CACHE_FILENAME


def load_cache(data_dir: Path) -> dict[str, Any]:
    return read_json(cache_path(data_dir), default={"values": {}, "meta": {}})


def save_cache(data_dir: Path, cache: dict) -> None:
    write_json_atomic(cache_path(data_dir), cache)


def update_value(cache: dict, key: str, value: float,
                 source: str, fetched_at: dt.datetime | None = None) -> None:
    if fetched_at is None:
        fetched_at = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc)
    cache.setdefault("values", {})[key] = value
    cache.setdefault("meta", {})[key] = {
        "source": source,
        "fetched_at": fetched_at.isoformat(),
        "stale": False,
    }


def mark_stale(cache: dict, key: str) -> None:
    meta = cache.setdefault("meta", {}).setdefault(key, {})
    meta["stale"] = True


def to_feature_dict(cache: dict) -> dict[str, float]:
    """Devuelve solo el dict de valores plano para broadcastear en features."""
    return dict(cache.get("values", {}))


def update_series(cache: dict, key: str,
                  points: list[tuple[str, float]], source: str) -> None:
    """Guarda una serie temporal corta [(iso, valor), ...] de una variable base.

    Se usa para las series que necesitan derivadas lag/roll (funding, VIX, NDX):
    el escalar no basta, hace falta la historia para reindexar al grid 15m y
    calcular lag/roll. `points` debe venir ordenado por tiempo ascendente.
    """
    clean = []
    for ts, v in points:
        try:
            clean.append([str(ts), float(v)])
        except (TypeError, ValueError):
            continue
    cache.setdefault("series", {})[key] = clean
    cache.setdefault("meta", {})[f"series:{key}"] = {
        "source": source,
        "fetched_at": dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat(),
        "n": len(clean),
    }


def get_series(cache: dict, key: str) -> list[list]:
    """Devuelve la serie [(iso, valor), ...] guardada o lista vacia."""
    return cache.get("series", {}).get(key, [])
