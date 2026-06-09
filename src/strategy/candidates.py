"""Generacion de candidatos de operacion a partir de la libreria barriers_v2.

La libreria contiene 100 filas (= 10 deciles x 2 sides x 5 ranks) por TF.
En live, dado un (vol_decile, side), elegimos los top-K candidatos del decile.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd


@lru_cache(maxsize=4)
def load_library(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def candidates_for(library: pd.DataFrame,
                   timeframe: str,
                   vol_decile: int,
                   sides: list[str] = ("long", "short"),
                   top_k: int = 5) -> pd.DataFrame:
    """Filtra la libreria por (tf, decile, sides) y devuelve top_k por rank."""
    mask = (library["timeframe"] == timeframe) & \
           (library["decile"] == vol_decile) & \
           (library["side"].isin(sides))
    sub = library[mask].copy()
    sub = sub.sort_values("rank").head(top_k * len(list(sides)))
    return sub.reset_index(drop=True)
