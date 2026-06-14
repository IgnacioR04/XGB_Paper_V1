"""IO helpers: lectura/escritura atomica de JSON, CSV, parquet."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import pandas as pd


def read_json(path: Path | str, default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json_atomic(path: Path | str, data: Any) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        os.replace(tmp, p)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def append_csv(path: Path | str, row: dict) -> None:
    """Anexa una fila a un CSV con seguridad de esquema.

    Si el archivo no existe -> escribe header + fila.
    Si existe y el header coincide con las columnas de la fila -> append rapido.
    Si el header NO coincide (p.ej. un reset previo creo el archivo con otras
    columnas) -> reescribe alineando por nombre (union), evitando que las filas
    queden desalineadas bajo un header equivocado.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if not p.exists():
        new.to_csv(p, index=False)
        return
    try:
        existing_cols = pd.read_csv(p, nrows=0).columns.tolist()
    except Exception:
        existing_cols = []
    if existing_cols == list(new.columns):
        new.to_csv(p, mode="a", header=False, index=False)
    else:
        old = pd.read_csv(p) if existing_cols else pd.DataFrame()
        combined = pd.concat([old, new], ignore_index=True, sort=False)
        combined.to_csv(p, index=False)


def append_parquet(path: Path | str, row: dict) -> None:
    """Anexa a un parquet. Implementacion simple: read-modify-write.

    Para volumenes pequenos de trades (decenas/dia) es suficiente.
    Si llega a miles/dia, migrar a pyarrow ParquetWriter incremental.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if p.exists():
        old = pd.read_parquet(p)
        combined = pd.concat([old, new], ignore_index=True)
    else:
        combined = new
    combined.to_parquet(p, index=False)
