"""Loader para los modelos XGBoost Approach B."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import xgboost as xgb

from ..utils.errors import ArtifactMissing


@lru_cache(maxsize=8)
def load_xgb_model(model_path: str) -> xgb.Booster:
    p = Path(model_path)
    if not p.exists():
        raise ArtifactMissing(f"XGB model not found: {p}")
    b = xgb.Booster()
    b.load_model(str(p))
    return b


def model_path_for_tf(models_dir: Path, template: str, tf: str) -> Path:
    return Path(models_dir) / template.format(tf=tf)
