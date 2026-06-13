"""Carga unificada de los YAML de config y resolucion de rutas."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _expand_env(value: Any) -> Any:
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], "")
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


@dataclass
class Config:
    repo_root: Path
    paper: dict[str, Any]
    data_sources: dict[str, Any]
    strategy: dict[str, Any]
    profile: str = ""   # "" = bot principal; "oof" = bot del modelo OOF

    @property
    def _sfx(self) -> str:
        return f"_{self.profile}" if self.profile else ""

    @property
    def artifacts_dir(self) -> Path:
        return self.repo_root / "artifacts"

    @property
    def models_dir(self) -> Path:
        return self.repo_root / self.strategy["model"]["models_dir"]

    @property
    def calibration_dir(self) -> Path:
        return self.repo_root / self.strategy["model"]["calibration_dir"]

    @property
    def candidates_dir(self) -> Path:
        return self.repo_root / self.strategy["candidates"]["library_dir"]

    @property
    def schemas_dir(self) -> Path:
        return self.repo_root / self.strategy["model"]["schemas_dir"]

    @property
    def data_dir(self) -> Path:
        return self.repo_root / "data"

    @property
    def state_dir(self) -> Path:
        return self.data_dir / f"state{self._sfx}"

    @property
    def logs_dir(self) -> Path:
        return self.data_dir / f"logs{self._sfx}"

    @property
    def trades_dir(self) -> Path:
        return self.data_dir / f"paper_trades{self._sfx}"

    # live_raw y live_features se COMPARTEN entre perfiles (klines/macro cache)
    @property
    def live_raw_dir(self) -> Path:
        return self.data_dir / "live_raw"

    @property
    def live_features_dir(self) -> Path:
        return self.data_dir / "live_features"

    @property
    def dashboard_data_dir(self) -> Path:
        return self.repo_root / "dashboard" / f"data{self._sfx}"

    def ensure_runtime_dirs(self) -> None:
        for d in (self.state_dir, self.logs_dir, self.trades_dir,
                  self.live_raw_dir, self.live_features_dir,
                  self.dashboard_data_dir):
            d.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=4)
def load_config(repo_root: Path | str | None = None, profile: str = "") -> Config:
    """profile="" carga el bot principal; profile="oof" carga el bot del modelo
    OOF (paper_trading_oof.yaml + strategy_oof.yaml; data_sources es comun)."""
    root = Path(repo_root) if repo_root else REPO_ROOT
    cfg_dir = root / "config"
    paper_file = f"paper_trading_{profile}.yaml" if profile else "paper_trading.yaml"
    strat_file = f"strategy_{profile}.yaml" if profile else "strategy.yaml"
    paper = _expand_env(_load_yaml(cfg_dir / paper_file))
    data_sources = _expand_env(_load_yaml(cfg_dir / "data_sources.yaml"))
    strategy = _expand_env(_load_yaml(cfg_dir / strat_file))
    return Config(repo_root=root, paper=paper, data_sources=data_sources,
                  strategy=strategy, profile=profile)
