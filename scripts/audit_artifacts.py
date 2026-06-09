"""Verifica que los artifacts y configs esten en su sitio."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


CHECK = "  [OK]"
WARN = "  [WARN]"
FAIL = "  [FAIL]"


def main() -> int:
    cfg = load_config()
    print("Repo root:", cfg.repo_root)
    n_fail = 0
    n_warn = 0

    # 1. Modelos
    print("\n1. XGBoost models")
    for tf in ("15m", "1h", "4h"):
        p = cfg.models_dir / cfg.strategy["model"]["model_filename"].format(tf=tf)
        if p.exists():
            print(f"{CHECK} {p.name} ({p.stat().st_size / 1024:.0f} KB)")
        else:
            print(f"{FAIL} {p.name} MISSING -> espera en {cfg.models_dir}")
            n_fail += 1

    # 2. Calibrador
    print("\n2. Calibrator (compact JSON)")
    p = cfg.calibration_dir / cfg.strategy["model"]["calibration_filename"]
    if p.exists():
        with p.open() as f:
            calib = json.load(f)
        keys = list(calib.keys())
        print(f"{CHECK} {p.name} contains tfs: {keys}")
    else:
        print(f"{FAIL} {p} MISSING")
        n_fail += 1

    # 3. Libraries de candidatos
    print("\n3. Candidate libraries")
    for tf in ("15m", "1h", "4h"):
        p = cfg.candidates_dir / cfg.strategy["candidates"]["library_filename"].format(tf=tf)
        if p.exists():
            print(f"{CHECK} {p.name} ({p.stat().st_size / 1024:.1f} KB)")
        else:
            print(f"{FAIL} {p.name} MISSING")
            n_fail += 1

    # 4. Feature schema
    print("\n4. Feature schema")
    p = cfg.schemas_dir / "feature_schema.json"
    if p.exists():
        with p.open() as f:
            schema = json.load(f)
        print(f"{CHECK} {p.name} ({schema['n_features']} features)")
    else:
        print(f"{FAIL} feature_schema.json MISSING")
        n_fail += 1

    # 5. Vol decile bounds
    print("\n5. Vol decile bounds")
    p = cfg.schemas_dir / "vol_decile_bounds.json"
    if p.exists():
        print(f"{CHECK} {p.name}")
    else:
        print(f"{FAIL} vol_decile_bounds.json MISSING")
        n_fail += 1

    # 6. Configs
    print("\n6. Configs")
    for name in ("paper_trading.yaml", "data_sources.yaml", "strategy.yaml"):
        p = cfg.repo_root / "config" / name
        if p.exists():
            print(f"{CHECK} config/{name}")
        else:
            print(f"{FAIL} config/{name} MISSING")
            n_fail += 1

    # 7. Macro cache
    print("\n7. External features cache")
    p = cfg.data_dir / "live_features" / "external_features_cache.json"
    if p.exists():
        with p.open() as f:
            cache = json.load(f)
        n = len(cache.get("values", {}))
        print(f"{CHECK} cache has {n} cached values, updated {cache.get('updated_at', '?')}")
    else:
        print(f"{WARN} no external cache yet. Run `scripts/update_external_data.py` first.")
        n_warn += 1

    # 8. Runtime dirs
    print("\n8. Runtime dirs")
    cfg.ensure_runtime_dirs()
    for d in (cfg.state_dir, cfg.logs_dir, cfg.trades_dir,
              cfg.live_raw_dir, cfg.dashboard_data_dir):
        print(f"{CHECK} {d.relative_to(cfg.repo_root)}/")

    print()
    print(f"SUMMARY: {n_fail} FAIL, {n_warn} WARN")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
