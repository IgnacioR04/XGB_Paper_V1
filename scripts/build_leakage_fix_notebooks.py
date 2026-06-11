# -*- coding: utf-8 -*-
"""Genera en el repo de Drive:
- 02_transform/FIX01_leakage_causal_features.ipynb (fix T03/T10 -> *_CAUSAL)
- 04_validate/V05_leakage_scan.ipynb (escaneo de correlacion con el futuro)
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]


def make_nb(cells):
    nb = {"cells": [], "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []}},
        "nbformat": 4, "nbformat_minor": 5}
    for kind, src in cells:
        nb["cells"].append({
            "cell_type": kind, "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {})})
    return nb


# ===================================================================== FIX01
FIX_MD = """# FIX01 - Correccion de leakage en features (T03 + T10)

**Hallazgo (2026-06-11)**: tres grupos de features del master contenian
informacion FUTURA:

| Feature | Problema | Evidencia |
|---|---|---|
| `ta_chikou_span` / `ta_chikou_dist` | pandas_ta usa `close.shift(-26)` en 1h = close de 26h en el FUTURO | corr -0.994 con el retorno futuro a 26h |
| `ta_dist_last_swing_high/low` | `rolling(20, center=True)` usa 10 velas futuras | formula centrada reproduce el master al 100% |
| `cx_*_return_1d`, `cx_btc_dom_chg_1d` y derivados | dato diario asignado al INICIO del dia (cada vela intradia ve el cierre del dia completo) | corr +0.24 a +0.43 con el retorno futuro |

Impacto en los modelos Approach B (suma de total_gain de features con leakage):
**~54% (15m), ~35% (1h), ~42% (4h)**. El backtest esta inflado y los modelos
deben reentrenarse tras este fix.

Este notebook regenera las columnas afectadas con definiciones CAUSALES y
escribe archivos NUEVOS (no toca los originales):
- `processed/M03_technicals_CAUSAL.parquet`
- `processed/M10_cross_crypto_CAUSAL.parquet`
- `master/master_15m_CAUSAL.parquet`

Para PROMOCIONAR el fix (tras revisar la verificacion de la ultima celda):
renombrar los originales a `*_LEAKY_backup.parquet` y quitar el sufijo
`_CAUSAL`. Despues seguir la cadena de reentrenamiento de
`docs/LEAKAGE_FIX_PLAN.md`.
"""

FIX_C1 = """# Celda 1
# Objetivo
# Setup.

from pathlib import Path
import numpy as np
import pandas as pd

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
    BASE = Path("/content/drive/MyDrive/Base de Datos BITCOIN")
except Exception:
    BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")

PROCESSED = BASE / "Data/processed"
m01 = pd.read_parquet(PROCESSED / "M01_ohlcv.parquet")
o, h, l, c = (m01["ohlcv_btc_open"], m01["ohlcv_btc_high"],
              m01["ohlcv_btc_low"], m01["ohlcv_btc_close"])
print("M01:", m01.shape)
"""

FIX_C2 = """# Celda 2
# Objetivo
# M03 causal: Ichimoku con nube VISIBLE (calculada hace 26 velas 1h) y
# chikou como close de hace 26 HORAS (momentum causal). Swings con pivote
# confirmado 10 velas despues (ventana [t-20, t], sin centro).

m03 = pd.read_parquet(PROCESSED / "M03_technicals.parquet")

h1 = h.resample("1h").max()
l1 = l.resample("1h").min()
c1 = c.resample("1h").last()
tenkan = (h1.rolling(9).max() + l1.rolling(9).min()) / 2
kijun = (h1.rolling(26).max() + l1.rolling(26).min()) / 2
senkou_a = ((tenkan + kijun) / 2).shift(26)
senkou_b = ((h1.rolling(52).max() + l1.rolling(52).min()) / 2).shift(26)
chikou = c1.shift(26)


def to15(s):
    return s.reindex(c.index, method="ffill")


m03["ta_tenkan_sen"] = to15(tenkan)
m03["ta_kijun_sen"] = to15(kijun)
m03["ta_senkou_a"] = to15(senkou_a)
m03["ta_senkou_b"] = to15(senkou_b)
m03["ta_chikou_span"] = to15(chikou)
m03["ta_chikou_dist"] = (c - m03["ta_chikou_span"]) / c
m03["ta_cloud_width"] = (m03["ta_senkou_a"] - m03["ta_senkou_b"]).abs() / c
m03["ta_price_above_cloud"] = (
    c > pd.concat([m03["ta_senkou_a"], m03["ta_senkou_b"]], axis=1).max(axis=1)
).astype("int8")

piv_h = h.shift(10).where(h.shift(10) >= h.rolling(21).max() - 1e-9)
piv_l = l.shift(10).where(l.shift(10) <= l.rolling(21).min() + 1e-9)
m03["ta_dist_last_swing_high"] = (c - piv_h.ffill()) / c
m03["ta_dist_last_swing_low"] = (c - piv_l.ffill()) / c

m03.to_parquet(PROCESSED / "M03_technicals_CAUSAL.parquet", compression="snappy")
print("M03 causal:", m03.shape)
"""

FIX_C3 = """# Celda 3
# Objetivo
# M10 causal: el dato diario del dia D esta disponible desde D+1 00:00.

raw = pd.read_parquet(BASE / "Data/raw/cross_crypto/total_mcap/cross_crypto_daily.parquet")
raw.index = pd.to_datetime(raw.index, utc=True)
raw_lag = raw.copy()
raw_lag.index = raw_lag.index + pd.Timedelta(days=1)

grid = m01.index
cx = pd.DataFrame(index=grid)
df = raw_lag.reindex(grid, method="ffill")
for col in df.columns:
    cx[col] = df[col]
cx["cx_btc_dom_chg_1d"] = cx["cx_btc_dominance"].diff(96)
cx["cx_total_mcap_return_1d"] = cx["cx_total_mcap"].pct_change(96)
cx["cx_total2_return_1d"] = cx["cx_total2_mcap"].pct_change(96)
cx["cx_total3_return_1d"] = cx["cx_total3_mcap"].pct_change(96)

cx["cx_eth_btc_ratio"] = m01["ohlcv_eth_close"] / m01["ohlcv_btc_close"]
cx["cx_eth_btc_return"] = np.log(cx["cx_eth_btc_ratio"]).diff()
cx["cx_xrp_btc_ratio"] = m01["ohlcv_xrp_close"] / m01["ohlcv_btc_close"]
cx["cx_xrp_btc_return"] = np.log(cx["cx_xrp_btc_ratio"]).diff()

cx["cx_altcoin_idx_return"] = cx["cx_total3_return_1d"]
cx["cx_altcoin_idx_vol"] = cx["cx_total3_return_1d"].rolling(96).std()
cx["cx_eth_outperf_btc"] = (cx["cx_eth_btc_return"] > 0).rolling(96).mean()
btc_ret = np.log(m01["ohlcv_btc_close"]).diff(96)
cx["cx_alt_outperf_btc"] = ((cx["cx_total3_return_1d"] - btc_ret) > 0).astype("int8")
cx["cx_btc_dom_breakdown"] = ((btc_ret < 0) & (cx["cx_total3_return_1d"] > 0)).astype("int8")

cx = cx[pd.read_parquet(PROCESSED / "M10_cross_crypto.parquet").columns]
cx.to_parquet(PROCESSED / "M10_cross_crypto_CAUSAL.parquet", compression="snappy")
print("M10 causal:", cx.shape)
"""

FIX_C4 = """# Celda 4
# Objetivo
# master_15m_CAUSAL.parquet: master original con las 30 columnas corregidas.

MASTER_P = BASE / "Data/master/master_15m.parquet"
master = pd.read_parquet(MASTER_P)
patch_m03 = ["ta_tenkan_sen", "ta_kijun_sen", "ta_senkou_a", "ta_senkou_b",
             "ta_chikou_span", "ta_chikou_dist", "ta_cloud_width",
             "ta_price_above_cloud", "ta_dist_last_swing_high",
             "ta_dist_last_swing_low"]
n = 0
for col in patch_m03:
    if col in master.columns:
        master[col] = m03[col].reindex(master.index)
        n += 1
for col in cx.columns:
    if col in master.columns:
        master[col] = cx[col].reindex(master.index)
        n += 1
OUT = BASE / "Data/master/master_15m_CAUSAL.parquet"
master.to_parquet(OUT, compression="snappy")
print(f"{n} columnas corregidas -> {OUT}")
"""

FIX_C5 = """# Celda 5
# Objetivo
# Verificacion: ninguna columna corregida debe correlacionar con el futuro.

fwd_26h = (c.shift(-104) / c - 1)
fwd_4h = (c.shift(-16) / c - 1)
ok = True
for col in ["ta_chikou_dist", "ta_dist_last_swing_high",
            "ta_dist_last_swing_low", "cx_total_mcap_return_1d",
            "cx_btc_dom_chg_1d", "cx_total2_return_1d",
            "cx_total3_return_1d", "cx_altcoin_idx_return"]:
    s = master[col].iloc[::4]
    c26 = s.corr(fwd_26h.iloc[::4])
    c4 = s.corr(fwd_4h.iloc[::4])
    bad = max(abs(c26 or 0), abs(c4 or 0)) >= 0.2
    ok = ok and not bad
    print(f"  {col:32s} fwd4h={c4:+.3f}  fwd26h={c26:+.3f}  "
          f"{'**SIGUE ALTO**' if bad else 'OK'}")
print()
print("VERIFICACION", "SUPERADA" if ok else "FALLIDA")
print()
print("Siguiente paso: revisar docs/LEAKAGE_FIX_PLAN.md y promocionar los")
print("archivos _CAUSAL antes de reentrenar (D01 -> D04 -> D09 -> RB01 -> RB13).")
"""

# ====================================================================== V05
V05_MD = """# V05 - Escaneo de leakage por correlacion con el futuro

Para CUALQUIER version del master: correlacion de cada feature contra el
retorno futuro de BTC a 4h y 26h. Una feature legitima no deberia superar
|corr| ~0.15-0.20 (los targets si, por definicion, y se excluyen).

Ejecutar despues de cada regeneracion del master y antes de reentrenar.
"""

V05_C1 = """# Celda 1
# Objetivo
# Setup y eleccion del master a escanear.

from pathlib import Path
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
    BASE = Path("/content/drive/MyDrive/Base de Datos BITCOIN")
except Exception:
    BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")

# Cambiar aqui si se quiere escanear otra version
MASTER_PATH = BASE / "Data/master/master_15m_CAUSAL.parquet"
if not MASTER_PATH.exists():
    MASTER_PATH = BASE / "Data/master/master_15m.parquet"
print("Escaneando:", MASTER_PATH.name)

m = pd.read_parquet(MASTER_PATH)
c = m["ohlcv_btc_close"]
print("master:", m.shape)
"""

V05_C2 = """# Celda 2
# Objetivo
# Escaneo completo. Umbral de alarma: |corr| > 0.20 con el retorno futuro.

THRESHOLD = 0.20
fwd = {"4h": (c.shift(-16) / c - 1), "26h": (c.shift(-104) / c - 1)}
feat_cols = [col for col in m.columns
             if not col.startswith(("tgt_", "qa_", "cost_"))]
sub = m[feat_cols].iloc[::4]
fsub = {k: v.iloc[::4] for k, v in fwd.items()}

rows = []
for col in feat_cols:
    s = sub[col]
    if s.dtype == "O" or s.notna().sum() < 1000 or s.std() == 0:
        continue
    cs = {k: s.corr(v) for k, v in fsub.items()}
    worst = max(abs(v) for v in cs.values() if pd.notna(v)) if cs else 0
    rows.append({"feature": col,
                 **{f"corr_fwd_{k}": round(v, 3) for k, v in cs.items()},
                 "max_abs": round(worst, 3)})
res = pd.DataFrame(rows).sort_values("max_abs", ascending=False)
flagged = res[res["max_abs"] > THRESHOLD]
print(f"Features con |corr| > {THRESHOLD}: {len(flagged)}")
print(flagged.to_string(index=False) if len(flagged) else "  (ninguna) OK")
print()
print("Top 20 (referencia):")
print(res.head(20).to_string(index=False))

OUT = BASE / "Data/logs/leakage_scan_results.csv"
res.to_csv(OUT, index=False)
print()
print("Resultados ->", OUT)
"""

nb_fix = make_nb([("markdown", FIX_MD), ("code", FIX_C1), ("code", FIX_C2),
                  ("code", FIX_C3), ("code", FIX_C4), ("code", FIX_C5)])
nb_v05 = make_nb([("markdown", V05_MD), ("code", V05_C1), ("code", V05_C2)])

p1 = BASE / "repo/notebooks/02_transform/FIX01_leakage_causal_features.ipynb"
p2 = BASE / "repo/notebooks/04_validate/V05_leakage_scan.ipynb"
p1.write_text(json.dumps(nb_fix, ensure_ascii=False, indent=1), encoding="utf-8")
p2.write_text(json.dumps(nb_v05, ensure_ascii=False, indent=1), encoding="utf-8")
print("Escritos:")
print(" -", p1)
print(" -", p2)
