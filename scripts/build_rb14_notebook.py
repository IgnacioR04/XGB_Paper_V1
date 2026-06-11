# -*- coding: utf-8 -*-
"""Genera RB14_portfolio_management.ipynb en el repo de Drive."""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parents[2]   # G:/Mi unidad/Base de Datos BITCOIN
OUT_NB = BASE / "repo/notebooks/08_real_backtest/RB14_portfolio_management.ipynb"

MD0 = """# RB14 - Gestion de cartera y estadisticas operativas

**Motivacion**: el bot de paper trading en vivo solo permite **1 posicion por
cartera** (15m/1h/4h independientes). Si llega una senal valida con la cartera
ocupada, se rechaza y NO se reintenta. Sin embargo, los backtests de los que
salieron las bandas (RB13) y la estrategia congelada (RB05) **no aplicaban esa
restriccion**: asumian que TODAS las senales en banda se convertian en trades.

Dia 1 real del bot (2026-06-10): 56 senales validas de 15m, solo 3 ejecutadas
(las otras 53 rechazadas por `WALLET_AT_MAX_POSITIONS`). Capture ratio ~5%.

**Preguntas que responde este notebook (todo con coste 0.0012, bandas
desplegadas: 15m 0.65-0.70, 1h 0.70-0.75, 4h 0.65-0.70, banda `[lo, hi)`):**

1. Cuanto retorno captura la regla actual (1 posicion, 100% de la cartera)
   frente a fraccionar: 50%/2 posiciones, 33%/3, 25%/4, y frente al backtest
   sin limite (referencia RB13).
2. Trades/dia por timeframe, PnL diario estimado en EUR (cartera de 100),
   maxima racha de perdidas consecutivas.
3. Proporcion long/short de las senales en banda por timeframe.
4. Como afecta la **tendencia** (alcista/bajista/lateral) a la rentabilidad
   por side - en particular si los longs pierden mas en tendencia bajista
   (situacion actual del mercado).

Reglas anti-leakage: solo se usan predicciones ya generadas (signals de RB01) y
labels precalculadas. No se reentrena ni recalibra nada. La eleccion formal de
configuracion se hace mirando VAL; TEST solo confirma.
"""

C1 = """# Celda 1
# Objetivo
# Imports, BASE, rutas, parametros de la estrategia desplegada.

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

try:
    from google.colab import drive  # type: ignore
    drive.mount("/content/drive", force_remount=False)
    BASE = Path("/content/drive/MyDrive/Base de Datos BITCOIN")
except Exception:
    BASE = Path("G:/Mi unidad/Base de Datos BITCOIN")

DIR_RB = BASE / "Data/model_outputs/real_backtest"
DIR_LBL = BASE / "Data/model_outputs/directional/datasets"
OUT_DIR = DIR_RB / "portfolio_mgmt"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR = DIR_RB / "plots/portfolio_mgmt"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

COST = 0.0012
CAPITAL_INI = 100.0
# Bandas desplegadas en el bot (config/paper_trading.yaml), semantica [lo, hi)
BANDS = {"15m": (0.65, 0.70), "1h": (0.70, 0.75), "4h": (0.65, 0.70)}
TF_MIN = {"15m": 15, "1h": 60, "4h": 240}
TFS = ["15m", "1h", "4h"]

print("BASE:", BASE)
print("OUT_DIR:", OUT_DIR)
print("Bandas:", BANDS, "| coste:", COST)
"""

C2 = """# Celda 2
# Objetivo
# Cargar signals val/test (predicciones calibradas de RB01) y enriquecer con
# labels_compact: H, gross_return, exit_reason y time_to_exit (velas hasta
# salida REAL). La duracion real es clave: el backtest RB04 bloqueaba la
# cartera el horizonte completo H, pero en vivo la cartera se libera cuando
# toca TP/SL, que suele ser antes.

val = pd.read_parquet(DIR_RB / "signals/signals_val.parquet")
test = pd.read_parquet(DIR_RB / "signals/signals_test.parquet")
for df in (val, test):
    df["timestamp"] = pd.to_datetime(df["timestamp"])

lookup_parts = []
for tf in TFS:
    lp = DIR_LBL / f"labels_compact_{tf}.parquet"
    h = pd.read_parquet(lp, columns=["timestamp", "candidate_id", "timeframe",
                                      "H", "gross_return", "exit_reason",
                                      "time_to_exit"])
    h["timestamp"] = pd.to_datetime(h["timestamp"])
    lookup_parts.append(h)
H_lookup = pd.concat(lookup_parts, ignore_index=True)


def enrich(df):
    out = df.merge(H_lookup, on=["timestamp", "candidate_id", "timeframe"],
                   how="left")
    out = out.rename(columns={"H": "H_bars"})
    out["net_return_cost012"] = out["gross_return"].astype(float) - COST
    tfm = out["timeframe"].map(TF_MIN)
    # duracion real (hasta TP/SL/timeout); fallback al horizonte completo
    out["duration_real_min"] = tfm * out["time_to_exit"].fillna(out["H_bars"])
    out["duration_H_min"] = tfm * out["H_bars"]
    return out


val = enrich(val)
test = enrich(test)
print("val:", val.shape, "| test:", test.shape)
print("time_to_exit NaN en test:", test["time_to_exit"].isna().sum())
print("Duracion real media vs H (test, min):",
      test["duration_real_min"].mean().round(1), "vs",
      test["duration_H_min"].mean().round(1))
"""

C3 = """# Celda 3
# Objetivo
# Reproducir el flujo de senales del bot: para cada vela (timestamp), filtrar
# candidatos en banda [lo, hi) y elegir el de mayor EV. Una senal por vela
# como maximo, por timeframe. Esto es exactamente lo que hace signal_engine
# en el repo XGB_Paper_V1.


def bot_signals(df, tf):
    lo, hi = BANDS[tf]
    sub = df[(df["timeframe"] == tf)
             & (df["p_win_isotonic"] >= lo)
             & (df["p_win_isotonic"] < hi)].copy()
    if len(sub) == 0:
        return sub
    idx = sub.groupby("timestamp")["EV_pred"].idxmax()
    return sub.loc[idx].sort_values("timestamp").reset_index(drop=True)


signals = {}
for split_name, df in [("val", val), ("test", test)]:
    for tf in TFS:
        s = bot_signals(df, tf)
        signals[(split_name, tf)] = s

# Estadisticas basicas de las senales en banda
rows = []
for (split_name, tf), s in signals.items():
    if len(s) == 0:
        rows.append({"split": split_name, "timeframe": tf, "n_senales": 0})
        continue
    days = max(1, (s["timestamp"].max() - s["timestamp"].min()).days)
    rows.append({
        "split": split_name, "timeframe": tf,
        "n_senales": len(s),
        "senales_dia": round(len(s) / days, 2),
        "pct_long": round((s["side"] == "long").mean() * 100, 1),
        "pct_short": round((s["side"] == "short").mean() * 100, 1),
        "win_rate": round((s["net_return_cost012"] > 0).mean(), 3),
        "expectancy": round(s["net_return_cost012"].mean(), 5),
        "dur_real_media_min": round(s["duration_real_min"].mean(), 0),
    })
sig_stats = pd.DataFrame(rows)
print("=== Senales en banda (flujo que ve el bot) ===")
print(sig_stats.to_string(index=False))
sig_stats.to_csv(OUT_DIR / "signal_stats_by_tf.csv", index=False)
"""

C4 = """# Celda 4
# Objetivo
# Simulador de cartera por timeframe. Event-driven:
#   - posiciones abiertas se cierran cuando ts >= entry_ts + duracion real
#   - una senal nueva se toma solo si hay hueco (max_pos) y cash suficiente
#   - notional = frac * equity_total en el momento de apertura
#   - PnL en EUR se realiza al cierre; equity compone
# Con frac=1.0 y max_pos=1 replica el bot actual.


def simulate_wallet(sig, frac, max_pos, capital0=CAPITAL_INI):
    sig = sig.sort_values("timestamp").reset_index(drop=True)
    cash = capital0
    open_pos = []          # lista de dicts {close_ts, notional, net_return}
    taken = []
    for _, row in sig.iterrows():
        ts = row["timestamp"]
        # 1) cerrar posiciones vencidas
        still = []
        for p in open_pos:
            if p["close_ts"] <= ts:
                cash += p["notional"] * (1.0 + p["net_return"])
            else:
                still.append(p)
        open_pos = still
        # 2) intentar abrir
        if len(open_pos) >= max_pos:
            continue
        equity_now = cash + sum(p["notional"] for p in open_pos)
        notional = frac * equity_now
        if notional > cash + 1e-9:      # sin apalancamiento
            continue
        dur = row["duration_real_min"]
        if not np.isfinite(dur):
            continue
        cash -= notional
        nr = float(row["net_return_cost012"])
        open_pos.append({"close_ts": ts + pd.Timedelta(minutes=float(dur)),
                         "notional": notional, "net_return": nr})
        taken.append({"timestamp": ts, "side": row["side"],
                      "net_return": nr, "pnl_eur": notional * nr,
                      "p_win": row["p_win_isotonic"]})
    # cerrar lo que quede
    for p in open_pos:
        cash += p["notional"] * (1.0 + p["net_return"])
    trades = pd.DataFrame(taken)
    if len(trades) == 0:
        return trades, {"n_taken": 0, "final_capital": capital0}
    eq = capital0 + trades["pnl_eur"].cumsum()
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    # racha maxima de perdidas consecutivas
    is_loss = (trades["net_return"] <= 0).astype(int).values
    streak = max_streak = 0
    for x in is_loss:
        streak = streak + 1 if x else 0
        max_streak = max(max_streak, streak)
    days = max(1, (trades["timestamp"].max() - trades["timestamp"].min()).days)
    m = {
        "n_senales": len(sig),
        "n_taken": len(trades),
        "capture_ratio": round(len(trades) / max(1, len(sig)), 3),
        "final_capital": round(float(cash), 2),
        "pnl_total_eur": round(float(cash - capital0), 2),
        "win_rate": round(float((trades["net_return"] > 0).mean()), 3),
        "max_dd_pct": round(float(dd.min() * 100), 2),
        "max_rachas_perdida": int(max_streak),
        "trades_dia": round(len(trades) / days, 2),
        "pnl_dia_eur": round(float((cash - capital0) / days), 3),
    }
    return trades, m


print("simulate_wallet listo.")
"""

C5 = """# Celda 5
# Objetivo
# Grid de configuraciones de sizing por timeframe, en VAL (eleccion) y TEST
# (confirmacion). Referencia adicional: 'sin limite' = lo que asumia RB13
# (cada senal sized al 100% sin restriccion; equity = suma de retornos, no
# componible en una cartera real - solo como cota superior teorica).

CONFIGS = [
    ("bot_actual_100pct_1pos", 1.00, 1),
    ("50pct_2pos",             0.50, 2),
    ("33pct_3pos",             0.33, 3),
    ("25pct_4pos",             0.25, 4),
]

all_rows = []
trades_store = {}
for split_name in ["val", "test"]:
    for tf in TFS:
        sig = signals[(split_name, tf)]
        if len(sig) == 0:
            continue
        days = max(1, (sig["timestamp"].max() - sig["timestamp"].min()).days)
        for name, frac, mp in CONFIGS:
            trades, m = simulate_wallet(sig, frac, mp)
            m.update({"config": name, "timeframe": tf, "split": split_name})
            all_rows.append(m)
            trades_store[(split_name, tf, name)] = trades
        # referencia sin limite (RB13): suma simple de retornos x 100 EUR
        ref_pnl = sig["net_return_cost012"].sum() * CAPITAL_INI
        all_rows.append({
            "config": "ref_RB13_sin_limite", "timeframe": tf,
            "split": split_name, "n_senales": len(sig), "n_taken": len(sig),
            "capture_ratio": 1.0,
            "final_capital": round(CAPITAL_INI + ref_pnl, 2),
            "pnl_total_eur": round(ref_pnl, 2),
            "win_rate": round((sig["net_return_cost012"] > 0).mean(), 3),
            "max_dd_pct": np.nan, "max_rachas_perdida": np.nan,
            "trades_dia": round(len(sig) / days, 2),
            "pnl_dia_eur": round(ref_pnl / days, 3),
        })

grid = pd.DataFrame(all_rows)
cols_show = ["split", "timeframe", "config", "n_senales", "n_taken",
             "capture_ratio", "final_capital", "pnl_total_eur", "win_rate",
             "max_dd_pct", "max_rachas_perdida", "trades_dia", "pnl_dia_eur"]
print("=== VAL (aqui se decide) ===")
print(grid[grid["split"] == "val"][cols_show].to_string(index=False))
print()
print("=== TEST (confirmacion) ===")
print(grid[grid["split"] == "test"][cols_show].to_string(index=False))
grid.to_csv(OUT_DIR / "wallet_config_grid.csv", index=False)
"""

C6 = """# Celda 6
# Objetivo
# Curvas de equity en TEST por timeframe y configuracion.

fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharey=False)
colors = {"bot_actual_100pct_1pos": "black", "50pct_2pos": "steelblue",
          "33pct_3pos": "darkorange", "25pct_4pos": "seagreen"}
for ax, tf in zip(axes, TFS):
    for name, frac, mp in CONFIGS:
        t = trades_store.get(("test", tf, name))
        if t is None or len(t) == 0:
            continue
        eq = CAPITAL_INI + t["pnl_eur"].cumsum()
        ax.plot(t["timestamp"], eq, label=name, color=colors[name],
                linewidth=1.2)
    ax.axhline(CAPITAL_INI, color="grey", linestyle="--", linewidth=0.6)
    ax.set_title(f"{tf} (TEST, banda {BANDS[tf]})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
axes[0].set_ylabel("Equity EUR (cartera 100)")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "01_equity_by_config_test.png", dpi=110)
plt.show()
"""

C7 = """# Celda 7
# Objetivo
# Detalle operativo de la mejor configuracion por TF (elegida en VAL por
# pnl_total con max_dd aceptable): distribucion de PnL diario, rachas,
# dias planos.

g_val = grid[(grid["split"] == "val")
             & (grid["config"] != "ref_RB13_sin_limite")].copy()
best_cfg = (g_val.sort_values(["timeframe", "pnl_total_eur"],
                              ascending=[True, False])
                 .groupby("timeframe").head(1)[["timeframe", "config"]])
print("Mejor configuracion por TF segun VAL:")
print(best_cfg.to_string(index=False))
print()

for _, brow in best_cfg.iterrows():
    tf, name = brow["timeframe"], brow["config"]
    t = trades_store.get(("test", tf, name))
    if t is None or len(t) == 0:
        print(f"{tf}/{name}: sin trades en test")
        continue
    daily = (t.set_index("timestamp")["pnl_eur"]
              .resample("1D").sum())
    active = daily[daily != 0]
    print(f"--- {tf} | {name} (TEST) ---")
    print(f"  dias con actividad: {len(active)} de {len(daily)}")
    print(f"  PnL diario medio: {daily.mean():+.3f} EUR | mediana dias "
          f"activos: {active.median():+.3f}")
    print(f"  p10 / p90 PnL diario: {daily.quantile(.1):+.3f} / "
          f"{daily.quantile(.9):+.3f}")
    print(f"  mejor dia: {daily.max():+.2f} | peor dia: {daily.min():+.2f}")
    wins = (t['net_return'] > 0)
    print(f"  win rate: {wins.mean():.3f} | trades: {len(t)}")
    print()
"""

C8 = """# Celda 8
# Objetivo
# Tendencia vs rentabilidad. Clasificamos el regimen de mercado con el close
# 15m del master: EMA50 vs EMA200 sobre velas 1h + posicion del precio.
#   alcista: close > EMA200 y EMA50 > EMA200
#   bajista: close < EMA200 y EMA50 < EMA200
#   lateral: resto
# Usamos TODAS las senales en banda (no solo las tomadas por la cartera) para
# tener muestra grande. Pregunta clave: pierden mas los longs en bajista?

mc = pd.read_parquet(DIR_LBL / "master_close_only.parquet")
mc["timestamp"] = pd.to_datetime(mc["timestamp"])
if getattr(mc["timestamp"].dt, "tz", None) is not None:
    mc["timestamp"] = mc["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
mc = mc.set_index("timestamp").sort_index()

h1 = mc["ohlcv_btc_close"].resample("1h").last().dropna()
ema50 = h1.ewm(span=50, adjust=False).mean()
ema200 = h1.ewm(span=200, adjust=False).mean()
regime = pd.Series("lateral", index=h1.index, name="regimen")
regime[(h1 > ema200) & (ema50 > ema200)] = "alcista"
regime[(h1 < ema200) & (ema50 < ema200)] = "bajista"
regime_df = regime.reset_index()

rows = []
for split_name, df in [("val", val), ("test", test)]:
    for tf in TFS:
        s = signals[(split_name, tf)]
        if len(s) == 0:
            continue
        s = pd.merge_asof(s.sort_values("timestamp"), regime_df,
                          on="timestamp", direction="backward")
        for (reg, side), g in s.groupby(["regimen", "side"]):
            if len(g) < 20:
                continue
            rows.append({
                "split": split_name, "timeframe": tf, "regimen": reg,
                "side": side, "n": len(g),
                "win_rate": round((g["net_return_cost012"] > 0).mean(), 3),
                "expectancy": round(g["net_return_cost012"].mean(), 5),
                "total_return": round(g["net_return_cost012"].sum(), 3),
            })
trend_tbl = pd.DataFrame(rows)
print("=== Rentabilidad por regimen y side (senales en banda) ===")
for split_name in ["val", "test"]:
    print(f"-- {split_name} --")
    sub = trend_tbl[trend_tbl["split"] == split_name]
    print(sub.pivot_table(index=["timeframe", "side"], columns="regimen",
                          values="expectancy").round(5).to_string())
    print()
print("Detalle completo:")
print(trend_tbl.to_string(index=False))
trend_tbl.to_csv(OUT_DIR / "trend_regime_table.csv", index=False)

# Distribucion de senales long/short por regimen (el modelo cambia de side?)
print()
print("=== Mix de sides por regimen (test, senales en banda) ===")
for tf in TFS:
    s = signals[("test", tf)]
    if len(s) == 0:
        continue
    s = pd.merge_asof(s.sort_values("timestamp"), regime_df,
                      on="timestamp", direction="backward")
    mix = s.groupby("regimen")["side"].value_counts(normalize=True).round(3)
    print(f"-- {tf} --")
    print(mix.to_string())
"""

C9 = """# Celda 9
# Objetivo
# Resumen ejecutivo en markdown con los numeros clave.

lines = ["# RB14 - Resumen ejecutivo", ""]
lines.append(f"_Generado: {pd.Timestamp.utcnow().isoformat()}_")
lines.append("")
lines.append("## Capture ratio y PnL por configuracion (TEST)")
lines.append("")
sub = grid[grid['split'] == 'test'][cols_show]
try:
    lines.append(sub.to_markdown(index=False))
except Exception:
    lines.append("```\\n" + sub.to_string(index=False) + "\\n```")
lines.append("")
lines.append("## Mejor configuracion por TF (elegida en VAL)")
lines.append("")
lines.append(best_cfg.to_string(index=False))
lines.append("")
lines.append("## Tendencia (expectancy por regimen, test)")
lines.append("")
sub2 = trend_tbl[trend_tbl['split'] == 'test']
piv = sub2.pivot_table(index=['timeframe', 'side'], columns='regimen',
                        values='expectancy').round(5)
lines.append("```\\n" + piv.to_string() + "\\n```")
lines.append("")
(OUT_DIR / "RB14_summary.md").write_text("\\n".join(lines), encoding="utf-8")
print("Resumen escrito en", OUT_DIR / "RB14_summary.md")
print()
print("RB14 OK")
"""

nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


def add_cell(kind, src):
    lines = src.splitlines(keepends=True)
    nb["cells"].append({
        "cell_type": kind,
        "metadata": {},
        "source": lines,
        **({"outputs": [], "execution_count": None} if kind == "code" else {}),
    })


add_cell("markdown", MD0)
for c in [C1, C2, C3, C4, C5, C6, C7, C8, C9]:
    add_cell("code", c)

OUT_NB.parent.mkdir(parents=True, exist_ok=True)
OUT_NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1),
                  encoding="utf-8")
print(f"Notebook escrito: {OUT_NB}")
print(f"Celdas: {len(nb['cells'])}")
