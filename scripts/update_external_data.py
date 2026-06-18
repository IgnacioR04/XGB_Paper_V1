"""Refresca el cache de fuentes externas (macro, onchain, sentiment, cross-crypto, funding).

Pensado para correr 1-4 veces al dia desde un cron separado. Es mucho mas lento
que un tick del paper trader, no es viable hacerlo cada 5 min.

Sugerencia: workflow Actions adicional con cron "0 */6 * * *" (cada 6h).
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.data import binance_futures as bfut
from src.data import blockchain_com as oc
from src.data import coingecko as cg
from src.data import cryptocompare as cc
from src.data import fear_greed as fg
from src.data import fred_client as fred
from src.data import yahoo_macro as yahoo
from src.features.macro_cache import load_cache, save_cache, update_value
from src.utils.logging_utils import setup_logger


def _prev(cache: dict, key: str):
    """Ultimo valor cacheado de `key` (para cambios dia a dia aproximados)."""
    return cache.get("values", {}).get(key)


def update_fear_greed(cache: dict, log) -> None:
    try:
        df = fg.fetch_fear_greed(limit=10)
        if df.empty:
            log.warning("F&G returned empty")
            return
        last = df.iloc[-1]
        update_value(cache, "sent_fear_greed", float(last["fear_greed"]), "alternative.me")
        prev_1d = df.iloc[-2]["fear_greed"] if len(df) > 1 else float("nan")
        prev_7d = df.iloc[-8]["fear_greed"] if len(df) > 7 else float("nan")
        update_value(cache, "sent_fg_change_1d", float(last["fear_greed"]) - float(prev_1d), "alternative.me")
        update_value(cache, "sent_fg_change_7d", float(last["fear_greed"]) - float(prev_7d), "alternative.me")
        fgv = float(last["fear_greed"])
        update_value(cache, "sent_fg_extreme_fear", 1.0 if fgv <= 25 else 0.0, "alternative.me")
        update_value(cache, "sent_fg_extreme_greed", 1.0 if fgv >= 75 else 0.0, "alternative.me")
        log.info("F&G updated: %s", last["fear_greed"])
    except Exception as e:
        log.error("F&G failed: %s", e)


def update_onchain(cache: dict, log) -> None:
    try:
        chains = oc.fetch_basic_onchain()
        mapping = {"hashrate": "oc_hashrate", "difficulty": "oc_difficulty",
                   "tx_count": "oc_tx_count_24h", "addresses": "oc_addresses_24h"}
        for k, df in chains.items():
            if df.empty:
                continue
            last_val = float(df.iloc[-1].iloc[1])  # 2da col es la metrica
            update_value(cache, mapping[k], last_val, "blockchain.com")
        log.info("On-chain updated: %d metrics", len(chains))
    except Exception as e:
        log.error("On-chain failed: %s", e)


def update_macro_yahoo(cache: dict, log, tickers: dict) -> None:
    try:
        results = yahoo.fetch_all(tickers)
        for name, df in results.items():
            if df.empty:
                continue
            last_close = float(df.iloc[-1]["close"])
            ret_1d = float(df.iloc[-1]["close"]) / float(df.iloc[-2]["close"]) - 1 \
                if len(df) > 1 else 0.0
            update_value(cache, f"ext_{name}_close", last_close, "yahoo")
            update_value(cache, f"ext_{name}_return_1d", ret_1d, "yahoo")
            update_value(cache, f"macro_{name}_return_1d", ret_1d, "yahoo")
        log.info("Yahoo updated: %d tickers", len(results))
    except Exception as e:
        log.error("Yahoo failed: %s", e)


def update_fred(cache: dict, log, series: dict) -> None:
    for name, sid in series.items():
        try:
            df = fred.fetch_series(sid, observation_start="2023-01-01")
            if df.empty:
                continue
            last_val = float(df.iloc[-1]["value"])
            update_value(cache, f"macro_{name}", last_val, "fred")
            log.info("FRED %s = %.4f", name, last_val)
        except Exception as e:
            log.warning("FRED %s failed: %s", name, e)


def update_cryptocompare(cache: dict, log, coins: list[str]) -> None:
    try:
        results = cc.fetch_all_coins(coins)
        # Sumas de mcap aproximadas: solo guardamos el close de cada coin
        for c, df in results.items():
            if df.empty:
                continue
            update_value(cache, f"cx_{c.lower()}_close", float(df.iloc[-1]["close"]), "cryptocompare")
        log.info("CryptoCompare updated: %d coins", len(results))
    except Exception as e:
        log.error("CryptoCompare failed: %s", e)


def update_macro_derived(cache: dict, log) -> None:
    """Features macro derivadas que el loop generico no cubre (nivel, cambio,
    zscore 1a, retornos 5d). Necesitan historia -> se piden 1 ano. Formulas y
    escalas verificadas contra el master."""
    try:
        import numpy as np
        hist = {n: yahoo.fetch_ticker_history(s, period="1y")
                for n, s in {"vix": "^VIX", "move": "^MOVE", "ndx": "^IXIC",
                             "spx": "^GSPC", "rut": "^RUT"}.items()}

        def close(n):
            df = hist.get(n)
            return df["close"].astype(float).dropna() if (df is not None and not df.empty
                                                           and "close" in df) else None

        vix = close("vix")
        if vix is not None and len(vix) > 30:
            update_value(cache, "macro_vix", float(vix.iloc[-1]), "yahoo")
            update_value(cache, "macro_vix_change_1d", float(vix.iloc[-1] - vix.iloc[-2]), "yahoo")
            w = vix.tail(252)
            sd = float(w.std())
            update_value(cache, "macro_vix_zscore_1y",
                         float((vix.iloc[-1] - w.mean()) / sd) if sd > 0 else 0.0, "yahoo")
        mv = close("move")
        if mv is not None and len(mv):
            update_value(cache, "macro_move", float(mv.iloc[-1]), "yahoo")
        ndx = close("ndx")
        if ndx is not None and len(ndx) > 5:
            update_value(cache, "macro_ndx_return_5d", float(ndx.iloc[-1] / ndx.iloc[-6] - 1), "yahoo")
        spx = close("spx")
        if spx is not None and len(spx) > 5:
            update_value(cache, "macro_spx_return_5d", float(spx.iloc[-1] / spx.iloc[-6] - 1), "yahoo")
        # growth-value spread (proxy): retorno 1d Nasdaq (growth) - Russell2000 (value/small)
        rut = close("rut")
        if ndx is not None and rut is not None and len(ndx) > 1 and len(rut) > 1:
            gv = (ndx.iloc[-1] / ndx.iloc[-2] - 1) - (rut.iloc[-1] / rut.iloc[-2] - 1)
            update_value(cache, "macro_growth_value_spread", float(gv), "yahoo")
        log.info("Macro derived updated")
    except Exception as e:
        log.error("Macro derived failed: %s", e)


def update_crypto_mcap(cache: dict, log) -> None:
    """Market cap total y dominancia (CoinGecko) + ratios cross. Escribe claves
    ext_ y cx_ (el modelo usa ambas). total2 = total*(1-btc_dom),
    total3 = total*(1-btc_dom-eth_dom) (verificado contra el master)."""
    try:
        g = cg.fetch_global()
        total, btc_d, eth_d, stab_d = (g["total_mcap"], g["btc_dominance"],
                                       g["eth_dominance"], g["stablecoin_dominance"])
        if total != total:  # NaN
            log.warning("CoinGecko global sin total_mcap"); return
        total2 = total * (1 - btc_d) if btc_d == btc_d else float("nan")
        total3 = total * (1 - btc_d - eth_d) if (btc_d == btc_d and eth_d == eth_d) else float("nan")
        stab_mcap = total * stab_d if stab_d == stab_d else float("nan")
        # cambios dia a dia aproximados (vs ultimo cacheado)
        p_tot, p_t2, p_t3, p_btcd, p_stab = (_prev(cache, "cx_total_mcap"),
            _prev(cache, "cx_total2_mcap"), _prev(cache, "cx_total3_mcap"),
            _prev(cache, "cx_btc_dominance"), _prev(cache, "ext_stablecoin_mcap"))
        pairs = {
            "total_mcap": total, "total2_mcap": total2, "total3_mcap": total3,
            "btc_dominance": btc_d, "eth_dominance": eth_d,
            "stablecoin_dominance": stab_d, "stablecoin_mcap": stab_mcap,
        }
        for k, v in pairs.items():
            update_value(cache, f"ext_{k}", v, "coingecko")
            update_value(cache, f"cx_{k}", v, "coingecko")
        if p_tot: update_value(cache, "cx_total_mcap_return_1d", total / p_tot - 1, "coingecko")
        if p_t2:  update_value(cache, "cx_total2_return_1d", total2 / p_t2 - 1, "coingecko")
        if p_t3:  update_value(cache, "cx_total3_return_1d", total3 / p_t3 - 1, "coingecko")
        if p_btcd is not None: update_value(cache, "cx_btc_dom_chg_1d", btc_d - p_btcd, "coingecko")
        if p_stab: update_value(cache, "ext_stablecoin_supply_chg_24h", stab_mcap / p_stab - 1, "coingecko")
        # ratios cross desde los closes de cryptocompare (ya cacheados)
        vals = cache.get("values", {})
        btc_c, eth_c, xrp_c = vals.get("cx_btc_close"), vals.get("cx_eth_close"), vals.get("cx_xrp_close")
        if btc_c:
            if eth_c:
                ebr = eth_c / btc_c; update_value(cache, "cx_eth_btc_ratio", ebr, "cryptocompare")
                pe = _prev(cache, "cx_eth_btc_ratio")
                if pe: update_value(cache, "cx_eth_btc_return", ebr / pe - 1, "cryptocompare")
            if xrp_c:
                xbr = xrp_c / btc_c; update_value(cache, "cx_xrp_btc_ratio", xbr, "cryptocompare")
                px = _prev(cache, "cx_xrp_btc_ratio")
                if px: update_value(cache, "cx_xrp_btc_return", xbr / px - 1, "cryptocompare")
        log.info("Crypto mcap updated: total=%.3g btc_dom=%.3f", total, btc_d)
    except Exception as e:
        log.error("Crypto mcap failed: %s", e)


def update_funding(cache: dict, log) -> None:
    try:
        df = bfut.fetch_funding_rate("BTCUSDT", limit=10)
        if df.empty:
            return
        last = df.iloc[-1]
        update_value(cache, "der_funding_rate", float(last["funding_rate"]), "binance_fut")
        update_value(cache, "der_funding_annualized",
                      float(last["funding_rate"]) * 3 * 365, "binance_fut")
        log.info("Funding updated: %.6f", last["funding_rate"])
    except Exception as e:
        log.error("Funding failed: %s", e)


def main() -> int:
    cfg = load_config()
    log = setup_logger(cfg.logs_dir / "external_update.log", name="external_update")
    cfg.ensure_runtime_dirs()
    cache = load_cache(cfg.data_dir)
    log.info("Updating external sources (cache had %d values)",
             len(cache.get("values", {})))

    ds = cfg.data_sources
    update_fear_greed(cache, log)
    update_onchain(cache, log)
    update_macro_yahoo(cache, log, ds["yahoo_macro"]["tickers"])
    update_macro_derived(cache, log)
    update_fred(cache, log, ds["fred"]["series"])
    update_cryptocompare(cache, log, ds["cryptocompare"]["coins"])
    update_crypto_mcap(cache, log)   # tras cryptocompare (usa sus closes para ratios)
    update_funding(cache, log)

    cache["updated_at"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
    save_cache(cfg.data_dir, cache)
    log.info("Cache saved (%d values total)", len(cache.get("values", {})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
