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
from src.data import cryptocompare as cc
from src.data import fear_greed as fg
from src.data import fred_client as fred
from src.data import yahoo_macro as yahoo
from src.features.macro_cache import load_cache, save_cache, update_value
from src.utils.logging_utils import setup_logger


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
    update_fred(cache, log, ds["fred"]["series"])
    update_cryptocompare(cache, log, ds["cryptocompare"]["coins"])
    update_funding(cache, log)

    cache["updated_at"] = dt.datetime.utcnow().replace(tzinfo=dt.timezone.utc).isoformat()
    save_cache(cfg.data_dir, cache)
    log.info("Cache saved (%d values total)", len(cache.get("values", {})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
