"""CoinGecko /global: market cap total y dominancias (gratis, sin API key).

Devuelve dominancias en FRACCION (0-1) y mcap en USD crudo, para que coincida
con la escala del master (verificado: ext_btc_dominance ~0.64, ext_total_mcap
~2e12, ext_total2_mcap = total*(1-btc_dom)).
"""
from __future__ import annotations

from ..utils.http import get_json
from ..utils.logging_utils import get_logger

log = get_logger("coingecko")

GLOBAL_URL = "https://api.coingecko.com/api/v3/global"

# Stablecoins habituales para sumar su dominancia
_STABLES = ["usdt", "usdc", "dai", "busd", "tusd", "usde", "fdusd", "usdd"]


def fetch_global(timeout: int = 15) -> dict:
    """Snapshot global. Claves: total_mcap (USD), btc_dominance, eth_dominance,
    stablecoin_dominance (fracciones 0-1). NaN si la fuente falla."""
    data = get_json(GLOBAL_URL, timeout=timeout)
    d = (data or {}).get("data", {}) or {}
    total = d.get("total_market_cap", {}).get("usd")
    pct = d.get("market_cap_percentage", {}) or {}

    def dom(key: str) -> float:
        v = pct.get(key)
        return float(v) / 100.0 if v is not None else float("nan")

    stable_dom = sum(float(pct.get(k, 0.0)) for k in _STABLES) / 100.0
    return {
        "total_mcap": float(total) if total is not None else float("nan"),
        "btc_dominance": dom("btc"),
        "eth_dominance": dom("eth"),
        "stablecoin_dominance": stable_dom if stable_dom > 0 else float("nan"),
    }
