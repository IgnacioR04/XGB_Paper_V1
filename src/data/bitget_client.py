"""Cliente minimo de Bitget USDT-M Futures (v2 API).

Auth: HMAC-SHA256 sobre timestamp+method+path+body, codificado base64.
Doc: https://www.bitget.com/api-doc/contract/intro

Endpoints que usa el bot 4h-live:
  - GET  /api/v2/mix/account/account       (saldo de la cuenta de futures)
  - POST /api/v2/mix/account/set-leverage  (ajustar leverage por symbol)
  - POST /api/v2/mix/order/place-order     (abrir/cerrar mercado)
  - POST /api/v2/mix/order/place-plan-order (TP/SL condicional, opcional)
  - GET  /api/v2/mix/position/single-position (posicion abierta del par)
  - GET  /api/v2/mix/order/orders-pending  (ordenes abiertas)
  - POST /api/v2/mix/order/cancel-plan-order (cancelar condicionales)

Las credenciales se leen de env vars: BITGET_API_KEY/SECRET/PASSPHRASE.
Si no estan, el cliente queda en modo no-autenticado (solo lecturas publicas).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import time
from typing import Any

import requests

from ..utils.logging_utils import get_logger

log = get_logger("bitget")

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"   # USDT-M perpetuals
MARGIN_COIN = "USDT"


class BitgetError(Exception):
    pass


class BitgetClient:
    def __init__(self, api_key: str | None = None, secret: str | None = None,
                 passphrase: str | None = None, timeout: int = 15) -> None:
        self.key = api_key or os.environ.get("BITGET_API_KEY", "")
        self.secret = secret or os.environ.get("BITGET_API_SECRET", "")
        self.passphrase = passphrase or os.environ.get("BITGET_PASSPHRASE", "")
        self.timeout = timeout
        self.authenticated = bool(self.key and self.secret and self.passphrase)
        if not self.authenticated:
            log.warning("Bitget client SIN credenciales (modo solo-lectura publica)")

    # ----- auth ---------------------------------------------------------
    def _sign(self, ts: str, method: str, path: str, body: str) -> str:
        msg = f"{ts}{method.upper()}{path}{body}"
        sig = hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).digest()
        return base64.b64encode(sig).decode()

    def _headers(self, ts: str, sign: str) -> dict[str, str]:
        return {
            "ACCESS-KEY": self.key,
            "ACCESS-SIGN": sign,
            "ACCESS-TIMESTAMP": ts,
            "ACCESS-PASSPHRASE": self.passphrase,
            "Content-Type": "application/json",
            "locale": "en-US",
        }

    def _request(self, method: str, path: str,
                 params: dict | None = None, body: dict | None = None) -> dict:
        if not self.authenticated:
            raise BitgetError("Bitget sin credenciales")
        query = ""
        if params:
            # ordenar claves para firma estable
            query = "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items())
                                   if v is not None)
        body_str = json.dumps(body, separators=(",", ":")) if body else ""
        ts = str(int(time.time() * 1000))
        sign = self._sign(ts, method, path + query, body_str)
        headers = self._headers(ts, sign)
        url = BASE_URL + path + query
        try:
            r = requests.request(method, url, headers=headers,
                                 data=body_str if body else None, timeout=self.timeout)
        except Exception as e:
            raise BitgetError(f"Bitget request {path} failed: {e}") from e
        if r.status_code != 200:
            raise BitgetError(f"Bitget {path} HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        if str(data.get("code")) != "00000":
            raise BitgetError(f"Bitget {path} api-code {data.get('code')}: {data.get('msg')}")
        return data.get("data") or {}

    # ----- account ------------------------------------------------------
    def account(self, symbol: str = "BTCUSDT") -> dict:
        """Devuelve el saldo de la cuenta de futures (USDT-M)."""
        return self._request("GET", "/api/v2/mix/account/account",
                             params={"symbol": symbol, "productType": PRODUCT_TYPE,
                                     "marginCoin": MARGIN_COIN})

    def equity_usdt(self, symbol: str = "BTCUSDT") -> float:
        """Equity total de la cuenta (USDT) — el numero que usamos como cartera."""
        a = self.account(symbol)
        # campos habituales: 'accountEquity' (equity total con PnL no realizado),
        # 'available', 'crossedMaxAvailable'. Usamos accountEquity.
        for k in ("accountEquity", "usdtEquity", "equity", "available"):
            if k in a and a[k] not in (None, ""):
                try: return float(a[k])
                except (TypeError, ValueError): continue
        raise BitgetError(f"No se pudo leer equity de la cuenta: {a}")

    # ----- leverage / margin mode --------------------------------------
    def set_leverage(self, symbol: str, leverage: int,
                     hold_side: str = "long") -> dict:
        """Ajusta leverage para un lado (long/short). En crossed margin afecta a ambos."""
        return self._request("POST", "/api/v2/mix/account/set-leverage", body={
            "symbol": symbol, "productType": PRODUCT_TYPE, "marginCoin": MARGIN_COIN,
            "leverage": str(int(leverage)), "holdSide": hold_side,
        })

    def set_margin_mode(self, symbol: str, mode: str = "crossed") -> dict:
        """mode: 'crossed' (recomendado para 1 sola pos) o 'isolated'."""
        return self._request("POST", "/api/v2/mix/account/set-margin-mode", body={
            "symbol": symbol, "productType": PRODUCT_TYPE,
            "marginCoin": MARGIN_COIN, "marginMode": mode,
        })

    # ----- ordenes ------------------------------------------------------
    def place_market_order(self, symbol: str, side: str, size: float,
                            reduce_only: bool = False,
                            client_oid: str | None = None) -> dict:
        """Order market. side='buy' o 'sell'. size en BTC (contracts coin)."""
        body = {
            "symbol": symbol, "productType": PRODUCT_TYPE, "marginCoin": MARGIN_COIN,
            "marginMode": "crossed",
            "size": str(size),
            "side": side,                  # buy / sell
            "orderType": "market",
            "reduceOnly": "YES" if reduce_only else "NO",
            "tradeSide": "close" if reduce_only else "open",
        }
        if client_oid: body["clientOid"] = client_oid
        return self._request("POST", "/api/v2/mix/order/place-order", body=body)

    def place_tp_sl(self, symbol: str, hold_side: str,
                    tp_price: float | None, sl_price: float | None,
                    size: float) -> list[dict]:
        """Coloca TP y SL como ordenes condicionales reduceOnly (plan order).

        hold_side: 'long' o 'short' (lado de la posicion abierta).
        Devuelve los responses (puede ser 1 o 2 segun cuales tengan precio).
        """
        out = []
        for plan_type, trigger in (("pos_profit", tp_price), ("pos_loss", sl_price)):
            if trigger is None: continue
            out.append(self._request("POST", "/api/v2/mix/order/place-tpsl-order", body={
                "symbol": symbol, "productType": PRODUCT_TYPE, "marginCoin": MARGIN_COIN,
                "planType": plan_type, "triggerType": "mark_price",
                "triggerPrice": str(round(trigger, 1)),
                "holdSide": hold_side, "size": str(size),
            }))
        return out

    def cancel_all_plan_orders(self, symbol: str) -> dict:
        """Cancela todas las TP/SL del par (usado al cerrar manualmente)."""
        return self._request("POST", "/api/v2/mix/order/cancel-plan-order", body={
            "symbol": symbol, "productType": PRODUCT_TYPE, "marginCoin": MARGIN_COIN,
            "planType": "profit_loss",
        })

    # ----- consultas ----------------------------------------------------
    def position(self, symbol: str) -> dict:
        """Posicion abierta del par (size=0 si no hay)."""
        return self._request("GET", "/api/v2/mix/position/single-position",
                             params={"symbol": symbol, "productType": PRODUCT_TYPE,
                                     "marginCoin": MARGIN_COIN})

    def has_open_position(self, symbol: str) -> bool:
        try:
            p = self.position(symbol)
            # Bitget devuelve lista o dict segun version
            items = p if isinstance(p, list) else [p]
            for it in items:
                if float(it.get("total", 0) or 0) > 0: return True
            return False
        except Exception as e:
            log.warning("position check failed: %s", e)
            return False

    def ticker_price(self, symbol: str) -> float:
        """Precio mark del par (publico, sin auth)."""
        r = requests.get(f"{BASE_URL}/api/v2/mix/market/ticker",
                          params={"symbol": symbol, "productType": PRODUCT_TYPE},
                          timeout=self.timeout)
        r.raise_for_status()
        d = r.json().get("data", [])
        if isinstance(d, list) and d:
            return float(d[0]["lastPr"])
        return float(d.get("lastPr"))
