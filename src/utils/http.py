"""HTTP cliente con retry/backoff exponencial."""
from __future__ import annotations

import time
from typing import Any

import requests

from .errors import DataFetchError
from .logging_utils import get_logger

log = get_logger("http")


def request_with_retry(method: str, url: str, *,
                       max_retries: int = 4,
                       backoff_base: float = 1.5,
                       timeout: int = 15,
                       **kwargs) -> Any:
    last_exc = None
    for attempt in range(max_retries):
        try:
            r = requests.request(method, url, timeout=timeout, **kwargs)
            if r.status_code == 429:
                wait = backoff_base ** (attempt + 2)
                log.warning("429 rate limited on %s, waiting %.1fs", url, wait)
                time.sleep(wait)
                continue
            if r.status_code >= 500:
                wait = backoff_base ** (attempt + 1)
                log.warning("HTTP %d on %s, retry in %.1fs", r.status_code, url, wait)
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r
        except (requests.ConnectionError, requests.Timeout) as e:
            last_exc = e
            wait = backoff_base ** (attempt + 1)
            log.warning("%s on %s, retry in %.1fs (%d/%d)",
                        type(e).__name__, url, wait, attempt + 1, max_retries)
            time.sleep(wait)
    raise DataFetchError(f"Failed {method} {url}: {last_exc}")


def get_json(url: str, params: dict | None = None, **kwargs) -> Any:
    r = request_with_retry("GET", url, params=params, **kwargs)
    return r.json()
