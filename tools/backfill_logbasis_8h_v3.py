#!/usr/bin/env python3
"""Retry-hardened + Unicode-safe launcher for backfill_logbasis_8h.py."""
from __future__ import annotations

import time
from urllib.parse import quote

import backfill_logbasis_8h as base

_original_get_url = base.get_url


def get_url_unicode_retry(url: str, timeout: int = 60):
    # urllib/http.client expects an ASCII request target. Binance/core symbol names can
    # contain non-ASCII characters, so percent-encode only the URL path characters
    # while preserving the URL structure.
    encoded_url = quote(url, safe=":/%?=&")
    last = ("error:unknown", b"")
    for attempt in range(4):
        status, data = _original_get_url(encoded_url, timeout=timeout)
        last = (status, data)
        if status in {"ok", "missing"}:
            return status, data
        if attempt < 3:
            time.sleep(0.5 * (2 ** attempt))
    return last


base.get_url = get_url_unicode_retry

if __name__ == "__main__":
    raise SystemExit(base.main())
