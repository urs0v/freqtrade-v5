#!/usr/bin/env python3
"""Retry-hardened launcher for backfill_logbasis_8h.py."""
from __future__ import annotations

import time

import backfill_logbasis_8h as base

_original_get_url = base.get_url


def get_url_retry(url: str, timeout: int = 60):
    last = ("error:unknown", b"")
    for attempt in range(4):
        status, data = _original_get_url(url, timeout=timeout)
        last = (status, data)
        if status in {"ok", "missing"}:
            return status, data
        if attempt < 3:
            time.sleep(0.5 * (2 ** attempt))
    return last


base.get_url = get_url_retry

if __name__ == "__main__":
    raise SystemExit(base.main())
