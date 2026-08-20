#!/usr/bin/env python3
"""Causal funding-window correction for reversal8w_perp_test.py.

The base test rebalances at the complete Sunday close. Funding for the new
weekly position must therefore start on Monday 00:00 UTC, not at Sunday 00:00.
This wrapper changes only that accounting boundary; signal, universe, returns,
costs, portfolio construction, and pre-registered gates remain unchanged.
"""
from __future__ import annotations

import pandas as pd

import reversal8w_perp_test as base

_orig_funding_sum = base.funding_sum


def funding_sum(pref, sym: str, entry: pd.Timestamp, exit_: pd.Timestamp):
    return _orig_funding_sum(pref, sym, pd.Timestamp(entry) + pd.Timedelta(days=1), exit_)


base.funding_sum = funding_sum


if __name__ == "__main__":
    raise SystemExit(base.main())
