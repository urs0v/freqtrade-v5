#!/usr/bin/env python3
"""
CTREND research V3.

This is a surgical correction on top of cttrend_research_v2.py.
V2 required an exact complete Sunday daily bar for every weekly exit. Since the
source is aggregated from 6h Binance archives and daily construction deliberately
keeps only days with exactly four 6h bars, one missing archive bar can remove the
whole Sunday and make a continuously traded contract look un-exitable.

V3 keeps the point-in-time universe unchanged and uses a pre-declared execution
rule that does not condition selection on future availability:

    planned exit = next Sunday close
    execution proxy = latest archived complete daily close strictly after entry
                      and no later than planned exit

If the contract has later data after the planned exit, an earlier proxy is marked
as an archive-gap fallback. If the contract's archive truly ends before the
planned exit, the same price is marked as a forced/delisting exit.

No signal, training, universe, cost, leverage, or CTREND parameter is changed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import cttrend_research_v2 as base


def attach_forward_exits(sun: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    sun = sun.copy()
    sun["planned_exit_date"] = sun["date"] + pd.Timedelta(days=7)
    sun["exit_close"] = np.nan
    # Keep the column explicitly UTC-aware. Pandas 3.x rejects assigning a
    # timezone-aware Timestamp into a tz-naive datetime64[ns] column.
    sun["actual_exit_date"] = pd.Series(
        pd.NaT, index=sun.index, dtype="datetime64[ns, UTC]"
    )
    sun["forced_exit"] = False
    sun["archive_gap_exit"] = False

    daily_sorted = daily[["symbol", "date", "close"]].sort_values(["symbol", "date"])
    daily_groups = {sym: g for sym, g in daily_sorted.groupby("symbol", sort=False)}

    fallback_count = 0
    forced_count = 0
    no_exit_count = 0

    for sym, idx in sun.groupby("symbol", sort=False).groups.items():
        g = daily_groups.get(sym)
        if g is None or g.empty:
            no_exit_count += len(idx)
            continue

        # searchsorted works on tz-naive numpy datetime64 values. Convert only
        # for the search; all DataFrame-facing timestamps remain UTC-aware.
        dates = g["date"].dt.tz_convert(None).to_numpy(dtype="datetime64[ns]")
        closes = g["close"].to_numpy(dtype=float)
        last_date = pd.Timestamp(g["date"].iloc[-1])

        for row_idx in idx:
            entry_date = pd.Timestamp(sun.at[row_idx, "date"])
            planned = pd.Timestamp(sun.at[row_idx, "planned_exit_date"])

            # Deterministic weekly execution proxy: latest complete archived day
            # at or before the planned exit. Never use a price after planned exit.
            planned_naive = planned.tz_convert(None) if planned.tzinfo is not None else planned
            pos = int(np.searchsorted(dates, planned_naive.to_datetime64(), side="right") - 1)
            if pos < 0:
                no_exit_count += 1
                continue

            actual = pd.Timestamp(dates[pos]).tz_localize("UTC")
            # A valid holding-period exit must occur strictly after the rebalance.
            if actual <= entry_date:
                no_exit_count += 1
                continue

            sun.at[row_idx, "exit_close"] = float(closes[pos])
            sun.at[row_idx, "actual_exit_date"] = actual

            if actual < planned:
                # Classification is diagnostic only; it never affects selection.
                # If the symbol trades after planned exit, this was merely an
                # archive gap. Otherwise its archive ended during the holding week.
                if last_date > planned:
                    sun.at[row_idx, "archive_gap_exit"] = True
                    fallback_count += 1
                elif last_date < planned:
                    sun.at[row_idx, "forced_exit"] = True
                    forced_count += 1

    sun["fwd_ret"] = sun["exit_close"] / sun["close"] - 1.0

    print(
        "Exit-price audit: "
        f"archive_gap_fallbacks={fallback_count:,} | "
        f"forced_contract_ends={forced_count:,} | "
        f"no_valid_exit={no_exit_count:,}",
        flush=True,
    )
    return sun


# weekly_panel() in V2 resolves this name from its module globals at call time.
base.attach_forward_exits = attach_forward_exits


if __name__ == "__main__":
    raise SystemExit(base.main())
