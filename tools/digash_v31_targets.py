#!/usr/bin/env python3
from __future__ import annotations

import bisect

import numpy as np
import pandas as pd

from digash_v3_common import Level, TF_MINUTES

TARGET_SAME_ZONE_PCT = 0.0025


def _next_price(nodes: list[tuple[float, int]], side: int, entry: float, source_level: float) -> float:
    """Nearest already-known horizontal level in trade direction, excluding the source zone."""
    if not nodes:
        return np.nan
    if side > 0:
        floor = max(entry, source_level * (1.0 + TARGET_SAME_ZONE_PCT))
        j = bisect.bisect_right(nodes, (floor, 10**18))
        return float(nodes[j][0]) if j < len(nodes) else np.nan
    ceil = min(entry, source_level * (1.0 - TARGET_SAME_ZONE_PCT))
    j = bisect.bisect_left(nodes, (ceil, -1)) - 1
    return float(nodes[j][0]) if j >= 0 else np.nan


def assign_targets(events, levels: list[Level], x5: pd.DataFrame) -> dict[int, dict[str, float]]:
    """
    Causal target lookup.

    Horizontal S/R roles can flip after a break, so target relevance is determined by
    current price direction rather than by the level's original S/R label. This keeps
    the robust part of V3 target semantics while adding a same-or-higher-TF target to
    stop tiny lower-timeframe duplicate levels from defining all structural room.
    """
    if not events or not levels:
        return {}

    levs = sorted(levels, key=lambda z: z.formed_time)
    lp = 0
    all_nodes: list[tuple[float, int]] = []
    thresholds = {m: [] for m in sorted(set(TF_MINUTES.values()))}
    times = pd.to_datetime(x5["signal_time"], utc=True)
    out: dict[int, dict[str, float]] = {}

    for ei, e in sorted(enumerate(events), key=lambda z: z[1].signal_idx):
        ts = pd.Timestamp(times.iloc[e.signal_idx])
        while lp < len(levs) and pd.Timestamp(levs[lp].formed_time) <= ts:
            lv = levs[lp]
            node = (float(lv.price), int(lv.level_id))
            bisect.insort(all_nodes, node)
            for m in thresholds:
                if lv.tf_minutes >= m:
                    bisect.insort(thresholds[m], node)
            lp += 1

        entry = float(x5.iloc[e.entry_idx]["open"]) if e.entry_idx < len(x5) else e.level_price
        out[ei] = {
            "any": _next_price(all_nodes, e.side, entry, e.level_price),
            "htf": _next_price(thresholds[e.tf_minutes], e.side, entry, e.level_price),
        }
    return out
