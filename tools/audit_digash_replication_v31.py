#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from digash_v3_common import *
from digash_v31_events import detect_events, dedup_events, assign_targets, simulate


def main() -> int:
    a = parse_args()
    cfg = json.loads(Path(a.config).read_text())
    pairs = list(cfg.get("exchange", {}).get("pair_whitelist", []))
    if len(pairs) != 1:
        raise RuntimeError("Worker config must contain one pair")
    pair = pairs[0]
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    datadir = Path(a.datadir)
    start = pd.Timestamp(a.start, tz="UTC")
    end = pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)
    warm = pd.Timedelta(days=45)
    t0 = time.monotonic()

    log(f"=== DIGASH REPLICATION V3.1 | {pair} ===")
    progress("load", 0, 1)
    raw15 = load_tf(cfg, datadir, pair, "15m")
    raw5, source = load_5m(cfg, datadir, pair)
    if raw15.empty or raw5.empty:
        raise RuntimeError(f"Missing cached 15m/detail for {pair}")
    x15 = prep_ohlcv(raw15, 15)
    x5raw = prep_ohlcv(raw5, 5)
    x15 = x15[(x15.date >= start-warm) & (x15.date < end+pd.Timedelta(hours=8))].reset_index(drop=True)
    x5raw = x5raw[(x5raw.date >= start-warm) & (x5raw.date < end+pd.Timedelta(hours=8))].reset_index(drop=True)
    activity = pd.read_pickle(a.activity_file)
    x5 = prepare_5m_with_activity(x5raw, activity)
    tfs = {
        "5m": x5[["date", "open", "high", "low", "close", "volume", "atr", "signal_time"]].copy(),
        "15m": x15,
        "1h": resample_from_15(x15, "1h", 60),
        "4h": resample_from_15(x15, "4h", 240),
    }
    progress("load", 1, 1)

    levels: list[Level] = []
    next_id = 0
    jobs = [(tf, p) for tf in TFS for p in PERIODS]
    for j, (tf, p) in enumerate(jobs, 1):
        z = build_levels(tfs[tf], tf, p, next_id)
        levels.extend(z)
        next_id += len(z)
        progress("levels", j, len(jobs))
    if not levels:
        raise RuntimeError(f"No levels for {pair}")

    # Keep the full causal event stream for level-role lifecycle (R->S / S->R on break,
    # restored on quick fakeout). Deduplication is only for the simulated trade sample.
    raw_events = detect_events(x5, levels)
    events = dedup_events(raw_events)
    targets = assign_targets(events, levels, x5, lifecycle_events=raw_events)
    level_map = {z.level_id: z for z in levels}

    progress("simulate", 0, max(len(events), 1))
    rows = []
    for i, e in enumerate(events):
        if i % 1000 == 0 or i+1 == len(events):
            progress("simulate", i+1, max(len(events), 1))
        row = simulate(x5, e, pair, targets.get(i, {}), level_map)
        if row is None:
            continue
        et = pd.Timestamp(row["entry_time"])
        if start <= et < end:
            rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError(f"No events inside requested period for {pair}")
    df.to_csv(outdir / "events.csv", index=False)

    pd.DataFrame([asdict(x) for x in levels]).to_csv(outdir / "levels.csv", index=False)
    pd.DataFrame([{
        "pair": pair, "detail_source": source, "bars5": len(x5), "bars15": len(x15),
        "levels": len(levels), "raw_events": len(raw_events), "dedup_events": len(df),
        "elapsed_s": time.monotonic()-t0,
    }]).to_csv(outdir / "coverage.csv", index=False)
    log(f"DONE|{pair}|levels={len(levels)}|raw_events={len(raw_events)}|events={len(df)}|elapsed={time.monotonic()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
