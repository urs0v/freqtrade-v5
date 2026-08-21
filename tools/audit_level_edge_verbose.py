#!/usr/bin/env python3
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_level_edge as a

_started = time.monotonic()
_scan_count = defaultdict(int)
_sim_count = defaultdict(int)
_level_total = {}


def log(msg: str):
    print(f"[stage +{time.monotonic()-_started:7.1f}s] {msg}", flush=True)


_orig_load_tf = a.load_tf
_orig_prepare_15m = a.prepare_15m
_orig_make_5m = a.make_5m
_orig_confirmed_levels = a.confirmed_levels
_orig_first_trade_at_level = a.first_trade_at_level
_orig_simulate_event = a.simulate_event


def load_tf(config, datadir, pair, timeframe):
    t = time.monotonic()
    log(f"{pair}: load {timeframe} START")
    out = _orig_load_tf(config, datadir, pair, timeframe)
    log(f"{pair}: load {timeframe} DONE rows={len(out):,} [{time.monotonic()-t:.1f}s]")
    return out


def prepare_15m(df):
    t = time.monotonic()
    log(f"prepare 15m features START rows={len(df):,}")
    out = _orig_prepare_15m(df)
    log(f"prepare 15m features DONE rows={len(out):,} [{time.monotonic()-t:.1f}s]")
    return out


def make_5m(config, datadir, pair):
    t = time.monotonic()
    log(f"{pair}: detail data START (prefer 5m, fallback existing 1m->5m)")
    out, source = _orig_make_5m(config, datadir, pair)
    log(f"{pair}: detail data DONE source={source} rows={len(out):,} [{time.monotonic()-t:.1f}s]")
    return out, source


def confirmed_levels(x15):
    t = time.monotonic()
    log(f"build confirmed 15m levels START bars={len(x15):,}")
    out = _orig_confirmed_levels(x15)
    log(f"build confirmed 15m levels DONE levels={len(out):,} [{time.monotonic()-t:.1f}s]")
    return out


def first_trade_at_level(pair, kind, level, level_time, known_time, x5):
    _scan_count[pair] += 1
    n = _scan_count[pair]
    if n == 1:
        log(f"{pair}: scan level interactions START")
    elif n % 500 == 0:
        log(f"{pair}: scan level interactions {n:,} levels checked")
    return _orig_first_trade_at_level(pair, kind, level, level_time, known_time, x5)


def simulate_event(ev, x5):
    pair = ev.pair
    _sim_count[pair] += 1
    n = _sim_count[pair]
    if n == 1:
        log(f"{pair}: simulate detected setups START")
    elif n % 500 == 0:
        log(f"{pair}: simulate detected setups {n:,} events")
    return _orig_simulate_event(ev, x5)


a.load_tf = load_tf
a.prepare_15m = prepare_15m
a.make_5m = make_5m
a.confirmed_levels = confirmed_levels
a.first_trade_at_level = first_trade_at_level
a.simulate_event = simulate_event

if __name__ == "__main__":
    log("verbose instrumentation enabled")
    raise SystemExit(a.main())
