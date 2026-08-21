#!/usr/bin/env python3
from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pandas as pd

import digash_v31_events as de


STATE_VERSION = 2


def level_key(lv) -> tuple:
    """Stable identity for a frozen level across append-only history extensions."""
    return (
        str(lv.tf),
        int(lv.period),
        str(lv.kind),
        pd.Timestamp(lv.formed_time).isoformat(),
        round(float(lv.price), 12),
        round(float(lv.init_price), 12),
        round(float(lv.touch_price), 12),
    )


def _fresh_state() -> dict[str, Any]:
    return {
        "approach": 0,
        "inside": False,
        "break_done": False,
        "bounce_done": False,
        "retest_done": False,
        "fakeout_done": False,
        "break_idx": None,
        "break_side": 0,
        "break_extreme": np.nan,
        "bounce_candidate": None,
    }


def score_event(e) -> tuple:
    return (
        int(e.confluence_tfs),
        int(e.tf_minutes),
        int(e.period),
        -float(e.touch_error_pct),
    )


def causal_dedup_incremental(events, seen_keys: set[tuple] | None = None):
    """V1.6 causal first-bar 3-candle dedup with persistent bucket memory."""
    seen = set(seen_keys or set())
    if not events:
        return [], seen

    ordered = sorted(events, key=lambda e: (int(e.signal_idx), str(e.setup), int(e.side)))
    by_idx: dict[int, list] = {}
    for e in ordered:
        by_idx.setdefault(int(e.signal_idx), []).append(e)

    out = []
    for idx in sorted(by_idx):
        same_bar = {}
        for e in by_idx[idx]:
            key = (str(e.setup), int(e.side), int(e.signal_idx) // 3)
            if key in seen:
                continue
            old = same_bar.get(key)
            if old is None or score_event(e) > score_event(old):
                same_bar[key] = e
        for key, e in same_bar.items():
            out.append(e)
            seen.add(key)
    return sorted(out, key=lambda e: (int(e.signal_idx), str(e.setup), int(e.side))), seen


def detect_events_incremental(
    x5: pd.DataFrame,
    levels: list,
    *,
    start_i: int = 1,
    stop_i: int | None = None,
    initial_state: dict[tuple, dict] | None = None,
    prior_signal_time: pd.Timestamp | None = None,
    index_offset: int = 0,
):
    """Exact resumable continuation of digash_v31_events.detect_events().

    ``start_i`` / ``stop_i`` refer to local dataframe positions. Event indices and
    persisted lifecycle indices are emitted in the global index space defined by
    ``index_offset``. The default offset is zero, preserving the historical/batch
    behavior. Live callers can therefore keep only a small OHLCV tail while age,
    causal 3-bar dedup and lifecycle state continue in the original global index
    space.
    """
    initial_state = initial_state or {}
    natural_end = len(x5) - 2
    if natural_end < 1:
        return [], dict(initial_state), natural_end
    end_i = natural_end if stop_i is None else min(natural_end, int(stop_i))
    start_i = max(1, int(start_i))
    if start_i > end_i:
        return [], dict(initial_state), end_i
    if not levels:
        return [], {}, end_i

    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    c = x5["close"].to_numpy(float)
    atr = x5["atr"].to_numpy(float)
    rz = x5["range_z"].to_numpy(float) if "range_z" in x5 else np.full(len(x5), np.nan)
    vs = x5["volume_spike_local"].to_numpy(float) if "volume_spike_local" in x5 else np.full(len(x5), np.nan)
    times = pd.to_datetime(x5["signal_time"], utc=True)
    time_ns = times.astype("int64").to_numpy()
    last_hi, last_lo = de._confirmed_structure_arrays(h, l)

    by_price = sorted(levels, key=lambda z: z.price)
    level_prices = np.fromiter((lv.price for lv in by_price), dtype=np.float64, count=len(by_price))
    level_formed_ns = np.fromiter(
        (pd.Timestamp(lv.formed_time).value for lv in by_price), dtype=np.int64, count=len(by_price)
    )
    get_conf = de._make_confluence_getter(levels)

    state_by_id = {}
    key_by_id = {}
    for lv in levels:
        k = level_key(lv)
        key_by_id[lv.level_id] = k
        if k in initial_state:
            state_by_id[lv.level_id] = copy.deepcopy(initial_state[k])
        else:
            if prior_signal_time is not None and pd.Timestamp(lv.formed_time) <= pd.Timestamp(prior_signal_time):
                raise RuntimeError(f"INCREMENTAL_REBOOTSTRAP_REQUIRED level={k}")
            state_by_id[lv.level_id] = _fresh_state()

    events = []
    for i in range(start_i, end_i + 1):
        gi = int(index_offset) + int(i)
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        lowq = l[i] / (1.0 + de.TOUCH_TOL_PCT)
        highq = h[i] / (1.0 - de.TOUCH_TOL_PCT)
        a0 = int(np.searchsorted(level_prices, lowq, side="left"))
        a1 = int(np.searchsorted(level_prices, highq, side="right"))
        if a0 == a1:
            continue

        ns = int(time_ns[i])
        impulse = bool((np.isfinite(rz[i]) and rz[i] >= 1.0) or (np.isfinite(vs[i]) and vs[i] >= 2.0))

        for kk in range(a0, a1):
            if level_formed_ns[kk] > ns:
                continue
            lv = by_price[kk]
            st = state_by_id[lv.level_id]
            level = lv.price

            dist_now = abs(c[i] / level - 1.0)
            near_now = dist_now <= de.PROTO_NEAR_PCT
            if near_now and not st["inside"]:
                st["approach"] += 1
                st["inside"] = True
            elif not near_now and dist_now > de.PROTO_NEAR_PCT * 1.5:
                st["inside"] = False

            bc = st.get("bounce_candidate")
            if bc is not None and not st["bounce_done"]:
                age = gi - int(bc["idx"])
                if 1 <= age <= de.BOUNCE_CONFIRM_BARS:
                    # A live tail is always retained long enough to cover the
                    # short micro-confirmation lookback used by the frozen rule.
                    if bc["side"] > 0:
                        micro = float(np.max(h[max(0, i - age - 2):i - age + 1]))
                        confirmed = c[i] > micro
                    else:
                        micro = float(np.min(l[max(0, i - age - 2):i - age + 1]))
                        confirmed = c[i] < micro
                    if confirmed:
                        near_bars, near_proxy = de._generic_near(c, i, level)
                        extreme = bc["extreme"]
                        stop = extreme - 0.05 * atr[i] if bc["side"] > 0 else extreme + 0.05 * atr[i]
                        events.append(de.V31Event(
                            "H_BOUNCE", gi, gi + 1, bc["side"], float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), near_proxy, near_bars, impulse,
                            confluence_tfs=get_conf(lv, ns), stop_source="reaction_extreme",
                        ))
                        st["bounce_done"] = True
                        st["bounce_candidate"] = None
                elif age > de.BOUNCE_CONFIRM_BARS:
                    st["bounce_candidate"] = None

            cross_up = c[i] > level and c[i - 1] <= level
            cross_dn = c[i] < level and c[i - 1] >= level
            if not st["break_done"]:
                side = +1 if lv.kind == "R" and cross_up else -1 if lv.kind == "S" and cross_dn else 0
                if side:
                    near_bars, proto = de._breakout_protor(c, i, level, side)
                    stop, stop_source = de._breakout_stop(i, side, level, atr, h, l, last_hi, last_lo)
                    break_atr = side * (c[i] - level) / atr[i]
                    events.append(de.V31Event(
                        "H_BREAK", gi, gi + 1, side, float(stop), lv.level_id, level, lv.kind,
                        lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                        int(st["approach"]), proto, near_bars, impulse,
                        confluence_tfs=get_conf(lv, ns), break_distance_atr=float(break_atr),
                        stop_source=stop_source,
                    ))
                    st["break_done"] = True
                    st["break_idx"] = gi
                    st["break_side"] = side
                    st["break_extreme"] = float(h[i] if side > 0 else l[i])
                    continue

                if lv.kind == "S" and l[i] <= level and c[i] >= level:
                    st["bounce_candidate"] = {"idx": gi, "side": +1, "extreme": float(l[i])}
                elif lv.kind == "R" and h[i] >= level and c[i] <= level:
                    st["bounce_candidate"] = {"idx": gi, "side": -1, "extreme": float(h[i])}
            else:
                bi = st["break_idx"]
                bside = st["break_side"]
                age = gi - int(bi)
                if age <= 0:
                    continue
                st["break_extreme"] = (
                    max(float(st["break_extreme"]), float(h[i])) if bside > 0
                    else min(float(st["break_extreme"]), float(l[i]))
                )

                if not st["fakeout_done"] and age <= de.FAKEOUT_MAX_BARS:
                    reclaim = (bside > 0 and c[i] < level) or (bside < 0 and c[i] > level)
                    if reclaim:
                        near_bars, near_proxy = de._generic_near(c, i, level)
                        side = -bside
                        ext = float(st["break_extreme"])
                        stop = ext + 0.05 * atr[i] if side < 0 else ext - 0.05 * atr[i]
                        events.append(de.V31Event(
                            "H_FAKEOUT", gi, gi + 1, side, float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), near_proxy, near_bars, impulse, reclaim_bars=age,
                            confluence_tfs=get_conf(lv, ns), stop_source="sweep_extreme",
                        ))
                        st["fakeout_done"] = True

                if not st["retest_done"] and age <= de.RETEST_MAX_BARS:
                    if bside > 0:
                        touched = l[i] <= level and c[i] > level
                        if touched:
                            stop = min(float(l[i] - 0.05 * atr[i]), float(level - 0.10 * atr[i]))
                    else:
                        touched = h[i] >= level and c[i] < level
                        if touched:
                            stop = max(float(h[i] + 0.05 * atr[i]), float(level + 0.10 * atr[i]))
                    if touched:
                        near_bars, near_proxy = de._generic_near(c, i, level)
                        events.append(de.V31Event(
                            "H_RETEST", gi, gi + 1, bside, float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), near_proxy, near_bars, impulse, reclaim_bars=age,
                            confluence_tfs=get_conf(lv, ns), stop_source="retest_extreme",
                        ))
                        st["retest_done"] = True

    next_state = {key_by_id[lid]: copy.deepcopy(st) for lid, st in state_by_id.items()}
    return events, next_state, end_i
