#!/usr/bin/env python3
from __future__ import annotations

import bisect
from dataclasses import dataclass

import numpy as np
import pandas as pd

from digash_v3_common import *

BREAK_FACT_ATR = 0.10
TARGET_SAME_ZONE_PCT = 0.0025
STRUCTURE_PIVOT_SPAN = 2


@dataclass
class V31Event:
    setup: str
    signal_idx: int
    entry_idx: int
    side: int
    stop: float
    level_id: int
    level_price: float
    level_kind: str
    tf: str
    tf_minutes: int
    period: int
    touch_error_pct: float
    clean_between: bool
    approach_no: int
    protor_proxy: bool
    near_bars_6: int
    impulse_proxy: bool
    reclaim_bars: int = 0
    confluence_tfs: int = 1
    break_distance_atr: float = np.nan
    stop_source: str = ""


def _make_confluence_getter(levels: list[Level]):
    indexes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tf in TFS:
        z = sorted((lv for lv in levels if lv.tf == tf), key=lambda lv: lv.price)
        if not z:
            continue
        prices = np.fromiter((lv.price for lv in z), dtype=np.float64, count=len(z))
        formed = np.fromiter((pd.Timestamp(lv.formed_time).value for lv in z), dtype=np.int64, count=len(z))
        indexes[tf] = (prices, formed)

    def get(lv: Level, cutoff_ns: int) -> int:
        p = lv.price
        n = 0
        for prices, formed in indexes.values():
            lo = int(np.searchsorted(prices, p * (1.0 - TARGET_SAME_ZONE_PCT), side="left"))
            hi = int(np.searchsorted(prices, p * (1.0 + TARGET_SAME_ZONE_PCT), side="right"))
            if hi > lo and np.any(formed[lo:hi] <= cutoff_ns):
                n += 1
        return max(1, n)

    return get


def _confirmed_structure_arrays(high: np.ndarray, low: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = len(high)
    last_hi = np.full(n, np.nan)
    last_lo = np.full(n, np.nan)
    cur_hi = np.nan
    cur_lo = np.nan
    s = STRUCTURE_PIVOT_SPAN
    for i in range(n):
        j = i - s
        if j >= s and j + s < n:
            if high[j] >= np.max(high[j-s:j+s+1]):
                cur_hi = float(high[j])
            if low[j] <= np.min(low[j-s:j+s+1]):
                cur_lo = float(low[j])
        last_hi[i] = cur_hi
        last_lo[i] = cur_lo
    return last_hi, last_lo


def _breakout_stop(
    i: int, side: int, level: float, atr: np.ndarray,
    high: np.ndarray, low: np.ndarray, last_hi: np.ndarray, last_lo: np.ndarray,
) -> tuple[float, str]:
    a = float(atr[i])
    if side > 0:
        p = float(last_lo[i]) if np.isfinite(last_lo[i]) else np.nan
        if np.isfinite(p) and p < level:
            return p, "confirmed_5m_swing"
        r = float(np.min(low[max(0, i-12):i+1]))
        if r < level:
            return r, "recent_5m_structure"
        return float(level - 0.25*a), "atr_fallback"
    p = float(last_hi[i]) if np.isfinite(last_hi[i]) else np.nan
    if np.isfinite(p) and p > level:
        return p, "confirmed_5m_swing"
    r = float(np.max(high[max(0, i-12):i+1]))
    if r > level:
        return r, "recent_5m_structure"
    return float(level + 0.25*a), "atr_fallback"


def _breakout_protor(close: np.ndarray, i: int, level: float, side: int) -> tuple[int, bool]:
    if i < PROTO_BARS:
        return 0, False
    z = close[i-PROTO_BARS:i]
    dist = np.abs(z / level - 1.0)
    near = dist <= PROTO_NEAR_PCT
    n = int(near.sum())
    if side > 0:
        one_sided = float(np.mean(z <= level)) >= (PROTO_BARS - 1) / PROTO_BARS
    else:
        one_sided = float(np.mean(z >= level)) >= (PROTO_BARS - 1) / PROTO_BARS
    contracting = float(np.median(dist[-3:])) <= float(np.median(dist[:3]))
    return n, bool(n >= PROTO_MIN_NEAR and one_sided and contracting)


def _generic_near(close: np.ndarray, i: int, level: float) -> tuple[int, bool]:
    if i < PROTO_BARS:
        return 0, False
    z = close[i-PROTO_BARS:i]
    n = int((np.abs(z / level - 1.0) <= PROTO_NEAR_PCT).sum())
    return n, n >= PROTO_MIN_NEAR


def detect_events(x5: pd.DataFrame, levels: list[Level]) -> list[V31Event]:
    if not levels:
        return []

    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    c = x5["close"].to_numpy(float)
    atr = x5["atr"].to_numpy(float)
    rz = x5["range_z"].to_numpy(float) if "range_z" in x5 else np.full(len(x5), np.nan)
    vs = x5["volume_spike_local"].to_numpy(float) if "volume_spike_local" in x5 else np.full(len(x5), np.nan)
    times = pd.to_datetime(x5["signal_time"], utc=True)
    time_ns = times.astype("int64").to_numpy()
    last_hi, last_lo = _confirmed_structure_arrays(h, l)

    by_price = sorted(levels, key=lambda z: z.price)
    level_prices = np.fromiter((lv.price for lv in by_price), dtype=np.float64, count=len(by_price))
    level_formed_ns = np.fromiter(
        (pd.Timestamp(lv.formed_time).value for lv in by_price), dtype=np.int64, count=len(by_price)
    )
    get_conf = _make_confluence_getter(levels)

    state = {
        lv.level_id: {
            "approach": 0, "inside": False, "break_done": False, "bounce_done": False,
            "retest_done": False, "fakeout_done": False, "break_idx": None, "break_side": 0,
            "break_extreme": np.nan, "bounce_candidate": None,
        } for lv in levels
    }
    events: list[V31Event] = []

    progress("scan", 0, len(x5))
    for i in range(1, len(x5)-1):
        if i % 5000 == 0:
            progress("scan", i, len(x5))
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        lowq = l[i] / (1.0 + TOUCH_TOL_PCT)
        highq = h[i] / (1.0 - TOUCH_TOL_PCT)
        a0 = int(np.searchsorted(level_prices, lowq, side="left"))
        a1 = int(np.searchsorted(level_prices, highq, side="right"))
        if a0 == a1:
            continue

        ns = int(time_ns[i])
        impulse = bool((np.isfinite(rz[i]) and rz[i] >= 1.0) or (np.isfinite(vs[i]) and vs[i] >= 2.0))

        for k in range(a0, a1):
            if level_formed_ns[k] > ns:
                continue
            lv = by_price[k]
            st = state[lv.level_id]
            level = lv.price

            dist_now = abs(c[i] / level - 1.0)
            near_now = dist_now <= PROTO_NEAR_PCT
            if near_now and not st["inside"]:
                st["approach"] += 1
                st["inside"] = True
            elif not near_now and dist_now > PROTO_NEAR_PCT * 1.5:
                st["inside"] = False

            bc = st.get("bounce_candidate")
            if bc is not None and not st["bounce_done"]:
                age = i - bc["idx"]
                if 1 <= age <= BOUNCE_CONFIRM_BARS:
                    if bc["side"] > 0:
                        micro = float(np.max(h[max(0, bc["idx"]-2):bc["idx"]+1]))
                        confirmed = c[i] > micro
                    else:
                        micro = float(np.min(l[max(0, bc["idx"]-2):bc["idx"]+1]))
                        confirmed = c[i] < micro
                    if confirmed:
                        near_bars, near_proxy = _generic_near(c, i, level)
                        extreme = bc["extreme"]
                        stop = extreme - 0.05*atr[i] if bc["side"] > 0 else extreme + 0.05*atr[i]
                        events.append(V31Event(
                            "H_BOUNCE", i, i+1, bc["side"], float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), near_proxy, near_bars, impulse,
                            confluence_tfs=get_conf(lv, ns), stop_source="reaction_extreme",
                        ))
                        st["bounce_done"] = True
                        st["bounce_candidate"] = None
                elif age > BOUNCE_CONFIRM_BARS:
                    st["bounce_candidate"] = None

            cross_up = c[i] > level and c[i-1] <= level
            cross_dn = c[i] < level and c[i-1] >= level
            if not st["break_done"]:
                side = +1 if lv.kind == "R" and cross_up else -1 if lv.kind == "S" and cross_dn else 0
                if side:
                    near_bars, proto = _breakout_protor(c, i, level, side)
                    stop, stop_source = _breakout_stop(i, side, level, atr, h, l, last_hi, last_lo)
                    break_atr = side * (c[i] - level) / atr[i]
                    events.append(V31Event(
                        "H_BREAK", i, i+1, side, float(stop), lv.level_id, level, lv.kind,
                        lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                        int(st["approach"]), proto, near_bars, impulse,
                        confluence_tfs=get_conf(lv, ns), break_distance_atr=float(break_atr),
                        stop_source=stop_source,
                    ))
                    st["break_done"] = True
                    st["break_idx"] = i
                    st["break_side"] = side
                    st["break_extreme"] = float(h[i] if side > 0 else l[i])
                    continue

                if lv.kind == "S" and l[i] <= level and c[i] >= level:
                    st["bounce_candidate"] = {"idx": i, "side": +1, "extreme": float(l[i])}
                elif lv.kind == "R" and h[i] >= level and c[i] <= level:
                    st["bounce_candidate"] = {"idx": i, "side": -1, "extreme": float(h[i])}
            else:
                bi = st["break_idx"]
                bside = st["break_side"]
                age = i - int(bi)
                if age <= 0:
                    continue
                st["break_extreme"] = (
                    max(float(st["break_extreme"]), float(h[i])) if bside > 0
                    else min(float(st["break_extreme"]), float(l[i]))
                )

                if not st["fakeout_done"] and age <= FAKEOUT_MAX_BARS:
                    reclaim = (bside > 0 and c[i] < level) or (bside < 0 and c[i] > level)
                    if reclaim:
                        near_bars, near_proxy = _generic_near(c, i, level)
                        side = -bside
                        ext = float(st["break_extreme"])
                        stop = ext + 0.05*atr[i] if side < 0 else ext - 0.05*atr[i]
                        events.append(V31Event(
                            "H_FAKEOUT", i, i+1, side, float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), near_proxy, near_bars, impulse, reclaim_bars=age,
                            confluence_tfs=get_conf(lv, ns), stop_source="sweep_extreme",
                        ))
                        st["fakeout_done"] = True

                if not st["retest_done"] and age <= RETEST_MAX_BARS:
                    if bside > 0:
                        touched = l[i] <= level and c[i] > level
                        if touched:
                            stop = min(float(l[i] - 0.05*atr[i]), float(level - 0.10*atr[i]))
                    else:
                        touched = h[i] >= level and c[i] < level
                        if touched:
                            stop = max(float(h[i] + 0.05*atr[i]), float(level + 0.10*atr[i]))
                    if touched:
                        near_bars, near_proxy = _generic_near(c, i, level)
                        events.append(V31Event(
                            "H_RETEST", i, i+1, bside, float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), near_proxy, near_bars, impulse, reclaim_bars=age,
                            confluence_tfs=get_conf(lv, ns), stop_source="retest_extreme",
                        ))
                        st["retest_done"] = True

    progress("scan", len(x5), len(x5))
    return events


def dedup_events(events: list[V31Event]) -> list[V31Event]:
    buckets: dict[tuple, V31Event] = {}
    for e in events:
        key = (e.setup, e.side, e.signal_idx // 3)
        old = buckets.get(key)
        score = (e.confluence_tfs, e.tf_minutes, e.period, -e.touch_error_pct)
        if old is None:
            buckets[key] = e
        else:
            oscore = (old.confluence_tfs, old.tf_minutes, old.period, -old.touch_error_pct)
            if score > oscore:
                buckets[key] = e
    return sorted(buckets.values(), key=lambda z: (z.signal_idx, z.setup, z.side))


def _next_price(nodes: list[tuple[float, int]], side: int, entry: float, level: float) -> float:
    if not nodes:
        return np.nan
    if side > 0:
        floor = max(entry, level * (1.0 + TARGET_SAME_ZONE_PCT))
        j = bisect.bisect_right(nodes, (floor, 10**18))
        return float(nodes[j][0]) if j < len(nodes) else np.nan
    ceil = min(entry, level * (1.0 - TARGET_SAME_ZONE_PCT))
    j = bisect.bisect_left(nodes, (ceil, -1)) - 1
    return float(nodes[j][0]) if j >= 0 else np.nan


def assign_targets(
    events: list[V31Event], levels: list[Level], x5: pd.DataFrame,
    lifecycle_events: list[V31Event] | None = None,
) -> dict[int, dict[str, float]]:
    if not events or not levels:
        return {}
    lifecycle_events = lifecycle_events or events
    level_map = {lv.level_id: lv for lv in levels}
    levs = sorted(levels, key=lambda z: z.formed_time)
    lp = 0
    any_nodes: dict[str, list[tuple[float, int]]] = {"R": [], "S": []}
    thresholds = {m: {"R": [], "S": []} for m in sorted(set(TF_MINUTES.values()))}
    role: dict[int, str] = {}

    def add_level(lv: Level, kind: str) -> None:
        node = (float(lv.price), int(lv.level_id))
        bisect.insort(any_nodes[kind], node)
        for m in thresholds:
            if lv.tf_minutes >= m:
                bisect.insort(thresholds[m][kind], node)
        role[lv.level_id] = kind

    def remove_level(lv: Level, kind: str) -> None:
        node = (float(lv.price), int(lv.level_id))
        arr = any_nodes[kind]
        j = bisect.bisect_left(arr, node)
        if j < len(arr) and arr[j] == node:
            arr.pop(j)
        for m in thresholds:
            if lv.tf_minutes < m:
                continue
            arr = thresholds[m][kind]
            j = bisect.bisect_left(arr, node)
            if j < len(arr) and arr[j] == node:
                arr.pop(j)

    actions = sorted(
        (e for e in lifecycle_events if e.setup in ("H_BREAK", "H_FAKEOUT")),
        key=lambda e: (e.signal_idx, 0 if e.setup == "H_BREAK" else 1),
    )
    ap = 0
    times = pd.to_datetime(x5["signal_time"], utc=True)
    out: dict[int, dict[str, float]] = {}

    for ei, e in sorted(enumerate(events), key=lambda z: z[1].signal_idx):
        ts = pd.Timestamp(times.iloc[e.signal_idx])
        while lp < len(levs) and pd.Timestamp(levs[lp].formed_time) <= ts:
            lv = levs[lp]
            add_level(lv, lv.kind)
            lp += 1

        while ap < len(actions) and actions[ap].signal_idx <= e.signal_idx:
            a = actions[ap]
            lv = level_map.get(a.level_id)
            if lv is not None and lv.level_id in role:
                old = role[lv.level_id]
                remove_level(lv, old)
                new = ("S" if a.side > 0 else "R") if a.setup == "H_BREAK" else lv.kind
                add_level(lv, new)
            ap += 1

        entry = float(x5.iloc[e.entry_idx]["open"]) if e.entry_idx < len(x5) else e.level_price
        target_kind = "R" if e.side > 0 else "S"
        out[ei] = {
            "any": _next_price(any_nodes[target_kind], e.side, entry, e.level_price),
            "htf": _next_price(thresholds[e.tf_minutes][target_kind], e.side, entry, e.level_price),
        }
    return out


def _safe_bool(v) -> bool:
    try:
        if pd.isna(v):
            return False
    except Exception:
        pass
    return bool(v)


def simulate(
    x5: pd.DataFrame, e: V31Event, pair: str, targets: dict[str, float], level_map: dict[int, Level]
) -> dict | None:
    if e.entry_idx >= len(x5):
        return None
    entry = float(x5.iloc[e.entry_idx]["open"])
    stop = float(e.stop)
    if not np.isfinite(entry) or not np.isfinite(stop) or entry <= 0:
        return None
    risk = abs(entry - stop)
    if risk <= 0 or (e.side > 0 and stop >= entry) or (e.side < 0 and stop <= entry):
        return None

    hi = x5["high"].to_numpy(float)
    lo = x5["low"].to_numpy(float)
    cl = x5["close"].to_numpy(float)
    end = min(e.entry_idx + MAX_HOLD_5M, len(x5))
    outcomes = {1: None, 2: None, 3: None}
    alive = {1: True, 2: True, 3: True}
    hits = {1: False, 2: False, 3: False}
    mfe = mae = 0.0
    last = entry
    for i in range(e.entry_idx, end):
        last = float(cl[i])
        fav = (hi[i]-entry) if e.side > 0 else (entry-lo[i])
        adv = (entry-lo[i]) if e.side > 0 else (hi[i]-entry)
        mfe = max(mfe, fav/risk)
        mae = max(mae, adv/risk)
        stop_hit = lo[i] <= stop if e.side > 0 else hi[i] >= stop
        for r in (1, 2, 3):
            if not alive[r]:
                continue
            tp = entry + e.side*r*risk
            target_hit = hi[i] >= tp if e.side > 0 else lo[i] <= tp
            if stop_hit:
                outcomes[r] = -1.0
                alive[r] = False
            elif target_hit:
                outcomes[r] = float(r)
                hits[r] = True
                alive[r] = False
    mtm = e.side*(last-entry)/risk
    for r in (1, 2, 3):
        if outcomes[r] is None:
            outcomes[r] = float(np.clip(mtm, -1.0, float(r)))

    risk_frac = risk/entry
    cost_r = (BASE_COST_BPS/10000.0)/risk_frac
    stress_r = (STRESS_COST_BPS/10000.0)/risk_frac
    target_any = float(targets.get("any", np.nan))
    target_htf = float(targets.get("htf", np.nan))
    rr_any = e.side*(target_any-entry)/risk if np.isfinite(target_any) else np.nan
    rr_htf = e.side*(target_htf-entry)/risk if np.isfinite(target_htf) else np.nan

    signal_time = pd.Timestamp(x5.iloc[e.signal_idx]["signal_time"])
    lv = level_map[e.level_id]
    level_age_h = max(0.0, (signal_time - pd.Timestamp(lv.formed_time)).total_seconds()/3600.0)
    srow = x5.iloc[e.signal_idx]
    fact_proxy = True
    if e.setup == "H_BREAK":
        fact_proxy = bool(e.protor_proxy and (e.impulse_proxy or e.break_distance_atr >= BREAK_FACT_ATR))

    return {
        "pair": pair, "setup": e.setup,
        "signal_time": signal_time,
        "entry_time": pd.Timestamp(x5.iloc[e.entry_idx]["date"]),
        "side": e.side, "entry": entry, "stop": stop, "risk_pct": risk_frac*100.0,
        "stop_source": e.stop_source,
        "level_id": e.level_id, "level_price": e.level_price, "level_kind": e.level_kind,
        "level_age_h": level_age_h,
        "tf": e.tf, "tf_minutes": e.tf_minutes, "period": e.period,
        "touch_error_pct": e.touch_error_pct, "clean_between": e.clean_between,
        "approach_no": e.approach_no, "protor_proxy": e.protor_proxy, "near_bars_6": e.near_bars_6,
        "impulse_proxy": e.impulse_proxy, "break_distance_atr": e.break_distance_atr,
        "fact_proxy": fact_proxy, "reclaim_bars": e.reclaim_bars,
        "confluence_tfs": e.confluence_tfs,
        "target_any": target_any, "target_htf": target_htf,
        "rr_available_any": rr_any, "rr_available_htf": rr_htf,
        "has_rr3_any": bool(np.isfinite(rr_any) and rr_any >= 3.0),
        "has_rr3_htf": bool(np.isfinite(rr_htf) and rr_htf >= 3.0),
        "active_any": _safe_bool(srow.get("active_any", False)),
        "active_strict": _safe_bool(srow.get("active_strict", False)),
        "active_votes": float(srow.get("active_votes", np.nan)),
        "top_growth": _safe_bool(srow.get("top_growth", False)),
        "top_decline": _safe_bool(srow.get("top_decline", False)),
        "top_volatility": _safe_bool(srow.get("top_volatility", False)),
        "top_volume": _safe_bool(srow.get("top_volume", False)),
        "spike_alert": _safe_bool(srow.get("spike_alert", False)),
        "volume_spike": float(srow.get("volume_spike", np.nan)),
        "ret_24h": float(srow.get("ret_24h", np.nan)),
        "natr_local": float(srow.get("natr_local", np.nan)),
        "quote_vol_24h": float(srow.get("quote_vol_24h", np.nan)),
        "mfe_r": float(mfe), "mae_r": float(mae),
        "hit_1r": hits[1], "hit_2r": hits[2], "hit_3r": hits[3],
        "gross_1r": outcomes[1], "gross_2r": outcomes[2], "gross_3r": outcomes[3],
        "net_1r": outcomes[1]-cost_r, "net_2r": outcomes[2]-cost_r, "net_3r": outcomes[3]-cost_r,
        "stress_net_3r": outcomes[3]-stress_r, "cost_r": cost_r,
    }
