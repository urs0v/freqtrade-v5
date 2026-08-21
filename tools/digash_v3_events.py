#!/usr/bin/env python3
from __future__ import annotations

import bisect
import numpy as np
import pandas as pd

from digash_v3_common import *


def detect_events(x5: pd.DataFrame, levels: list[Level]) -> list[Event]:
    if not levels:
        return []
    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    c = x5["close"].to_numpy(float)
    atr = x5["atr"].to_numpy(float)
    times = pd.to_datetime(x5["signal_time"], utc=True)
    time_ns = times.astype("int64").to_numpy()

    conf = confluence_counts(levels)
    formed = sorted(levels, key=lambda z: z.formed_time)
    formed_ns = np.array([int(pd.Timestamp(z.formed_time).value) for z in formed], dtype=np.int64)
    state = {
        lv.level_id: {
            "approach": 0, "inside": False, "break_done": False, "bounce_done": False,
            "retest_done": False, "fakeout_done": False, "break_idx": None, "break_side": 0,
            "break_extreme": np.nan, "bounce_candidate": None,
        } for lv in levels
    }
    by_id = {lv.level_id: lv for lv in levels}
    active_prices: list[float] = []
    active_ids: list[int] = []
    fp = 0
    events: list[Event] = []

    for i in range(1, len(x5)-1):
        if i % 25000 == 0:
            progress("scan", i, len(x5))
        ns = int(time_ns[i])
        while fp < len(formed) and formed_ns[fp] <= ns:
            lv = formed[fp]
            pos = bisect.bisect_right(active_prices, lv.price)
            active_prices.insert(pos, lv.price)
            active_ids.insert(pos, lv.level_id)
            fp += 1
        if not active_prices or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue

        lowq = l[i] / (1.0 + TOUCH_TOL_PCT)
        highq = h[i] / (1.0 - TOUCH_TOL_PCT)
        a0 = bisect.bisect_left(active_prices, lowq)
        a1 = bisect.bisect_right(active_prices, highq)
        if a0 == a1:
            continue

        for k in range(a0, a1):
            lv = by_id[active_ids[k]]
            st = state[lv.level_id]
            level = lv.price
            dist_now = abs(c[i]/level - 1.0)
            near_now = dist_now <= PROTO_NEAR_PCT
            if near_now and not st["inside"]:
                st["approach"] += 1
                st["inside"] = True
            elif not near_now and dist_now > PROTO_NEAR_PCT*1.5:
                st["inside"] = False

            near_bars, proto = protor_features(c, i, level)
            impulse = bool(
                (np.isfinite(x5.iloc[i].get("range_z", np.nan)) and float(x5.iloc[i]["range_z"]) >= 1.0)
                or (np.isfinite(x5.iloc[i].get("volume_spike_local", np.nan)) and float(x5.iloc[i]["volume_spike_local"]) >= 2.0)
            )

            bc = st.get("bounce_candidate")
            if bc is not None and not st["bounce_done"]:
                age = i - bc["idx"]
                if 1 <= age <= BOUNCE_CONFIRM_BARS:
                    if bc["side"] > 0:
                        micro = float(np.max(h[max(0,bc["idx"]-2):bc["idx"]+1]))
                        confirmed = c[i] > micro
                    else:
                        micro = float(np.min(l[max(0,bc["idx"]-2):bc["idx"]+1]))
                        confirmed = c[i] < micro
                    if confirmed:
                        extreme = bc["extreme"]
                        stop = extreme - 0.05*atr[i] if bc["side"] > 0 else extreme + 0.05*atr[i]
                        events.append(Event(
                            "H_BOUNCE", i, i+1, bc["side"], float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), proto, near_bars, impulse, confluence_tfs=conf[lv.level_id]
                        ))
                        st["bounce_done"] = True
                        st["bounce_candidate"] = None
                elif age > BOUNCE_CONFIRM_BARS:
                    st["bounce_candidate"] = None

            cross_up = c[i] > level and c[i-1] <= level
            cross_dn = c[i] < level and c[i-1] >= level
            if not st["break_done"]:
                if lv.kind == "R" and cross_up:
                    side = +1
                elif lv.kind == "S" and cross_dn:
                    side = -1
                else:
                    side = 0
                if side:
                    stop = recent_structure_stop(x5, i, side, level)
                    events.append(Event(
                        "H_BREAK", i, i+1, side, float(stop), lv.level_id, level, lv.kind,
                        lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                        int(st["approach"]), proto, near_bars, impulse, confluence_tfs=conf[lv.level_id]
                    ))
                    st["break_done"] = True
                    st["break_idx"] = i
                    st["break_side"] = side
                    st["break_extreme"] = float(h[i] if side>0 else l[i])
                    continue

                if lv.kind == "S" and l[i] <= level and c[i] >= level:
                    st["bounce_candidate"] = {"idx":i, "side":+1, "extreme":float(l[i])}
                elif lv.kind == "R" and h[i] >= level and c[i] <= level:
                    st["bounce_candidate"] = {"idx":i, "side":-1, "extreme":float(h[i])}
            else:
                bi = st["break_idx"]
                bside = st["break_side"]
                age = i - int(bi)
                if age <= 0:
                    continue
                st["break_extreme"] = (
                    max(float(st["break_extreme"]), float(h[i])) if bside>0
                    else min(float(st["break_extreme"]), float(l[i]))
                )

                if not st["fakeout_done"] and age <= FAKEOUT_MAX_BARS:
                    reclaim = (bside > 0 and c[i] < level) or (bside < 0 and c[i] > level)
                    if reclaim:
                        side = -bside
                        ext = float(st["break_extreme"])
                        stop = ext + 0.05*atr[i] if side < 0 else ext - 0.05*atr[i]
                        events.append(Event(
                            "H_FAKEOUT", i, i+1, side, float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), proto, near_bars, impulse, reclaim_bars=age,
                            confluence_tfs=conf[lv.level_id]
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
                        events.append(Event(
                            "H_RETEST", i, i+1, bside, float(stop), lv.level_id, level, lv.kind,
                            lv.tf, lv.tf_minutes, lv.period, lv.touch_error_pct, lv.clean_between,
                            int(st["approach"]), proto, near_bars, impulse, reclaim_bars=age,
                            confluence_tfs=conf[lv.level_id]
                        ))
                        st["retest_done"] = True

    progress("scan", len(x5), len(x5))
    return events


def dedup_events(events: list[Event]) -> list[Event]:
    buckets: dict[tuple, Event] = {}
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
    return sorted(buckets.values(), key=lambda z:(z.signal_idx,z.setup,z.side))


def assign_targets(events: list[Event], levels: list[Level], x5: pd.DataFrame) -> dict[int,float]:
    if not events or not levels:
        return {}
    levs = sorted(levels, key=lambda l:l.formed_time)
    lp = 0
    prices: list[float] = []
    times = pd.to_datetime(x5["signal_time"], utc=True)
    out = {}
    for ei, e in sorted(enumerate(events), key=lambda z:z[1].signal_idx):
        ts = pd.Timestamp(times.iloc[e.signal_idx])
        while lp < len(levs) and levs[lp].formed_time <= ts:
            bisect.insort(prices, levs[lp].price)
            lp += 1
        if not prices:
            out[ei] = np.nan
            continue
        entry_guess = float(x5.iloc[e.entry_idx]["open"]) if e.entry_idx < len(x5) else e.level_price
        if e.side > 0:
            j = bisect.bisect_right(prices, max(entry_guess, e.level_price*(1+0.001)))
            out[ei] = float(prices[j]) if j < len(prices) else np.nan
        else:
            j = bisect.bisect_left(prices, min(entry_guess, e.level_price*(1-0.001))) - 1
            out[ei] = float(prices[j]) if j >= 0 else np.nan
    return out


def simulate(x5: pd.DataFrame, e: Event, pair: str, target_level: float) -> dict | None:
    if e.entry_idx >= len(x5):
        return None
    entry = float(x5.iloc[e.entry_idx]["open"])
    stop = float(e.stop)
    if not np.isfinite(entry) or not np.isfinite(stop) or entry <= 0:
        return None
    risk = abs(entry-stop)
    if risk <= 0 or (e.side>0 and stop>=entry) or (e.side<0 and stop<=entry):
        return None

    hi = x5["high"].to_numpy(float)
    lo = x5["low"].to_numpy(float)
    cl = x5["close"].to_numpy(float)
    end = min(e.entry_idx + MAX_HOLD_5M, len(x5))
    outcomes = {1:None,2:None,3:None}
    alive = {1:True,2:True,3:True}
    hits = {1:False,2:False,3:False}
    mfe = mae = 0.0
    last = entry
    for i in range(e.entry_idx, end):
        last = float(cl[i])
        fav = (hi[i]-entry) if e.side>0 else (entry-lo[i])
        adv = (entry-lo[i]) if e.side>0 else (hi[i]-entry)
        mfe = max(mfe, fav/risk)
        mae = max(mae, adv/risk)
        sh = lo[i] <= stop if e.side>0 else hi[i] >= stop
        for r in (1,2,3):
            if not alive[r]:
                continue
            tp = entry + e.side*r*risk
            th = hi[i] >= tp if e.side>0 else lo[i] <= tp
            if sh:
                outcomes[r] = -1.0
                alive[r] = False
            elif th:
                outcomes[r] = float(r)
                hits[r] = True
                alive[r] = False
    mtm = e.side*(last-entry)/risk
    for r in (1,2,3):
        if outcomes[r] is None:
            outcomes[r] = float(np.clip(mtm, -1.0, float(r)))

    risk_frac = risk/entry
    cost_r = (BASE_COST_BPS/10000.0)/risk_frac
    stress_r = (STRESS_COST_BPS/10000.0)/risk_frac
    rr_avail = e.side*(target_level-entry)/risk if np.isfinite(target_level) else np.nan
    srow = x5.iloc[e.signal_idx]
    return {
        "pair":pair, "setup":e.setup,
        "signal_time":pd.Timestamp(x5.iloc[e.signal_idx]["signal_time"]),
        "entry_time":pd.Timestamp(x5.iloc[e.entry_idx]["date"]),
        "side":e.side, "entry":entry, "stop":stop, "risk_pct":risk_frac*100.0,
        "level_id":e.level_id, "level_price":e.level_price, "level_kind":e.level_kind,
        "tf":e.tf, "tf_minutes":e.tf_minutes, "period":e.period,
        "touch_error_pct":e.touch_error_pct, "clean_between":e.clean_between,
        "approach_no":e.approach_no, "protor_proxy":e.protor_proxy, "near_bars_6":e.near_bars_6,
        "impulse_proxy":e.impulse_proxy, "reclaim_bars":e.reclaim_bars,
        "confluence_tfs":e.confluence_tfs, "target_level":target_level,
        "rr_available":rr_avail, "has_rr3":bool(np.isfinite(rr_avail) and rr_avail>=3.0),
        "active_any":bool(srow.get("active_any",False)),
        "active_strict":bool(srow.get("active_strict",False)),
        "active_votes":float(srow.get("active_votes",np.nan)),
        "top_growth":bool(srow.get("top_growth",False)),
        "top_decline":bool(srow.get("top_decline",False)),
        "top_volatility":bool(srow.get("top_volatility",False)),
        "spike_alert":bool(srow.get("spike_alert",False)),
        "volume_spike":float(srow.get("volume_spike",np.nan)),
        "ret_24h":float(srow.get("ret_24h",np.nan)),
        "natr_local":float(srow.get("natr_local",np.nan)),
        "quote_vol_24h":float(srow.get("quote_vol_24h",np.nan)),
        "mfe_r":float(mfe), "mae_r":float(mae),
        "hit_1r":hits[1], "hit_2r":hits[2], "hit_3r":hits[3],
        "gross_1r":outcomes[1], "gross_2r":outcomes[2], "gross_3r":outcomes[3],
        "net_1r":outcomes[1]-cost_r, "net_2r":outcomes[2]-cost_r, "net_3r":outcomes[3]-cost_r,
        "stress_net_3r":outcomes[3]-stress_r, "cost_r":cost_r,
    }
