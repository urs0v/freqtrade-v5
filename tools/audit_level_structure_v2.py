#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

BAR5_MIN = 5
BAR15_MIN = 15

# Predeclared rules. These are intentionally not optimized on the result set.
PIVOT_SPAN_15 = 2
PIVOT_SPAN_5 = 2
DEPARTURE_LOOKAHEAD_15 = 4          # only to confirm that a past reaction was meaningful
MIN_DEPARTURE_ATR = 0.80
LEVEL_CLUSTER_ATR = 0.15
LEVEL_INVALIDATE_ATR = 0.45
LEVEL_MIN_TOUCH_SEP_15 = 4          # 1 hour
LEVEL_MAX_IDLE_15 = 96 * 14         # 14 days since last qualifying touch
LEVEL_MIN_TOUCHES = 2
LEVEL_STRONG_TOUCHES = 3

BREAK_ATR = 0.10
SWEEP_MIN_ATR = 0.10
SWEEP_MAX_ATR = 1.25
RETEST_BARS_5 = 6
REACTION_BARS_5 = 3
MAX_HOLD_BARS_5 = 48                # 4 hours
COST_BPS = 8.0
STRESS_COST_BPS = 12.0
MIN_RR = 3.0
TIGHT_STOP_PCT = 0.75

SETUPS = (
    "LEVEL_BREAKOUT",
    "LEVEL_BREAK_RETEST",
    "CONSOLIDATION_BREAKOUT",
    "CONFIRMED_BOUNCE",
    "SWEEP_RECLAIM",
    "STRUCTURE_BREAK_RETEST",
)

def log(msg: str) -> None:
    print(msg, flush=True)

def progress(phase: str, done: int = 0, total: int = 1) -> None:
    print(f"PROGRESS|{phase}|{done}|{total}", flush=True)

def parse_args():
    p = argparse.ArgumentParser(description="Level/Structure Edge V2 causal event study")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--activity-file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    return p.parse_args()

def as_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")

def load_tf(config: dict, datadir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    return load_pair_history(
        pair=pair,
        timeframe=timeframe,
        datadir=datadir,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"),
        candle_type=CandleType.FUTURES,
    )

def prepare_15m(df: pd.DataFrame) -> pd.DataFrame:
    x = df[["date", "open", "high", "low", "close", "volume"]].copy()
    x["date"] = as_ns(x["date"])
    x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    prev = x["close"].shift()
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14, min_periods=14).mean()
    x["signal_time"] = x["date"] + pd.Timedelta(minutes=15)
    return x

def load_5m(config: dict, datadir: Path, pair: str) -> tuple[pd.DataFrame, str]:
    d5 = load_tf(config, datadir, pair, "5m")
    if not d5.empty:
        x = d5[["date", "open", "high", "low", "close", "volume"]].copy()
        x["date"] = as_ns(x["date"])
        x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        return x, "5m"

    d1 = load_tf(config, datadir, pair, "1m")
    if d1.empty:
        return pd.DataFrame(), "none"
    x = d1[["date", "open", "high", "low", "close", "volume"]].copy()
    x["date"] = as_ns(x["date"])
    y = (
        x.set_index("date").sort_index()
        .resample("5min", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna()
        .reset_index()
    )
    return y, "1m->5m"

def robust_z_np(values: np.ndarray, window: int, min_periods: int) -> np.ndarray:
    s = pd.Series(values)
    med = s.rolling(window, min_periods=min_periods).median()
    mad = (s - med).abs().rolling(window, min_periods=min_periods).median()
    z = (s - med) / (1.4826 * mad.replace(0, np.nan))
    return z.clip(-8, 8).to_numpy(float)

def prepare_5m(x5: pd.DataFrame, x15: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    x = x5.copy().sort_values("date").reset_index(drop=True)
    x["signal_time"] = x["date"] + pd.Timedelta(minutes=5)
    prev = x["close"].shift()
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    x["atr5"] = tr.rolling(42, min_periods=21).mean()  # ~3.5h, smoother than 14x5m
    x["quote"] = x["volume"] * x["close"]
    x["range"] = x["high"] - x["low"]
    x["volume_z"] = robust_z_np(x["quote"].to_numpy(float), 288, 72)
    x["range_z"] = robust_z_np(x["range"].to_numpy(float), 288, 72)

    ctx = x15[["signal_time", "atr"]].copy().rename(columns={"signal_time": "ctx_time", "atr": "atr15"})
    x = pd.merge_asof(
        x.sort_values("signal_time"),
        ctx.sort_values("ctx_time"),
        left_on="signal_time",
        right_on="ctx_time",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
    )
    a = activity.copy()
    a["signal_time"] = pd.to_datetime(a["signal_time"], utc=True)
    x = pd.merge_asof(
        x.sort_values("signal_time"),
        a.sort_values("signal_time"),
        on="signal_time",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
    )
    x["atr"] = x["atr15"].fillna(x["atr5"])
    return x.reset_index(drop=True)

def _first_departure(high: np.ndarray, low: np.ndarray, atr: np.ndarray, i: int, kind: str):
    if not np.isfinite(atr[i]) or atr[i] <= 0:
        return None
    for j in range(1, DEPARTURE_LOOKAHEAD_15 + 1):
        k = i + j
        if k >= len(high):
            return None
        if kind == "R":
            dep = (high[i] - low[k]) / atr[i]
        else:
            dep = (high[k] - low[i]) / atr[i]
        if np.isfinite(dep) and dep >= MIN_DEPARTURE_ATR:
            return j, float(dep)
    return None

def pivot_candidates_15(x15: pd.DataFrame) -> list[dict]:
    h = x15["high"].to_numpy(float)
    l = x15["low"].to_numpy(float)
    atr = x15["atr"].to_numpy(float)
    n = len(x15)
    s = PIVOT_SPAN_15
    out = []
    for i in range(s, n - max(s, DEPARTURE_LOOKAHEAD_15) - 1):
        if not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        is_hi = h[i] >= np.max(h[i-s:i+s+1])
        is_lo = l[i] <= np.min(l[i-s:i+s+1])
        if is_hi:
            dep = _first_departure(h, l, atr, i, "R")
            if dep is not None:
                j, d = dep
                avail_idx = max(i + s, i + j)
                out.append({"kind": "R", "price": float(h[i]), "pivot_idx": i,
                            "avail_idx": avail_idx, "atr": float(atr[i]), "departure": d})
        if is_lo:
            dep = _first_departure(h, l, atr, i, "S")
            if dep is not None:
                j, d = dep
                avail_idx = max(i + s, i + j)
                out.append({"kind": "S", "price": float(l[i]), "pivot_idx": i,
                            "avail_idx": avail_idx, "atr": float(atr[i]), "departure": d})
    return sorted(out, key=lambda z: (z["avail_idx"], z["pivot_idx"]))

def build_level_versions(x15: pd.DataFrame) -> pd.DataFrame:
    """
    Causal level zones.
    A pivot can join a zone only after:
      1) the right-side pivot bars have closed, and
      2) a >=0.8 ATR departure has actually occurred.
    Every later touch creates a new version; earlier versions never use future centers/touch counts.
    """
    cands = pivot_candidates_15(x15)
    if not cands:
        return pd.DataFrame()

    close = x15["close"].to_numpy(float)
    atr = x15["atr"].to_numpy(float)
    sig = pd.to_datetime(x15["signal_time"], utc=True).tolist()
    by_avail: dict[int, list[dict]] = {}
    for c in cands:
        by_avail.setdefault(c["avail_idx"], []).append(c)

    zones: list[dict] = []
    active: set[int] = set()
    versions: list[dict] = []
    open_version: dict[int, int] = {}

    def close_version(zid: int, end_idx: int):
        vi = open_version.pop(zid, None)
        if vi is not None:
            versions[vi]["valid_to_idx"] = end_idx
            versions[vi]["valid_to"] = sig[end_idx] if end_idx < len(sig) else pd.Timestamp.max.tz_localize("UTC")

    def open_strong_version(zid: int, start_idx: int):
        z = zones[zid]
        if z["touches"] < LEVEL_MIN_TOUCHES:
            return
        if z["last_touch_idx"] - z["first_touch_idx"] < LEVEL_MIN_TOUCH_SEP_15:
            return
        row = {
            "zone_id": zid,
            "kind": z["kind"],
            "center": z["center"],
            "halfwidth": max(LEVEL_CLUSTER_ATR * z["last_atr"], z["center"] * 0.00025),
            "touches": z["touches"],
            "avg_departure": z["dep_sum"] / z["touches"],
            "first_touch_idx": z["first_touch_idx"],
            "last_touch_idx": z["last_touch_idx"],
            "valid_from_idx": start_idx,
            "valid_to_idx": len(x15),
            "valid_from": sig[start_idx],
            "valid_to": pd.Timestamp.max.tz_localize("UTC"),
        }
        versions.append(row)
        open_version[zid] = len(versions) - 1

    for bi in range(len(x15)):
        ai = float(atr[bi]) if np.isfinite(atr[bi]) else np.nan
        ci = float(close[bi])

        # Invalidate/expire before adding newly confirmed historical reactions at this close.
        dead = []
        for zid in tuple(active):
            z = zones[zid]
            expired = bi - z["last_touch_idx"] > LEVEL_MAX_IDLE_15
            invalid = False
            if np.isfinite(ai) and ai > 0:
                if z["kind"] == "R":
                    invalid = ci > z["center"] + LEVEL_INVALIDATE_ATR * ai
                else:
                    invalid = ci < z["center"] - LEVEL_INVALIDATE_ATR * ai
            if expired or invalid:
                close_version(zid, bi)
                z["active"] = False
                dead.append(zid)
        for zid in dead:
            active.discard(zid)

        for cand in by_avail.get(bi, []):
            best = None
            best_dist = float("inf")
            for zid in active:
                z = zones[zid]
                if z["kind"] != cand["kind"]:
                    continue
                if cand["pivot_idx"] - z["last_touch_idx"] < LEVEL_MIN_TOUCH_SEP_15:
                    continue
                tol = max(LEVEL_CLUSTER_ATR * cand["atr"], z["center"] * 0.00025)
                dist = abs(cand["price"] - z["center"])
                if dist <= tol and dist < best_dist:
                    best, best_dist = zid, dist

            if best is None:
                zid = len(zones)
                zones.append({
                    "kind": cand["kind"],
                    "center": cand["price"],
                    "touches": 1,
                    "dep_sum": cand["departure"],
                    "first_touch_idx": cand["pivot_idx"],
                    "last_touch_idx": cand["pivot_idx"],
                    "last_atr": cand["atr"],
                    "active": True,
                })
                active.add(zid)
                continue

            z = zones[best]
            # Close old causal snapshot before changing its center/strength.
            close_version(best, bi)
            w = z["touches"]
            z["center"] = (z["center"] * w + cand["price"]) / (w + 1)
            z["touches"] += 1
            z["dep_sum"] += cand["departure"]
            z["last_touch_idx"] = cand["pivot_idx"]
            z["last_atr"] = cand["atr"]
            open_strong_version(best, bi)

    for zid in tuple(open_version):
        close_version(zid, len(x15) - 1)

    return pd.DataFrame(versions)

@dataclass
class Event:
    setup: str
    signal_idx: int
    side: int
    entry_idx: int
    stop: float
    level_price: float | None = None
    level_kind: str | None = None
    level_touches: int = 0
    level_departure: float = np.nan
    interaction_no: int = 0
    compression_score: float = np.nan
    reclaim_bars: int = 0
    sweep_depth_atr: float = np.nan
    structure_subtype: str = ""
    box_bars: int = 0

class TargetFinder:
    def __init__(self, versions: pd.DataFrame, x15: pd.DataFrame):
        if versions.empty:
            self.empty = True
            return
        self.empty = False
        self.kind = versions["kind"].to_numpy(str)
        self.center = versions["center"].to_numpy(float)
        self.vf = versions["valid_from_idx"].to_numpy(int)
        self.vt = versions["valid_to_idx"].to_numpy(int)
        self.x15_times = pd.to_datetime(x15["signal_time"], utc=True).astype("int64").to_numpy()

    def bar15_index(self, ts: pd.Timestamp) -> int:
        ns = int(pd.Timestamp(ts).value)
        return int(np.searchsorted(self.x15_times, ns, side="right") - 1)

    def nearest(self, ts: pd.Timestamp, entry: float, side: int) -> float:
        if self.empty:
            return np.nan
        bi = self.bar15_index(ts)
        if bi < 0:
            return np.nan
        active = (self.vf <= bi) & (self.vt > bi)
        if side > 0:
            m = active & (self.kind == "R") & (self.center > entry)
            if not np.any(m):
                return np.nan
            return float(np.min(self.center[m]))
        m = active & (self.kind == "S") & (self.center < entry)
        if not np.any(m):
            return np.nan
        return float(np.max(self.center[m]))

def _compression_score(x5: pd.DataFrame, i: int, level: float, side: int) -> float:
    # side is breakout direction: +1 through resistance, -1 through support
    if i < 12:
        return 0.0
    atr = float(x5.iloc[i]["atr"])
    if not np.isfinite(atr) or atr <= 0:
        return 0.0
    recent = x5.iloc[i-6:i]
    prior = x5.iloc[i-12:i-6]
    score = 0.0
    rr = float((recent["high"] - recent["low"]).mean())
    pr = float((prior["high"] - prior["low"]).mean())
    if np.isfinite(rr) and np.isfinite(pr) and pr > 0 and rr <= 0.85 * pr:
        score += 1.0
    closes = recent["close"].to_numpy(float)
    dist = (level - closes) * side
    if np.isfinite(dist).all() and dist[-1] < dist[0] - 0.10 * atr:
        score += 1.0
    lows = recent["low"].to_numpy(float)
    highs = recent["high"].to_numpy(float)
    if side > 0 and np.polyfit(np.arange(len(lows)), lows, 1)[0] > 0.02 * atr:
        score += 1.0
    if side < 0 and np.polyfit(np.arange(len(highs)), highs, 1)[0] < -0.02 * atr:
        score += 1.0
    return score

def _event_stop_breakout(x5: pd.DataFrame, i: int, level: float, side: int) -> float:
    atr = float(x5.iloc[i]["atr"])
    lo = float(x5.iloc[max(0, i-5):i+1]["low"].min())
    hi = float(x5.iloc[max(0, i-5):i+1]["high"].max())
    if side > 0:
        return max(lo, level - 0.35 * atr)
    return min(hi, level + 0.35 * atr)

def level_events(x5: pd.DataFrame, versions: pd.DataFrame) -> list[Event]:
    if versions.empty:
        return []
    times_ns = pd.to_datetime(x5["signal_time"], utc=True).astype("int64").to_numpy()
    o = x5["open"].to_numpy(float)
    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    c = x5["close"].to_numpy(float)
    atr = x5["atr"].to_numpy(float)

    events: list[Event] = []
    interaction_count: dict[int, int] = {}

    grouped = versions.sort_values(["zone_id", "valid_from_idx"]).groupby("zone_id", sort=False)
    total = len(grouped)
    for gi, (zid, g) in enumerate(grouped, 1):
        if gi % 100 == 0 or gi == total:
            progress("levels", gi, total)
        interactions = interaction_count.get(int(zid), 0)
        cooldown_until = -1
        for v in g.itertuples(index=False):
            start_ns = int(pd.Timestamp(v.valid_from).value)
            end_ns = int(pd.Timestamp(v.valid_to).value)
            si = int(np.searchsorted(times_ns, start_ns, side="left"))
            ei = int(np.searchsorted(times_ns, end_ns, side="left"))
            if si >= len(x5):
                continue
            ei = min(ei, len(x5) - 1)
            level = float(v.center)
            width = float(v.halfwidth)
            kind = str(v.kind)

            i = max(si, cooldown_until)
            while i < ei - 2:
                if not np.isfinite(atr[i]) or atr[i] <= 0:
                    i += 1
                    continue
                a = atr[i]
                prevc = c[i-1] if i > 0 else c[i]
                made = False

                if kind == "R":
                    # Fakeout/sweep first: price trades through resistance, then reclaims below.
                    depth = (h[i] - level) / a
                    if SWEEP_MIN_ATR <= depth <= SWEEP_MAX_ATR and c[i] < level:
                        interactions += 1
                        stop = h[i] + 0.05 * a
                        events.append(Event("SWEEP_RECLAIM", i, -1, i+1, stop, level, kind,
                                            int(v.touches), float(v.avg_departure), interactions,
                                            reclaim_bars=0, sweep_depth_atr=float(depth)))
                        made = True
                    # Confirmed level breakout.
                    elif c[i] > level + BREAK_ATR * a and prevc <= level + width:
                        interactions += 1
                        comp = _compression_score(x5, i, level, +1)
                        stop = _event_stop_breakout(x5, i, level, +1)
                        events.append(Event("LEVEL_BREAKOUT", i, +1, i+1, stop, level, kind,
                                            int(v.touches), float(v.avg_departure), interactions,
                                            compression_score=comp))
                        # First retest variant of the same confirmed break.
                        for k in range(i+1, min(i+1+RETEST_BARS_5, ei-1)):
                            if l[k] <= level + width and c[k] > level:
                                rs = min(l[k] - 0.05 * atr[k], level - 0.20 * atr[k])
                                events.append(Event("LEVEL_BREAK_RETEST", k, +1, k+1, rs, level, kind,
                                                    int(v.touches), float(v.avg_departure), interactions,
                                                    compression_score=comp, reclaim_bars=k-i))
                                break
                            if c[k] < level - 0.5 * atr[k]:
                                break
                        made = True
                    # Confirmed defense: short only after the local micro support breaks.
                    elif h[i] >= level - width and c[i] <= level:
                        touch_hi = h[i]
                        for k in range(i+1, min(i+1+REACTION_BARS_5, ei-1)):
                            micro = np.min(l[max(si, i-2):i+1])
                            if c[k] < micro:
                                interactions += 1
                                stop = max(touch_hi + 0.05 * atr[k], level + 0.15 * atr[k])
                                events.append(Event("CONFIRMED_BOUNCE", k, -1, k+1, stop, level, kind,
                                                    int(v.touches), float(v.avg_departure), interactions,
                                                    reclaim_bars=k-i))
                                made = True
                                break
                        if not made and h[i] > level + SWEEP_MIN_ATR*a:
                            for k in range(i+1, min(i+3, ei-1)):
                                if c[k] < level:
                                    depth2 = (max(h[i:k+1]) - level) / a
                                    if depth2 <= SWEEP_MAX_ATR:
                                        interactions += 1
                                        stop = max(h[i:k+1]) + 0.05*atr[k]
                                        events.append(Event("SWEEP_RECLAIM", k, -1, k+1, stop, level, kind,
                                                            int(v.touches), float(v.avg_departure), interactions,
                                                            reclaim_bars=k-i, sweep_depth_atr=float(depth2)))
                                        made = True
                                        break
                else:
                    depth = (level - l[i]) / a
                    if SWEEP_MIN_ATR <= depth <= SWEEP_MAX_ATR and c[i] > level:
                        interactions += 1
                        stop = l[i] - 0.05 * a
                        events.append(Event("SWEEP_RECLAIM", i, +1, i+1, stop, level, kind,
                                            int(v.touches), float(v.avg_departure), interactions,
                                            reclaim_bars=0, sweep_depth_atr=float(depth)))
                        made = True
                    elif c[i] < level - BREAK_ATR * a and prevc >= level - width:
                        interactions += 1
                        comp = _compression_score(x5, i, level, -1)
                        stop = _event_stop_breakout(x5, i, level, -1)
                        events.append(Event("LEVEL_BREAKOUT", i, -1, i+1, stop, level, kind,
                                            int(v.touches), float(v.avg_departure), interactions,
                                            compression_score=comp))
                        for k in range(i+1, min(i+1+RETEST_BARS_5, ei-1)):
                            if h[k] >= level - width and c[k] < level:
                                rs = max(h[k] + 0.05 * atr[k], level + 0.20 * atr[k])
                                events.append(Event("LEVEL_BREAK_RETEST", k, -1, k+1, rs, level, kind,
                                                    int(v.touches), float(v.avg_departure), interactions,
                                                    compression_score=comp, reclaim_bars=k-i))
                                break
                            if c[k] > level + 0.5 * atr[k]:
                                break
                        made = True
                    elif l[i] <= level + width and c[i] >= level:
                        touch_lo = l[i]
                        for k in range(i+1, min(i+1+REACTION_BARS_5, ei-1)):
                            micro = np.max(h[max(si, i-2):i+1])
                            if c[k] > micro:
                                interactions += 1
                                stop = min(touch_lo - 0.05 * atr[k], level - 0.15 * atr[k])
                                events.append(Event("CONFIRMED_BOUNCE", k, +1, k+1, stop, level, kind,
                                                    int(v.touches), float(v.avg_departure), interactions,
                                                    reclaim_bars=k-i))
                                made = True
                                break
                        if not made and l[i] < level - SWEEP_MIN_ATR*a:
                            for k in range(i+1, min(i+3, ei-1)):
                                if c[k] > level:
                                    depth2 = (level - min(l[i:k+1])) / a
                                    if depth2 <= SWEEP_MAX_ATR:
                                        interactions += 1
                                        stop = min(l[i:k+1]) - 0.05*atr[k]
                                        events.append(Event("SWEEP_RECLAIM", k, +1, k+1, stop, level, kind,
                                                            int(v.touches), float(v.avg_departure), interactions,
                                                            reclaim_bars=k-i, sweep_depth_atr=float(depth2)))
                                        made = True
                                        break
                if made:
                    cooldown_until = i + 6
                    i += 6
                else:
                    i += 1
        interaction_count[int(zid)] = interactions
    return events

def _touch_clusters(vals: np.ndarray) -> int:
    if len(vals) == 0:
        return 0
    idx = np.flatnonzero(vals)
    if len(idx) == 0:
        return 0
    return int(1 + np.sum(np.diff(idx) > 2))

def consolidation_events(x5: pd.DataFrame) -> list[Event]:
    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    c = x5["close"].to_numpy(float)
    atr = x5["atr"].to_numpy(float)
    n = len(x5)
    look = 24  # 2 hours
    # Candidate breakouts only; inspect the preceding box in numpy.
    roll_hi = pd.Series(h).shift(1).rolling(look, min_periods=look).max().to_numpy(float)
    roll_lo = pd.Series(l).shift(1).rolling(look, min_periods=look).min().to_numpy(float)
    events = []
    last_event = -100
    candidates = np.flatnonzero(
        ((c > roll_hi + BREAK_ATR * atr) | (c < roll_lo - BREAK_ATR * atr))
        & np.isfinite(roll_hi) & np.isfinite(roll_lo) & np.isfinite(atr)
    )
    total = len(candidates)
    for ci, i in enumerate(candidates, 1):
        if ci % 1000 == 0 or ci == total:
            progress("consolidations", ci, max(total, 1))
        if i <= look or i - last_event < 12 or i + 1 >= n:
            continue
        a = atr[i]
        if not np.isfinite(a) or a <= 0:
            continue
        bh, bl = roll_hi[i], roll_lo[i]
        width = bh - bl
        if not (0.8 * a <= width <= 3.0 * a):
            continue
        wh = h[i-look:i]
        wl = l[i-look:i]
        top = wh >= bh - 0.15*a
        bot = wl <= bl + 0.15*a
        if _touch_clusters(top) < 2 or _touch_clusters(bot) < 2:
            continue
        # A real box should spend most closes inside its range.
        inside = np.mean((c[i-look:i] >= bl) & (c[i-look:i] <= bh))
        if inside < 0.85:
            continue
        if c[i] > bh + BREAK_ATR*a:
            side = +1
            stop = max(bl, np.min(l[i-5:i+1]))
            level = bh
        elif c[i] < bl - BREAK_ATR*a:
            side = -1
            stop = min(bh, np.max(h[i-5:i+1]))
            level = bl
        else:
            continue
        if side > 0 and stop >= x5.iloc[i+1].open:
            continue
        if side < 0 and stop <= x5.iloc[i+1].open:
            continue
        events.append(Event(
            "CONSOLIDATION_BREAKOUT", i, side, i+1, float(stop),
            level_price=float(level), level_kind="BOX", box_bars=look,
            compression_score=_compression_score(x5, i, level, side),
        ))
        last_event = i
    return events

def micro_pivots_5(x5: pd.DataFrame):
    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    s = PIVOT_SPAN_5
    piv = []
    for i in range(s, len(x5)-s):
        if h[i] >= np.max(h[i-s:i+s+1]):
            piv.append((i+s, i, "H", float(h[i])))
        if l[i] <= np.min(l[i-s:i+s+1]):
            piv.append((i+s, i, "L", float(l[i])))
    return sorted(piv)

def structure_break_events(x5: pd.DataFrame) -> list[Event]:
    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    c = x5["close"].to_numpy(float)
    atr = x5["atr"].to_numpy(float)
    piv = micro_pivots_5(x5)
    by_avail: dict[int, list[tuple]] = {}
    for p in piv:
        by_avail.setdefault(p[0], []).append(p)
    highs: list[tuple[int,float]] = []
    lows: list[tuple[int,float]] = []
    events = []
    last_break_level_h = np.nan
    last_break_level_l = np.nan
    last_event = -100
    total = len(x5)
    for i in range(total-RETEST_BARS_5-2):
        if i % 10000 == 0:
            progress("structure", i, total)
        for _, pi, kind, price in by_avail.get(i, []):
            if kind == "H":
                highs.append((pi, price))
                highs = highs[-4:]
            else:
                lows.append((pi, price))
                lows = lows[-4:]
        if i < 2 or not np.isfinite(atr[i]) or atr[i] <= 0 or i-last_event < 6:
            continue

        uptrend = len(highs) >= 2 and len(lows) >= 2 and highs[-1][1] > highs[-2][1] and lows[-1][1] > lows[-2][1]
        downtrend = len(highs) >= 2 and len(lows) >= 2 and highs[-1][1] < highs[-2][1] and lows[-1][1] < lows[-2][1]

        if highs:
            level = highs[-1][1]
            if c[i] > level + 0.05*atr[i] and c[i-1] <= level and (not np.isfinite(last_break_level_h) or abs(level-last_break_level_h) > 0.1*atr[i]):
                subtype = "REVERSAL" if downtrend else ("CONTINUATION" if uptrend else "NEUTRAL")
                for k in range(i+1, min(i+1+RETEST_BARS_5, total-1)):
                    if l[k] <= level + 0.15*atr[k] and c[k] > level:
                        stop = min(l[k] - 0.05*atr[k], level - 0.20*atr[k])
                        events.append(Event("STRUCTURE_BREAK_RETEST", k, +1, k+1, float(stop),
                                            level_price=float(level), level_kind="STRUCT_H",
                                            reclaim_bars=k-i, structure_subtype=subtype))
                        last_event = k
                        break
                    if c[k] < level - 0.5*atr[k]:
                        break
                last_break_level_h = level

        if lows:
            level = lows[-1][1]
            if c[i] < level - 0.05*atr[i] and c[i-1] >= level and (not np.isfinite(last_break_level_l) or abs(level-last_break_level_l) > 0.1*atr[i]):
                subtype = "REVERSAL" if uptrend else ("CONTINUATION" if downtrend else "NEUTRAL")
                for k in range(i+1, min(i+1+RETEST_BARS_5, total-1)):
                    if h[k] >= level - 0.15*atr[k] and c[k] < level:
                        stop = max(h[k] + 0.05*atr[k], level + 0.20*atr[k])
                        events.append(Event("STRUCTURE_BREAK_RETEST", k, -1, k+1, float(stop),
                                            level_price=float(level), level_kind="STRUCT_L",
                                            reclaim_bars=k-i, structure_subtype=subtype))
                        last_event = k
                        break
                    if c[k] > level + 0.5*atr[k]:
                        break
                last_break_level_l = level
    progress("structure", total, total)
    return events

def dedup_events(events: list[Event]) -> list[Event]:
    # One event of the same family+direction per 15 minutes; variants remain separate.
    events = sorted(events, key=lambda e: (e.signal_idx, e.setup, e.side))
    out = []
    last: dict[tuple[str,int], int] = {}
    for e in events:
        key = (e.setup, e.side)
        if e.entry_idx <= e.signal_idx:
            continue
        prev = last.get(key, -1000)
        if e.signal_idx - prev < 3:
            continue
        out.append(e)
        last[key] = e.signal_idx
    return out

def simulate_path(x5: pd.DataFrame, e: Event, target_finder: TargetFinder, pair: str) -> dict | None:
    if e.entry_idx >= len(x5):
        return None
    entry = float(x5.iloc[e.entry_idx]["open"])
    if not np.isfinite(entry) or entry <= 0 or not np.isfinite(e.stop):
        return None
    risk = abs(entry - e.stop)
    if risk <= 0:
        return None
    if e.side > 0 and e.stop >= entry:
        return None
    if e.side < 0 and e.stop <= entry:
        return None

    signal_time = pd.Timestamp(x5.iloc[e.signal_idx]["signal_time"])
    target = target_finder.nearest(signal_time, entry, e.side)
    rr_avail = e.side * (target - entry) / risk if np.isfinite(target) else np.nan
    risk_pct = risk / entry * 100.0

    end = min(e.entry_idx + MAX_HOLD_BARS_5, len(x5))
    hi = x5["high"].to_numpy(float)
    lo = x5["low"].to_numpy(float)
    close = x5["close"].to_numpy(float)

    mfe = 0.0
    mae = 0.0
    hits = {1: False, 2: False, 3: False}
    outcomes = {1: None, 2: None, 3: None}
    alive = {1: True, 2: True, 3: True}
    last_close = entry

    for i in range(e.entry_idx, end):
        last_close = close[i]
        fav = (hi[i] - entry) if e.side > 0 else (entry - lo[i])
        adv = (entry - lo[i]) if e.side > 0 else (hi[i] - entry)
        mfe = max(mfe, fav / risk)
        mae = max(mae, adv / risk)

        stop_hit = lo[i] <= e.stop if e.side > 0 else hi[i] >= e.stop
        for r in (1, 2, 3):
            if not alive[r]:
                continue
            tp = entry + e.side * r * risk
            tp_hit = hi[i] >= tp if e.side > 0 else lo[i] <= tp
            # Conservative ambiguity: stop first when both are inside one 5m candle.
            if stop_hit:
                outcomes[r] = -1.0
                alive[r] = False
            elif tp_hit:
                hits[r] = True
                outcomes[r] = float(r)
                alive[r] = False

    mtm_r = e.side * (last_close - entry) / risk
    for r in (1, 2, 3):
        if outcomes[r] is None:
            outcomes[r] = float(np.clip(mtm_r, -1.0, float(r)))

    cost_r = (COST_BPS / 10000.0) / (risk / entry)
    stress_cost_r = (STRESS_COST_BPS / 10000.0) / (risk / entry)

    row = {
        "pair": pair,
        "setup": e.setup,
        "signal_time": signal_time,
        "entry_time": pd.Timestamp(x5.iloc[e.entry_idx]["date"]),
        "side": e.side,
        "entry": entry,
        "stop": e.stop,
        "risk_pct": risk_pct,
        "target_level": target,
        "rr_available": rr_avail,
        "has_rr3": bool(np.isfinite(rr_avail) and rr_avail >= MIN_RR),
        "tight_stop": bool(risk_pct <= TIGHT_STOP_PCT),
        "level_price": e.level_price,
        "level_kind": e.level_kind,
        "level_touches": e.level_touches,
        "level_departure": e.level_departure,
        "interaction_no": e.interaction_no,
        "compression_score": e.compression_score,
        "reclaim_bars": e.reclaim_bars,
        "sweep_depth_atr": e.sweep_depth_atr,
        "structure_subtype": e.structure_subtype,
        "box_bars": e.box_bars,
        "activity_rank": float(x5.iloc[e.signal_idx].get("activity_rank", np.nan)),
        "active_top5": bool(x5.iloc[e.signal_idx].get("active_top5", False)),
        "active_top10": bool(x5.iloc[e.signal_idx].get("active_top10", False)),
        "activity_score": float(x5.iloc[e.signal_idx].get("activity_score", np.nan)),
        "ret_4h": float(x5.iloc[e.signal_idx].get("ret_4h", np.nan)),
        "ret_24h": float(x5.iloc[e.signal_idx].get("ret_24h", np.nan)),
        "volume_rank": float(x5.iloc[e.signal_idx].get("volume_rank", np.nan)),
        "volume_z": float(x5.iloc[e.signal_idx]["volume_z"]),
        "range_z": float(x5.iloc[e.signal_idx]["range_z"]),
        "mfe_r": float(mfe),
        "mae_r": float(mae),
        "hit_1r": hits[1],
        "hit_2r": hits[2],
        "hit_3r": hits[3],
        "gross_1r": outcomes[1],
        "gross_2r": outcomes[2],
        "gross_3r": outcomes[3],
        "net_1r": outcomes[1] - cost_r,
        "net_2r": outcomes[2] - cost_r,
        "net_3r": outcomes[3] - cost_r,
        "stress_net_3r": outcomes[3] - stress_cost_r,
        "cost_r": cost_r,
    }
    return row

def summarize_pair(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    variants = {
        "ALL": np.ones(len(df), dtype=bool),
        "TOP10": df["active_top10"].to_numpy(bool),
        "TOP5": df["active_top5"].to_numpy(bool),
        "RR3": df["has_rr3"].to_numpy(bool),
        "TOP10_RR3": (df["active_top10"] & df["has_rr3"]).to_numpy(bool),
        "TOP5_RR3": (df["active_top5"] & df["has_rr3"]).to_numpy(bool),
        "TOP10_RR3_TIGHT": (df["active_top10"] & df["has_rr3"] & df["tight_stop"]).to_numpy(bool),
    }
    rows = []
    for setup in SETUPS:
        base = df["setup"].eq(setup).to_numpy()
        for name, mask in variants.items():
            z = df.loc[base & mask]
            if z.empty:
                continue
            pos = z["net_3r"].clip(lower=0).sum()
            neg = -z["net_3r"].clip(upper=0).sum()
            rows.append({
                "setup": setup,
                "variant": name,
                "n": len(z),
                "hit1_pct": z["hit_1r"].mean()*100,
                "hit2_pct": z["hit_2r"].mean()*100,
                "hit3_pct": z["hit_3r"].mean()*100,
                "mean_net3_r": z["net_3r"].mean(),
                "median_net3_r": z["net_3r"].median(),
                "pf3_r": (pos / neg) if neg > 0 else math.inf,
                "mean_mfe_r": z["mfe_r"].mean(),
                "mean_mae_r": z["mae_r"].mean(),
                "mean_rr_available": z["rr_available"].mean(),
            })
    return pd.DataFrame(rows)

def main() -> int:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(args.config).read_text())
    pairs = list(cfg.get("exchange", {}).get("pair_whitelist", []))
    if len(pairs) != 1:
        raise RuntimeError("Worker config must contain exactly one pair")
    pair = pairs[0]
    datadir = Path(args.datadir)
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)

    t0 = time.monotonic()
    log(f"=== LEVEL/STRUCTURE EDGE V2 | {pair} ===")
    progress("load", 0, 1)
    d15 = load_tf(cfg, datadir, pair, "15m")
    x5, source = load_5m(cfg, datadir, pair)
    if d15.empty or x5.empty:
        raise RuntimeError(f"Missing 15m/detail data for {pair}")
    x15 = prepare_15m(d15)
    activity = pd.read_pickle(args.activity_file)
    progress("load", 1, 1)

    warm = pd.Timedelta(days=35)
    x15 = x15[(x15["date"] >= start-warm) & (x15["date"] < end + pd.Timedelta(hours=4))].reset_index(drop=True)
    x5 = x5[(x5["date"] >= start-warm) & (x5["date"] < end + pd.Timedelta(hours=4))].reset_index(drop=True)
    x5 = prepare_5m(x5, x15, activity)
    x5 = x5.sort_values("date").reset_index(drop=True)

    progress("zones", 0, 1)
    versions = build_level_versions(x15)
    versions.to_csv(outdir / "level_versions.csv", index=False)
    progress("zones", 1, 1)

    tf = TargetFinder(versions, x15)

    events = []
    events.extend(level_events(x5, versions))
    progress("consolidations", 0, 1)
    events.extend(consolidation_events(x5))
    events.extend(structure_break_events(x5))
    events = dedup_events(events)
    progress("simulate", 0, max(len(events), 1))

    rows = []
    for j, e in enumerate(events, 1):
        if j % 1000 == 0 or j == len(events):
            progress("simulate", j, max(len(events), 1))
        row = simulate_path(x5, e, tf, pair)
        if row is None:
            continue
        et = pd.Timestamp(row["entry_time"])
        if start <= et < end:
            rows.append(row)

    trades = pd.DataFrame(rows)
    if trades.empty:
        raise RuntimeError(f"No V2 events found for {pair}")
    trades.to_csv(outdir / "events.csv", index=False)
    summarize_pair(trades).to_csv(outdir / "pair_summary.csv", index=False)
    pd.DataFrame([{
        "pair": pair,
        "detail_source": source,
        "bars15": len(x15),
        "bars5": len(x5),
        "level_versions": len(versions),
        "events": len(trades),
        "elapsed_s": time.monotonic() - t0,
    }]).to_csv(outdir / "coverage.csv", index=False)

    log(f"DONE|{pair}|events={len(trades)}|levels={len(versions)}|elapsed={time.monotonic()-t0:.1f}s")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
