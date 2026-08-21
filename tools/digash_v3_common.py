#!/usr/bin/env python3
from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

PIVOT_RIGHT = 2                  # mechanical implementation detail, not a claimed Digash parameter
TOUCH_TOL_PCT = 0.01            # frozen replication uncertainty: screener setting "1", corroborated as 1%
PERIODS = (20, 30)              # public walkthrough: local/global profiles
TFS = ("5m", "15m", "1h", "4h")
TF_MINUTES = {"5m": 5, "15m": 15, "1h": 60, "4h": 240}
MAX_HOLD_5M = 48                # 4h diagnostic horizon
BASE_COST_BPS = 8.0
STRESS_COST_BPS = 12.0
RETEST_MAX_BARS = 12            # mechanical proxy (1h on 5m)
FAKEOUT_MAX_BARS = 2
BOUNCE_CONFIRM_BARS = 3
PROTO_NEAR_PCT = 0.005          # explicit proxy, not claimed as Digash exact rule
PROTO_BARS = 6
PROTO_MIN_NEAR = 3

SETUPS = ("H_BREAK", "H_RETEST", "H_BOUNCE", "H_FAKEOUT")


def parse_args():
    p = argparse.ArgumentParser(description="Causal Digash horizontal-level replication V3")
    p.add_argument("--config", required=True)
    p.add_argument("--datadir", required=True)
    p.add_argument("--activity-file", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    return p.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def progress(phase: str, done: int, total: int) -> None:
    print(f"PROGRESS|{phase}|{done}|{max(total,1)}", flush=True)


def as_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def load_tf(config: dict, datadir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    return load_pair_history(
        pair=pair, timeframe=timeframe, datadir=datadir,
        fill_up_missing=False, drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"), candle_type=CandleType.FUTURES,
    )


def load_5m(config: dict, datadir: Path, pair: str) -> tuple[pd.DataFrame, str]:
    d5 = load_tf(config, datadir, pair, "5m")
    if not d5.empty:
        x = d5[["date", "open", "high", "low", "close", "volume"]].copy()
        x["date"] = as_ns(x["date"])
        return x.sort_values("date").drop_duplicates("date").reset_index(drop=True), "5m"
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
        .dropna().reset_index()
    )
    return y, "1m->5m"


def prep_ohlcv(x: pd.DataFrame, minutes: int) -> pd.DataFrame:
    y = x[["date", "open", "high", "low", "close", "volume"]].copy()
    y["date"] = as_ns(y["date"])
    y = y.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    prev = y["close"].shift()
    tr = pd.concat([
        y["high"] - y["low"],
        (y["high"] - prev).abs(),
        (y["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    y["atr"] = tr.rolling(14, min_periods=7).mean()
    y["signal_time"] = y["date"] + pd.Timedelta(minutes=minutes)
    return y


def resample_from_15(x15: pd.DataFrame, rule: str, minutes: int) -> pd.DataFrame:
    z = (
        x15.set_index("date")[["open", "high", "low", "close", "volume"]]
        .resample(rule, label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna().reset_index()
    )
    return prep_ohlcv(z, minutes)


def prepare_5m_with_activity(x5: pd.DataFrame, activity: pd.DataFrame) -> pd.DataFrame:
    x = prep_ohlcv(x5, 5)
    q = x["volume"] * x["close"]
    x["volume_spike_local"] = q / q.shift(1).rolling(24, min_periods=12).mean().replace(0, np.nan)
    rng = x["high"] - x["low"]
    med = rng.shift(1).rolling(288, min_periods=72).median()
    mad = (rng.shift(1) - med).abs().rolling(288, min_periods=72).median()
    x["range_z"] = ((rng - med) / (1.4826 * mad.replace(0, np.nan))).clip(-8, 8)
    a = activity.copy()
    a["signal_time"] = as_ns(a["signal_time"])
    x["signal_time"] = as_ns(x["signal_time"])
    x = pd.merge_asof(
        x.sort_values("signal_time"), a.sort_values("signal_time"),
        on="signal_time", direction="backward", tolerance=pd.Timedelta("10min"),
    )
    for c in ["active_any", "active_strict", "spike_alert", "top_growth", "top_decline", "top_volatility"]:
        if c in x:
            x[c] = x[c].fillna(False).astype(bool)
    return x.reset_index(drop=True)


@dataclass
class Level:
    level_id: int
    tf: str
    tf_minutes: int
    period: int
    kind: str
    price: float
    init_price: float
    touch_price: float
    touch_error_pct: float
    init_idx: int
    touch_idx: int
    formed_time: pd.Timestamp
    clean_between: bool
    counted_touches: int = 1


@dataclass
class Event:
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


def local_pivots(x: pd.DataFrame) -> list[dict]:
    h = x["high"].to_numpy(float)
    l = x["low"].to_numpy(float)
    sig = pd.to_datetime(x["signal_time"], utc=True).tolist()
    s = PIVOT_RIGHT
    out = []
    for i in range(s, len(x)-s):
        avail = i + s
        if h[i] >= np.max(h[i-s:i+s+1]):
            out.append({"kind":"R", "idx":i, "avail":avail, "price":float(h[i]), "formed":sig[avail]})
        if l[i] <= np.min(l[i-s:i+s+1]):
            out.append({"kind":"S", "idx":i, "avail":avail, "price":float(l[i]), "formed":sig[avail]})
    return sorted(out, key=lambda r: (r["avail"], r["idx"], r["kind"]))


def one_sided_between(x: pd.DataFrame, a: dict, b: dict, price: float) -> bool:
    if b["idx"] <= a["idx"] + 1:
        return False
    closes = x["close"].iloc[a["idx"]+1:b["idx"]].to_numpy(float)
    if len(closes) == 0:
        return False
    band = price * TOUCH_TOL_PCT
    if a["kind"] == "R":
        departed = np.min(closes) <= price - band
        wrong = np.mean(closes > price + 0.25*band)
    else:
        departed = np.max(closes) >= price + band
        wrong = np.mean(closes < price - 0.25*band)
    return bool(departed and wrong <= 0.05)


def build_levels(x: pd.DataFrame, tf: str, period: int, id_start: int) -> list[Level]:
    piv = local_pivots(x)
    pending = {"R": [], "S": []}
    levels: list[Level] = []
    formed_state: list[dict] = []
    lid = id_start
    for p in piv:
        best_existing = None
        best_err = float("inf")
        for st in formed_state:
            lv = st["level"]
            if lv.kind != p["kind"] or p["idx"] - st["last_touch_idx"] < period:
                continue
            err = abs(p["price"] - lv.price) / max(abs(lv.price), 1e-12)
            if err <= TOUCH_TOL_PCT and err < best_err:
                best_existing, best_err = st, err
        if best_existing is not None:
            best_existing["last_touch_idx"] = int(p["idx"])
            best_existing["level"].counted_touches += 1
            continue
        arr = pending[p["kind"]]
        best = None
        best_err = float("inf")
        best_pos = None
        for pos in range(len(arr)-1, -1, -1):
            q = arr[pos]
            if p["idx"] - q["idx"] < period:
                continue
            err = abs(p["price"] - q["price"]) / max(abs(q["price"]), 1e-12)
            if err <= TOUCH_TOL_PCT and err < best_err:
                center = (p["price"] + q["price"]) / 2.0
                if one_sided_between(x, q, p, center):
                    best, best_err, best_pos = q, err, pos
                    if err <= TOUCH_TOL_PCT * 0.25:
                        break
        if best is not None:
            center = (p["price"] + best["price"]) / 2.0
            lv = Level(
                level_id=lid, tf=tf, tf_minutes=TF_MINUTES[tf], period=period,
                kind=p["kind"], price=float(center), init_price=float(best["price"]),
                touch_price=float(p["price"]), touch_error_pct=float(best_err*100.0),
                init_idx=int(best["idx"]), touch_idx=int(p["idx"]), formed_time=pd.Timestamp(p["formed"]),
                clean_between=True,
            )
            levels.append(lv)
            formed_state.append({"level": lv, "last_touch_idx": int(p["idx"])})
            lid += 1
            if best_pos is not None:
                arr.pop(best_pos)
            continue
        arr.append(p)
        if len(arr) > 5000:
            del arr[:2000]
    return levels


def confluence_counts(levels: list[Level]) -> dict[int,int]:
    out = {}
    by_price = sorted([(l.price, l) for l in levels], key=lambda z:z[0])
    prices = [p for p,_ in by_price]
    for p, lvl in by_price:
        lo = bisect.bisect_left(prices, p*(1-0.0025))
        hi = bisect.bisect_right(prices, p*(1+0.0025))
        tfs = {by_price[j][1].tf for j in range(lo,hi) if by_price[j][1].formed_time <= lvl.formed_time}
        out[lvl.level_id] = max(1, len(tfs))
    return out


def recent_structure_stop(x5: pd.DataFrame, i: int, side: int, level: float) -> float:
    a = float(x5.iloc[i]["atr"])
    z = x5.iloc[max(0,i-6):i+1]
    if side > 0:
        candidate = float(z["low"].min())
        if candidate < level:
            return candidate
        return level - 0.25*a
    candidate = float(z["high"].max())
    if candidate > level:
        return candidate
    return level + 0.25*a


def protor_features(close: np.ndarray, i: int, level: float) -> tuple[int,bool]:
    if i < PROTO_BARS:
        return 0, False
    z = close[i-PROTO_BARS:i]
    near = np.abs(z/level - 1.0) <= PROTO_NEAR_PCT
    n = int(near.sum())
    return n, n >= PROTO_MIN_NEAR
