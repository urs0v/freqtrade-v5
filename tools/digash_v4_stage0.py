#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import random
import zipfile
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_PAIRS = (
    "AAVE", "ADA", "ATOM", "AVAX", "BCH", "BNB", "BTC", "DOGE", "DOT", "ETC",
    "ETH", "FIL", "LINK", "LTC", "NEAR", "SOL", "TRX", "UNI", "XLM", "XRP",
)
MIN_RR = 3.0
LEVEL_MAX_AGE_DAYS = 90
LEVEL_BREAK_VALID_DAYS = 14
RETEST_WINDOW_MIN = 30
RANDOM_SEED = 4404


def parse_args():
    p = argparse.ArgumentParser(
        description="Digash V4 Stage-0 causal detector: visual parity before PnL."
    )
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance/futures")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_v4_stage0")
    p.add_argument("--start", default="2025-11-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--pairs", default=",".join(DEFAULT_PAIRS))
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--sample", type=int, default=100)
    p.add_argument("--no-render", action="store_true")
    p.add_argument("--rebuild-activity", action="store_true")
    return p.parse_args()


def log(msg: str) -> None:
    print(msg, flush=True)


def _ts(s: str) -> pd.Timestamp:
    return pd.Timestamp(s, tz="UTC")


def _atr14(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev).abs(),
            (df["low"] - prev).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _data_path(datadir: Path, symbol: str, tf: str = "1m") -> Path:
    return datadir / f"{symbol}_USDT_USDT-{tf}-futures.feather"


def _load_1m(
    datadir: Path,
    symbol: str,
    start: pd.Timestamp,
    end_day: pd.Timestamp,
    warm_days: int = 100,
) -> pd.DataFrame:
    path = _data_path(datadir, symbol, "1m")
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_feather(path)
    need = {"date", "open", "high", "low", "close", "volume"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    x = df[["date", "open", "high", "low", "close", "volume"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values("date").drop_duplicates("date", keep="last")
    warm = start - pd.Timedelta(days=warm_days)
    end_excl = end_day + pd.Timedelta(days=1)
    x = x[(x["date"] >= warm) & (x["date"] < end_excl)].reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    x = x.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)
    return x


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    z = df.set_index("date")
    out = z.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    out = out.dropna().reset_index()
    out["atr14"] = _atr14(out)
    return out


def _rule_delta(rule: str) -> pd.Timedelta:
    return pd.Timedelta(rule)


def _swing_state(df: pd.DataFrame, rule: str, span: int) -> pd.DataFrame:
    """Causal swing state. A pivot at j becomes known only after j+span bar closes."""
    n = len(df)
    state = np.zeros(n, dtype=np.int8)
    highs: list[tuple[int, float]] = []
    lows: list[tuple[int, float]] = []
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)

    current = 0
    for k in range(n):
        j = k - span
        if j >= span and j + span < n:
            wh = h[j - span : j + span + 1]
            wl = l[j - span : j + span + 1]
            if np.isfinite(h[j]) and h[j] >= np.nanmax(wh):
                if np.sum(np.isclose(wh, h[j], rtol=0, atol=max(abs(h[j]) * 1e-10, 1e-12))) <= 2:
                    highs.append((j, float(h[j])))
            if np.isfinite(l[j]) and l[j] <= np.nanmin(wl):
                if np.sum(np.isclose(wl, l[j], rtol=0, atol=max(abs(l[j]) * 1e-10, 1e-12))) <= 2:
                    lows.append((j, float(l[j])))
            if len(highs) >= 2 and len(lows) >= 2:
                hh = highs[-1][1] >= highs[-2][1]
                hl = lows[-1][1] >= lows[-2][1]
                lh = highs[-1][1] <= highs[-2][1]
                ll = lows[-1][1] <= lows[-2][1]
                if hh and hl and not (lh and ll):
                    current = 1
                elif lh and ll and not (hh and hl):
                    current = -1
                else:
                    current = 0
        state[k] = current

    return pd.DataFrame(
        {
            "available_time": df["date"] + _rule_delta(rule),
            "dir": state,
            "atr": df["atr14"].to_numpy(float),
        }
    )


def _prominent_levels(
    df: pd.DataFrame,
    rule: str,
    span: int,
    min_prom: float = 0.75,
    min_depart: float = 0.50,
) -> list[dict]:
    """Confirmed prominent pivots. Right-side bars are part of confirmation."""
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    atr = df["atr14"].to_numpy(float)
    dates = pd.to_datetime(df["date"], utc=True).reset_index(drop=True)
    out: list[dict] = []
    n = len(df)
    delta = _rule_delta(rule)

    for j in range(span, n - span):
        a = atr[j]
        if not (np.isfinite(a) and a > 0):
            continue
        wh = h[j - span : j + span + 1]
        wl = l[j - span : j + span + 1]
        avail = pd.Timestamp(dates.iloc[j + span]) + delta

        if h[j] >= np.nanmax(wh):
            left_floor = np.nanmin(l[j - span : j])
            right_floor = np.nanmin(l[j + 1 : j + span + 1])
            shoulder = max(left_floor, right_floor)
            prom = (h[j] - shoulder) / a
            depart = (h[j] - right_floor) / a
            if prom >= min_prom and depart >= min_depart:
                out.append(
                    {
                        "id": f"{rule}-H-{j}",
                        "kind": "H",
                        "tf": rule,
                        "price": float(h[j]),
                        "atr": float(a),
                        "prominence_atr": float(prom),
                        "departure_atr": float(depart),
                        "pivot_time": pd.Timestamp(dates.iloc[j]),
                        "available_time": avail,
                    }
                )

        if l[j] <= np.nanmin(wl):
            left_ceiling = np.nanmax(h[j - span : j])
            right_ceiling = np.nanmax(h[j + 1 : j + span + 1])
            shoulder = min(left_ceiling, right_ceiling)
            prom = (shoulder - l[j]) / a
            depart = (right_ceiling - l[j]) / a
            if prom >= min_prom and depart >= min_depart:
                out.append(
                    {
                        "id": f"{rule}-L-{j}",
                        "kind": "L",
                        "tf": rule,
                        "price": float(l[j]),
                        "atr": float(a),
                        "prominence_atr": float(prom),
                        "departure_atr": float(depart),
                        "pivot_time": pd.Timestamp(dates.iloc[j]),
                        "available_time": avail,
                    }
                )

    out.sort(key=lambda r: r["available_time"])
    return out


def _level_snapshots(levels: list[dict]) -> list[dict]:
    """Causal multi-touch level snapshots. Snapshot center only uses touches known then."""
    clusters: list[dict] = []
    snaps: list[dict] = []
    next_id = 1
    for lv in sorted(levels, key=lambda r: r["available_time"]):
        best = None
        best_dist = math.inf
        for c in clusters:
            if c["kind"] != lv["kind"]:
                continue
            tol = 0.20 * max(float(c["atr"]), float(lv["atr"]))
            d = abs(float(lv["price"]) - float(c["center"]))
            min_gap = pd.Timedelta(hours=2) if lv["tf"] == "1h" else pd.Timedelta(hours=8)
            if d <= tol and lv["pivot_time"] - c["last_pivot_time"] >= min_gap and d < best_dist:
                best = c
                best_dist = d
        if best is None:
            clusters.append(
                {
                    "cluster_id": f"L{next_id:05d}",
                    "kind": lv["kind"],
                    "center": float(lv["price"]),
                    "atr": float(lv["atr"]),
                    "touches": 1,
                    "last_pivot_time": lv["pivot_time"],
                    "source_tfs": {lv["tf"]},
                    "prominence_max": float(lv["prominence_atr"]),
                }
            )
            next_id += 1
            continue

        n = int(best["touches"])
        best["center"] = (best["center"] * n + float(lv["price"])) / (n + 1)
        best["atr"] = (best["atr"] * n + float(lv["atr"])) / (n + 1)
        best["touches"] = n + 1
        best["last_pivot_time"] = lv["pivot_time"]
        best["source_tfs"].add(lv["tf"])
        best["prominence_max"] = max(best["prominence_max"], float(lv["prominence_atr"]))

        if best["touches"] >= 2:
            snaps.append(
                {
                    "level_id": best["cluster_id"],
                    "kind": best["kind"],
                    "center": float(best["center"]),
                    "width": float(0.15 * best["atr"]),
                    "atr": float(best["atr"]),
                    "touch_count": int(best["touches"]),
                    "prominence_atr": float(best["prominence_max"]),
                    "active_from": lv["available_time"],
                    "source_tfs": "+".join(sorted(best["source_tfs"])),
                }
            )
    return snaps


def _build_activity(
    datadir: Path,
    outpath: Path,
    pairs: list[str],
    start: pd.Timestamp,
    end_day: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for sym in pairs:
        try:
            x = _load_1m(datadir, sym, start - pd.Timedelta(days=2), end_day, warm_days=2)
        except FileNotFoundError:
            continue
        h1 = _resample(x, "1h")
        if h1.empty:
            continue
        lr = np.log(h1["close"] / h1["close"].shift(1))
        h1["ret24_abs"] = (h1["close"] / h1["close"].shift(24) - 1.0).abs()
        h1["vol24"] = h1["volume"].rolling(24, min_periods=18).sum()
        h1["rv24"] = lr.rolling(24, min_periods=18).std()
        h1["available_time"] = h1["date"] + pd.Timedelta(hours=1)
        h1["pair"] = sym
        rows.append(h1[["pair", "available_time", "ret24_abs", "vol24", "rv24"]])
        log(f"activity {sym}: hours={len(h1)}")

    if not rows:
        raise RuntimeError("No 1m data found for activity model.")

    a = pd.concat(rows, ignore_index=True).dropna()
    for c in ("ret24_abs", "vol24", "rv24"):
        a[c + "_pct"] = a.groupby("available_time")[c].rank(pct=True, method="average")
    a["activity_hits"] = (
        (a["ret24_abs_pct"] >= 0.75).astype(int)
        + (a["vol24_pct"] >= 0.75).astype(int)
        + (a["rv24_pct"] >= 0.75).astype(int)
    )
    a["activity_class"] = np.select(
        [a["activity_hits"] >= 2, a["activity_hits"] >= 1],
        ["SUPER_ACTIVE", "ACTIVE"],
        default="INACTIVE",
    )
    outpath.parent.mkdir(parents=True, exist_ok=True)
    a.reset_index(drop=True).to_feather(outpath)
    return a


def _merge_asof_feature(
    x: pd.DataFrame,
    feat: pd.DataFrame,
    value_cols: list[str],
    prefix: str,
) -> pd.DataFrame:
    f = feat[["available_time"] + value_cols].copy().sort_values("available_time")
    f = f.rename(columns={c: f"{prefix}{c}" for c in value_cols})
    left = x.sort_values("asof_time")
    return pd.merge_asof(
        left,
        f,
        left_on="asof_time",
        right_on="available_time",
        direction="backward",
    ).drop(columns=["available_time"], errors="ignore")


def _quality_features(x: pd.DataFrame) -> pd.DataFrame:
    z = x.copy()
    z["atr1m"] = _atr14(z)
    z["range_hi10"] = z["high"].shift(1).rolling(10, min_periods=10).max()
    z["range_lo10"] = z["low"].shift(1).rolling(10, min_periods=10).min()
    z["range10"] = z["range_hi10"] - z["range_lo10"]
    hi30 = z["high"].shift(1).rolling(30, min_periods=24).max()
    lo30 = z["low"].shift(1).rolling(30, min_periods=24).min()
    z["compression"] = z["range10"] / (hi30 - lo30).replace(0, np.nan)
    path = z["close"].diff().abs().shift(1).rolling(10, min_periods=10).sum()
    net = (z["close"].shift(1) - z["close"].shift(11)).abs()
    z["efficiency"] = net / path.replace(0, np.nan)

    prev_h = z["high"].shift(1)
    prev_l = z["low"].shift(1)
    inter = (np.minimum(z["high"], prev_h) - np.maximum(z["low"], prev_l)).clip(lower=0)
    denom = np.maximum(z["high"] - z["low"], prev_h - prev_l).replace(0, np.nan)
    pair_overlap = inter / denom
    z["overlap10"] = pair_overlap.shift(1).rolling(10, min_periods=8).mean()

    rng = (z["high"] - z["low"]).replace(0, np.nan)
    body = (z["close"] - z["open"]).abs()
    wick = (rng - body).clip(lower=0) / rng
    z["wickiness10"] = wick.shift(1).rolling(10, min_periods=8).mean()
    z["prior_ret20"] = z["close"].shift(1) / z["close"].shift(21) - 1.0

    z["clean_structure"] = (
        (z["compression"] <= 0.60)
        & (z["efficiency"] <= 0.48)
        & (z["overlap10"] >= 0.22)
        & (z["wickiness10"] <= 0.72)
    )
    return z


def _htf_direction(row: pd.Series) -> tuple[int, str]:
    d1 = int(row.get("h1_dir", 0) if pd.notna(row.get("h1_dir", 0)) else 0)
    d4 = int(row.get("h4_dir", 0) if pd.notna(row.get("h4_dir", 0)) else 0)
    if d1 and d4 and d1 != d4:
        return 0, "CONFLICT"
    if d4:
        return d4, "4h" if not d1 else "1h+4h"
    if d1:
        return d1, "1h"
    return 0, "NONE"


def _touch_count(levels: list[dict], target: dict, asof: pd.Timestamp) -> int:
    tol = 0.20 * max(float(target["atr"]), 1e-12)
    return sum(
        1
        for lv in levels
        if lv["kind"] == target["kind"]
        and lv["available_time"] <= asof
        and abs(float(lv["price"]) - float(target["price"])) <= tol
    )


def _nearest_targets(
    levels: list[dict],
    asof: pd.Timestamp,
    entry: float,
    side: int,
) -> list[dict]:
    kind = "H" if side > 0 else "L"
    cutoff = asof - pd.Timedelta(days=LEVEL_MAX_AGE_DAYS)
    cand = []
    for lv in levels:
        if lv["kind"] != kind or lv["available_time"] > asof or lv["available_time"] < cutoff:
            continue
        p = float(lv["price"])
        if (side > 0 and p <= entry) or (side < 0 and p >= entry):
            continue
        cand.append(lv)
    cand.sort(key=lambda r: side * (float(r["price"]) - entry))

    uniq: list[dict] = []
    for lv in cand:
        if not uniq:
            uniq.append(lv)
            continue
        tol = 0.15 * max(float(lv["atr"]), float(uniq[-1]["atr"]))
        if abs(float(lv["price"]) - float(uniq[-1]["price"])) > tol:
            uniq.append(lv)
        if len(uniq) >= 6:
            break
    return uniq


def _cascade_from_targets(targets: list[dict]) -> list[dict]:
    if not targets:
        return []
    out = [targets[0]]
    for lv in targets[1:]:
        prev = out[-1]
        gap = abs(float(lv["price"]) - float(prev["price"]))
        if gap <= 1.75 * max(float(lv["atr"]), float(prev["atr"])):
            out.append(lv)
        else:
            break
        if len(out) >= 3:
            break
    return out


def _safe_float(v):
    try:
        f = float(v)
        return f if np.isfinite(f) else None
    except Exception:
        return None


def _activity_components(row: pd.Series) -> dict:
    return {
        "source": "OHLCV_PROXY",
        "ret24_abs_pct": _safe_float(row.get("act_ret24_abs_pct")),
        "vol24_pct": _safe_float(row.get("act_vol24_pct")),
        "rv24_pct": _safe_float(row.get("act_rv24_pct")),
        "hits": int(row.get("act_activity_hits", 0) or 0),
    }


def _quality_dict(row: pd.Series) -> dict:
    return {
        "compression": _safe_float(row.get("compression")),
        "efficiency": _safe_float(row.get("efficiency")),
        "overlap10": _safe_float(row.get("overlap10")),
        "wickiness10": _safe_float(row.get("wickiness10")),
        "clean": bool(row.get("clean_structure", False)),
    }


def _evaluate_common(
    row: pd.Series,
    side: int,
    entry: float,
    stop: float,
    levels: list[dict],
    asof: pd.Timestamp,
    require_local_counter: bool = True,
) -> tuple[list[str], dict | None]:
    veto = []
    activity = str(row.get("act_activity_class", "INACTIVE"))
    if activity not in ("ACTIVE", "SUPER_ACTIVE"):
        veto.append("NOT_ACTIVE")

    htf, htf_tf = _htf_direction(row)
    if htf != side:
        veto.append("NO_HTF_DIRECTION")

    if require_local_counter:
        d5v = row.get("m5_dir", 0)
        d5 = int(d5v) if pd.notna(d5v) else 0
        r20 = row.get("prior_ret20")
        r20 = float(r20) if pd.notna(r20) else 0.0
        if not (d5 == -side or side * r20 < 0):
            veto.append("NO_LOCAL_COUNTERTREND")

    if not bool(row.get("clean_structure", False)):
        veto.append("TOO_NOISY")

    risk = side * (entry - stop)
    if not (np.isfinite(risk) and risk > 0):
        veto.append("NO_VALID_INVALIDATION")
        return veto, None

    risk_bps = risk / entry * 10000.0
    if risk_bps < 8 or risk_bps > 700:
        veto.append("INVALID_STOP_GEOMETRY")

    targets = _nearest_targets(levels, asof, entry, side)
    if not targets:
        veto.append("NO_MEANINGFUL_TARGET")
        return veto, None
    cascade = _cascade_from_targets(targets)
    final = cascade[-1]
    rr = side * (float(final["price"]) - entry) / risk
    if rr < MIN_RR:
        veto.append("AVAILABLE_R_LT_3")

    data = {
        "htf_direction": htf,
        "htf_tf": htf_tf,
        "risk_abs": risk,
        "risk_bps": risk_bps,
        "targets": cascade,
        "available_R": rr,
    }
    return veto, data


def _event_row(
    symbol: str,
    x: pd.DataFrame,
    signal_idx: int,
    entry_idx: int,
    side: int,
    family: str,
    stop: float,
    trigger_price: float,
    levels: list[dict],
    common: dict,
    level_meta: dict | None = None,
    sweep_depth_atr: float | None = None,
    retest_price: float | None = None,
) -> dict:
    row = x.iloc[signal_idx]
    entry = float(x.iloc[entry_idx]["open"])
    targets = common["targets"]
    t1 = targets[0]
    final = targets[-1]
    asof = pd.Timestamp(row["asof_time"])
    touch_count = _touch_count(levels, t1, asof)
    level_meta = level_meta or {}
    return {
        "pair": symbol,
        "asof_time": asof,
        "entry_time": pd.Timestamp(x.iloc[entry_idx]["date"]),
        "side": "LONG" if side > 0 else "SHORT",
        "side_i": side,
        "entry_family": family,
        "activity_class": str(row.get("act_activity_class", "INACTIVE")),
        "activity_source": "OHLCV_PROXY",
        "activity_components": json.dumps(_activity_components(row), sort_keys=True),
        "htf_direction": "LONG" if side > 0 else "SHORT",
        "htf_tf": common["htf_tf"],
        "level_type": level_meta.get("level_type", f"TARGET_{t1['kind']}"),
        "level_id": level_meta.get("level_id", t1["id"]),
        "level_center": float(level_meta.get("level_center", t1["price"])),
        "level_width": float(level_meta.get("level_width", 0.15 * t1["atr"])),
        "level_touch_count": int(level_meta.get("level_touch_count", touch_count)),
        "level_prominence_atr": float(level_meta.get("level_prominence_atr", t1["prominence_atr"])),
        "cascade_count": len(targets),
        "local_structure_type": "PROTORGOVKA_BREAK" if family in ("BOS_BREAK", "RETEST_REACTION") else family,
        "protorgovka_duration_min": 10 if family in ("BOS_BREAK", "RETEST_REACTION", "LEVEL_BREAK") else None,
        "bos_price": float(trigger_price) if family in ("BOS_BREAK", "RETEST_REACTION", "LEVEL_BREAK") else None,
        "retest_price": retest_price,
        "sweep_depth_atr": sweep_depth_atr,
        "entry_price": entry,
        "initial_stop": float(stop),
        "initial_stop_bps": float(common["risk_bps"]),
        "initial_stop_atr": float(common["risk_abs"] / max(float(row["atr1m"]), 1e-12)),
        "target_1": float(t1["price"]),
        "final_target": float(final["price"]),
        "available_R": float(common["available_R"]),
        "structure_quality": json.dumps(_quality_dict(row), sort_keys=True),
        "veto_flags": "",
        "_signal_idx": int(signal_idx),
    }


def _prepare_pair(
    symbol: str,
    datadir: Path,
    activity_path: Path,
    start: pd.Timestamp,
    end_day: pd.Timestamp,
):
    x = _load_1m(datadir, symbol, start, end_day)
    if x.empty:
        raise RuntimeError("empty data")
    x["asof_time"] = x["date"] + pd.Timedelta(minutes=1)

    h5 = _resample(x, "5min")
    h1 = _resample(x, "1h")
    h4 = _resample(x, "4h")

    s5 = _swing_state(h5, "5min", 2)
    s1 = _swing_state(h1, "1h", 3)
    s4 = _swing_state(h4, "4h", 2)

    x = _merge_asof_feature(x, s5, ["dir"], "m5_")
    x = _merge_asof_feature(x, s1, ["dir", "atr"], "h1_")
    x = _merge_asof_feature(x, s4, ["dir", "atr"], "h4_")

    act = pd.read_feather(activity_path)
    act = act[act["pair"] == symbol].copy()
    act_cols = [
        "activity_class", "activity_hits",
        "ret24_abs_pct", "vol24_pct", "rv24_pct",
    ]
    x = _merge_asof_feature(x, act, act_cols, "act_")
    x = _quality_features(x)

    levels = _prominent_levels(h1, "1h", 3) + _prominent_levels(h4, "4h", 2)
    levels.sort(key=lambda r: r["available_time"])
    snaps = _level_snapshots(levels)
    return x.reset_index(drop=True), h5, h1, h4, levels, snaps


def _valid_time(row: pd.Series, start: pd.Timestamp, end_excl: pd.Timestamp) -> bool:
    t = pd.Timestamp(row["asof_time"])
    return start <= t < end_excl


def _add_veto(veto_counts: Counter, family: str, reasons: list[str]) -> None:
    for r in set(reasons):
        veto_counts[f"{family}:{r}"] += 1


def _detect_bos(
    symbol: str,
    x: pd.DataFrame,
    levels: list[dict],
    start: pd.Timestamp,
    end_excl: pd.Timestamp,
    veto_counts: Counter,
) -> list[dict]:
    events = []
    atr = x["atr1m"].to_numpy(float)
    c = x["close"].to_numpy(float)
    hi = x["range_hi10"].to_numpy(float)
    lo = x["range_lo10"].to_numpy(float)
    clean = x["clean_structure"].fillna(False).to_numpy(bool)

    long_mask = clean & np.isfinite(atr) & np.isfinite(hi) & (c > hi + 0.03 * atr)
    short_mask = clean & np.isfinite(atr) & np.isfinite(lo) & (c < lo - 0.03 * atr)
    idxs = np.flatnonzero(long_mask | short_mask)

    for i in idxs:
        if i + 1 >= len(x):
            continue
        row = x.iloc[i]
        if not _valid_time(row, start, end_excl):
            continue
        side = 1 if long_mask[i] else -1
        trigger = float(hi[i] if side > 0 else lo[i])
        stop = float(lo[i] if side > 0 else hi[i])
        entry = float(x.iloc[i + 1]["open"])
        asof = pd.Timestamp(row["asof_time"])
        reasons, common = _evaluate_common(row, side, entry, stop, levels, asof, True)
        if reasons:
            _add_veto(veto_counts, "BOS_BREAK", reasons)
            continue
        events.append(
            _event_row(symbol, x, i, i + 1, side, "BOS_BREAK", stop, trigger, levels, common)
        )
    return events


def _detect_retest_reaction(
    symbol: str,
    x: pd.DataFrame,
    levels: list[dict],
    bos_events: list[dict],
    veto_counts: Counter,
) -> list[dict]:
    out = []
    for ev in bos_events:
        i = int(ev["_signal_idx"])
        side = int(ev["side_i"])
        bos = float(ev["bos_price"])
        original_stop = float(ev["initial_stop"])
        target_final = float(ev["final_target"])
        end = min(len(x) - 2, i + RETEST_WINDOW_MIN)

        for j in range(i + 2, end + 1):
            bar = x.iloc[j]
            atr = float(bar["atr1m"])
            if not (np.isfinite(atr) and atr > 0):
                continue
            if side > 0 and float(bar["low"]) <= original_stop:
                break
            if side < 0 and float(bar["high"]) >= original_stop:
                break

            touched = (
                float(bar["low"]) <= bos + 0.15 * atr if side > 0
                else float(bar["high"]) >= bos - 0.15 * atr
            )
            if not touched:
                continue

            prev = x.iloc[j - 1]
            reaction = (
                float(bar["close"]) > bos
                and float(bar["close"]) > float(prev["close"])
                and float(bar["close"]) >= float(bar["open"])
                if side > 0
                else float(bar["close"]) < bos
                and float(bar["close"]) < float(prev["close"])
                and float(bar["close"]) <= float(bar["open"])
            )
            if not reaction:
                continue

            entry_idx = j + 1
            entry = float(x.iloc[entry_idx]["open"])
            if side > 0:
                stop = min(float(x.iloc[j - 1]["low"]), float(bar["low"])) - 0.03 * atr
            else:
                stop = max(float(x.iloc[j - 1]["high"]), float(bar["high"])) + 0.03 * atr

            asof = pd.Timestamp(bar["asof_time"])
            reasons, common = _evaluate_common(
                bar, side, entry, stop, levels, asof, require_local_counter=False
            )
            if common is not None:
                risk = side * (entry - stop)
                rr_orig = side * (target_final - entry) / risk if risk > 0 else -math.inf
                if rr_orig >= MIN_RR:
                    common["targets"] = [
                        {
                            "id": str(ev["level_id"]),
                            "kind": "H" if side > 0 else "L",
                            "tf": str(ev["htf_tf"]),
                            "price": float(ev["target_1"]),
                            "atr": max(atr, 1e-12),
                            "prominence_atr": float(ev["level_prominence_atr"]),
                        },
                        {
                            "id": str(ev["level_id"]) + "-final",
                            "kind": "H" if side > 0 else "L",
                            "tf": str(ev["htf_tf"]),
                            "price": target_final,
                            "atr": max(atr, 1e-12),
                            "prominence_atr": float(ev["level_prominence_atr"]),
                        },
                    ] if float(ev["target_1"]) != target_final else [{
                        "id": str(ev["level_id"]),
                        "kind": "H" if side > 0 else "L",
                        "tf": str(ev["htf_tf"]),
                        "price": target_final,
                        "atr": max(atr, 1e-12),
                        "prominence_atr": float(ev["level_prominence_atr"]),
                    }]
                    common["available_R"] = rr_orig
                    common["risk_abs"] = risk
                    common["risk_bps"] = risk / entry * 10000.0

            if reasons:
                _add_veto(veto_counts, "RETEST_REACTION", reasons)
                continue
            out.append(
                _event_row(
                    symbol, x, j, entry_idx, side, "RETEST_REACTION",
                    stop, bos, levels, common,
                    retest_price=float(bar["low"] if side > 0 else bar["high"]),
                )
            )
            break
    return out


def _find_crossings(arr_prev, arr_cur, level: float, side: int, start_i: int, end_i: int):
    if end_i <= start_i:
        return np.array([], dtype=int)
    p = arr_prev[start_i:end_i]
    c = arr_cur[start_i:end_i]
    if side > 0:
        m = (p <= level) & (c > level)
    else:
        m = (p >= level) & (c < level)
    return np.flatnonzero(m) + start_i


def _detect_level_breaks_and_sweeps(
    symbol: str,
    x: pd.DataFrame,
    levels: list[dict],
    snaps: list[dict],
    start: pd.Timestamp,
    end_excl: pd.Timestamp,
    veto_counts: Counter,
) -> tuple[list[dict], list[dict]]:
    level_events: list[dict] = []
    sweep_events: list[dict] = []
    dates = pd.to_datetime(x["date"], utc=True)
    dt64 = dates.to_numpy(dtype="datetime64[ns]")
    close = x["close"].to_numpy(float)
    prev_close = x["close"].shift(1).to_numpy(float)
    high = x["high"].to_numpy(float)
    low = x["low"].to_numpy(float)
    atr = x["atr1m"].to_numpy(float)
    rhi = x["range_hi10"].to_numpy(float)
    rlo = x["range_lo10"].to_numpy(float)

    for snap in snaps:
        active = pd.Timestamp(snap["active_from"])
        if active >= end_excl or active < start - pd.Timedelta(days=LEVEL_MAX_AGE_DAYS):
            continue
        start_i = int(np.searchsorted(
            dt64,
            np.datetime64(max(active, start - pd.Timedelta(days=1)).to_datetime64()),
            side="left",
        ))
        valid_end = min(end_excl, active + pd.Timedelta(days=LEVEL_BREAK_VALID_DAYS))
        end_i = int(np.searchsorted(dt64, np.datetime64(valid_end.to_datetime64()), side="left"))
        if start_i >= len(x) - 2:
            continue
        end_i = min(end_i, len(x) - 2)
        center = float(snap["center"])

        side_break = 1 if snap["kind"] == "H" else -1
        crosses = _find_crossings(prev_close, close, center, side_break, max(1, start_i), end_i)
        for i in crosses[:12]:
            if i + 1 >= len(x):
                continue
            row = x.iloc[i]
            if not _valid_time(row, start, end_excl):
                continue
            if not np.isfinite(atr[i]) or atr[i] <= 0:
                continue
            near_accum = (
                abs(float(rhi[i]) - center) <= 0.40 * atr[i] if side_break > 0
                else abs(float(rlo[i]) - center) <= 0.40 * atr[i]
            )
            if not near_accum:
                _add_veto(veto_counts, "LEVEL_BREAK", ["NO_NEAR_LEVEL_ACCUMULATION"])
                continue
            stop = float(rlo[i] if side_break > 0 else rhi[i])
            entry = float(x.iloc[i + 1]["open"])
            asof = pd.Timestamp(row["asof_time"])
            reasons, common = _evaluate_common(
                row, side_break, entry, stop, levels, asof, require_local_counter=False
            )
            if common:
                targets = [t for t in common["targets"] if side_break * (float(t["price"]) - center) > 0]
                if not targets:
                    reasons.append("NO_TARGET_BEYOND_BROKEN_LEVEL")
                else:
                    cascade = _cascade_from_targets(targets)
                    common["targets"] = cascade
                    risk = side_break * (entry - stop)
                    common["available_R"] = side_break * (float(cascade[-1]["price"]) - entry) / risk
                    if common["available_R"] < MIN_RR:
                        reasons.append("AVAILABLE_R_LT_3")
            if reasons:
                _add_veto(veto_counts, "LEVEL_BREAK", reasons)
                continue
            meta = {
                "level_type": "MULTITOUCH_LEVEL_BREAK",
                "level_id": snap["level_id"],
                "level_center": center,
                "level_width": float(snap["width"]),
                "level_touch_count": int(snap["touch_count"]),
                "level_prominence_atr": float(snap["prominence_atr"]),
            }
            level_events.append(
                _event_row(
                    symbol, x, i, i + 1, side_break, "LEVEL_BREAK",
                    stop, center, levels, common, level_meta=meta,
                )
            )
            break

        side_sweep = 1 if snap["kind"] == "L" else -1
        ss = max(1, start_i)
        ee = min(end_i, len(x) - 2)
        if ee > ss:
            aa = atr[ss:ee]
            finite = np.isfinite(aa) & (aa > 0)
            if side_sweep > 0:
                depth_arr = (center - low[ss:ee]) / aa
                swept_mask = finite & (low[ss:ee] < center - 0.05 * aa) & (close[ss:ee] > center)
                confirm_mask = (close[ss + 1 : ee + 1] > close[ss:ee]) & (close[ss + 1 : ee + 1] > center)
            else:
                depth_arr = (high[ss:ee] - center) / aa
                swept_mask = finite & (high[ss:ee] > center + 0.05 * aa) & (close[ss:ee] < center)
                confirm_mask = (close[ss + 1 : ee + 1] < close[ss:ee]) & (close[ss + 1 : ee + 1] < center)
            candidates = np.flatnonzero(swept_mask & confirm_mask & (depth_arr <= 1.5)) + ss
        else:
            candidates = np.array([], dtype=int)

        for i in candidates[:10]:
            signal_idx = i + 1
            entry_idx = i + 2
            row = x.iloc[signal_idx]
            if not _valid_time(row, start, end_excl):
                continue
            a = float(row["atr1m"])
            if not (np.isfinite(a) and a > 0):
                continue
            depth = float((center - low[i]) / atr[i] if side_sweep > 0 else (high[i] - center) / atr[i])
            stop = float(low[i] - 0.03 * a if side_sweep > 0 else high[i] + 0.03 * a)
            entry = float(x.iloc[entry_idx]["open"])
            asof = pd.Timestamp(row["asof_time"])
            reasons, common = _evaluate_common(
                row, side_sweep, entry, stop, levels, asof, require_local_counter=False
            )
            if reasons:
                _add_veto(veto_counts, "SWEEP_RETURN", reasons)
                continue
            meta = {
                "level_type": "MULTITOUCH_SWEEP",
                "level_id": snap["level_id"],
                "level_center": center,
                "level_width": float(snap["width"]),
                "level_touch_count": int(snap["touch_count"]),
                "level_prominence_atr": float(snap["prominence_atr"]),
            }
            sweep_events.append(
                _event_row(
                    symbol, x, signal_idx, entry_idx, side_sweep, "SWEEP_RETURN",
                    stop, center, levels, common, level_meta=meta,
                    sweep_depth_atr=depth,
                    retest_price=float(low[i] if side_sweep > 0 else high[i]),
                )
            )
            break

    return level_events, sweep_events


def process_pair(
    symbol: str,
    datadir_s: str,
    activity_path_s: str,
    start_s: str,
    end_s: str,
):
    datadir = Path(datadir_s)
    activity_path = Path(activity_path_s)
    start = _ts(start_s)
    end_day = _ts(end_s)
    end_excl = end_day + pd.Timedelta(days=1)
    veto = Counter()

    try:
        x, h5, h1, h4, levels, snaps = _prepare_pair(
            symbol, datadir, activity_path, start, end_day
        )
    except FileNotFoundError:
        return [], {}, {"pair": symbol, "status": "NO_1M_DATA"}
    except Exception as e:
        return [], {}, {"pair": symbol, "status": "ERROR", "error": repr(e)}

    bos = _detect_bos(symbol, x, levels, start, end_excl, veto)
    ret = _detect_retest_reaction(symbol, x, levels, bos, veto)
    lvl, swp = _detect_level_breaks_and_sweeps(
        symbol, x, levels, snaps, start, end_excl, veto
    )
    events = bos + ret + lvl + swp

    dedup = {}
    for ev in events:
        key = (ev["entry_time"], ev["entry_family"], ev["side"])
        old = dedup.get(key)
        if old is None or ev["available_R"] > old["available_R"]:
            dedup[key] = ev
    events = list(dedup.values())
    events.sort(key=lambda r: (r["entry_time"], r["entry_family"]))

    meta = {
        "pair": symbol,
        "status": "OK",
        "bars_1m": int(len(x)),
        "bars_5m": int(len(h5)),
        "bars_1h": int(len(h1)),
        "bars_4h": int(len(h4)),
        "prominent_levels": int(len(levels)),
        "multitouch_snapshots": int(len(snaps)),
        "events": int(len(events)),
        "family_counts": dict(Counter(e["entry_family"] for e in events)),
    }
    return events, dict(veto), meta


def _strip_internal(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in df.columns if c.startswith("_")], errors="ignore")


def _stratified_sample(events: pd.DataFrame, n: int, seed: int = RANDOM_SEED) -> pd.DataFrame:
    if len(events) <= n:
        return events.copy().sort_values("entry_time")
    rng = random.Random(seed)
    z = events.copy()
    z["month"] = pd.to_datetime(z["entry_time"], utc=True).dt.strftime("%Y-%m")
    z["stratum"] = (
        z["entry_family"].astype(str) + "|" + z["pair"].astype(str) + "|"
        + z["side"].astype(str) + "|" + z["month"].astype(str)
    )
    groups = list(z.groupby("stratum", sort=False).groups.values())
    rng.shuffle(groups)
    chosen = []
    for idxs in groups:
        chosen.append(rng.choice(list(idxs)))
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        chosen_set = set(chosen)
        rest = [i for i in z.index if i not in chosen_set]
        rng.shuffle(rest)
        chosen.extend(rest[: n - len(chosen)])
    return z.loc[chosen].drop(columns=["month", "stratum"]).sort_values("entry_time")


def _fmt_price(v: float) -> str:
    if abs(v) >= 1000:
        return f"{v:,.2f}"
    if abs(v) >= 1:
        return f"{v:.4f}"
    return f"{v:.7f}"


def _svg_candles(
    df: pd.DataFrame,
    x0: int,
    y0: int,
    width: int,
    height: int,
    title: str,
    lines: list[tuple[float, str, str]],
    entry_time: pd.Timestamp,
) -> str:
    if df.empty:
        return f'<text x="{x0+10}" y="{y0+25}" font-size="16">No data: {html.escape(title)}</text>'
    q = df.reset_index(drop=True)
    lo = float(q["low"].min())
    hi = float(q["high"].max())
    for p, _, _ in lines:
        if np.isfinite(p):
            lo = min(lo, p)
            hi = max(hi, p)
    pad = max((hi - lo) * 0.06, abs(hi) * 1e-6, 1e-9)
    lo -= pad
    hi += pad
    n = len(q)
    cw = width / max(n, 1)
    bodyw = max(1.0, min(5.0, cw * 0.65))

    def sy(p):
        return y0 + height - (p - lo) / max(hi - lo, 1e-12) * height

    parts = [
        f'<rect x="{x0}" y="{y0}" width="{width}" height="{height}" fill="#ffffff" stroke="#cccccc"/>',
        f'<text x="{x0+8}" y="{y0+18}" font-size="14" font-weight="bold">{html.escape(title)}</text>',
    ]
    for k in range(1, 4):
        yy = y0 + k * height / 4
        parts.append(f'<line x1="{x0}" y1="{yy:.1f}" x2="{x0+width}" y2="{yy:.1f}" stroke="#eeeeee"/>')

    for i, r in q.iterrows():
        xx = x0 + (i + 0.5) * cw
        o, h, l, c = map(float, (r["open"], r["high"], r["low"], r["close"]))
        color = "#1a7f37" if c >= o else "#cf222e"
        parts.append(f'<line x1="{xx:.2f}" y1="{sy(h):.2f}" x2="{xx:.2f}" y2="{sy(l):.2f}" stroke="{color}" stroke-width="1"/>')
        top = min(sy(o), sy(c))
        bh = max(1.0, abs(sy(o) - sy(c)))
        parts.append(f'<rect x="{xx-bodyw/2:.2f}" y="{top:.2f}" width="{bodyw:.2f}" height="{bh:.2f}" fill="{color}"/>')

    for p, label, color in lines:
        if not np.isfinite(p):
            continue
        yy = sy(p)
        parts.append(f'<line x1="{x0}" y1="{yy:.2f}" x2="{x0+width}" y2="{yy:.2f}" stroke="{color}" stroke-width="1.5" stroke-dasharray="6 4"/>')
        parts.append(f'<text x="{x0+width-5}" y="{yy-3:.2f}" text-anchor="end" font-size="11" fill="{color}">{html.escape(label)} {_fmt_price(p)}</text>')

    dates = pd.to_datetime(q["date"], utc=True)
    if len(dates):
        pos = np.searchsorted(dates.to_numpy(dtype="datetime64[ns]"), np.datetime64(entry_time.to_datetime64()))
        xx = x0 + width - 1 if pos >= n else x0 + (pos + 0.5) * cw
        parts.append(f'<line x1="{xx:.2f}" y1="{y0}" x2="{xx:.2f}" y2="{y0+height}" stroke="#8250df" stroke-width="2"/>')
        parts.append(f'<text x="{xx-4:.2f}" y="{y0+height-5}" text-anchor="end" font-size="10" fill="#8250df">ENTRY</text>')

    parts.append(f'<text x="{x0+5}" y="{y0+height-5}" font-size="10" fill="#666">{html.escape(str(dates.iloc[0]))[:19]} → {html.escape(str(dates.iloc[-1]))[:19]}</text>')
    return "".join(parts)


def _window(df: pd.DataFrame, t: pd.Timestamp, bars: int) -> pd.DataFrame:
    return df[df["date"] < t].tail(bars).copy()


def _render_event_svg(
    event: pd.Series,
    x1m: pd.DataFrame,
    h5: pd.DataFrame,
    h1: pd.DataFrame,
    outpath: Path,
) -> None:
    entry_time = pd.Timestamp(event["entry_time"])
    lines = [
        (float(event["level_center"]), "DECISIVE", "#0969da"),
        (float(event["initial_stop"]), "STOP", "#cf222e"),
        (float(event["target_1"]), "T1", "#1a7f37"),
        (float(event["final_target"]), "FINAL", "#116329"),
    ]
    if pd.notna(event.get("bos_price")):
        lines.append((float(event["bos_price"]), "BOS", "#bf8700"))

    w1 = _window(h1, entry_time, 120)
    w5 = _window(h5, entry_time, 144)
    wm = _window(x1m, entry_time, 180)

    header = (
        f"{event['pair']}  {event['side']}  {event['entry_family']}  "
        f"{entry_time.strftime('%Y-%m-%d %H:%M UTC')}  "
        f"activity={event['activity_class']}  HTF={event['htf_tf']}  "
        f"RR={float(event['available_R']):.2f}  touches={int(event['level_touch_count'])}"
    )
    subtitle = (
        f"entry={_fmt_price(float(event['entry_price']))}  "
        f"stop={_fmt_price(float(event['initial_stop']))}  "
        f"T1={_fmt_price(float(event['target_1']))}  "
        f"final={_fmt_price(float(event['final_target']))}"
    )

    width, height = 1500, 1040
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f6f8fa"/>',
        f'<text x="25" y="32" font-size="20" font-weight="bold">{html.escape(header)}</text>',
        f'<text x="25" y="55" font-size="13">{html.escape(subtitle)}</text>',
        _svg_candles(w1, 25, 75, 1450, 280, "1h — HTF context / targets", lines, entry_time),
        _svg_candles(w5, 25, 385, 1450, 280, "5m — local construction", lines, entry_time),
        _svg_candles(wm, 25, 695, 1450, 280, "1m — precise entry structure", lines, entry_time),
        '<text x="25" y="1015" font-size="12" fill="#555">Stage-0 parity chart: only candles before entry are rendered. No PnL/outcome is shown.</text>',
        "</svg>",
    ]
    outpath.write_text("".join(parts), encoding="utf-8")


def _render_review_bundle(
    sample: pd.DataFrame,
    datadir: Path,
    outdir: Path,
    start: pd.Timestamp,
    end_day: pd.Timestamp,
) -> Path:
    charts = outdir / "charts"
    charts.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    html_cards = []

    for sym, grp in sample.groupby("pair", sort=True):
        x = _load_1m(datadir, sym, start, end_day)
        h5 = _resample(x, "5min")
        h1 = _resample(x, "1h")
        for _, ev in grp.iterrows():
            seq = int(ev["_sample_no"])
            stamp = pd.Timestamp(ev["entry_time"]).strftime("%Y%m%d_%H%M")
            fname = f"{seq:03d}_{sym}_{stamp}_{ev['entry_family']}_{ev['side']}.svg"
            path = charts / fname
            _render_event_svg(ev, x, h5, h1, path)
            manifest_rows.append(
                {
                    "sample_no": seq,
                    "file": fname,
                    "pair": sym,
                    "entry_time": ev["entry_time"],
                    "side": ev["side"],
                    "family": ev["entry_family"],
                    "available_R": ev["available_R"],
                    "activity_class": ev["activity_class"],
                }
            )
            html_cards.append(
                f'<div class="card"><div><b>#{seq:03d} {html.escape(sym)} '
                f'{html.escape(str(ev["side"]))} {html.escape(str(ev["entry_family"]))}</b></div>'
                f'<img src="{html.escape(fname)}" loading="lazy"/></div>'
            )

    pd.DataFrame(manifest_rows).sort_values("sample_no").to_csv(
        outdir / "review_manifest.csv", index=False
    )
    page = """<!doctype html><html><head><meta charset="utf-8">
<title>Digash V4 Stage-0 Visual Review</title>
<style>
body{font-family:Arial,sans-serif;background:#f6f8fa;margin:20px}
.card{background:white;border:1px solid #d0d7de;border-radius:8px;padding:10px;margin:14px 0}
.card img{width:100%;max-width:1500px;height:auto}
</style></head><body>
<h1>Digash V4 Stage-0 — visual parity review</h1>
<p>Charts contain pre-entry candles only. Review whether the detected setup resembles the source strategy semantics before any PnL backtest.</p>
""" + "\n".join(html_cards) + "</body></html>"
    (charts / "index.html").write_text(page, encoding="utf-8")

    zip_path = outdir / "digash_v4_stage0_review.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(outdir / "review_manifest.csv", arcname="review_manifest.csv")
        zf.write(charts / "index.html", arcname="charts/index.html")
        for p in charts.glob("*.svg"):
            zf.write(p, arcname=f"charts/{p.name}")
    return zip_path


def main():
    args = parse_args()
    datadir = Path(args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    start = _ts(args.start)
    end_day = _ts(args.end)
    pairs_requested = [p.strip().upper() for p in args.pairs.split(",") if p.strip()]
    pairs = [p for p in pairs_requested if _data_path(datadir, p, "1m").exists()]
    missing = [p for p in pairs_requested if p not in pairs]

    log("=== DIGASH V4 STAGE-0 — VISUAL PARITY DETECTOR ===")
    log("Research only. No PnL/backtest is computed in Stage 0.")
    log(f"Window: {start.date()} -> {end_day.date()}")
    log(f"1m pairs: {len(pairs)}/{len(pairs_requested)}")
    if missing:
        log("Missing 1m: " + " ".join(missing))
    if not pairs:
        raise SystemExit("No requested 1m files found.")

    activity_path = outdir / "activity_hourly.feather"
    if args.rebuild_activity or not activity_path.exists():
        log("Building causal cross-sectional OHLCV activity proxy...")
        activity = _build_activity(datadir, activity_path, pairs, start, end_day)
        log(f"Activity rows: {len(activity)}")
    else:
        activity = pd.read_feather(activity_path)
        log(f"Using cached activity: {activity_path} rows={len(activity)}")

    all_events = []
    veto_total = Counter()
    meta = []
    workers = max(1, min(int(args.workers), len(pairs)))
    log(f"Scanning pairs with workers={workers}...")
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(
                process_pair, sym, str(datadir), str(activity_path),
                args.start, args.end,
            ): sym
            for sym in pairs
        }
        for fut in as_completed(futs):
            sym = futs[fut]
            try:
                events, veto, m = fut.result()
            except Exception as e:
                events, veto, m = [], {}, {"pair": sym, "status": "ERROR", "error": repr(e)}
            all_events.extend(events)
            veto_total.update(veto)
            meta.append(m)
            log(
                f"pair {sym}: {m.get('status')} "
                f"events={m.get('events', 0)} "
                f"levels={m.get('prominent_levels', 0)} "
                f"families={m.get('family_counts', {})}"
            )

    if all_events:
        ev = pd.DataFrame(all_events)
        ev["entry_time"] = pd.to_datetime(ev["entry_time"], utc=True)
        ev["asof_time"] = pd.to_datetime(ev["asof_time"], utc=True)
        ev = ev.sort_values(["entry_time", "pair", "entry_family"]).reset_index(drop=True)
    else:
        ev = pd.DataFrame()

    events_path = outdir / "events.csv"
    _strip_internal(ev).to_csv(events_path, index=False)

    family_counts = dict(Counter(ev["entry_family"])) if not ev.empty else {}
    pair_counts = dict(Counter(ev["pair"])) if not ev.empty else {}
    side_counts = dict(Counter(ev["side"])) if not ev.empty else {}

    summary = {
        "stage": "Digash V4 Stage-0 visual parity",
        "pnl_computed": False,
        "start": str(start),
        "end": str(end_day),
        "pairs_requested": pairs_requested,
        "pairs_scanned": pairs,
        "missing_1m": missing,
        "events": int(len(ev)),
        "family_counts": family_counts,
        "pair_counts": pair_counts,
        "side_counts": side_counts,
        "veto_counts": dict(veto_total.most_common()),
        "pair_meta": sorted(meta, key=lambda x: x.get("pair", "")),
        "spec_contract": {
            "activity": "OHLCV_PROXY cross-sectional trailing 24h ranks; not author-exact trade count",
            "htf": "causal confirmed swing structure on 1h/4h",
            "levels": "causal prominent 1h/4h pivots; multi-touch snapshots",
            "local": "5m swing direction + 1m compression/overlap structure",
            "rr_gate": MIN_RR,
            "future_outcome_used": False,
        },
    }

    zip_path = None
    if not ev.empty and not args.no_render:
        sample = _stratified_sample(ev, min(args.sample, len(ev))).reset_index(drop=True)
        sample["_sample_no"] = np.arange(1, len(sample) + 1)
        _strip_internal(sample).to_csv(outdir / "review_sample.csv", index=False)
        log(f"Rendering {len(sample)} pre-entry visual-parity charts (dependency-free SVG)...")
        zip_path = _render_review_bundle(sample, datadir, outdir, start, end_day)
        summary["review_sample"] = int(len(sample))
        summary["review_bundle"] = str(zip_path)
    else:
        summary["review_sample"] = 0
        summary["review_bundle"] = None

    (outdir / "summary.json").write_text(
        json.dumps(summary, indent=2, default=str), encoding="utf-8"
    )

    log("")
    log("=== STAGE-0 RESULT ===")
    log(f"events={len(ev)} families={family_counts}")
    log(f"sides={side_counts}")
    log("top vetoes:")
    for k, v in veto_total.most_common(15):
        log(f"  {k}: {v}")
    log(f"events_csv: {events_path}")
    log(f"summary: {outdir / 'summary.json'}")
    if zip_path:
        log(f"review_bundle: {zip_path}")
        log(f"review_index: {outdir / 'charts' / 'index.html'}")
    log("NEXT GATE: manually review charts. Do NOT interpret PnL until visual parity is acceptable.")


if __name__ == "__main__":
    main()
