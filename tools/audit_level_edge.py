#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

LEVEL_SPAN = 2
LEVEL_MAX_AGE_H = 72
ZONE_ATR = 0.15
BREAK_ATR = 0.10
STOP_ATR = 0.25
RETEST_BARS = 6
MAX_HOLD_BARS = 24
R_TARGETS = (1.0, 1.5, 2.0)
COSTS_BPS = (8.0, 12.0)
MIN_QUOTE_VOL_24H = 10_000_000.0


@dataclass
class Event:
    pair: str
    setup: str
    level_kind: str
    level_price: float
    level_time: pd.Timestamp
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    side: int
    entry: float
    stop: float
    active: bool
    compression: bool
    quote_vol_24h: float
    atr_pct: float


def parse_args():
    p = argparse.ArgumentParser(description="Causal horizontal-level bounce and break-retest audit")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/level_edge_audit")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    return p.parse_args()


def as_ns(s):
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def load_tf(config: dict, datadir: Path, pair: str, timeframe: str):
    return load_pair_history(
        pair=pair,
        timeframe=timeframe,
        datadir=datadir,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"),
        candle_type=CandleType.FUTURES,
    )


def prepare_15m(df: pd.DataFrame):
    x = df[["date","open","high","low","close","volume"]].copy()
    x["date"] = as_ns(x["date"])
    x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    prev = x["close"].shift()
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    x["atr"] = tr.rolling(14, min_periods=14).mean()
    x["atr_pct"] = x["atr"] / x["close"]
    x["quote"] = x["volume"] * x["close"]
    x["quote_vol_24h"] = x["quote"].rolling(96, min_periods=48).sum()
    x["atr_pct_med30d"] = x["atr_pct"].rolling(96*30, min_periods=96*7).median()
    x["quote_med30d"] = x["quote_vol_24h"].rolling(96*30, min_periods=96*7).median()
    return x


def make_5m(config: dict, datadir: Path, pair: str):
    d5 = load_tf(config, datadir, pair, "5m")
    if not d5.empty:
        x = d5[["date","open","high","low","close","volume"]].copy()
        x["date"] = as_ns(x["date"])
        return x.sort_values("date").drop_duplicates("date").reset_index(drop=True), "5m"

    d1 = load_tf(config, datadir, pair, "1m")
    if d1.empty:
        return pd.DataFrame(), "none"
    x = d1[["date","open","high","low","close","volume"]].copy()
    x["date"] = as_ns(x["date"])
    x = x.set_index("date").sort_index()
    y = x.resample("5min", label="left", closed="left").agg(
        open=("open","first"),
        high=("high","max"),
        low=("low","min"),
        close=("close","last"),
        volume=("volume","sum"),
    ).dropna()
    return y.reset_index(), "1m->5m"


def confirmed_levels(x15: pd.DataFrame):
    h = x15["high"].to_numpy(float)
    l = x15["low"].to_numpy(float)
    rows = []
    s = LEVEL_SPAN
    for i in range(s, len(x15)-s):
        winh = h[i-s:i+s+1]
        winl = l[i-s:i+s+1]
        known_idx = i + s
        known_time = pd.Timestamp(x15.iloc[known_idx]["date"])
        if np.isfinite(h[i]) and h[i] >= np.nanmax(winh):
            rows.append(("R", float(h[i]), pd.Timestamp(x15.iloc[i]["date"]), known_time))
        if np.isfinite(l[i]) and l[i] <= np.nanmin(winl):
            rows.append(("S", float(l[i]), pd.Timestamp(x15.iloc[i]["date"]), known_time))
    return rows


def merge_context(x5: pd.DataFrame, x15: pd.DataFrame):
    ctx = x15[["date","atr","atr_pct","quote_vol_24h","atr_pct_med30d","quote_med30d"]].copy()
    ctx["available"] = ctx["date"] + pd.Timedelta(minutes=15)
    y = x5.sort_values("date").copy()
    y["signal_time"] = y["date"] + pd.Timedelta(minutes=5)
    y = pd.merge_asof(
        y.sort_values("signal_time"),
        ctx.sort_values("available"),
        left_on="signal_time",
        right_on="available",
        direction="backward",
        tolerance=pd.Timedelta("30min"),
        suffixes=("","_15"),
    )
    return y.drop(columns=[c for c in ["date_15","available"] if c in y.columns])


def is_active(row):
    q = float(row.quote_vol_24h) if np.isfinite(row.quote_vol_24h) else 0.0
    ap = float(row.atr_pct) if np.isfinite(row.atr_pct) else np.nan
    qm = float(row.quote_med30d) if np.isfinite(row.quote_med30d) else np.nan
    am = float(row.atr_pct_med30d) if np.isfinite(row.atr_pct_med30d) else np.nan
    return (
        q >= MIN_QUOTE_VOL_24H
        and np.isfinite(ap) and np.isfinite(am) and ap >= am
        and np.isfinite(qm) and q >= qm
    )


def compression_before(x5: pd.DataFrame, i: int, level: float, side: int):
    if i < 18:
        return False
    recent = x5.iloc[i-6:i]
    prior = x5.iloc[i-18:i-6]
    rr = (recent["high"] - recent["low"]).mean()
    pr = (prior["high"] - prior["low"]).mean()
    if not np.isfinite(rr) or not np.isfinite(pr) or pr <= 0 or rr > 0.85 * pr:
        return False
    c = recent["close"].to_numpy(float)
    d = (level - c) * side
    return np.isfinite(d).all() and d[-1] < d[0] and np.median(np.diff(d)) <= 0


def first_trade_at_level(pair, kind, level, level_time, known_time, x5):
    start = int(x5["signal_time"].searchsorted(known_time, side="left"))
    end_time = known_time + pd.Timedelta(hours=LEVEL_MAX_AGE_H)
    end = int(x5["signal_time"].searchsorted(end_time, side="right"))
    if start >= len(x5):
        return []
    end = min(end, len(x5))
    if end - start < 3:
        return []

    events = []
    broken_at = None
    break_side = None

    for i in range(start, end-1):
        r = x5.iloc[i]
        if not np.isfinite(r.atr) or r.atr <= 0:
            continue
        atr = float(r.atr)
        zone = ZONE_ATR * atr
        br = BREAK_ATR * atr
        px = float(level)

        if kind == "R":
            if float(r.close) > px + br and float(r.open) <= px + zone:
                broken_at, break_side = i, +1
                break
            if float(r.high) >= px - zone and float(r.close) < px:
                j = i + 1
                entry = float(x5.iloc[j].open)
                stop = max(px + STOP_ATR*atr, float(r.high) + 0.05*atr)
                if stop > entry:
                    events.append(Event(
                        pair, "BOUNCE", kind, px, level_time, pd.Timestamp(r.signal_time),
                        pd.Timestamp(x5.iloc[j].date), -1, entry, stop, is_active(r), False,
                        float(r.quote_vol_24h), float(r.atr_pct)
                    ))
                return events
            if float(r.close) > px + 0.5*atr:
                return []

        else:
            if float(r.close) < px - br and float(r.open) >= px - zone:
                broken_at, break_side = i, -1
                break
            if float(r.low) <= px + zone and float(r.close) > px:
                j = i + 1
                entry = float(x5.iloc[j].open)
                stop = min(px - STOP_ATR*atr, float(r.low) - 0.05*atr)
                if stop < entry:
                    events.append(Event(
                        pair, "BOUNCE", kind, px, level_time, pd.Timestamp(r.signal_time),
                        pd.Timestamp(x5.iloc[j].date), +1, entry, stop, is_active(r), False,
                        float(r.quote_vol_24h), float(r.atr_pct)
                    ))
                return events
            if float(r.close) < px - 0.5*atr:
                return []

    if broken_at is None:
        return []

    i = broken_at
    comp = compression_before(x5, i, px, +1 if break_side == +1 else -1)
    for k in range(i+1, min(i+1+RETEST_BARS, end-1)):
        r = x5.iloc[k]
        if not np.isfinite(r.atr) or r.atr <= 0:
            continue
        atr = float(r.atr)
        zone = ZONE_ATR * atr
        j = k + 1
        if break_side == +1:
            if float(r.low) <= px + zone and float(r.close) > px:
                entry = float(x5.iloc[j].open)
                stop = min(px - STOP_ATR*atr, float(r.low) - 0.05*atr)
                if stop < entry:
                    events.append(Event(
                        pair, "BREAK_RETEST", kind, px, level_time, pd.Timestamp(r.signal_time),
                        pd.Timestamp(x5.iloc[j].date), +1, entry, stop, is_active(r), comp,
                        float(r.quote_vol_24h), float(r.atr_pct)
                    ))
                return events
            if float(r.close) < px - 0.5*atr:
                return []
        else:
            if float(r.high) >= px - zone and float(r.close) < px:
                entry = float(x5.iloc[j].open)
                stop = max(px + STOP_ATR*atr, float(r.high) + 0.05*atr)
                if stop > entry:
                    events.append(Event(
                        pair, "BREAK_RETEST", kind, px, level_time, pd.Timestamp(r.signal_time),
                        pd.Timestamp(x5.iloc[j].date), -1, entry, stop, is_active(r), comp,
                        float(r.quote_vol_24h), float(r.atr_pct)
                    ))
                return events
            if float(r.close) > px + 0.5*atr:
                return []
    return []


def simulate_event(ev: Event, x5: pd.DataFrame):
    start = int(x5["date"].searchsorted(ev.entry_time, side="left"))
    if start >= len(x5):
        return []
    risk = abs(ev.entry - ev.stop)
    if not np.isfinite(risk) or risk <= 0:
        return []

    out = []
    for rt in R_TARGETS:
        target = ev.entry + ev.side * rt * risk
        exit_px = None
        reason = "TIME"
        end = min(start + MAX_HOLD_BARS, len(x5))
        last_close = ev.entry
        for i in range(start, end):
            r = x5.iloc[i]
            hi, lo = float(r.high), float(r.low)
            last_close = float(r.close)
            stop_hit = lo <= ev.stop if ev.side > 0 else hi >= ev.stop
            tp_hit = hi >= target if ev.side > 0 else lo <= target
            if stop_hit:
                exit_px = ev.stop
                reason = "SL"
                break
            if tp_hit:
                exit_px = target
                reason = "TP"
                break
        if exit_px is None:
            exit_px = last_close

        gross = ev.side * (exit_px / ev.entry - 1.0)
        risk_frac = risk / ev.entry
        row = {
            "pair": ev.pair,
            "setup": ev.setup,
            "level_kind": ev.level_kind,
            "level_time": ev.level_time,
            "signal_time": ev.signal_time,
            "entry_time": ev.entry_time,
            "side": ev.side,
            "entry": ev.entry,
            "stop": ev.stop,
            "risk_pct": risk_frac * 100.0,
            "active": ev.active,
            "compression": ev.compression,
            "quote_vol_24h": ev.quote_vol_24h,
            "atr_pct": ev.atr_pct,
            "target_r": rt,
            "exit_reason": reason,
            "gross_ret": gross,
            "gross_bps": gross * 10000.0,
            "gross_r": gross / risk_frac if risk_frac > 0 else np.nan,
        }
        for c in COSTS_BPS:
            net = gross - c/10000.0
            row[f"net{int(c)}_bps"] = net * 10000.0
            row[f"net{int(c)}_r"] = net / risk_frac if risk_frac > 0 else np.nan
        out.append(row)
    return out


def summarize(df: pd.DataFrame, mask_name: str, mask):
    x = df.loc[mask].copy()
    rows = []
    if x.empty:
        return rows
    for setup in ["BOUNCE","BREAK_RETEST"]:
        for rt in R_TARGETS:
            z = x[(x.setup == setup) & (x.target_r == rt)]
            if z.empty:
                continue
            for c in COSTS_BPS:
                netcol = f"net{int(c)}_r"
                rows.append({
                    "subset": mask_name,
                    "setup": setup,
                    "target_r": rt,
                    "cost_bps": c,
                    "n": len(z),
                    "win_pct": float((z[netcol] > 0).mean()*100),
                    "mean_gross_bps": float(z.gross_bps.mean()),
                    "mean_net_r": float(z[netcol].mean()),
                    "median_net_r": float(z[netcol].median()),
                    "profit_factor_r": (
                        float(z.loc[z[netcol] > 0, netcol].sum() / -z.loc[z[netcol] < 0, netcol].sum())
                        if (z[netcol] < 0).any() else math.inf
                    ),
                    "positive_years": int((z.assign(year=z.entry_time.dt.year)
                                           .groupby("year")[netcol].mean() > 0).sum()),
                    "years": int(z.entry_time.dt.year.nunique()),
                })
    return rows


def main():
    cfg = parse_args()
    config = json.loads(Path(cfg.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair whitelist in config")

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    datadir = Path(cfg.datadir)
    outdir = Path(cfg.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("=== CAUSAL LEVEL EDGE AUDIT: BOUNCE vs BREAK->RETEST ===")
    print("15m confirmed swing levels; execution on existing 5m data (or 1m aggregated to 5m).")
    print("One first interaction per level. No future-confirmed levels. No parameter optimization.")
    print("Active subset: 24h quote volume >= $10m AND quote-volume & ATR% >= own trailing-30d median.")
    print("Breakout compression: last 30m ranges <=85% of prior hour and distance to level contracts.")
    print("Stops live beyond level/signal extreme. Targets: 1R/1.5R/2R. Max hold: 2h.")
    print("Intrabar ambiguity is conservative: SL before TP. Costs: 8/12 bps round trip.")
    print(f"Range: {cfg.start} .. {cfg.end} | pairs={len(pairs)}\n")

    all_rows = []
    coverage = []
    for pi, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        d15 = load_tf(config, datadir, pair, "15m")
        if d15.empty:
            print(f"[{pi:02d}/{len(pairs)}] {pair}: NO 15m")
            continue
        x15 = prepare_15m(d15)
        x5, source = make_5m(config, datadir, pair)
        if x5.empty:
            print(f"[{pi:02d}/{len(pairs)}] {pair}: NO 5m/1m detail")
            coverage.append({"pair": pair, "detail": "none", "events": 0})
            continue

        x5["date"] = as_ns(x5["date"])
        x5 = x5[(x5.date >= start - pd.Timedelta(days=35)) & (x5.date <= end + pd.Timedelta(hours=3))].copy()
        x15 = x15[(x15.date >= start - pd.Timedelta(days=35)) & (x15.date <= end + pd.Timedelta(hours=3))].copy()
        if x5.empty or x15.empty:
            continue
        x5 = merge_context(x5, x15).sort_values("date").reset_index(drop=True)
        levels = confirmed_levels(x15)

        pair_events = []
        for kind, level, level_time, known_time in levels:
            if known_time < start or known_time > end:
                continue
            pair_events.extend(first_trade_at_level(pair, kind, level, level_time, known_time, x5))

        pair_events.sort(key=lambda e: e.entry_time)
        dedup = []
        last_key = {}
        for ev in pair_events:
            key = (ev.setup, ev.side)
            prev = last_key.get(key)
            if prev is not None and ev.entry_time - prev < pd.Timedelta(minutes=15):
                continue
            dedup.append(ev)
            last_key[key] = ev.entry_time

        for ev in dedup:
            if start <= ev.entry_time <= end:
                all_rows.extend(simulate_event(ev, x5))

        coverage.append({"pair": pair, "detail": source, "events": len(dedup)})
        print(f"[{pi:02d}/{len(pairs)}] {pair}: detail={source}, levels={len(levels)}, events={len(dedup)} [{time.monotonic()-t0:.1f}s]", flush=True)

    trades = pd.DataFrame(all_rows)
    pd.DataFrame(coverage).to_csv(outdir / "coverage.csv", index=False)
    if trades.empty:
        raise RuntimeError("No events found. Check detail data coverage.")

    trades["entry_time"] = pd.to_datetime(trades["entry_time"], utc=True)
    trades["signal_time"] = pd.to_datetime(trades["signal_time"], utc=True)
    trades["level_time"] = pd.to_datetime(trades["level_time"], utc=True)
    trades["year"] = trades.entry_time.dt.year
    trades["month"] = trades.entry_time.dt.strftime("%Y-%m")
    trades.to_csv(outdir / "trades.csv", index=False)

    rows = []
    rows += summarize(trades, "ALL", np.ones(len(trades), dtype=bool))
    rows += summarize(trades, "ACTIVE", trades.active)
    rows += summarize(trades, "ACTIVE_COMP", trades.active & ((trades.setup=="BOUNCE") | trades.compression))
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary.csv", index=False)

    yr = []
    for setup in ["BOUNCE","BREAK_RETEST"]:
        for rt in R_TARGETS:
            z = trades[(trades.active) & (trades.setup==setup) & (trades.target_r==rt)]
            for y, g in z.groupby("year"):
                for c in COSTS_BPS:
                    col = f"net{int(c)}_r"
                    yr.append({
                        "setup": setup, "target_r": rt, "year": int(y), "cost_bps": c,
                        "n": len(g), "win_pct": float((g[col]>0).mean()*100),
                        "mean_net_r": float(g[col].mean()),
                        "mean_net_bps": float((g.gross_ret.mean()-c/10000.0)*10000.0),
                    })
    yearly = pd.DataFrame(yr)
    yearly.to_csv(outdir / "yearly_active.csv", index=False)

    print("\n=== SUMMARY: ACTIVE SUBSET / 8 BPS ===")
    v = summary[(summary.subset=="ACTIVE") & (summary.cost_bps==8.0)]
    for r0 in v.itertuples(index=False):
        print(
            f"{r0.setup:12s} {r0.target_r:>3.1f}R | N={r0.n:5d} "
            f"WR={r0.win_pct:5.1f}% gross={r0.mean_gross_bps:+7.2f}bps "
            f"net={r0.mean_net_r:+.3f}R PF={r0.profit_factor_r:.2f} "
            f"positive_years={r0.positive_years}/{r0.years}"
        )

    print("\n=== SUMMARY: ACTIVE + COMPRESSION FOR BREAKOUTS / 8 BPS ===")
    v = summary[(summary.subset=="ACTIVE_COMP") & (summary.cost_bps==8.0)]
    for r0 in v.itertuples(index=False):
        print(
            f"{r0.setup:12s} {r0.target_r:>3.1f}R | N={r0.n:5d} "
            f"WR={r0.win_pct:5.1f}% gross={r0.mean_gross_bps:+7.2f}bps "
            f"net={r0.mean_net_r:+.3f}R PF={r0.profit_factor_r:.2f} "
            f"positive_years={r0.positive_years}/{r0.years}"
        )

    print("\n=== YEARLY ACTIVE / 1.5R / 8 BPS ===")
    yv = yearly[(yearly.target_r==1.5) & (yearly.cost_bps==8.0)]
    for r0 in yv.itertuples(index=False):
        print(f"{r0.setup:12s} {r0.year}: N={r0.n:4d} WR={r0.win_pct:5.1f}% net={r0.mean_net_r:+.3f}R ({r0.mean_net_bps:+.2f}bps)")

    print(f"\nReports: {outdir}")
    print("Interpretation: a real candidate should survive 8bps, multiple R targets, and multiple years; 2026 must not be the only profitable period.")


if __name__ == "__main__":
    raise SystemExit(main())
