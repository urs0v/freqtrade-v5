#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc
import digash_v31_events as de
import breakout_retest_profit_v1 as v1

THRESH = 1.5
RISK_MIN_BPS = 160.0
RR = 3.0
HOLD_BARS = 48
BASE_COST_BPS = 8.0
STRESS_COST_BPS = 12.0
WARMUP_DAYS = 60
LEVEL_TFS = ("15m", "1h", "4h")
LEVEL_PERIODS = (20, 30)


def parse_args():
    p = argparse.ArgumentParser(description="Profit V1.6: causal dedup correction for the frozen FAKEOUT signal")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v16")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def _score(e):
    return (int(e.confluence_tfs), int(e.tf_minutes), int(e.period), -float(e.touch_error_pct))


def causal_dedup_events(events):
    """Live-feasible version of the old 3-bar bucket dedup.

    The legacy batch function picks the highest-score event from the whole 3-bar
    bucket, including events that occur on later bars. A live system cannot know
    those later events when the first signal appears. Here the first signal bar in
    each (setup, side, 3-bar bucket) wins; if multiple levels fire on that exact
    bar, choosing the best of those is causal because all are known simultaneously.
    """
    if not events:
        return []
    ordered = sorted(events, key=lambda e: (int(e.signal_idx), str(e.setup), int(e.side)))
    by_idx = {}
    for e in ordered:
        by_idx.setdefault(int(e.signal_idx), []).append(e)

    seen = set()
    out = []
    for idx in sorted(by_idx):
        same_bar = {}
        for e in by_idx[idx]:
            key = (str(e.setup), int(e.side), int(e.signal_idx) // 3)
            if key in seen:
                continue
            old = same_bar.get(key)
            if old is None or _score(e) > _score(old):
                same_bar[key] = e
        for key, e in same_bar.items():
            out.append(e)
            seen.add(key)
    return sorted(out, key=lambda e: (int(e.signal_idx), str(e.setup), int(e.side)))


def _future_replacement_count(raw_events, legacy_events, setup="H_FAKEOUT"):
    earliest = {}
    for e in raw_events:
        if e.setup != setup:
            continue
        key = (str(e.setup), int(e.side), int(e.signal_idx) // 3)
        earliest[key] = min(int(e.signal_idx), earliest.get(key, 10**18))
    n = 0
    lag_bars = []
    for e in legacy_events:
        if e.setup != setup:
            continue
        key = (str(e.setup), int(e.side), int(e.signal_idx) // 3)
        first = earliest.get(key, int(e.signal_idx))
        lag = int(e.signal_idx) - int(first)
        if lag > 0:
            n += 1
            lag_bars.append(lag)
    return n, lag_bars


def _event_rows(pair, mode, events, x5, start, end):
    rows = []
    for e in events:
        if e.setup != "H_FAKEOUT":
            continue
        si = int(e.signal_idx)
        ei = int(e.entry_idx)
        if si < 0 or ei < 0 or si >= len(x5) or ei >= len(x5):
            continue
        et = pd.Timestamp(x5.iloc[ei]["date"])
        if not (start <= et < end):
            continue
        ev = v1._event_dict(e)
        if ev is None:
            continue
        rb, reason_b, risk_bps, _ = v1._simulate_one(x5, ev, RR, HOLD_BARS, BASE_COST_BPS)
        rs, reason_s, _, _ = v1._simulate_one(x5, ev, RR, HOLD_BARS, STRESS_COST_BPS)
        activity = float(x5.iloc[si].get("activity_score", np.nan))
        rows.append({
            "mode": mode,
            "pair": pair,
            "entry_time": et,
            "signal_time": pd.Timestamp(x5.iloc[si]["signal_time"]),
            "signal_idx": si,
            "entry_idx": ei,
            "side": int(e.side),
            "level_id": int(e.level_id),
            "level_price": float(e.level_price),
            "tf": str(e.tf),
            "period": int(e.period),
            "approach_no": int(e.approach_no),
            "confluence_tfs": int(e.confluence_tfs),
            "touch_error_pct": float(e.touch_error_pct),
            "stop": float(e.stop),
            "activity_score": activity,
            "risk_bps": float(risk_bps) if np.isfinite(risk_bps) else np.nan,
            "net8_r": float(rb) if np.isfinite(rb) else np.nan,
            "stress12_r": float(rs) if np.isfinite(rs) else np.nan,
            "exit_reason_8": reason_b,
            "exit_reason_12": reason_s,
        })
    return rows


def process_pair(pair, cfg_path, datadir_s, start_s, end_s):
    t0 = time.monotonic()
    cfg = json.loads(Path(cfg_path).read_text())
    datadir = Path(datadir_s)
    start = pd.Timestamp(start_s, tz="UTC")
    end = pd.Timestamp(end_s, tz="UTC") + pd.Timedelta(days=1)
    warm = pd.Timedelta(days=WARMUP_DAYS)

    raw15 = dc.load_tf(cfg, datadir, pair, "15m")
    raw5, detail_source = dc.load_5m(cfg, datadir, pair)
    if raw15.empty or raw5.empty:
        return [], {"pair": pair, "status": "NO_DATA", "elapsed_s": time.monotonic() - t0}

    x15 = dc.prep_ohlcv(raw15, 15)
    x5 = v1._prep_exec(raw5)
    x15 = x15[(x15.date >= start - warm) & (x15.date < end + pd.Timedelta(hours=16))].reset_index(drop=True)
    x5 = x5[(x5.date >= start - warm) & (x5.date < end + pd.Timedelta(hours=16))].reset_index(drop=True)
    if x15.empty or x5.empty:
        return [], {"pair": pair, "status": "NO_RANGE", "elapsed_s": time.monotonic() - t0}

    # Entry-time causal activity for a signal on bar i is the 15m information
    # available at x5.signal_time[i] == next bar open / intended entry time.
    x5 = v1._add_activity(x5, v1._activity15(x15))

    tfs = {
        "15m": x15,
        "1h": dc.resample_from_15(x15, "1h", 60),
        "4h": dc.resample_from_15(x15, "4h", 240),
    }
    levels = []
    lid = 0
    for tf in LEVEL_TFS:
        for period in LEVEL_PERIODS:
            zz = dc.build_levels(tfs[tf], tf, period, lid)
            levels.extend(zz)
            lid += len(zz)
    if not levels:
        return [], {"pair": pair, "status": "NO_LEVELS", "elapsed_s": time.monotonic() - t0}

    raw_events = de.detect_events(x5, levels)
    legacy_events = de.dedup_events(raw_events)
    causal_events = causal_dedup_events(raw_events)
    future_n, lag_bars = _future_replacement_count(raw_events, legacy_events)

    rows = []
    rows.extend(_event_rows(pair, "LEGACY_DEDUP", legacy_events, x5, start, end))
    rows.extend(_event_rows(pair, "CAUSAL_DEDUP", causal_events, x5, start, end))
    meta = {
        "pair": pair,
        "status": "OK",
        "detail_source": detail_source,
        "bars5": len(x5),
        "bars15": len(x15),
        "levels": len(levels),
        "raw_events": len(raw_events),
        "legacy_events": len(legacy_events),
        "causal_events": len(causal_events),
        "raw_fakeouts": sum(e.setup == "H_FAKEOUT" for e in raw_events),
        "legacy_fakeouts": sum(e.setup == "H_FAKEOUT" for e in legacy_events),
        "causal_fakeouts": sum(e.setup == "H_FAKEOUT" for e in causal_events),
        "legacy_fakeout_future_replacements": int(future_n),
        "replacement_lag_bars_mean": float(np.mean(lag_bars)) if lag_bars else 0.0,
        "replacement_lag_bars_max": int(max(lag_bars)) if lag_bars else 0,
        "elapsed_s": time.monotonic() - t0,
    }
    return rows, meta


def metric(g, col):
    r = pd.to_numeric(g[col], errors="coerce").dropna().astype(float)
    if r.empty:
        return {"N": 0, "PF": np.nan, "WR": np.nan, "EXP": np.nan, "DD": np.nan}
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    curve = r.cumsum()
    dd = float((curve.cummax() - curve).max()) if len(curve) else 0.0
    return {"N": int(len(r)), "PF": float(pf), "WR": float((r > 0).mean() * 100.0), "EXP": float(r.mean()), "DD": dd}


def fmt(m):
    return f"N={m['N']:4d} PF={m['PF']:5.2f} WR={m['WR']:5.1f}% EXP={m['EXP']:+.3f}R DD={m['DD']:6.1f}R"


def selected(g):
    return g[
        (pd.to_numeric(g.activity_score, errors="coerce") >= THRESH)
        & (pd.to_numeric(g.risk_bps, errors="coerce") >= RISK_MIN_BPS)
    ].copy()


def trade_ids(g):
    return set(
        (str(r.pair), pd.Timestamp(r.entry_time).value, int(r.side), int(r.level_id))
        for r in g.itertuples(index=False)
    )


def main():
    a = parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    pairs = list(cfg.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        # Frozen research universe used throughout V1.x.
        pairs = [
            "AAVE/USDT:USDT", "ADA/USDT:USDT", "ATOM/USDT:USDT", "AVAX/USDT:USDT",
            "BCH/USDT:USDT", "BNB/USDT:USDT", "BTC/USDT:USDT", "DOGE/USDT:USDT",
            "DOT/USDT:USDT", "ETC/USDT:USDT", "ETH/USDT:USDT", "FIL/USDT:USDT",
            "LINK/USDT:USDT", "LTC/USDT:USDT", "NEAR/USDT:USDT", "SOL/USDT:USDT",
            "TRX/USDT:USDT", "UNI/USDT:USDT", "XLM/USDT:USDT", "XRP/USDT:USDT",
        ]

    print("=== BREAKOUT / RETEST PROFIT V1.6 — CAUSAL DEDUP AUDIT ===", flush=True)
    print("Frozen FAKEOUT_RISK160P only. Combines the V1.5 causal activity fix with a live-feasible event dedup. No tuning.", flush=True)
    print("Legacy dedup can choose a later event from the same 3-bar bucket and retroactively suppress an earlier event; causal dedup cannot.", flush=True)
    print(f"pairs={len(pairs)} workers={a.workers} threshold={THRESH} risk>={RISK_MIN_BPS:.0f}bps RR={RR:g} hold={HOLD_BARS}x5m", flush=True)

    rows = []
    metas = []
    with ProcessPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
        futs = {ex.submit(process_pair, p, a.config, a.datadir, a.start, a.end): p for p in pairs}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                rr, meta = fut.result()
            except Exception as e:
                rr, meta = [], {"pair": p, "status": f"ERROR:{type(e).__name__}:{e}"}
            rows.extend(rr)
            metas.append(meta)
            done += 1
            print(
                f"pair {done:2d}/{len(pairs)} {p:24s} {meta.get('status')} "
                f"legacyF={meta.get('legacy_fakeouts',0)} causalF={meta.get('causal_fakeouts',0)} "
                f"futureReplace={meta.get('legacy_fakeout_future_replacements',0)} elapsed={meta.get('elapsed_s',0):.1f}s",
                flush=True,
            )

    md = pd.DataFrame(metas)
    md.to_csv(out / "dedup_pair_coverage.csv", index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No V1.6 rows produced")
    df["entry_time"] = pd.to_datetime(df.entry_time, utc=True)
    df["signal_time"] = pd.to_datetime(df.signal_time, utc=True)
    bounds = v1._split_bounds(a.start, a.end)
    df["split"] = v1._split_name(df.entry_time, bounds)
    df.to_csv(out / "dedup_all_fakeouts.csv", index=False)

    legacy = selected(df[df.mode.eq("LEGACY_DEDUP")].sort_values("entry_time"))
    causal = selected(df[df.mode.eq("CAUSAL_DEDUP")].sort_values("entry_time"))
    legacy.to_csv(out / "legacy_selected.csv", index=False)
    causal.to_csv(out / "causal_selected.csv", index=False)

    print("\n=== DEDUP LOOKAHEAD SANITY ===")
    okm = md[md.status.eq("OK")].copy() if "status" in md else md.iloc[0:0]
    future_total = int(pd.to_numeric(okm.get("legacy_fakeout_future_replacements", 0), errors="coerce").fillna(0).sum()) if len(okm) else 0
    legacy_f = int(pd.to_numeric(okm.get("legacy_fakeouts", 0), errors="coerce").fillna(0).sum()) if len(okm) else 0
    causal_f = int(pd.to_numeric(okm.get("causal_fakeouts", 0), errors="coerce").fillna(0).sum()) if len(okm) else 0
    print(f"legacy fakeouts={legacy_f:,} | causal fakeouts={causal_f:,} | legacy selections that came from a later bar in their bucket={future_total:,}")
    if legacy_f:
        print(f"future-replacement share of legacy fakeouts={future_total/legacy_f*100:.2f}%")
    lid = trade_ids(legacy); cid = trade_ids(causal)
    inter = len(lid & cid); union = len(lid | cid)
    print(f"frozen selected overlap={inter} legacy-only={len(lid-cid)} causal-only={len(cid-lid)} Jaccard={(inter/union if union else np.nan):.3f}")

    print("\n=== FROZEN SIGNAL: LEGACY VS FULLY CAUSAL ===")
    report = []
    for split in ("TRAIN", "VALID", "HOLDOUT"):
        lg = legacy[legacy.split.eq(split)]
        cg = causal[causal.split.eq(split)]
        lm = metric(lg, "net8_r"); ls = metric(lg, "stress12_r")
        cm = metric(cg, "net8_r"); cs = metric(cg, "stress12_r")
        print(f"{split:7s} LEGACY {fmt(lm)} stressEXP={ls['EXP']:+.3f}R")
        print(f"{split:7s} CAUSAL {fmt(cm)} stressEXP={cs['EXP']:+.3f}R")
        report += [
            {"split": split, "mode": "LEGACY_DEDUP", **lm, "STRESS_EXP": ls["EXP"]},
            {"split": split, "mode": "CAUSAL_DEDUP", **cm, "STRESS_EXP": cs["EXP"]},
        ]
    pd.DataFrame(report).to_csv(out / "dedup_metrics.csv", index=False)

    print("\n=== FULLY CAUSAL CALENDAR STABILITY (8bps) ===")
    for year, g in causal.groupby(causal.entry_time.dt.year):
        print(f"year {year}: {fmt(metric(g, 'net8_r'))}")
    qq = causal.entry_time.dt.tz_localize(None).dt.to_period("Q").astype(str)
    for q, g in causal.groupby(qq):
        m = metric(g, "net8_r")
        print(f"quarter {q}: N={m['N']:4d} PF={m['PF']:5.2f} EXP={m['EXP']:+.3f}R")

    tr = metric(causal[causal.split.eq("TRAIN")], "net8_r")
    va = metric(causal[causal.split.eq("VALID")], "net8_r")
    ho = metric(causal[causal.split.eq("HOLDOUT")], "net8_r")
    hs = metric(causal[causal.split.eq("HOLDOUT")], "stress12_r")
    survives = (
        tr["N"] >= 50 and tr["EXP"] > 0
        and va["N"] >= 50 and va["EXP"] > 0
        and ho["N"] >= 50 and ho["PF"] > 1 and ho["EXP"] > 0
        and hs["EXP"] >= 0
    )
    print("\n=== V1.6 VERDICT ===")
    print("SURVIVES_CAUSAL_DEDUP_CORRECTION" if survives else "FAILS_CAUSAL_DEDUP_CORRECTION")
    print("This correction is not a new holdout. If it survives, the next evidence must be prospective paper/dry-run data with the causal implementation frozen.")
    print(f"Reports: {out}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
