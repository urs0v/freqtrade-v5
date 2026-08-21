#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc
from audit_digash_fidelity_v4 import load_published_tf

MATCH_BPS = (10.0, 25.0, 50.0, 100.0)


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.1: null-controlled level fidelity + alert timing")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v4dir", default="/freqtrade/user_data/digash_fidelity_v4")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v41")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--warmup-days", type=int, default=120)
    p.add_argument("--band-pct", type=float, default=0.10, help="near-market band for chance-coverage baseline, e.g. 0.10 = +/-10%%")
    return p.parse_args()


def _merge_union_fraction(prices: list[float], lo: float, hi: float, bps: float) -> float:
    if not prices or not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.nan
    eps = bps / 10000.0
    intervals = []
    for p in prices:
        a = max(lo, p * (1.0 - eps))
        b = min(hi, p * (1.0 + eps))
        if b > a:
            intervals.append((a, b))
    if not intervals:
        return 0.0
    intervals.sort()
    total = 0.0
    ca, cb = intervals[0]
    for a, b in intervals[1:]:
        if a <= cb:
            cb = max(cb, b)
        else:
            total += cb - ca
            ca, cb = a, b
    total += cb - ca
    return float(total / (hi - lo))


def _nearest(levels: list[dc.Level], t: pd.Timestamp, p: float):
    best = None
    for lv in levels:
        if pd.Timestamp(lv.formed_time) > t:
            continue
        e = abs(float(lv.price) / p - 1.0) * 10000.0
        if best is None or e < best[0]:
            best = (e, float(lv.price), lv.kind, pd.Timestamp(lv.formed_time))
    return best


def _any_range_touch(high: np.ndarray, low: np.ndarray, p: float, a: int, b: int) -> bool:
    a = max(0, a); b = min(len(high) - 1, b)
    if b < a:
        return False
    return bool(np.any((low[a:b+1] <= p) & (high[a:b+1] >= p)))


def _any_close_cross(close: np.ndarray, p: float, a: int, b: int) -> bool:
    a = max(1, a); b = min(len(close) - 1, b)
    if b < a:
        return False
    prev = close[a-1:b]
    cur = close[a:b+1]
    return bool(np.any(((prev <= p) & (cur > p)) | ((prev >= p) & (cur < p))))


def audit_group(records: list[dict], config_path: str, datadir_s: str, warmup_days: int, band_pct: float):
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    pair = records[0]["pair"]
    tf = records[0]["tf"]
    dc.TF_MINUTES.setdefault("1m", 1)
    try:
        x, source = load_published_tf(config, datadir, pair, tf)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_DATA", "elapsed_s": time.monotonic()-t0}
        times = pd.to_datetime([r["post_time"] for r in records], utc=True)
        lo_t = times.min() - pd.Timedelta(days=warmup_days)
        hi_t = times.max() + pd.Timedelta(days=2)
        x = x[(x.date >= lo_t) & (x.date < hi_t)].reset_index(drop=True)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_RANGE", "elapsed_s": time.monotonic()-t0}

        p20 = dc.build_levels(x, tf, 20, 0)
        p30 = dc.build_levels(x, tf, 30, len(p20))
        all_levels = p20 + p30
        sig = pd.to_datetime(x.signal_time, utc=True)
        sig_ns = sig.astype("int64").to_numpy()
        high = x.high.to_numpy(float); low = x.low.to_numpy(float); close = x.close.to_numpy(float)
        rows = []
        for r in records:
            t = pd.Timestamp(r["post_time"])
            pub = float(r["published_level"])
            idx = int(np.searchsorted(sig_ns, t.value, side="right") - 1)
            if idx < 1 or idx >= len(x):
                continue
            c0 = float(close[idx])
            if not np.isfinite(c0) or c0 <= 0:
                continue

            best = _nearest(all_levels, t, pub)
            if best is None:
                nearest_bps, nearest_price, nearest_kind, nearest_formed = np.nan, np.nan, "", pd.NaT
            else:
                nearest_bps, nearest_price, nearest_kind, nearest_formed = best

            causal_prices = [float(lv.price) for lv in all_levels if pd.Timestamp(lv.formed_time) <= t]
            band_lo = c0 * (1.0 - band_pct)
            band_hi = c0 * (1.0 + band_pct)
            in_band = [p for p in causal_prices if band_lo <= p <= band_hi]
            uniq = sorted(set(round(p, 12) for p in in_band))
            if len(uniq) >= 2:
                arr = np.asarray(uniq, float)
                spacing = np.diff(arr) / arr[:-1] * 10000.0
                median_spacing_bps = float(np.median(spacing)) if len(spacing) else np.nan
            else:
                median_spacing_bps = np.nan

            z = dict(r)
            z.update({
                "data_source": source,
                "post_close": c0,
                "published_distance_bps": (pub / c0 - 1.0) * 10000.0,
                "published_in_baseline_band": bool(band_lo <= pub <= band_hi),
                "nearest_bps": nearest_bps,
                "nearest_price": nearest_price,
                "nearest_kind": nearest_kind,
                "nearest_formed_time": nearest_formed.isoformat() if pd.notna(nearest_formed) else "",
                "causal_levels_total": len(causal_prices),
                "causal_levels_in_band": len(uniq),
                "median_spacing_bps_in_band": median_spacing_bps,
                "touch_prev1": _any_range_touch(high, low, pub, idx, idx),
                "touch_prev3": _any_range_touch(high, low, pub, idx-2, idx),
                "touch_next1": _any_range_touch(high, low, pub, idx+1, idx+1),
                "touch_next3": _any_range_touch(high, low, pub, idx+1, idx+3),
                "touch_next6": _any_range_touch(high, low, pub, idx+1, idx+6),
                "cross_prev1": _any_close_cross(close, pub, idx, idx),
                "cross_prev3": _any_close_cross(close, pub, idx-2, idx),
                "cross_next1": _any_close_cross(close, pub, idx+1, idx+1),
                "cross_next3": _any_close_cross(close, pub, idx+1, idx+3),
                "cross_next6": _any_close_cross(close, pub, idx+1, idx+6),
            })
            for b in MATCH_BPS:
                z[f"actual_{int(b)}bps"] = bool(np.isfinite(nearest_bps) and nearest_bps <= b)
                z[f"baseline_{int(b)}bps"] = _merge_union_fraction(uniq, band_lo, band_hi, b)
            rows.append(z)
        return rows, {
            "pair": pair, "tf": tf, "status": "OK", "rows": len(rows), "bars": len(x),
            "levels20": len(p20), "levels30": len(p30), "elapsed_s": time.monotonic()-t0,
        }
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic()-t0}


def _pct(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").mean() * 100.0) if len(s) else np.nan


def report_block(name: str, g: pd.DataFrame):
    print(f"\n{name} N={len(g)}", flush=True)
    inb = g[g.published_in_baseline_band.astype(bool)]
    print(f"baseline-band coverage: {len(inb)}/{len(g)} levels", flush=True)
    for b in MATCH_BPS:
        actual = _pct(g[f"actual_{int(b)}bps"])
        base = float(pd.to_numeric(inb[f"baseline_{int(b)}bps"], errors="coerce").mean() * 100.0) if len(inb) else np.nan
        lift = actual / base if np.isfinite(actual) and np.isfinite(base) and base > 0 else np.nan
        print(f"<={int(b):3d}bps actual={actual:5.1f}% | chance-coverage={base:5.1f}% | lift={lift:4.2f}x", flush=True)
    print(
        f"levels in +/-band median={pd.to_numeric(g.causal_levels_in_band, errors='coerce').median():.0f} | "
        f"median spacing={pd.to_numeric(g.median_spacing_bps_in_band, errors='coerce').median():.1f}bps",
        flush=True,
    )


def main() -> int:
    a = parse_args()
    v4dir = Path(a.v4dir); outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    parsed_path = v4dir / "digash_public_breakouts.csv"
    coverage_path = v4dir / "cache_coverage.csv"
    if not parsed_path.exists() or not coverage_path.exists():
        raise FileNotFoundError("Run Digash Fidelity V4 first; V4.1 consumes its frozen public-source snapshot and cache coverage.")
    parsed = pd.read_csv(parsed_path)
    parsed["post_time"] = pd.to_datetime(parsed.post_time, utc=True, errors="coerce")
    cov = pd.read_csv(coverage_path)
    ok = cov[cov.status.eq("OK")][["pair", "tf"]].drop_duplicates()
    work = parsed.merge(ok, on=["pair", "tf"], how="inner")
    groups = [g.to_dict("records") for _, g in work.groupby(["pair", "tf"], sort=True)]

    print("=== DIGASH FIDELITY V4.1 — NULL-CONTROL + ALERT TIMING ===", flush=True)
    print("NO PnL. NO detector tuning. NO downloads.", flush=True)
    print("Purpose: verify that V4 level matches beat chance from detector level density, then locate the public alert relative to the actual breakout/touch.", flush=True)
    print(f"covered source levels={len(work)} | pair×TF groups={len(groups)} | baseline band=+/-{a.band_pct*100:.1f}% | workers={a.workers}", flush=True)

    results = []; metas = []; t0 = time.monotonic(); done = 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(audit_group, recs, a.config, a.datadir, a.warmup_days, a.band_pct) for recs in groups]
        for f in as_completed(futs):
            rows, meta = f.result(); done += 1; results.extend(rows); metas.append(meta)
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(futs) - done) / done if done else np.nan
            print(f"V4.1 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} rows={len(rows)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(outdir / "coverage.csv", index=False)
    z = pd.DataFrame(results)
    z.to_csv(outdir / "fidelity_rows.csv", index=False)
    if z.empty:
        print("No covered rows; cannot diagnose fidelity.", flush=True)
        return 2

    print("\n=== CHANCE-CONTROLLED LEVEL FIDELITY ===", flush=True)
    report_block("ALL", z)
    lowtf = z[z.tf.isin(["1m", "5m", "15m"])]
    if not lowtf.empty:
        report_block("LOW_TF 1m+5m+15m", lowtf)
    for tf in ["1m", "5m", "15m", "1h", "4h"]:
        g = z[z.tf.eq(tf)]
        if not g.empty:
            report_block(tf, g)

    print("\n=== PUBLIC ALERT MARKET GEOMETRY ===", flush=True)
    for label, g in [("ALL", z)] + [(tf, z[z.tf.eq(tf)]) for tf in ["1m", "5m", "15m", "1h", "4h"] if (z.tf == tf).any()]:
        d = pd.to_numeric(g.published_distance_bps, errors="coerce").abs()
        print(
            f"{label:4s} N={len(g):3d} | abs(level-close) med={d.median():6.1f}bps p75={d.quantile(.75):6.1f} | "
            f"touch prev1={_pct(g.touch_prev1):5.1f}% next1={_pct(g.touch_next1):5.1f}% next3={_pct(g.touch_next3):5.1f}% next6={_pct(g.touch_next6):5.1f}%",
            flush=True,
        )

    print("\n=== CLOSE-CROSS TIMING AROUND PUBLIC ALERT ===", flush=True)
    for label, g in [("ALL", z)] + [(tf, z[z.tf.eq(tf)]) for tf in ["1m", "5m", "15m", "1h", "4h"] if (z.tf == tf).any()]:
        print(
            f"{label:4s} N={len(g):3d} | cross prev1={_pct(g.cross_prev1):5.1f}% prev3={_pct(g.cross_prev3):5.1f}% | "
            f"next1={_pct(g.cross_next1):5.1f}% next3={_pct(g.cross_next3):5.1f}% next6={_pct(g.cross_next6):5.1f}%",
            flush=True,
        )

    print("\n=== DECISION RULE ===", flush=True)
    print("If low-TF match lift over chance is strong, stop modifying the low-TF level detector: the missing Digash fidelity is selection/timing/market context, not level placement.", flush=True)
    print("If lift is near 1x, V4 was mostly a dense-level coincidence and level reconstruction must be fixed before formation logic.", flush=True)
    print("4h is reported separately because V4 already showed materially worse absolute reconstruction there.", flush=True)
    print(f"Reports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
