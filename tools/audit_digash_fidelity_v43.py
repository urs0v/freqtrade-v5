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
from audit_digash_fidelity_v4 import load_published_tf
from audit_digash_fidelity_v41 import _merge_union_fraction

MATCH_BPS = (10.0, 25.0, 50.0, 100.0)


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.3: touch-rearmed level lifecycle")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v4dir", default="/freqtrade/user_data/digash_fidelity_v4")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v43")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--warmup-days", type=int, default=120)
    p.add_argument("--band-pct", type=float, default=0.10)
    return p.parse_args()


def _first_break_after(lv, touch_ns, close, sig_ns):
    start = int(np.searchsorted(sig_ns, touch_ns, side="right"))
    if start >= len(close):
        return np.iinfo(np.int64).max
    q = np.flatnonzero(close[start:] > float(lv.price)) if lv.kind == "R" else np.flatnonzero(close[start:] < float(lv.price))
    if len(q) == 0:
        return np.iinfo(np.int64).max
    return int(sig_ns[start + int(q[0])])


def _touch_schedule(lv, pivots, sig_ns, close):
    touches = [(pd.Timestamp(lv.formed_time).value, int(lv.touch_idx))]
    last_idx = int(lv.touch_idx)
    for p in pivots:
        if p["kind"] != lv.kind or int(p["idx"]) <= last_idx:
            continue
        if int(p["idx"]) - last_idx < int(lv.period):
            continue
        err = abs(float(p["price"]) - float(lv.price)) / max(abs(float(lv.price)), 1e-12)
        if err <= dc.TOUCH_TOL_PCT:
            touches.append((pd.Timestamp(p["formed"]).value, int(p["idx"])))
            last_idx = int(p["idx"])
    return [(int(t_ns), idx, _first_break_after(lv, int(t_ns), close, sig_ns)) for t_ns, idx in touches]


def _nearest(levels, pub):
    if not levels:
        return np.nan, np.nan, ""
    best = min(levels, key=lambda lv: abs(float(lv.price) / pub - 1.0))
    return abs(float(best.price) / pub - 1.0) * 10000.0, float(best.price), best.kind


def _prices(levels, lo, hi):
    return sorted(set(round(float(lv.price), 12) for lv in levels if lo <= float(lv.price) <= hi))


def _active_rearmed(schedule, post_ns):
    prior = [q for q in schedule if q[0] <= post_ns]
    return bool(prior and prior[-1][2] > post_ns)


def audit_group(records, config_path, datadir_s, warmup_days, band_pct):
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    pair, tf = records[0]["pair"], records[0]["tf"]
    dc.TF_MINUTES.setdefault("1m", 1)
    try:
        x, source = load_published_tf(config, datadir, pair, tf)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_DATA", "elapsed_s": time.monotonic()-t0}
        times = pd.to_datetime([r["post_time"] for r in records], utc=True)
        x = x[(x.date >= times.min()-pd.Timedelta(days=warmup_days)) & (x.date < times.max()+pd.Timedelta(days=2))].reset_index(drop=True)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_RANGE", "elapsed_s": time.monotonic()-t0}

        p20 = dc.build_levels(x, tf, 20, 0)
        p30 = dc.build_levels(x, tf, 30, len(p20))
        levels = p20 + p30
        pivots = dc.local_pivots(x)
        sig_ns = pd.to_datetime(x.signal_time, utc=True).astype("int64").to_numpy()
        close = x.close.to_numpy(float)
        schedules = {lv.level_id: _touch_schedule(lv, pivots, sig_ns, close) for lv in levels}

        rows = []
        for r in records:
            t = pd.Timestamp(r["post_time"]); post_ns = int(t.value); pub = float(r["published_level"])
            idx = int(np.searchsorted(sig_ns, post_ns, side="right") - 1)
            if idx < 1 or idx >= len(close) or not np.isfinite(close[idx]) or close[idx] <= 0:
                continue
            c0 = float(close[idx])
            expected_kind = "R" if pub >= c0 else "S"
            formed = [lv for lv in levels if pd.Timestamp(lv.formed_time).value <= post_ns]
            rearm = [lv for lv in formed if _active_rearmed(schedules[lv.level_id], post_ns)]
            rearm_side = [lv for lv in rearm if lv.kind == expected_kind and ((lv.kind == "R" and lv.price >= c0) or (lv.kind == "S" and lv.price <= c0))]
            band_lo, band_hi = c0*(1.0-band_pct), c0*(1.0+band_pct)

            z = dict(r)
            z.update({"data_source": source, "post_close": c0, "expected_kind": expected_kind, "published_in_band": bool(band_lo <= pub <= band_hi)})
            for mode, arr in {"raw": formed, "rearm": rearm, "rearm_side": rearm_side}.items():
                e, nearp, kind = _nearest(arr, pub)
                prices = _prices(arr, band_lo, band_hi)
                z[f"{mode}_nearest_bps"] = e
                z[f"{mode}_nearest_price"] = nearp
                z[f"{mode}_nearest_kind"] = kind
                z[f"{mode}_candidate_n"] = len(prices)
                for b in MATCH_BPS:
                    z[f"{mode}_actual_{int(b)}"] = bool(np.isfinite(e) and e <= b)
                    z[f"{mode}_chance_{int(b)}"] = _merge_union_fraction(prices, band_lo, band_hi, b)

            if formed:
                best_lv = min(formed, key=lambda lv: abs(float(lv.price) / pub - 1.0))
                sched = [q for q in schedules[best_lv.level_id] if q[0] <= post_ns]
                if sched:
                    last_touch_ns, _, last_break_ns = sched[-1]
                    z.update({
                        "raw_match_error_bps": abs(float(best_lv.price) / pub - 1.0) * 10000.0,
                        "raw_match_period": int(best_lv.period),
                        "raw_match_kind": best_lv.kind,
                        "raw_match_counted_touches": int(best_lv.counted_touches),
                        "qualifying_touch_count_to_post": len(sched),
                        "prior_break_count": int(sum(1 for q in sched if q[2] <= post_ns)),
                        "last_qualifying_touch_age_h": (post_ns-last_touch_ns)/3.6e12,
                        "rearmed_active_for_raw_match": bool(last_break_ns > post_ns),
                    })
            rows.append(z)

        return rows, {"pair": pair, "tf": tf, "status": "OK", "rows": len(rows), "bars": len(x), "levels": len(levels), "elapsed_s": time.monotonic()-t0}
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic()-t0}


def _pct(s):
    return float(pd.to_numeric(s, errors="coerce").mean() * 100.0) if len(s) else np.nan


def _mode_report(g, mode):
    inb = g[g.published_in_band.astype(bool)]
    cand = pd.to_numeric(g[f"{mode}_candidate_n"], errors="coerce")
    print(f"  {mode:10s} candidates median={cand.median():5.0f} p75={cand.quantile(.75):5.0f}", flush=True)
    for b in MATCH_BPS:
        actual = _pct(g[f"{mode}_actual_{int(b)}"])
        base = float(pd.to_numeric(inb[f"{mode}_chance_{int(b)}"], errors="coerce").mean() * 100.0) if len(inb) else np.nan
        lift = actual/base if np.isfinite(actual) and np.isfinite(base) and base > 0 else np.nan
        print(f"    <={int(b):3d}bps actual={actual:5.1f}% chance={base:5.1f}% lift={lift:4.2f}x", flush=True)


def _lifecycle_report(name, g):
    err = pd.to_numeric(g.get("raw_match_error_bps"), errors="coerce")
    q = g[err <= 25.0].copy()
    print(f"{name:6s} raw<=25bps N={len(q)}", flush=True)
    if q.empty:
        return
    had_break = pd.to_numeric(q.prior_break_count, errors="coerce") > 0
    print(f"  rearmed-active={_pct(q.rearmed_active_for_raw_match):5.1f}% | had prior break={_pct(had_break):5.1f}% | touches median={pd.to_numeric(q.qualifying_touch_count_to_post, errors='coerce').median():.1f} | last-touch age med={pd.to_numeric(q.last_qualifying_touch_age_h, errors='coerce').median():.2f}h", flush=True)


def main():
    a = parse_args(); v4dir = Path(a.v4dir); outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    parsed_path = v4dir / "digash_public_breakouts.csv"; cov_path = v4dir / "cache_coverage.csv"
    if not parsed_path.exists() or not cov_path.exists():
        raise FileNotFoundError("Run Fidelity V4 first; V4.3 consumes its frozen public-source snapshot.")
    parsed = pd.read_csv(parsed_path); parsed["post_time"] = pd.to_datetime(parsed.post_time, utc=True, errors="coerce")
    cov = pd.read_csv(cov_path); ok = cov[cov.status.eq("OK")][["pair", "tf"]].drop_duplicates()
    work = parsed.merge(ok, on=["pair", "tf"], how="inner")
    groups = [g.to_dict("records") for _, g in work.groupby(["pair", "tf"], sort=True)]

    print("=== DIGASH FIDELITY V4.3 — TOUCH-REARMED LEVEL LIFECYCLE ===", flush=True)
    print("NO PnL. NO trading-rule tuning. NO downloads.", flush=True)
    print("V4.2 permanent close-through invalidation had high lift but low recall. V4.3 tests whether later qualifying touches re-arm the same horizontal zone.", flush=True)
    print("Re-arm is a source-fidelity diagnostic proxy, not a claim about Digash's private implementation.", flush=True)
    print(f"covered source rows={len(work)} | groups={len(groups)} | workers={a.workers}", flush=True)

    results=[]; metas=[]; t0=time.monotonic(); done=0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs=[ex.submit(audit_group, recs, a.config, a.datadir, a.warmup_days, a.band_pct) for recs in groups]
        for f in as_completed(futs):
            rows, meta=f.result(); done+=1; results.extend(rows); metas.append(meta)
            elapsed=time.monotonic()-t0; eta=elapsed*(len(futs)-done)/done if done else np.nan
            print(f"V4.3 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} rows={len(rows)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(outdir/"coverage.csv", index=False)
    z=pd.DataFrame(results); z.to_csv(outdir/"rearm_rows.csv", index=False)
    if z.empty:
        print("No rows.", flush=True); return 2

    print("\n=== REARMED CHANCE-CONTROLLED FIDELITY ===", flush=True)
    cohorts=[("ALL",z), ("LOW_TF",z[z.tf.isin(["1m","5m","15m"])])]
    cohorts += [(tf,z[z.tf.eq(tf)]) for tf in ["1m","5m","15m","1h","4h"] if (z.tf==tf).any()]
    for name,g in cohorts:
        print(f"\n{name} N={len(g)}", flush=True)
        for mode in ["raw","rearm","rearm_side"]:
            _mode_report(g,mode)

    print("\n=== WHY V4.2 LOST SOURCE LEVELS ===", flush=True)
    for name,g in cohorts:
        _lifecycle_report(name,g)

    print("\n=== DECISION RULE ===", flush=True)
    print("If re-arm materially restores <=25bps source recall while keeping candidate density low and lift high, permanent invalidation was too strict; carry touch-rearmed levels into formation-selection reconstruction.", flush=True)
    print("If recall stays low, do not tune PnL: reconstruct the horizontal-level formation/period semantics themselves on held-out public alerts.", flush=True)
    print(f"Reports: {outdir}", flush=True); print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
