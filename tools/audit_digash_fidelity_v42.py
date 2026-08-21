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
HORIZONS_H = (1, 4, 24, 72)


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.2: active/unbroken level lifecycle audit")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v4dir", default="/freqtrade/user_data/digash_fidelity_v4")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v42")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--warmup-days", type=int, default=120)
    p.add_argument("--band-pct", type=float, default=0.10)
    return p.parse_args()


def _first_close_invalidation(lv: dc.Level, close: np.ndarray, sig_ns: np.ndarray) -> int:
    """First completed candle AFTER formation that closes through the level in breakout direction."""
    start = int(np.searchsorted(sig_ns, pd.Timestamp(lv.formed_time).value, side="right"))
    if start >= len(close):
        return np.iinfo(np.int64).max
    if lv.kind == "R":
        q = np.flatnonzero(close[start:] > float(lv.price))
    else:
        q = np.flatnonzero(close[start:] < float(lv.price))
    if len(q) == 0:
        return np.iinfo(np.int64).max
    return int(sig_ns[start + int(q[0])])


def _nearest(levels: list[dc.Level], pub: float) -> tuple[float, float, str]:
    if not levels:
        return np.nan, np.nan, ""
    best = min(levels, key=lambda lv: abs(float(lv.price) / pub - 1.0))
    return abs(float(best.price) / pub - 1.0) * 10000.0, float(best.price), best.kind


def _unique_prices(levels: list[dc.Level], lo: float, hi: float) -> list[float]:
    return sorted(set(round(float(lv.price), 12) for lv in levels if lo <= float(lv.price) <= hi))


def _future_event_hours(
    pub: float, expected_kind: str, post_ns: int,
    sig_ns: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray,
    max_h: int = 72,
) -> tuple[float, float]:
    start = int(np.searchsorted(sig_ns, post_ns, side="right"))
    end_ns = post_ns + int(max_h * 3600 * 1e9)
    end = int(np.searchsorted(sig_ns, end_ns, side="right"))
    end = min(end, len(sig_ns))
    if start >= end:
        return np.nan, np.nan

    if expected_kind == "R":
        tq = np.flatnonzero(high[start:end] >= pub)
    else:
        tq = np.flatnonzero(low[start:end] <= pub)
    touch_h = np.nan
    if len(tq):
        j = start + int(tq[0])
        touch_h = (int(sig_ns[j]) - post_ns) / 3.6e12

    cross_h = np.nan
    a = max(1, start)
    if a < end:
        prev = close[a-1:end-1]
        cur = close[a:end]
        if expected_kind == "R":
            cq = np.flatnonzero((prev <= pub) & (cur > pub))
        else:
            cq = np.flatnonzero((prev >= pub) & (cur < pub))
        if len(cq):
            j = a + int(cq[0])
            cross_h = (int(sig_ns[j]) - post_ns) / 3.6e12
    return touch_h, cross_h


def audit_group(records: list[dict], config_path: str, datadir_s: str, warmup_days: int, band_pct: float):
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
        lo_t = times.min() - pd.Timedelta(days=warmup_days)
        hi_t = times.max() + pd.Timedelta(hours=73)
        x = x[(x.date >= lo_t) & (x.date < hi_t)].reset_index(drop=True)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_RANGE", "elapsed_s": time.monotonic()-t0}

        p20 = dc.build_levels(x, tf, 20, 0)
        p30 = dc.build_levels(x, tf, 30, len(p20))
        levels = p20 + p30
        sig_ns = pd.to_datetime(x.signal_time, utc=True).astype("int64").to_numpy()
        close = x.close.to_numpy(float); high = x.high.to_numpy(float); low = x.low.to_numpy(float)
        invalid_ns = {lv.level_id: _first_close_invalidation(lv, close, sig_ns) for lv in levels}

        rows = []
        for r in records:
            t = pd.Timestamp(r["post_time"]); post_ns = t.value; pub = float(r["published_level"])
            idx = int(np.searchsorted(sig_ns, post_ns, side="right") - 1)
            if idx < 1 or idx >= len(close) or not np.isfinite(close[idx]) or close[idx] <= 0:
                continue
            c0 = float(close[idx])
            expected_kind = "R" if pub >= c0 else "S"
            formed = [lv for lv in levels if pd.Timestamp(lv.formed_time).value <= post_ns]
            live = [lv for lv in formed if invalid_ns.get(lv.level_id, 0) > post_ns]
            live_side = [lv for lv in live if lv.kind == expected_kind and ((lv.kind == "R" and lv.price >= c0) or (lv.kind == "S" and lv.price <= c0))]

            band_lo, band_hi = c0 * (1.0 - band_pct), c0 * (1.0 + band_pct)
            mode_levels = {"raw": formed, "live": live, "live_side": live_side}
            z = dict(r)
            z.update({
                "data_source": source,
                "post_close": c0,
                "expected_kind": expected_kind,
                "published_distance_bps": (pub / c0 - 1.0) * 10000.0,
                "published_in_band": bool(band_lo <= pub <= band_hi),
            })
            for mode, arr in mode_levels.items():
                e, np_, k = _nearest(arr, pub)
                prices = _unique_prices(arr, band_lo, band_hi)
                z[f"{mode}_nearest_bps"] = e
                z[f"{mode}_nearest_price"] = np_
                z[f"{mode}_nearest_kind"] = k
                z[f"{mode}_candidate_n"] = len(prices)
                for b in MATCH_BPS:
                    z[f"{mode}_actual_{int(b)}"] = bool(np.isfinite(e) and e <= b)
                    z[f"{mode}_chance_{int(b)}"] = _merge_union_fraction(prices, band_lo, band_hi, b)

            th, ch = _future_event_hours(pub, expected_kind, post_ns, sig_ns, high, low, close)
            z["first_touch_h"] = th; z["first_directional_cross_h"] = ch
            rows.append(z)

        return rows, {
            "pair": pair, "tf": tf, "status": "OK", "rows": len(rows), "bars": len(x),
            "levels": len(levels), "elapsed_s": time.monotonic()-t0,
        }
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic()-t0}


def _pct(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").mean() * 100.0) if len(s) else np.nan


def _mode_report(g: pd.DataFrame, mode: str):
    inb = g[g.published_in_band.astype(bool)]
    cand = pd.to_numeric(g[f"{mode}_candidate_n"], errors="coerce")
    print(f"  {mode:9s} candidates median={cand.median():5.0f} p75={cand.quantile(.75):5.0f}", flush=True)
    for b in MATCH_BPS:
        act = _pct(g[f"{mode}_actual_{int(b)}"])
        base = float(pd.to_numeric(inb[f"{mode}_chance_{int(b)}"], errors="coerce").mean() * 100.0) if len(inb) else np.nan
        lift = act / base if np.isfinite(act) and np.isfinite(base) and base > 0 else np.nan
        print(f"    <={int(b):3d}bps actual={act:5.1f}% chance={base:5.1f}% lift={lift:4.2f}x", flush=True)


def _horizon_pct(s: pd.Series, h: int) -> float:
    x = pd.to_numeric(s, errors="coerce")
    return float((x <= h).mean() * 100.0)


def main() -> int:
    a = parse_args(); v4dir = Path(a.v4dir); outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    parsed_path = v4dir / "digash_public_breakouts.csv"; cov_path = v4dir / "cache_coverage.csv"
    if not parsed_path.exists() or not cov_path.exists():
        raise FileNotFoundError("Run Fidelity V4 first; V4.2 consumes its frozen source snapshot and coverage.")
    parsed = pd.read_csv(parsed_path); parsed["post_time"] = pd.to_datetime(parsed.post_time, utc=True, errors="coerce")
    cov = pd.read_csv(cov_path); ok = cov[cov.status.eq("OK")][["pair", "tf"]].drop_duplicates()
    work = parsed.merge(ok, on=["pair", "tf"], how="inner")
    groups = [g.to_dict("records") for _, g in work.groupby(["pair", "tf"], sort=True)]

    print("=== DIGASH FIDELITY V4.2 — LIVE LEVEL LIFECYCLE ===", flush=True)
    print("NO PnL. NO parameter tuning. NO downloads.", flush=True)
    print("Tests whether V4's dense chance baseline was caused by counting already-broken historical levels as current candidates.", flush=True)
    print("LIVE = formed before post and not yet closed through in breakout direction; LIVE_SIDE also requires the level to be on the published side of current price and matching R/S kind.", flush=True)
    print(f"covered source rows={len(work)} | groups={len(groups)} | workers={a.workers}", flush=True)

    results=[]; metas=[]; t0=time.monotonic(); done=0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs=[ex.submit(audit_group, recs, a.config, a.datadir, a.warmup_days, a.band_pct) for recs in groups]
        for f in as_completed(futs):
            rows, meta=f.result(); done+=1; results.extend(rows); metas.append(meta)
            elapsed=time.monotonic()-t0; eta=elapsed*(len(futs)-done)/done if done else np.nan
            print(f"V4.2 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} rows={len(rows)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)
    pd.DataFrame(metas).to_csv(outdir/"coverage.csv", index=False)
    z=pd.DataFrame(results); z.to_csv(outdir/"fidelity_live_rows.csv", index=False)
    if z.empty:
        print("No rows.", flush=True); return 2

    print("\n=== SAMPLE DEPENDENCE ===", flush=True)
    unique_exact = z[["pair","tf","published_level"]].drop_duplicates().shape[0]
    print(f"rows={len(z)} | unique exact pair×TF×level={unique_exact} | repeated rows={len(z)-unique_exact}", flush=True)

    print("\n=== ACTIVE-LEVEL CHANCE-CONTROLLED FIDELITY ===", flush=True)
    cohorts=[("ALL",z), ("LOW_TF",z[z.tf.isin(["1m","5m","15m"])])]
    cohorts += [(tf,z[z.tf.eq(tf)]) for tf in ["1m","5m","15m","1h","4h"] if (z.tf==tf).any()]
    for name,g in cohorts:
        print(f"\n{name} N={len(g)}", flush=True)
        for mode in ["raw","live","live_side"]:
            _mode_report(g, mode)

    print("\n=== FORMATION IS ADVANCE WATCH, NOT FIRST-CROSS ENTRY ===", flush=True)
    for name,g in cohorts:
        th=pd.to_numeric(g.first_touch_h,errors="coerce"); ch=pd.to_numeric(g.first_directional_cross_h,errors="coerce")
        print(f"{name:6s} N={len(g):3d} | touch median={th.median():6.2f}h | cross median={ch.median():6.2f}h", flush=True)
        print("  touch: " + " ".join(f"<={h}h {_horizon_pct(g.first_touch_h,h):5.1f}%" for h in HORIZONS_H), flush=True)
        print("  cross: " + " ".join(f"<={h}h {_horizon_pct(g.first_directional_cross_h,h):5.1f}%" for h in HORIZONS_H), flush=True)

    print("\n=== DECISION RULE ===", flush=True)
    print("If LIVE_SIDE sharply reduces candidate density and raises fidelity lift, use active/unbroken side-consistent levels as the source-faithful level universe for the next formation-selection reconstruction.", flush=True)
    print("If lift remains near 1x even after LIVE_SIDE, our horizontal-level construction itself is not selective enough and must be reconstructed before any PnL test.", flush=True)
    print("Do not reinterpret the Telegram post timestamp as a trade entry: the source rows are being tested as formation/watch candidates ahead of the eventual touch/cross.", flush=True)
    print(f"Reports: {outdir}", flush=True); print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
