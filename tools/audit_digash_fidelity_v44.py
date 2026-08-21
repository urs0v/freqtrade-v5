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
from audit_digash_fidelity_v43 import _touch_schedule, _active_rearmed

MATCH_BPS = 25.0
TOPK = (1, 3, 5, 10)


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.4: formation-selector diagnostics")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v4dir", default="/freqtrade/user_data/digash_fidelity_v4")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v44")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--warmup-days", type=int, default=120)
    return p.parse_args()


def _prior_schedule(schedule, post_ns):
    return [q for q in schedule if q[0] <= post_ns]


def _candidate_meta(lv, schedules, post_ns, close0):
    prior = _prior_schedule(schedules[lv.level_id], post_ns)
    if prior:
        last_touch_ns = prior[-1][0]
        touch_n = len(prior)
        last_touch_age_h = (post_ns - last_touch_ns) / 3.6e12
    else:
        touch_n = 0
        last_touch_age_h = np.nan
    dist_bps = abs(float(lv.price) / close0 - 1.0) * 10000.0
    level_age_h = (post_ns - pd.Timestamp(lv.formed_time).value) / 3.6e12
    return {
        "lv": lv,
        "distance_bps": float(dist_bps),
        "touch_n": int(touch_n),
        "last_touch_age_h": float(last_touch_age_h) if np.isfinite(last_touch_age_h) else np.nan,
        "level_age_h": float(level_age_h),
        "touch_error_pct": float(lv.touch_error_pct),
        "period": int(lv.period),
    }


def _rank_of(target_id, metas, key, reverse=False):
    vals = [m for m in metas if np.isfinite(m[key])]
    vals.sort(key=lambda m: m[key], reverse=reverse)
    for i, m in enumerate(vals, start=1):
        if m["lv"].level_id == target_id:
            return i
    return np.nan


def _best_match(metas, pub):
    if not metas:
        return None, np.nan
    best = min(metas, key=lambda m: abs(float(m["lv"].price) / pub - 1.0))
    err = abs(float(best["lv"].price) / pub - 1.0) * 10000.0
    return best, float(err)


def _activity_frame(config, datadir, pair, lo, hi):
    x, source = load_published_tf(config, datadir, pair, "15m")
    if x.empty:
        return pd.DataFrame(), "none"
    x = x[(x.date >= lo - pd.Timedelta(days=31)) & (x.date < hi + pd.Timedelta(days=1))].copy().reset_index(drop=True)
    if x.empty:
        return x, source
    close = x.close.astype(float)
    q = x.volume.astype(float) * close
    prev = close.shift()
    tr = pd.concat([
        x.high.astype(float) - x.low.astype(float),
        (x.high.astype(float) - prev).abs(),
        (x.low.astype(float) - prev).abs(),
    ], axis=1).max(axis=1)
    x["qvol24"] = q.rolling(96, min_periods=48).sum()
    x["absret24"] = close.pct_change(96).abs()
    x["natr15"] = tr.rolling(14, min_periods=7).mean() / close
    return x, source


def _causal_percentile(arr, idx, sig_ns, post_ns, days=30):
    if idx < 0 or idx >= len(arr) or not np.isfinite(arr[idx]):
        return np.nan
    lo_ns = post_ns - int(days * 86400 * 1e9)
    a = int(np.searchsorted(sig_ns, lo_ns, side="left"))
    w = arr[a:idx+1]
    w = w[np.isfinite(w)]
    if len(w) < 20:
        return np.nan
    return float(np.mean(w <= arr[idx]) * 100.0)


def audit_group(records, config_path, datadir_s, warmup_days):
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    pair, tf = records[0]["pair"], records[0]["tf"]
    dc.TF_MINUTES.setdefault("1m", 1)
    try:
        times = pd.to_datetime([r["post_time"] for r in records], utc=True)
        x, source = load_published_tf(config, datadir, pair, tf)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_DATA", "elapsed_s": time.monotonic()-t0}
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

        ax, asource = _activity_frame(config, datadir, pair, times.min(), times.max())
        if not ax.empty:
            asig_ns = pd.to_datetime(ax.signal_time, utc=True).astype("int64").to_numpy()
            aq = ax.qvol24.to_numpy(float)
            ar = ax.absret24.to_numpy(float)
            av = ax.natr15.to_numpy(float)
        else:
            asig_ns = np.array([], dtype=np.int64); aq = ar = av = np.array([], dtype=float)

        rows = []
        for r in records:
            t = pd.Timestamp(r["post_time"]); post_ns = int(t.value); pub = float(r["published_level"])
            idx = int(np.searchsorted(sig_ns, post_ns, side="right") - 1)
            if idx < 1 or idx >= len(close) or not np.isfinite(close[idx]) or close[idx] <= 0:
                continue
            c0 = float(close[idx])
            expected_kind = "R" if pub >= c0 else "S"
            formed = [lv for lv in levels if pd.Timestamp(lv.formed_time).value <= post_ns]
            raw_side = [lv for lv in formed if lv.kind == expected_kind and ((lv.kind == "R" and lv.price >= c0) or (lv.kind == "S" and lv.price <= c0))]
            rearm = [lv for lv in formed if _active_rearmed(schedules[lv.level_id], post_ns)]
            rearm_side = [lv for lv in rearm if lv.kind == expected_kind and ((lv.kind == "R" and lv.price >= c0) or (lv.kind == "S" and lv.price <= c0))]

            z = dict(r)
            z.update({
                "data_source": source,
                "post_close": c0,
                "expected_kind": expected_kind,
                "published_distance_bps": abs(pub / c0 - 1.0) * 10000.0,
            })
            for mode, arr in {"raw_side": raw_side, "rearm_side": rearm_side}.items():
                metas = [_candidate_meta(lv, schedules, post_ns, c0) for lv in arr]
                best, err = _best_match(metas, pub)
                z[f"{mode}_candidate_n"] = len(metas)
                z[f"{mode}_match_bps"] = err
                z[f"{mode}_matched25"] = bool(np.isfinite(err) and err <= MATCH_BPS)
                if best is not None:
                    lv = best["lv"]
                    z[f"{mode}_match_price"] = float(lv.price)
                    z[f"{mode}_match_period"] = int(best["period"])
                    z[f"{mode}_match_touch_n"] = int(best["touch_n"])
                    z[f"{mode}_match_last_touch_age_h"] = best["last_touch_age_h"]
                    z[f"{mode}_match_level_age_h"] = best["level_age_h"]
                    z[f"{mode}_rank_distance"] = _rank_of(lv.level_id, metas, "distance_bps", False)
                    z[f"{mode}_rank_touch_recency"] = _rank_of(lv.level_id, metas, "last_touch_age_h", False)
                    z[f"{mode}_rank_touch_count"] = _rank_of(lv.level_id, metas, "touch_n", True)

            if len(asig_ns):
                ai = int(np.searchsorted(asig_ns, post_ns, side="right") - 1)
                q_pct = _causal_percentile(aq, ai, asig_ns, post_ns)
                r_pct = _causal_percentile(ar, ai, asig_ns, post_ns)
                v_pct = _causal_percentile(av, ai, asig_ns, post_ns)
                z["activity_source"] = asource
                z["qvol24_pct30d"] = q_pct
                z["absret24_pct30d"] = r_pct
                z["natr15_pct30d"] = v_pct
                vals = [q_pct, r_pct, v_pct]
                vals = [v for v in vals if np.isfinite(v)]
                z["activity_max_pct30d"] = max(vals) if vals else np.nan
            rows.append(z)
        return rows, {"pair": pair, "tf": tf, "status": "OK", "rows": len(rows), "levels": len(levels), "elapsed_s": time.monotonic()-t0}
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic()-t0}


def _pct_bool(s):
    return float(pd.to_numeric(s, errors="coerce").mean() * 100.0) if len(s) else np.nan


def _selector_report(name, g, mode):
    m = g[f"{mode}_matched25"].astype(bool)
    q = g[m].copy()
    cand = pd.to_numeric(g[f"{mode}_candidate_n"], errors="coerce")
    print(f"{name:6s} {mode:10s} N={len(g):3d} candidates med={cand.median():5.0f} | source<=25bps={m.mean()*100:5.1f}%", flush=True)
    if q.empty:
        return
    for rank_col, label in [
        (f"{mode}_rank_distance", "nearest-price"),
        (f"{mode}_rank_touch_recency", "recent-touch"),
        (f"{mode}_rank_touch_count", "touch-count"),
    ]:
        r = pd.to_numeric(q[rank_col], errors="coerce")
        cond = " ".join(f"top{k}={(r <= k).mean()*100:5.1f}%" for k in TOPK)
        capture = " ".join(f"all@{k}={(m & (pd.to_numeric(g[rank_col], errors='coerce') <= k)).mean()*100:5.1f}%" for k in TOPK)
        print(f"  {label:13s} rank med={r.median():5.1f} | {cond} | {capture}", flush=True)
    touch = pd.to_numeric(q[f"{mode}_match_touch_n"], errors="coerce")
    age = pd.to_numeric(q[f"{mode}_match_last_touch_age_h"], errors="coerce")
    per = pd.to_numeric(q[f"{mode}_match_period"], errors="coerce")
    print(f"  matched source level: touches med={touch.median():.1f} | last-touch age med={age.median():.2f}h | p20={(per==20).mean()*100:.1f}% p30={(per==30).mean()*100:.1f}%", flush=True)


def _activity_report(name, g):
    print(f"{name:6s} activity N={len(g):3d}", flush=True)
    for c, label in [
        ("qvol24_pct30d", "24h quote-volume"),
        ("absret24_pct30d", "24h abs-return"),
        ("natr15_pct30d", "15m NATR"),
        ("activity_max_pct30d", "MAX activity"),
    ]:
        s = pd.to_numeric(g.get(c), errors="coerce").dropna()
        if s.empty:
            print(f"  {label:16s}: no coverage", flush=True)
        else:
            print(f"  {label:16s}: med={s.median():5.1f}pct | >=80={((s>=80).mean()*100):5.1f}% >=90={((s>=90).mean()*100):5.1f}%", flush=True)


def main():
    a = parse_args(); v4dir = Path(a.v4dir); outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    parsed_path = v4dir / "digash_public_breakouts.csv"; cov_path = v4dir / "cache_coverage.csv"
    if not parsed_path.exists() or not cov_path.exists():
        raise FileNotFoundError("Run Fidelity V4 first; V4.4 consumes its frozen public-source snapshot.")
    parsed = pd.read_csv(parsed_path); parsed["post_time"] = pd.to_datetime(parsed.post_time, utc=True, errors="coerce")
    cov = pd.read_csv(cov_path); ok = cov[cov.status.eq("OK")][["pair", "tf"]].drop_duplicates()
    work = parsed.merge(ok, on=["pair", "tf"], how="inner")
    groups = [g.to_dict("records") for _, g in work.groupby(["pair", "tf"], sort=True)]

    print("=== DIGASH FIDELITY V4.4 — FORMATION SELECTOR AUDIT ===", flush=True)
    print("NO PnL. NO selector fitting. NO downloads.", flush=True)
    print("Tests which candidate Digash appears to publish: nearest-to-price, recent-touch, touch-count, plus causal 30d OHLCV activity percentiles.", flush=True)
    print("Activity is only an OHLCV proxy; it does NOT reproduce Digash trade-count/orderbook/200+ technical inputs.", flush=True)
    print(f"covered source rows={len(work)} | groups={len(groups)} | workers={a.workers}", flush=True)

    results=[]; metas=[]; t0=time.monotonic(); done=0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs=[ex.submit(audit_group, recs, a.config, a.datadir, a.warmup_days) for recs in groups]
        for f in as_completed(futs):
            rows, meta=f.result(); done+=1; results.extend(rows); metas.append(meta)
            elapsed=time.monotonic()-t0; eta=elapsed*(len(futs)-done)/done if done else np.nan
            print(f"V4.4 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} rows={len(rows)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(outdir/"coverage.csv", index=False)
    z=pd.DataFrame(results); z.to_csv(outdir/"selector_rows.csv", index=False)
    if z.empty:
        print("No rows.", flush=True); return 2

    cohorts=[("ALL",z), ("LOW_TF",z[z.tf.isin(["1m","5m","15m"])])]
    cohorts += [(tf,z[z.tf.eq(tf)]) for tf in ["1m","5m","15m","1h","4h"] if (z.tf==tf).any()]

    print("\n=== SOURCE LEVEL RANK INSIDE CANDIDATE SET ===", flush=True)
    for name,g in cohorts:
        for mode in ["raw_side", "rearm_side"]:
            _selector_report(name,g,mode)

    print("\n=== SOURCE-COIN ACTIVITY AT FORMATION TIME ===", flush=True)
    for name,g in cohorts:
        _activity_report(name,g)

    print("\n=== DECISION RULE ===", flush=True)
    print("If source levels cluster in top-1/top-3 nearest candidates, distance-to-level is a major formation selector and should be reconstructed before any entry logic.", flush=True)
    print("If recent-touch or touch-count ranks dominate instead, carry that source-selection mechanism forward.", flush=True)
    print("If formation posts also cluster at high causal activity percentiles, combine level selection with active-coin selection; do not confuse the Telegram alert with the later trade entry.", flush=True)
    print(f"Reports: {outdir}", flush=True); print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
