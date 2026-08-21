#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from audit_digash_fidelity_v4 import load_published_tf

LOW_TFS = ("1m", "5m", "15m")
HORIZONS_H = (1.0, 4.0, 24.0)
CROSS_WINDOWS_H = (1.0, 4.0, 24.0, 72.0)


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.6: public source-level follow-through vs matched controls")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v45dir", default="/freqtrade/user_data/digash_fidelity_v45")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v46")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--controls", type=int, default=3)
    p.add_argument("--exclude-source-bps", type=float, default=25.0)
    p.add_argument("--train-frac", type=float, default=0.70)
    return p.parse_args()


def _truthy(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def build_source_control_rows(c: pd.DataFrame, controls: int, exclude_source_bps: float) -> pd.DataFrame:
    rows = []
    for qid, g in c.groupby("query_id", sort=False):
        g = g.copy()
        r0 = g.iloc[0]
        pub = float(r0.published_level)
        close0 = float(r0.post_close)
        source_dist = abs(pub / close0 - 1.0) * 10000.0
        source = {
            "query_id": qid,
            "post_id": int(r0.post_id),
            "post_time": pd.Timestamp(r0.post_time),
            "pair": str(r0.pair),
            "tf": str(r0.tf),
            "published_level_no": int(r0.published_level_no),
            "published_level": pub,
            "post_close": close0,
            "role": "SOURCE",
            "control_no": 0,
            "level_price": pub,
            "source_distance_bps": source_dist,
            "control_distance_gap_bps": 0.0,
        }
        rows.append(source)

        valid = g[_truthy(g.candidate_valid)].copy()
        if valid.empty:
            continue
        valid["source_zone_error_bps"] = (pd.to_numeric(valid.level_price, errors="coerce") / pub - 1.0).abs() * 10000.0
        valid = valid[pd.to_numeric(valid.source_zone_error_bps, errors="coerce") > exclude_source_bps].copy()
        if valid.empty:
            continue
        valid["distance_gap"] = (pd.to_numeric(valid.distance_bps, errors="coerce") - source_dist).abs()
        valid = valid.sort_values(["distance_gap", "last_touch_age_h", "level_id"], ascending=[True, True, True], na_position="last")
        # Avoid selecting duplicate p20/p30 representations of effectively the same control zone.
        chosen = []
        for rr in valid.itertuples(index=False):
            lp = float(rr.level_price)
            if any(abs(lp / p - 1.0) * 10000.0 <= exclude_source_bps for p in chosen):
                continue
            chosen.append(lp)
            rows.append({
                "query_id": qid,
                "post_id": int(r0.post_id),
                "post_time": pd.Timestamp(r0.post_time),
                "pair": str(r0.pair),
                "tf": str(r0.tf),
                "published_level_no": int(r0.published_level_no),
                "published_level": pub,
                "post_close": close0,
                "role": "CONTROL",
                "control_no": len(chosen),
                "level_price": lp,
                "source_distance_bps": source_dist,
                "control_distance_gap_bps": float(rr.distance_gap),
            })
            if len(chosen) >= max(1, controls):
                break
    return pd.DataFrame(rows)


def _first_touch_cross(x: pd.DataFrame, post_time: pd.Timestamp, level: float, direction: int):
    sig = pd.to_datetime(x.signal_time, utc=True)
    start = int(np.searchsorted(sig.astype("int64").to_numpy(), post_time.value, side="right"))
    if start >= len(x):
        return None
    deadline = post_time + pd.Timedelta(hours=72)
    end = int(np.searchsorted(sig.astype("int64").to_numpy(), deadline.value, side="right"))
    end = min(end, len(x))
    if end <= start:
        return None
    high = x.high.to_numpy(float)
    low = x.low.to_numpy(float)
    close = x.close.to_numpy(float)

    touch_i = None
    cross_i = None
    for i in range(start, end):
        if touch_i is None and low[i] <= level <= high[i]:
            touch_i = i
        prev_close = close[i - 1] if i > 0 else np.nan
        if direction > 0:
            crossed = np.isfinite(prev_close) and prev_close < level and close[i] >= level
        else:
            crossed = np.isfinite(prev_close) and prev_close > level and close[i] <= level
        if crossed:
            cross_i = i
            break
    return touch_i, cross_i


def _post_cross_metrics(x: pd.DataFrame, post_time: pd.Timestamp, level: float, post_close: float) -> dict:
    direction = 1 if level >= post_close else -1
    z = {"direction": direction}
    tc = _first_touch_cross(x, post_time, level, direction)
    if tc is None:
        for h in CROSS_WINDOWS_H:
            z[f"touch_le_{int(h)}h"] = False
            z[f"cross_le_{int(h)}h"] = False
        return z
    touch_i, cross_i = tc
    sig = pd.to_datetime(x.signal_time, utc=True)
    if touch_i is not None:
        touch_time = pd.Timestamp(sig.iloc[touch_i])
        touch_h = (touch_time - post_time).total_seconds() / 3600.0
        z["touch_time"] = touch_time
        z["touch_delay_h"] = touch_h
    else:
        touch_h = np.nan
    if cross_i is not None:
        cross_time = pd.Timestamp(sig.iloc[cross_i])
        cross_h = (cross_time - post_time).total_seconds() / 3600.0
        cross_close = float(x.close.iloc[cross_i])
        z["cross_time"] = cross_time
        z["cross_delay_h"] = cross_h
        z["cross_close"] = cross_close
        z["cross_overshoot_bps"] = direction * (cross_close / level - 1.0) * 10000.0
    else:
        cross_h = np.nan
        cross_close = np.nan

    for h in CROSS_WINDOWS_H:
        z[f"touch_le_{int(h)}h"] = bool(np.isfinite(touch_h) and touch_h <= h)
        z[f"cross_le_{int(h)}h"] = bool(np.isfinite(cross_h) and cross_h <= h)

    if cross_i is None or not np.isfinite(cross_close) or cross_close <= 0:
        return z

    sig_ns = sig.astype("int64").to_numpy()
    high = x.high.to_numpy(float)
    low = x.low.to_numpy(float)
    close = x.close.to_numpy(float)
    for h in HORIZONS_H:
        deadline_ns = (pd.Timestamp(sig.iloc[cross_i]) + pd.Timedelta(hours=h)).value
        end = int(np.searchsorted(sig_ns, deadline_ns, side="right")) - 1
        end = min(end, len(x) - 1)
        a = cross_i + 1
        key = int(h)
        if end < a:
            z[f"ret_{key}h_bps"] = np.nan
            z[f"mfe_{key}h_bps"] = np.nan
            z[f"mae_{key}h_bps"] = np.nan
            continue
        last = float(close[end])
        hh = float(np.max(high[a:end + 1]))
        ll = float(np.min(low[a:end + 1]))
        if direction > 0:
            ret = (last / cross_close - 1.0) * 10000.0
            mfe = (hh / cross_close - 1.0) * 10000.0
            mae = (1.0 - ll / cross_close) * 10000.0
        else:
            ret = (1.0 - last / cross_close) * 10000.0
            mfe = (1.0 - ll / cross_close) * 10000.0
            mae = (hh / cross_close - 1.0) * 10000.0
        z[f"ret_{key}h_bps"] = ret
        z[f"mfe_{key}h_bps"] = max(0.0, mfe)
        z[f"mae_{key}h_bps"] = max(0.0, mae)
    return z


def process_group(records: list[dict], config_path: str, datadir_s: str):
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    pair = records[0]["pair"]
    tf = records[0]["tf"]
    try:
        x, source = load_published_tf(config, datadir, pair, tf)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_DATA", "elapsed_s": time.monotonic() - t0}
        lo = min(pd.Timestamp(r["post_time"]) for r in records) - pd.Timedelta(days=1)
        hi = max(pd.Timestamp(r["post_time"]) for r in records) + pd.Timedelta(hours=97)
        x = x[(x.date >= lo) & (x.date <= hi)].reset_index(drop=True)
        out = []
        for r in records:
            z = dict(r)
            z["data_source"] = source
            z.update(_post_cross_metrics(x, pd.Timestamp(r["post_time"]), float(r["level_price"]), float(r["post_close"])))
            out.append(z)
        return out, {"pair": pair, "tf": tf, "status": "OK", "rows": len(out), "elapsed_s": time.monotonic() - t0}
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic() - t0}


def _pct(s: pd.Series) -> float:
    return float(pd.to_numeric(s, errors="coerce").fillna(0).mean() * 100.0) if len(s) else np.nan


def _med(s: pd.Series) -> float:
    q = pd.to_numeric(s, errors="coerce").dropna()
    return float(q.median()) if len(q) else np.nan


def _summary(label: str, g: pd.DataFrame):
    src = g[g.role.eq("SOURCE")]
    ctl = g[g.role.eq("CONTROL")]
    print(f"\n{label} queries={src.query_id.nunique()} | source rows={len(src)} control rows={len(ctl)}", flush=True)
    if len(ctl):
        print(f"  control distance-gap median={_med(ctl.control_distance_gap_bps):.1f}bps", flush=True)
    for h in CROSS_WINDOWS_H:
        k = int(h)
        print(
            f"  <= {k:2d}h touch source={_pct(src[f'touch_le_{k}h']):5.1f}% control={_pct(ctl[f'touch_le_{k}h']):5.1f}% | "
            f"cross source={_pct(src[f'cross_le_{k}h']):5.1f}% control={_pct(ctl[f'cross_le_{k}h']):5.1f}%",
            flush=True,
        )
    print(f"  cross-delay median source={_med(src.cross_delay_h):.2f}h control={_med(ctl.cross_delay_h):.2f}h", flush=True)
    for h in HORIZONS_H:
        k = int(h)
        print(
            f"  post-cross {k:2d}h RET med source={_med(src[f'ret_{k}h_bps']):7.1f}bps control={_med(ctl[f'ret_{k}h_bps']):7.1f} | "
            f"MFE { _med(src[f'mfe_{k}h_bps']):7.1f}/{_med(ctl[f'mfe_{k}h_bps']):7.1f} | "
            f"MAE { _med(src[f'mae_{k}h_bps']):7.1f}/{_med(ctl[f'mae_{k}h_bps']):7.1f}",
            flush=True,
        )


def _paired_queries(z: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for qid, g in z.groupby("query_id", sort=False):
        s = g[g.role.eq("SOURCE")]
        c = g[g.role.eq("CONTROL")]
        if s.empty:
            continue
        s0 = s.iloc[0]
        r = {
            "query_id": qid,
            "post_id": int(s0.post_id),
            "post_time": s0.post_time,
            "pair": s0.pair,
            "tf": s0.tf,
            "controls_n": len(c),
        }
        for h in CROSS_WINDOWS_H:
            k = int(h)
            r[f"source_cross_{k}h"] = bool(s0.get(f"cross_le_{k}h", False))
            r[f"control_cross_{k}h_mean"] = float(pd.to_numeric(c.get(f"cross_le_{k}h"), errors="coerce").mean()) if len(c) else np.nan
        for h in HORIZONS_H:
            k = int(h)
            for stem in ("ret", "mfe", "mae"):
                col = f"{stem}_{k}h_bps"
                sv = pd.to_numeric(pd.Series([s0.get(col)]), errors="coerce").iloc[0]
                cv = pd.to_numeric(c.get(col), errors="coerce").mean() if len(c) else np.nan
                r[f"source_{col}"] = sv
                r[f"control_mean_{col}"] = cv
                r[f"delta_{col}"] = sv - cv if np.isfinite(sv) and np.isfinite(cv) else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def _paired_report(label: str, p: pd.DataFrame):
    print(f"\n{label} paired queries={len(p)}", flush=True)
    for h in HORIZONS_H:
        k = int(h)
        for stem, good_sign in (("ret", 1), ("mfe", 1), ("mae", -1)):
            col = f"delta_{stem}_{k}h_bps"
            d = pd.to_numeric(p.get(col), errors="coerce").dropna()
            if d.empty:
                continue
            win = (d * good_sign > 0).mean() * 100.0
            print(f"  {stem.upper():3s} {k:2d}h source-control delta med={d.median():7.1f}bps | source-better={win:5.1f}% | N={len(d)}", flush=True)


def main() -> int:
    a = parse_args()
    v45dir = Path(a.v45dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = v45dir / "candidate_rows.csv"
    if not path.exists():
        raise FileNotFoundError("Run Fidelity V4.5 first; V4.6 consumes its frozen candidate_rows.csv.")

    c = pd.read_csv(path)
    c["post_time"] = pd.to_datetime(c.post_time, utc=True, errors="coerce")
    c = c[c.tf.isin(LOW_TFS)].copy()
    if c.empty:
        print("No LOW_TF V4.5 rows.", flush=True)
        return 2
    sc = build_source_control_rows(c, a.controls, a.exclude_source_bps)
    sc.to_csv(outdir / "source_control_design.csv", index=False)

    uq = sc[["post_id", "post_time"]].drop_duplicates().sort_values(["post_time", "post_id"])
    cut_i = max(1, min(len(uq) - 1, int(np.floor(len(uq) * a.train_frac))))
    hold_ids = set(uq.iloc[cut_i:].post_id.astype(int))
    cutoff = uq.iloc[cut_i].post_time
    sc["cohort"] = np.where(sc.post_id.astype(int).isin(hold_ids), "HOLDOUT", "EARLY")

    print("=== DIGASH FIDELITY V4.6 — SOURCE-LEVEL FOLLOW-THROUGH ===", flush=True)
    print("NO PnL optimization. NO stop/target fitting. NO downloads.", flush=True)
    print("Exact public Digash level is compared with up to distance-matched non-source V4.5 candidate levels from the same pair, timeframe, timestamp and market side.", flush=True)
    print("Measures touch/cross timing and post-cross directional RET/MFE/MAE only; this isolates formation-selection quality from trade-management assumptions.", flush=True)
    print(f"queries={sc.query_id.nunique()} | rows={len(sc):,} | controls/query<={a.controls} | holdout starts={cutoff}", flush=True)

    groups = [g.to_dict("records") for _, g in sc.groupby(["pair", "tf"], sort=True)]
    results = []
    metas = []
    t0 = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(process_group, recs, a.config, a.datadir) for recs in groups]
        for f in as_completed(futs):
            rows, meta = f.result()
            done += 1
            results.extend(rows)
            metas.append(meta)
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(futs) - done) / done if done else np.nan
            print(f"V4.6 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} rows={len(rows)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(outdir / "coverage.csv", index=False)
    z = pd.DataFrame(results)
    if z.empty:
        print("No result rows.", flush=True)
        return 2
    z["post_time"] = pd.to_datetime(z.post_time, utc=True, errors="coerce")
    z.to_csv(outdir / "followthrough_rows.csv", index=False)
    p = _paired_queries(z)
    p.to_csv(outdir / "paired_queries.csv", index=False)

    print("\n=== SOURCE VS MATCHED CONTROLS ===", flush=True)
    _summary("ALL LOW_TF", z)
    _summary("LATER HOLDOUT", z[z.cohort.eq("HOLDOUT")])
    for tf in LOW_TFS:
        _summary(f"HOLDOUT {tf}", z[z.cohort.eq("HOLDOUT") & z.tf.eq(tf)])

    print("\n=== PAIRED FOLLOW-THROUGH DELTA ===", flush=True)
    _paired_report("ALL LOW_TF", p)
    hold_qids = set(z[z.cohort.eq("HOLDOUT")].query_id.unique())
    _paired_report("LATER HOLDOUT", p[p.query_id.isin(hold_qids)])
    for tf in LOW_TFS:
        _paired_report(f"HOLDOUT {tf}", p[p.query_id.isin(hold_qids) & p.tf.eq(tf)])

    print("\n=== DECISION RULE ===", flush=True)
    print("If public SOURCE levels cross at a similar rate but show materially stronger post-cross RET/MFE or lower MAE than same-time same-side matched controls, Digash's formation selector contains predictive information that our generic level universe does not.", flush=True)
    print("If SOURCE and controls behave similarly on the later holdout, the public formation level itself is not enough; the missing edge is more likely in entry timing, orderbook/density context, breakout quality, or trade management.", flush=True)
    print("Do not convert these excursion diagnostics into stop/target rules from the same sample.", flush=True)
    print(f"Reports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
