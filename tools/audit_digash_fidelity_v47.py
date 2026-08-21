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

LOW_TFS = ("1m", "5m", "15m")
HORIZONS = ((0.25, "15m"), (1.0, "1h"), (4.0, "4h"), (24.0, "24h"))


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.7: source-level breakout-quality reconstruction")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v46dir", default="/freqtrade/user_data/digash_fidelity_v46")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v47")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--train-frac", type=float, default=0.70)
    return p.parse_args()


def _load_detail(config: dict, datadir: Path, pair: str, published_tf: str):
    d1 = dc.load_tf(config, datadir, pair, "1m")
    if not d1.empty:
        return dc.prep_ohlcv(d1, 1), "1m"
    d5, src5 = dc.load_5m(config, datadir, pair)
    if not d5.empty:
        return dc.prep_ohlcv(d5, 5), src5
    x, src = load_published_tf(config, datadir, pair, published_tf)
    return x, src


def _atr_prev(x: pd.DataFrame) -> np.ndarray:
    close = x.close.astype(float)
    prev = close.shift(1)
    tr = pd.concat([
        x.high.astype(float) - x.low.astype(float),
        (x.high.astype(float) - prev).abs(),
        (x.low.astype(float) - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=7).mean().shift(1).to_numpy(float)


def _first_directed_cross(x: pd.DataFrame, post_time: pd.Timestamp, level: float, direction: int):
    sig = pd.to_datetime(x.signal_time, utc=True)
    sig_ns = sig.astype("int64").to_numpy()
    start = int(np.searchsorted(sig_ns, post_time.value, side="right"))
    deadline = (post_time + pd.Timedelta(hours=72)).value
    end = min(len(x), int(np.searchsorted(sig_ns, deadline, side="right")))
    if start >= end:
        return None
    close = x.close.to_numpy(float)
    for i in range(max(1, start), end):
        p = close[i - 1]
        c = close[i]
        if direction > 0:
            crossed = np.isfinite(p) and np.isfinite(c) and p < level <= c
        else:
            crossed = np.isfinite(p) and np.isfinite(c) and p > level >= c
        if crossed:
            return i
    return None


def _pre_cross_features(x: pd.DataFrame, i: int, level: float, direction: int, atr_prev: np.ndarray) -> dict:
    z = {}
    if i < 7 or i >= len(x):
        return z
    atr0 = float(atr_prev[i]) if i < len(atr_prev) else np.nan
    if not np.isfinite(atr0) or atr0 <= 0:
        return z

    o = x.open.to_numpy(float)
    h = x.high.to_numpy(float)
    l = x.low.to_numpy(float)
    c = x.close.to_numpy(float)

    pre6 = c[i - 6:i]
    dist6 = np.abs(pre6 - level)
    side6 = (pre6 < level) if direction > 0 else (pre6 > level)
    near6 = dist6 <= 0.5 * atr0
    first3 = float(np.median(dist6[:3]))
    last3 = float(np.median(dist6[3:]))

    a12 = max(0, i - 12)
    signs = np.sign(c[a12:i] - level)
    signs = signs[np.isfinite(signs)]
    flips = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) >= 2 else 0
    touch12 = int(np.sum((l[a12:i] <= level) & (h[a12:i] >= level)))

    rng = float(h[i] - l[i])
    body = float(abs(c[i] - o[i]))
    through = float(direction * (c[i] - level) / atr0)
    range_atr = rng / atr0
    body_atr = body / atr0
    close_loc = ((c[i] - l[i]) / rng if direction > 0 else (h[i] - c[i]) / rng) if rng > 0 else np.nan

    approach_side_ratio = float(np.mean(side6))
    near_ratio = float(np.mean(near6))
    compression = bool(last3 <= first3)

    # Fixed mechanical proxies only. These are not claimed Digash private rules.
    clean_proxy = bool(flips <= 1)
    impulse_proxy = bool(through >= 0.10 and range_atr >= 1.0 and np.isfinite(close_loc) and close_loc >= 0.60)
    prebuilt_proxy = bool(approach_side_ratio >= (5.0 / 6.0) and near_ratio >= 0.50 and compression)

    z.update({
        "atr_pre": atr0,
        "approach_side_ratio6": approach_side_ratio,
        "near_05atr_ratio6": near_ratio,
        "compression_3v3": compression,
        "pre_cross_flips12": flips,
        "pre_touch_bars12": touch12,
        "cross_close_through_atr": through,
        "cross_range_atr": range_atr,
        "cross_body_atr": body_atr,
        "cross_close_location": float(close_loc) if np.isfinite(close_loc) else np.nan,
        "clean_proxy": clean_proxy,
        "impulse_proxy": impulse_proxy,
        "prebuilt_proxy": prebuilt_proxy,
        "impulse_clean_proxy": bool(impulse_proxy and clean_proxy),
        "prebuilt_clean_proxy": bool(prebuilt_proxy and clean_proxy),
    })
    return z


def _outcomes(x: pd.DataFrame, i: int, direction: int) -> dict:
    z = {}
    sig = pd.to_datetime(x.signal_time, utc=True)
    sig_ns = sig.astype("int64").to_numpy()
    h = x.high.to_numpy(float)
    l = x.low.to_numpy(float)
    c = x.close.to_numpy(float)
    entry = float(c[i])
    if not np.isfinite(entry) or entry <= 0:
        return z
    cross_time = pd.Timestamp(sig.iloc[i])
    z["cross_time"] = cross_time
    z["cross_close"] = entry
    for hours, label in HORIZONS:
        deadline = (cross_time + pd.Timedelta(hours=hours)).value
        end = int(np.searchsorted(sig_ns, deadline, side="right")) - 1
        end = min(end, len(x) - 1)
        a = i + 1
        if end < a:
            z[f"ret_{label}_bps"] = np.nan
            z[f"mfe_{label}_bps"] = np.nan
            z[f"mae_{label}_bps"] = np.nan
            continue
        last = float(c[end])
        hh = float(np.max(h[a:end + 1]))
        ll = float(np.min(l[a:end + 1]))
        if direction > 0:
            ret = (last / entry - 1.0) * 10000.0
            mfe = (hh / entry - 1.0) * 10000.0
            mae = (1.0 - ll / entry) * 10000.0
        else:
            ret = (1.0 - last / entry) * 10000.0
            mfe = (1.0 - ll / entry) * 10000.0
            mae = (hh / entry - 1.0) * 10000.0
        z[f"ret_{label}_bps"] = ret
        z[f"mfe_{label}_bps"] = max(0.0, mfe)
        z[f"mae_{label}_bps"] = max(0.0, mae)
    return z


def process_group(records, config_path, datadir_s):
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    pair = records[0]["pair"]
    tf = records[0]["tf"]
    try:
        x, src = _load_detail(config, datadir, pair, tf)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_DATA", "elapsed_s": time.monotonic() - t0}
        times = pd.to_datetime([r["post_time"] for r in records], utc=True)
        x = x[(x.date >= times.min() - pd.Timedelta(days=2)) & (x.date <= times.max() + pd.Timedelta(hours=98))].reset_index(drop=True)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_RANGE", "elapsed_s": time.monotonic() - t0}
        atrp = _atr_prev(x)
        sig = pd.to_datetime(x.signal_time, utc=True)
        rows = []
        for r in records:
            t = pd.Timestamp(r["post_time"])
            level = float(r["level_price"])
            post_close = float(r["post_close"])
            direction = 1 if level >= post_close else -1
            i = _first_directed_cross(x, t, level, direction)
            z = dict(r)
            z.update({"detail_source": src, "direction": direction, "cross_found": bool(i is not None)})
            if i is not None:
                z["cross_delay_h"] = (pd.Timestamp(sig.iloc[i]) - t).total_seconds() / 3600.0
                z.update(_pre_cross_features(x, i, level, direction, atrp))
                z.update(_outcomes(x, i, direction))
            rows.append(z)
        return rows, {"pair": pair, "tf": tf, "status": "OK", "rows": len(rows), "detail": src, "elapsed_s": time.monotonic() - t0}
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic() - t0}


def _truthy(z: pd.DataFrame, name: str) -> pd.Series:
    if name not in z:
        return pd.Series(False, index=z.index)
    s = z[name]
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _med(g: pd.DataFrame, col: str):
    if col not in g:
        return np.nan
    s = pd.to_numeric(g[col], errors="coerce").dropna()
    return float(s.median()) if len(s) else np.nan


def _report(label: str, g: pd.DataFrame):
    print(f"\n{label} N={len(g)}", flush=True)
    if g.empty:
        return
    print(f"  cross-delay median={_med(g, 'cross_delay_h'):.2f}h", flush=True)
    for _, hlabel in HORIZONS:
        print(
            f"  {hlabel:3s} RET={_med(g, f'ret_{hlabel}_bps'):7.1f}bps | "
            f"MFE={_med(g, f'mfe_{hlabel}_bps'):7.1f} | MAE={_med(g, f'mae_{hlabel}_bps'):7.1f}",
            flush=True,
        )


def main() -> int:
    a = parse_args()
    v46dir = Path(a.v46dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    design = v46dir / "source_control_design.csv"
    if not design.exists():
        raise FileNotFoundError("Run Fidelity V4.6 first; V4.7 consumes source_control_design.csv.")
    d = pd.read_csv(design)
    d["post_time"] = pd.to_datetime(d.post_time, utc=True, errors="coerce")
    s = d[d.role.eq("SOURCE") & d.tf.isin(LOW_TFS)].copy().drop_duplicates(["query_id"])
    if s.empty:
        print("No LOW_TF source rows.", flush=True)
        return 2

    uq = s[["post_id", "post_time"]].drop_duplicates().sort_values(["post_time", "post_id"])
    cut_i = max(1, min(len(uq) - 1, int(np.floor(len(uq) * a.train_frac))))
    hold_ids = set(uq.iloc[cut_i:].post_id.astype(int))
    cutoff = uq.iloc[cut_i].post_time
    s["cohort"] = np.where(s.post_id.astype(int).isin(hold_ids), "HOLDOUT", "EARLY")

    print("=== DIGASH FIDELITY V4.7 — SOURCE BREAKOUT QUALITY ===", flush=True)
    print("NO PnL fitting. NO stop/target fitting. NO downloads.", flush=True)
    print("Uses exact public Digash low-TF levels and the finest cached detail available to inspect the first directed close-cross after the formation alert.", flush=True)
    print("CLEAN / IMPULSE / PREBUILT are fixed mechanical OHLCV proxies only; they are NOT claimed Digash private formulas.", flush=True)
    print("IMPULSE: >=0.10 pre-cross ATR close-through, >=1.0 ATR crossing range, directional close-location >=0.60.", flush=True)
    print("PREBUILT: >=5/6 prior closes on approach side, >=3/6 within 0.5 ATR, last3 median distance <= first3. CLEAN: <=1 close-side flip in prior 12 detail bars.", flush=True)
    print(f"source queries={len(s)} | unique posts={len(uq)} | holdout starts={cutoff} | workers={a.workers}", flush=True)

    groups = [g.to_dict("records") for _, g in s.groupby(["pair", "tf"], sort=True)]
    rows = []
    metas = []
    t0 = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(process_group, recs, a.config, a.datadir) for recs in groups]
        for f in as_completed(futs):
            rr, meta = f.result()
            done += 1
            rows.extend(rr)
            metas.append(meta)
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(futs) - done) / done if done else np.nan
            print(f"V4.7 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} detail={meta.get('detail','')} rows={len(rr)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(outdir / "coverage.csv", index=False)
    z = pd.DataFrame(rows)
    if z.empty:
        print("No result rows.", flush=True)
        return 2
    z["post_time"] = pd.to_datetime(z.post_time, utc=True, errors="coerce")
    z.to_csv(outdir / "breakout_quality_rows.csv", index=False)

    crossed = z[_truthy(z, "cross_found")].copy()
    print("\n=== DETAIL / CROSS COVERAGE ===", flush=True)
    print(f"rows={len(z)} | crosses<=72h={len(crossed)} ({len(crossed)/len(z)*100:.1f}%)", flush=True)
    if not crossed.empty:
        print("detail sources: " + ", ".join(f"{k}={v}" for k, v in crossed.detail_source.value_counts().items()), flush=True)

    selectors = [
        ("ALL_CROSSES", None),
        ("CLEAN_PROXY", "clean_proxy"),
        ("IMPULSE_PROXY", "impulse_proxy"),
        ("PREBUILT_PROXY", "prebuilt_proxy"),
        ("IMPULSE_CLEAN", "impulse_clean_proxy"),
        ("PREBUILT_CLEAN", "prebuilt_clean_proxy"),
    ]

    print("\n=== PREDECLARED BREAKOUT-QUALITY PROXIES ===", flush=True)
    for cohort_name in ("EARLY", "HOLDOUT"):
        cohort = crossed[crossed.cohort.eq(cohort_name)]
        print(f"\n--- {cohort_name} ---", flush=True)
        for label, col in selectors:
            g = cohort if col is None else cohort[_truthy(cohort, col)]
            _report(label, g)

    print("\n=== HOLDOUT BY PUBLISHED TF ===", flush=True)
    hold = crossed[crossed.cohort.eq("HOLDOUT")]
    for tf in LOW_TFS:
        gtf = hold[hold.tf.eq(tf)]
        _report(f"{tf} ALL", gtf)
        _report(f"{tf} IMPULSE_CLEAN", gtf[_truthy(gtf, "impulse_clean_proxy")])
        _report(f"{tf} PREBUILT_CLEAN", gtf[_truthy(gtf, "prebuilt_clean_proxy")])

    print("\n=== DECISION RULE ===", flush=True)
    print("Do not choose a proxy because it is best on EARLY. A source-faithful entry clue must show the same directional improvement on the later chronological HOLDOUT with adequate N.", flush=True)
    print("If none of these public-motivated OHLCV proxies improves HOLDOUT follow-through, stop inventing more candle filters: the missing Digash entry edge is likely in information absent from OHLCV, especially density/orderbook context or richer movement-quality inputs.", flush=True)
    print("If a proxy does hold up, freeze it and validate prospectively before converting excursions into trade-management rules.", flush=True)
    print(f"Reports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
