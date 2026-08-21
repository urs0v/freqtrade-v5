#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_digash_breakout_v32 import build_broad_activity, discover_cached_15m_usdt_futures
from analyze_digash_breakout_v33 import alt_net, pf, stats, fmt, clustered_bootstrap
from digash_v3_common import (
    PERIODS, TFS, Level, build_levels, load_5m, load_tf, prep_ohlcv,
    prepare_5m_with_activity, resample_from_15,
)
from digash_v31_events import detect_events, dedup_events, assign_targets, simulate

SEED = 20260821


def parse_args():
    p = argparse.ArgumentParser(description="Digash breakout V3.4 cross-asset holdout")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_breakout_v34")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=5000)
    return p.parse_args()


def _prepare_5m_without_activity(raw5: pd.DataFrame) -> pd.DataFrame:
    pre = prep_ohlcv(raw5, 5)
    dummy = pre[["signal_time"]].copy()
    return prepare_5m_with_activity(raw5, dummy)


def generate_pair_candidate(
    pair: str,
    config: dict,
    datadir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[str, pd.DataFrame | None, dict]:
    t0 = time.monotonic()
    try:
        raw15 = load_tf(config, datadir, pair, "15m")
        raw5, detail_source = load_5m(config, datadir, pair)
        if raw15.empty or raw5.empty:
            return pair, None, {
                "pair": pair, "status": "NO_DETAIL" if not raw15.empty else "NO_15M",
                "detail_source": detail_source, "elapsed_s": time.monotonic() - t0,
            }

        warm = pd.Timedelta(days=45)
        x15 = prep_ohlcv(raw15, 15)
        x5raw = prep_ohlcv(raw5, 5)
        x15 = x15[(x15.date >= start - warm) & (x15.date < end + pd.Timedelta(hours=8))].reset_index(drop=True)
        x5raw = x5raw[(x5raw.date >= start - warm) & (x5raw.date < end + pd.Timedelta(hours=8))].reset_index(drop=True)
        if x15.empty or x5raw.empty:
            return pair, None, {
                "pair": pair, "status": "NO_RANGE", "detail_source": detail_source,
                "elapsed_s": time.monotonic() - t0,
            }

        x5 = _prepare_5m_without_activity(x5raw)
        tfs = {
            "5m": x5[["date", "open", "high", "low", "close", "volume", "atr", "signal_time"]].copy(),
            "15m": x15,
            "1h": resample_from_15(x15, "1h", 60),
            "4h": resample_from_15(x15, "4h", 240),
        }

        levels: list[Level] = []
        next_id = 0
        for tf in TFS:
            for period in PERIODS:
                z = build_levels(tfs[tf], tf, period, next_id)
                levels.extend(z)
                next_id += len(z)
        if not levels:
            return pair, None, {
                "pair": pair, "status": "NO_LEVELS", "detail_source": detail_source,
                "elapsed_s": time.monotonic() - t0,
            }

        raw_events = detect_events(x5, levels)
        deduped = dedup_events(raw_events)
        selected = [
            e for e in deduped
            if e.setup == "H_BREAK" and e.tf == "4h" and int(e.period) == 30
        ]
        if not selected:
            return pair, pd.DataFrame(), {
                "pair": pair, "status": "OK", "detail_source": detail_source,
                "bars5": len(x5), "levels": len(levels), "raw_events": len(raw_events),
                "selected_events": 0, "candidate_events": 0, "elapsed_s": time.monotonic() - t0,
            }

        targets = assign_targets(selected, levels, x5)
        level_map = {z.level_id: z for z in levels}
        rows = []
        for i, e in enumerate(selected):
            row = simulate(x5, e, pair, targets.get(i, {}), level_map)
            if row is None:
                continue
            et = pd.Timestamp(row["entry_time"])
            if start <= et < end:
                rows.append(row)

        df = pd.DataFrame(rows)
        if not df.empty:
            fact = df["fact_proxy"].astype(bool)
            approach = pd.to_numeric(df["approach_no"], errors="coerce") >= 3
            df = df[fact & approach].copy()
            if not df.empty:
                df["ctx_time"] = pd.to_datetime(df["signal_time"], utc=True).dt.floor("15min")

        return pair, df, {
            "pair": pair, "status": "OK", "detail_source": detail_source,
            "bars5": len(x5), "levels": len(levels), "raw_events": len(raw_events),
            "selected_events": len(selected), "candidate_events": len(df),
            "elapsed_s": time.monotonic() - t0,
        }
    except Exception as e:
        return pair, None, {
            "pair": pair, "status": "ERROR", "error": f"{type(e).__name__}: {e}",
            "elapsed_s": time.monotonic() - t0,
        }


def _print_progress(done: int, total: int, usable: int, started: float) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed * (total - done) / done if done else None
    width = 24
    nbar = int(round(width * done / max(total, 1)))
    bar = "[" + "█" * nbar + "░" * (width - nbar) + "]"
    def ft(v):
        if v is None:
            return "--:--"
        v = int(v)
        h, r = divmod(v, 3600)
        m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    sys.stdout.write(
        "\r\033[K"
        + f"{bar} holdout {done}/{total} | usable {usable} | elapsed {ft(elapsed)} | ETA {ft(eta)}"
    )
    sys.stdout.flush()


def summarize_holdout(z: pd.DataFrame, outdir: Path, bootstrap: int) -> None:
    print("\n=== FROZEN HOLDOUT CANDIDATE ===", flush=True)
    s = stats(z)
    print(fmt("4H P30 + 3+ + TOP10", s), flush=True)

    split_rows = []
    print("\n=== HOLDOUT TIME ROBUSTNESS ===", flush=True)
    split_map = {
        "2022-2024": z[(z.entry_time >= "2022-01-01") & (z.entry_time < "2025-01-01")],
        "2025": z[(z.entry_time >= "2025-01-01") & (z.entry_time < "2026-01-01")],
        "2026": z[z.entry_time >= "2026-01-01"],
        "2025-2026": z[z.entry_time >= "2025-01-01"],
    }
    for name, g in split_map.items():
        ss = stats(g)
        split_rows.append({"split": name, **ss})
        print(fmt(name, ss), flush=True)
    for year, g in z.groupby(z.entry_time.dt.year):
        ss = stats(g)
        split_rows.append({"split": str(int(year)), **ss})
        print(fmt(str(int(year)), ss), flush=True)
    pd.DataFrame(split_rows).to_csv(outdir / "holdout_time_splits.csv", index=False)

    print("\n=== HOLDOUT SIDE ROBUSTNESS ===", flush=True)
    side_rows = []
    for side, g in z.groupby("side"):
        name = "LONG" if int(side) > 0 else "SHORT"
        ss = stats(g)
        side_rows.append({"side": name, **ss})
        print(fmt(name, ss), flush=True)
    pd.DataFrame(side_rows).to_csv(outdir / "holdout_sides.csv", index=False)

    print("\n=== HOLDOUT PAIRS ===", flush=True)
    pair_rows = []
    for pair, g in z.groupby("pair"):
        ss = stats(g)
        pair_rows.append({
            "pair": pair,
            "sum_net8_r": float(alt_net(g, 8.0).sum()),
            **ss,
        })
    pairdf = pd.DataFrame(pair_rows).sort_values("sum_net8_r", ascending=False)
    pairdf.to_csv(outdir / "holdout_by_pair.csv", index=False)
    for r in pairdf.itertuples(index=False):
        print(
            f"{r.pair:20s} N={r.n:3d} 8b={r.net8:+.3f}R/PF={r.pf8:.2f} "
            f"12b={r.net12:+.3f}R/PF={r.pf12:.2f} sum8={r.sum_net8_r:+.2f}R",
            flush=True,
        )

    loo_rows = []
    for pair in sorted(z.pair.unique()):
        g = z[z.pair != pair]
        ss = stats(g)
        loo_rows.append({"excluded": pair, **ss})
    loo = pd.DataFrame(loo_rows)
    loo.to_csv(outdir / "holdout_leave_one_pair_out.csv", index=False)
    if not loo.empty:
        print(
            f"pair-LOO 8bps PF range {loo.pf8.min():.2f} .. {loo.pf8.max():.2f}; "
            f"net range {loo.net8.min():+.3f}R .. {loo.net8.max():+.3f}R",
            flush=True,
        )
        print(
            f"pair-LOO 12bps PF range {loo.pf12.min():.2f} .. {loo.pf12.max():.2f}",
            flush=True,
        )

    print("\n=== HOLDOUT MONTH-CLUSTER BOOTSTRAP ===", flush=True)
    boot_rows = []
    for bps in (8.0, 12.0, 16.0):
        mean, lo, hi = clustered_bootstrap(z, bps, bootstrap)
        boot_rows.append({"bps": bps, "mean_r": mean, "ci025": lo, "ci975": hi, "reps": bootstrap})
        print(
            f"{bps:>4.0f} bps: mean={mean:+.3f}R | month-cluster 95% CI [{lo:+.3f}, {hi:+.3f}]",
            flush=True,
        )
    pd.DataFrame(boot_rows).to_csv(outdir / "holdout_bootstrap.csv", index=False)

    s8 = stats(z)
    pos_years = 0
    years = 0
    if not z.empty:
        yy = z.assign(_n8=alt_net(z, 8.0)).groupby(z.entry_time.dt.year)["_n8"].mean()
        pos_years = int((yy > 0).sum())
        years = int(len(yy))
    recent = stats(split_map["2025-2026"])
    loo_min_pf8 = float(loo.pf8.min()) if not loo.empty else np.nan

    gate = (
        s8["n"] >= 40
        and s8["net8"] > 0 and s8["pf8"] >= 1.15
        and s8["net12"] > 0 and s8["pf12"] >= 1.05
        and recent["n"] >= 15 and recent["net8"] > 0
        and pos_years >= min(2, years)
        and np.isfinite(loo_min_pf8) and loo_min_pf8 >= 1.00
    )
    print("\n=== PREDECLARED CROSS-ASSET HOLDOUT GATE ===", flush=True)
    print(
        "Requires N>=40, PF8>=1.15 with positive 8bps expectancy, PF12>=1.05 with positive "
        "12bps expectancy, positive 2025-2026 (N>=15), >=2 positive years when available, "
        "and pair-LOO min PF8>=1.00.",
        flush=True,
    )
    print(f"FROZEN_4H_P30_APPROACH3PLUS {'PASS' if gate else 'FAIL'}", flush=True)
    print("Cross-asset holdout is stronger than another development-sample slice, but is not future-time OOS proof.", flush=True)


def main() -> int:
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(a.config).read_text())
    datadir = Path(a.datadir)
    dev_pairs = set(config.get("exchange", {}).get("pair_whitelist", []))
    cached = discover_cached_15m_usdt_futures(config, datadir)
    holdout_pairs = [p for p in cached if p not in dev_pairs]

    print("=== DIGASH BREAKOUT V3.4 — CROSS-ASSET HOLDOUT ===", flush=True)
    print("FROZEN CANDIDATE: 4h / period30 / factual breakout / approach>=3 / broad-market top10.", flush=True)
    print("No parameter tuning. No age filter. No side filter. No downloads.", flush=True)
    print(f"development universe={len(dev_pairs)} | cached 15m universe={len(cached)} | raw holdout={len(holdout_pairs)}", flush=True)
    if not holdout_pairs:
        raise RuntimeError("No cached 15m pairs outside the V3/V3.1 development whitelist")

    start = pd.Timestamp(a.start, tz="UTC")
    end = pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)
    workers = max(1, min(a.workers, len(holdout_pairs)))
    results = []
    coverage = []
    t0 = time.monotonic()
    usable = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(generate_pair_candidate, p, config, datadir, start, end): p
            for p in holdout_pairs
        }
        done = 0
        for f in as_completed(futs):
            pair, df, meta = f.result()
            done += 1
            coverage.append(meta)
            if df is not None:
                usable += 1
                if not df.empty:
                    results.append(df)
            _print_progress(done, len(holdout_pairs), usable, t0)
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()

    cov = pd.DataFrame(coverage)
    cov.to_csv(outdir / "holdout_coverage.csv", index=False)
    ok_pairs = cov[cov.status.eq("OK")]["pair"].nunique() if not cov.empty else 0
    candidate_rows = int(cov.get("candidate_events", pd.Series(dtype=float)).fillna(0).sum()) if not cov.empty else 0
    print(
        f"Holdout generation done: usable-detail pairs={ok_pairs}/{len(holdout_pairs)} | "
        f"pre-activity candidate rows={candidate_rows} | elapsed={time.monotonic()-t0:.1f}s",
        flush=True,
    )
    if not results:
        print("No frozen candidate events on usable holdout pairs.", flush=True)
        print("=== PREDECLARED CROSS-ASSET HOLDOUT GATE ===\nFROZEN_4H_P30_APPROACH3PLUS FAIL", flush=True)
        return 0

    events = pd.concat(results, ignore_index=True)
    events["signal_time"] = pd.to_datetime(events["signal_time"], utc=True)
    events["entry_time"] = pd.to_datetime(events["entry_time"], utc=True)
    events["ctx_time"] = pd.to_datetime(events["ctx_time"], utc=True)

    broad_dir = outdir / "broad_activity"
    broad_dir.mkdir(parents=True, exist_ok=True)
    broad, universe_meta = build_broad_activity(events, config, datadir, a.workers, broad_dir)
    merged = events.merge(broad, on=["pair", "ctx_time"], how="left", validate="many_to_one")
    merged["broad_active_top10"] = merged["broad_active_top10"].fillna(False).astype(bool)
    merged["broad_active_top5"] = merged["broad_active_top5"].fillna(False).astype(bool)
    merged.to_csv(outdir / "holdout_candidate_all_activity.csv", index=False)

    frozen = merged[merged.broad_active_top10].copy()
    frozen.to_csv(outdir / "holdout_frozen_trades.csv", index=False)

    print("\n=== HOLDOUT BROAD-RANK COVERAGE ===", flush=True)
    print(
        f"broad discovered={universe_meta['discovered_pairs']} contributing={universe_meta['contributing_pairs']} "
        f"median universe={universe_meta['median_universe_n']:.0f} "
        f"median liquid70={universe_meta['median_liquid_n']:.0f} "
        f"candidate broad coverage={merged.universe_n.notna().mean()*100:.1f}%",
        flush=True,
    )
    print(f"candidate rows before top10={len(merged)} | after frozen broad top10={len(frozen)}", flush=True)

    if frozen.empty:
        print("\n=== PREDECLARED CROSS-ASSET HOLDOUT GATE ===", flush=True)
        print("FROZEN_4H_P30_APPROACH3PLUS FAIL (no broad-top10 holdout trades)", flush=True)
        return 0

    summarize_holdout(frozen, outdir, a.bootstrap)
    print(f"\nReports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
