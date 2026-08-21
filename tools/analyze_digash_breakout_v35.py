#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import get_datahandler
from freqtrade.enums import TradingMode

from analyze_digash_breakout_v34 import summarize_holdout, _prepare_5m_without_activity
from digash_v3_common import (
    PERIODS, TFS, Level, build_levels, load_5m, load_tf, prep_ohlcv,
    resample_from_15,
)
from digash_v31_events import detect_events, dedup_events, assign_targets, simulate

VOL_FLOOR_24H = 70_000_000.0


def parse_args():
    p = argparse.ArgumentParser(description="Digash breakout V3.5 cache-union untouched-asset holdout")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_breakout_v35")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--bootstrap", type=int, default=5000)
    return p.parse_args()


def _ct_value(v) -> str:
    return str(getattr(v, "value", v)).lower()


def discover_cached_sets(config: dict, datadir: Path) -> dict[str, set[str]]:
    dh = get_datahandler(datadir, config.get("dataformat_ohlcv"))
    combos = dh.ohlcv_get_available_data(datadir, TradingMode.FUTURES)
    out = {"1m": set(), "5m": set(), "15m": set()}
    for pair, timeframe, candle_type in combos:
        if timeframe not in out:
            continue
        if _ct_value(candle_type) != "futures":
            continue
        if not pair.endswith("/USDT:USDT"):
            continue
        out[timeframe].add(pair)
    return out


def resample_raw(x: pd.DataFrame, rule: str) -> pd.DataFrame:
    if x.empty:
        return pd.DataFrame()
    z = x[["date", "open", "high", "low", "close", "volume"]].copy()
    z["date"] = pd.to_datetime(z["date"], utc=True)
    z = z.sort_values("date").drop_duplicates("date")
    y = (
        z.set_index("date")
        .resample(rule, label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna().reset_index()
    )
    return y


def load_15m_union(config: dict, datadir: Path, pair: str) -> tuple[pd.DataFrame, str]:
    d15 = load_tf(config, datadir, pair, "15m")
    if not d15.empty:
        return d15[["date", "open", "high", "low", "close", "volume"]].copy(), "15m"
    d5, src = load_5m(config, datadir, pair)
    if d5.empty:
        return pd.DataFrame(), "none"
    return resample_raw(d5, "15min"), f"{src}->15m"


def generate_pair_candidate(
    pair: str,
    config: dict,
    datadir: Path,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[str, pd.DataFrame | None, dict]:
    t0 = time.monotonic()
    try:
        raw5, detail_source = load_5m(config, datadir, pair)
        if raw5.empty:
            return pair, None, {
                "pair": pair, "status": "NO_DETAIL", "detail_source": detail_source,
                "elapsed_s": time.monotonic() - t0,
            }
        raw15, fifteen_source = load_15m_union(config, datadir, pair)
        if raw15.empty:
            return pair, None, {
                "pair": pair, "status": "NO_15M_DERIVABLE", "detail_source": detail_source,
                "fifteen_source": fifteen_source, "elapsed_s": time.monotonic() - t0,
            }

        warm = pd.Timedelta(days=45)
        raw5 = raw5.copy()
        raw5["date"] = pd.to_datetime(raw5["date"], utc=True)
        raw15 = raw15.copy()
        raw15["date"] = pd.to_datetime(raw15["date"], utc=True)
        raw5 = raw5[(raw5.date >= start - warm) & (raw5.date < end + pd.Timedelta(hours=8))].reset_index(drop=True)
        raw15 = raw15[(raw15.date >= start - warm) & (raw15.date < end + pd.Timedelta(hours=8))].reset_index(drop=True)
        if raw5.empty or raw15.empty:
            return pair, None, {
                "pair": pair, "status": "NO_RANGE", "detail_source": detail_source,
                "fifteen_source": fifteen_source, "elapsed_s": time.monotonic() - t0,
            }

        x5 = _prepare_5m_without_activity(raw5)
        x15 = prep_ohlcv(raw15, 15)
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
                "fifteen_source": fifteen_source, "elapsed_s": time.monotonic() - t0,
            }

        raw_events = detect_events(x5, levels)
        deduped = dedup_events(raw_events)
        selected = [e for e in deduped if e.setup == "H_BREAK" and e.tf == "4h" and int(e.period) == 30]
        if not selected:
            return pair, pd.DataFrame(), {
                "pair": pair, "status": "OK", "detail_source": detail_source,
                "fifteen_source": fifteen_source, "bars5": len(x5), "levels": len(levels),
                "raw_events": len(raw_events), "selected_events": 0, "candidate_events": 0,
                "elapsed_s": time.monotonic() - t0,
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
            "fifteen_source": fifteen_source, "bars5": len(x5), "levels": len(levels),
            "raw_events": len(raw_events), "selected_events": len(selected),
            "candidate_events": len(df), "elapsed_s": time.monotonic() - t0,
        }
    except Exception as e:
        return pair, None, {
            "pair": pair, "status": "ERROR", "error": f"{type(e).__name__}: {e}",
            "elapsed_s": time.monotonic() - t0,
        }


def load_activity_rows(
    pair: str,
    config: dict,
    datadir: Path,
    needed_ctx: pd.DatetimeIndex,
    min_ctx: pd.Timestamp,
    max_ctx: pd.Timestamp,
) -> tuple[str, pd.DataFrame | None, str | None]:
    try:
        d15, src = load_15m_union(config, datadir, pair)
        if d15.empty:
            return pair, None, "no_15m"
        x = d15[["date", "open", "high", "low", "close", "volume"]].copy()
        x["date"] = pd.to_datetime(x["date"], utc=True)
        x = x[(x.date >= min_ctx - pd.Timedelta(days=2)) & (x.date <= max_ctx)].copy()
        if len(x) < 20:
            return pair, None, "too_short"
        x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)
        q = x["volume"].to_numpy(float) * x["close"].to_numpy(float)
        close = x["close"].astype(float)
        prev = close.shift()
        tr = pd.concat([
            x["high"].astype(float) - x["low"].astype(float),
            (x["high"].astype(float) - prev).abs(),
            (x["low"].astype(float) - prev).abs(),
        ], axis=1).max(axis=1)
        x["ret_24h"] = close.pct_change(96)
        x["quote_vol_24h"] = pd.Series(q, index=x.index).rolling(96, min_periods=48).sum()
        x["natr_15m"] = tr.rolling(14, min_periods=7).mean() / close * 100.0
        x["ctx_time"] = x["date"] + pd.Timedelta(minutes=15)
        x = x[x.ctx_time.isin(needed_ctx)][["ctx_time", "ret_24h", "quote_vol_24h", "natr_15m"]].copy()
        if x.empty:
            return pair, None, "no_requested_times"
        x["pair"] = pair
        x["activity_source"] = src
        return pair, x, None
    except Exception as e:
        return pair, None, f"{type(e).__name__}: {e}"


def build_union_activity(
    events: pd.DataFrame,
    config: dict,
    datadir: Path,
    universe_pairs: list[str],
    workers: int,
    outdir: Path,
) -> tuple[pd.DataFrame, dict]:
    needed_ctx = pd.DatetimeIndex(events["ctx_time"].drop_duplicates().sort_values())
    min_ctx, max_ctx = needed_ctx.min(), needed_ctx.max()
    frames = []
    failures = []
    workers = max(1, min(int(workers), len(universe_pairs)))
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(load_activity_rows, p, config, datadir, needed_ctx, min_ctx, max_ctx): p
            for p in universe_pairs
        }
        done = 0
        for f in as_completed(futs):
            pair, x, err = f.result()
            done += 1
            if x is not None and not x.empty:
                frames.append(x)
            elif err not in ("no_requested_times", "too_short", "no_15m"):
                failures.append((pair, err or "unknown"))
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(universe_pairs) - done) / done if done else None
            width = 24
            nbar = int(round(width * done / max(len(universe_pairs), 1)))
            bar = "[" + "█" * nbar + "░" * (width - nbar) + "]"
            def ft(v):
                if v is None:
                    return "--:--"
                v = int(v); h, r = divmod(v, 3600); m, s = divmod(r, 60)
                return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            sys.stdout.write(
                "\r\033[K" + f"{bar} union-activity {done}/{len(universe_pairs)} | loaded {len(frames)} | elapsed {ft(elapsed)} | ETA {ft(eta)}"
            )
            sys.stdout.flush()
    sys.stdout.write("\r\033[K"); sys.stdout.flush()
    if failures:
        pd.DataFrame(failures, columns=["pair", "error"]).to_csv(outdir / "union_activity_failures.csv", index=False)
    if not frames:
        raise RuntimeError("No union activity rows")

    panel = pd.concat(frames, ignore_index=True)
    panel["liquid70"] = panel["quote_vol_24h"] >= VOL_FLOOR_24H
    panel["universe_n"] = panel.groupby("ctx_time")["pair"].transform("nunique")
    panel["liquid_n"] = panel.groupby("ctx_time")["liquid70"].transform("sum")

    specs = {
        "growth_rank": ("ret_24h", False),
        "decline_rank": ("ret_24h", True),
        "volatility_rank": ("natr_15m", False),
        "volume_rank": ("quote_vol_24h", False),
    }
    for rank_name, (col, ascending) in specs.items():
        masked = panel[col].where(panel.liquid70)
        panel[rank_name] = masked.groupby(panel.ctx_time).rank(ascending=ascending, method="first")
    cols = []
    for stem in ("growth", "decline", "volatility", "volume"):
        c = f"union_top10_{stem}"
        panel[c] = panel.liquid70 & (panel[f"{stem}_rank"] <= 10)
        cols.append(c)
    panel["union_active_top10"] = panel[cols].any(axis=1)

    needed_pairs = set(events.pair.unique())
    keep = [
        "pair", "ctx_time", "universe_n", "liquid_n", "liquid70",
        "growth_rank", "decline_rank", "volatility_rank", "volume_rank",
        *cols, "union_active_top10",
    ]
    flags = panel[panel.pair.isin(needed_pairs)][keep].copy()
    coverage = panel.groupby("ctx_time", as_index=False).agg(
        universe_n=("pair", "nunique"), liquid_n=("liquid70", "sum")
    )
    coverage.to_csv(outdir / "union_universe_coverage.csv", index=False)
    return flags, {
        "ranking_pairs_requested": len(universe_pairs),
        "ranking_pairs_contributing": len(frames),
        "median_universe_n": float(coverage.universe_n.median()),
        "median_liquid_n": float(coverage.liquid_n.median()),
        "min_universe_n": int(coverage.universe_n.min()),
        "max_universe_n": int(coverage.universe_n.max()),
    }


def print_progress(done: int, total: int, usable: int, started: float) -> None:
    elapsed = time.monotonic() - started
    eta = elapsed * (total - done) / done if done else None
    width = 24
    nbar = int(round(width * done / max(total, 1)))
    bar = "[" + "█" * nbar + "░" * (width - nbar) + "]"
    def ft(v):
        if v is None:
            return "--:--"
        v = int(v); h, r = divmod(v, 3600); m, s = divmod(r, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
    sys.stdout.write(
        "\r\033[K" + f"{bar} NEW holdout {done}/{total} | usable {usable} | elapsed {ft(elapsed)} | naive ETA {ft(eta)}"
    )
    sys.stdout.flush()


def main() -> int:
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(a.config).read_text())
    datadir = Path(a.datadir)
    dev_pairs = set(config.get("exchange", {}).get("pair_whitelist", []))
    sets = discover_cached_sets(config, datadir)
    direct15 = sets["15m"]
    detail = sets["5m"] | sets["1m"]
    rank_universe = sorted(direct15 | detail)

    # V3.4 already exposed every non-development pair with direct 15m cache.
    # V3.5 primary holdout is therefore ONLY previously unseen assets that have detail
    # (5m or 1m) but lacked a direct 15m file. Their 15m is causally derived from detail.
    seen_before = dev_pairs | direct15
    new_pairs = sorted(detail - seen_before)

    inventory_rows = []
    for p in rank_universe:
        inventory_rows.append({
            "pair": p, "has_1m": p in sets["1m"], "has_5m": p in sets["5m"],
            "has_15m": p in direct15, "development": p in dev_pairs,
            "new_v35_holdout": p in new_pairs,
        })
    pd.DataFrame(inventory_rows).to_csv(outdir / "cache_inventory.csv", index=False)

    print("=== DIGASH BREAKOUT V3.5 — CACHE-UNION NEW-ASSET HOLDOUT ===", flush=True)
    print("FROZEN CANDIDATE unchanged: 4h / p30 / factual breakout / approach>=3 / broad top10.", flush=True)
    print("No age/side/pair tuning. No downloads. 15m may only be causally resampled from existing 5m/1m.", flush=True)
    print(
        f"cache counts: 1m={len(sets['1m'])} | 5m={len(sets['5m'])} | 15m={len(direct15)} | "
        f"union rank universe={len(rank_universe)} | NEW untouched assets={len(new_pairs)}",
        flush=True,
    )
    if not new_pairs:
        print("No previously unseen detail-only assets exist in local cache; V3.5 cannot enlarge the holdout without new data.", flush=True)
        print("=== PREDECLARED NEW-ASSET HOLDOUT GATE ===\nFROZEN_4H_P30_APPROACH3PLUS FAIL (N=0 new assets)", flush=True)
        return 0

    start = pd.Timestamp(a.start, tz="UTC")
    end = pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=1)
    workers = max(1, min(a.workers, len(new_pairs)))
    results = []
    coverage = []
    t0 = time.monotonic()
    usable = 0

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(generate_pair_candidate, p, config, datadir, start, end): p for p in new_pairs}
        done = 0
        for f in as_completed(futs):
            pair, df, meta = f.result()
            done += 1
            coverage.append(meta)
            if df is not None:
                usable += 1
                if not df.empty:
                    results.append(df)
            print_progress(done, len(new_pairs), usable, t0)
    sys.stdout.write("\r\033[K"); sys.stdout.flush()

    cov = pd.DataFrame(coverage)
    cov.to_csv(outdir / "new_holdout_coverage.csv", index=False)
    ok_pairs = cov[cov.status.eq("OK")].pair.nunique() if not cov.empty else 0
    candidate_rows = int(cov.get("candidate_events", pd.Series(dtype=float)).fillna(0).sum()) if not cov.empty else 0
    print(
        f"New-asset generation done: usable={ok_pairs}/{len(new_pairs)} | pre-activity candidates={candidate_rows} | elapsed={time.monotonic()-t0:.1f}s",
        flush=True,
    )
    if not results:
        print("No frozen candidate events on new cached assets.", flush=True)
        print("=== PREDECLARED NEW-ASSET HOLDOUT GATE ===\nFROZEN_4H_P30_APPROACH3PLUS FAIL", flush=True)
        return 0

    events = pd.concat(results, ignore_index=True)
    events["signal_time"] = pd.to_datetime(events.signal_time, utc=True)
    events["entry_time"] = pd.to_datetime(events.entry_time, utc=True)
    events["ctx_time"] = pd.to_datetime(events.ctx_time, utc=True)

    broad_dir = outdir / "union_activity"
    broad_dir.mkdir(parents=True, exist_ok=True)
    broad, meta = build_union_activity(events, config, datadir, rank_universe, a.workers, broad_dir)
    merged = events.merge(broad, on=["pair", "ctx_time"], how="left", validate="many_to_one")
    merged["union_active_top10"] = merged["union_active_top10"].fillna(False).astype(bool)
    merged.to_csv(outdir / "new_candidate_all_activity.csv", index=False)
    frozen = merged[merged.union_active_top10].copy()
    frozen.to_csv(outdir / "new_holdout_frozen_trades.csv", index=False)

    print("\n=== V3.5 UNION-RANK COVERAGE ===", flush=True)
    print(
        f"ranking requested={meta['ranking_pairs_requested']} contributing={meta['ranking_pairs_contributing']} | "
        f"median universe={meta['median_universe_n']:.0f} median liquid70={meta['median_liquid_n']:.0f} | "
        f"candidate coverage={merged.universe_n.notna().mean()*100:.1f}%",
        flush=True,
    )
    print(f"NEW candidate rows before top10={len(merged)} | after frozen union top10={len(frozen)}", flush=True)

    if frozen.empty:
        print("\n=== PREDECLARED NEW-ASSET HOLDOUT GATE ===", flush=True)
        print("FROZEN_4H_P30_APPROACH3PLUS FAIL (no top10 new-asset trades)", flush=True)
        return 0

    # Reuse the exact V3.4 reporting/gate. This is still a new-only untouched-asset sample.
    summarize_holdout(frozen, outdir, a.bootstrap)
    print("\nNOTE: V3.5 primary sample excludes all 20 development pairs AND all 14 V3.4 direct-15m holdout pairs.", flush=True)
    print(f"Reports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
