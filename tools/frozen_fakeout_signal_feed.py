#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

import digash_v3_common as dc
import prospective_fakeout_v2 as p2


DEFAULT_OUT = "/freqtrade/user_data/frozen_fakeout_feed"
DEFAULT_CUTOFF_STATE = "/freqtrade/user_data/prospective_fakeout_v2/state.json"


def parse_args():
    p = argparse.ArgumentParser(description="Executable signal feed for FrozenFakeoutV1 Freqtrade dry-run")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--cutoff-state", default=DEFAULT_CUTOFF_STATE)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=5)
    return p.parse_args()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _cutoff(outdir: Path, source: Path) -> pd.Timestamp:
    state_path = outdir / "state.json"
    if state_path.exists():
        z = json.loads(state_path.read_text())
        return pd.Timestamp(z["cutoff"])

    if source.exists():
        src = json.loads(source.read_text())
        cutoff = pd.Timestamp(src["cutoff"])
        origin = str(source)
    else:
        cutoff = p2.next_5m_boundary(utc_now())
        origin = "new_next_5m_boundary"

    outdir.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps({
        "experiment": "frozen_fakeout_freqtrade_feed",
        "created_at": utc_now().isoformat(),
        "cutoff": cutoff.isoformat(),
        "cutoff_origin": origin,
        "signal": {
            "setup": "FAKEOUT",
            "activity_min": p2.THRESH,
            "risk_min_bps": p2.RISK_MIN_BPS,
            "risk_max_bps": 3000.0,
            "rr": p2.RR,
            "hold_bars_5m": p2.HOLD_BARS,
            "causal_activity": True,
            "causal_dedup": True,
        },
    }, indent=2))
    return cutoff


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _sync_tf(cfg: dict, datadir: Path, outdir: Path, pair: str, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    minutes = 5 if tf == "5m" else 15
    cache = outdir / "market_cache" / tf / f"{p2.symbol(pair)}.csv"
    live = p2.read_csv_candles(cache)
    base = dc.load_tf(cfg, datadir, pair, tf)
    if not base.empty:
        base = base[["date", "open", "high", "low", "close", "volume"]].copy()
        base["date"] = pd.to_datetime(base["date"], utc=True).astype("datetime64[ns, UTC]")
        base = base.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    latest = []
    if not base.empty:
        latest.append(pd.Timestamp(base.date.max()))
    if not live.empty:
        latest.append(pd.Timestamp(live.date.max()))
    fetch_start = max(latest) - pd.Timedelta(minutes=minutes) if latest else now - pd.Timedelta(days=75)

    fetched = p2.fetch_klines(
        pair,
        tf,
        int(fetch_start.timestamp() * 1000),
        int(now.timestamp() * 1000),
    )
    if not fetched.empty:
        if tf == "5m":
            # Keep the currently-open 5m bar as an ENTRY STUB. detect_events()
            # never processes the final row, so the previous closed candle can
            # use this row's immutable open without reading its future close.
            current_open = now.floor("5min")
            fetched = fetched[fetched.date <= current_open].copy()
        else:
            # Higher-timeframe context remains completed-candle-only.
            fetched = fetched[fetched.date + pd.Timedelta(minutes=minutes) <= now].copy()

    parts = [z for z in (live, fetched) if z is not None and not z.empty]
    if parts:
        live_new = (
            pd.concat(parts, ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
            .reset_index(drop=True)
        )
        live_new = live_new[live_new.date >= now - pd.Timedelta(days=75)].reset_index(drop=True)
        _write_csv(live_new, cache)
    else:
        live_new = live

    combined = [z for z in (base, live_new) if z is not None and not z.empty]
    if not combined:
        return pd.DataFrame()
    return (
        pd.concat(combined, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _load_cached_combined(cfg: dict, datadir: Path, outdir: Path, pair: str, tf: str) -> pd.DataFrame:
    cache = outdir / "market_cache" / tf / f"{p2.symbol(pair)}.csv"
    live = p2.read_csv_candles(cache)
    base = dc.load_tf(cfg, datadir, pair, tf)
    if not base.empty:
        base = base[["date", "open", "high", "low", "close", "volume"]].copy()
        base["date"] = pd.to_datetime(base["date"], utc=True).astype("datetime64[ns, UTC]")
        base = base.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    parts = [z for z in (base, live) if z is not None and not z.empty]
    if not parts:
        return pd.DataFrame()
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def _worker(pair: str, config: str, datadir_s: str, outdir_s: str, cutoff_s: str, now_s: str):
    cfg = json.loads(Path(config).read_text())
    datadir = Path(datadir_s)
    outdir = Path(outdir_s)
    now = pd.Timestamp(now_s)
    cutoff = pd.Timestamp(cutoff_s)
    raw5 = _load_cached_combined(cfg, datadir, outdir, pair, "5m")
    raw15 = _load_cached_combined(cfg, datadir, outdir, pair, "15m")
    return p2.compute_pair(pair, raw5, raw15, cutoff, now)


def _still_executable(r: pd.Series, checked_at: pd.Timestamp) -> tuple[bool, str]:
    """Reject a signal if stop/target was already touched before Freqtrade can enter."""
    try:
        entry = pd.Timestamp(r["entry_time"])
        pair = str(r["pair"])
        side = int(float(r["side"]))
        stop = float(r["stop_price"])
        target = float(r["target_price"])
        if checked_at.floor("5min") != entry:
            return False, "STALE_BUCKET"
        k = p2.fetch_klines(
            pair,
            "5m",
            int(entry.timestamp() * 1000),
            int(checked_at.timestamp() * 1000),
        )
        k = k[k.date.eq(entry)]
        if k.empty:
            return False, "NO_CURRENT_KLINE"
        c = k.iloc[-1]
        if side > 0:
            if float(c.low) <= stop:
                return False, "STOP_TOUCHED_PRE_ENTRY"
            if float(c.high) >= target:
                return False, "TARGET_TOUCHED_PRE_ENTRY"
        else:
            if float(c.high) >= stop:
                return False, "STOP_TOUCHED_PRE_ENTRY"
            if float(c.low) <= target:
                return False, "TARGET_TOUCHED_PRE_ENTRY"
        return True, "OK"
    except Exception as e:
        return False, f"RECHECK_ERROR:{type(e).__name__}"


def run_once(a, cutoff: pd.Timestamp):
    out = Path(a.outdir)
    cfg = json.loads(Path(a.config).read_text())
    datadir = Path(a.datadir)
    scan_started = utc_now().floor("s")
    bucket = scan_started.floor("5min")

    sync_status = []
    for n, pair in enumerate(p2.FROZEN_PAIRS, 1):
        try:
            x5 = _sync_tf(cfg, datadir, out, pair, "5m", scan_started)
            _sync_tf(cfg, datadir, out, pair, "15m", scan_started)
            has_stub = bool(len(x5) and pd.Timestamp(x5.date.max()) == bucket)
            sync_status.append({"pair": pair, "status": "OK", "has_entry_stub": has_stub})
            print(f"sync {n:2d}/{len(p2.FROZEN_PAIRS)} {pair:24s} stub={int(has_stub)}", flush=True)
        except Exception as e:
            sync_status.append({"pair": pair, "status": f"ERROR:{type(e).__name__}:{e}", "has_entry_stub": False})
            print(f"sync {n:2d}/{len(p2.FROZEN_PAIRS)} {pair:24s} ERROR {type(e).__name__}: {e}", flush=True)

    good = [r["pair"] for r in sync_status if r["status"] == "OK" and r["has_entry_stub"]]
    rows = []
    with ProcessPoolExecutor(max_workers=max(1, min(int(a.workers), len(good) or 1))) as ex:
        futs = {
            ex.submit(_worker, pair, a.config, a.datadir, a.outdir, cutoff.isoformat(), scan_started.isoformat()): pair
            for pair in good
        }
        done = 0
        for fut in as_completed(futs):
            pair = futs[fut]
            done += 1
            try:
                rr = fut.result()
                rows.extend(rr)
                print(f"scan {done:2d}/{len(good)} {pair:24s} signals={len(rr)}", flush=True)
            except Exception as e:
                print(f"scan {done:2d}/{len(good)} {pair:24s} ERROR {type(e).__name__}: {e}", flush=True)

    z = pd.DataFrame(rows)
    if z.empty:
        z = pd.DataFrame(columns=[
            "signal_id", "pair", "signal_time", "entry_time", "entry_price", "side",
            "stop_price", "target_price", "risk_bps", "activity_score", "status", "exit_reason",
        ])
    else:
        z["signal_time"] = pd.to_datetime(z["signal_time"], utc=True)
        z["entry_time"] = pd.to_datetime(z["entry_time"], utc=True)
        z = z.drop_duplicates("signal_id", keep="last").sort_values(["entry_time", "pair"]).reset_index(drop=True)

    checked_at = utc_now().floor("s")
    z["published_at"] = checked_at.isoformat()
    z["feed_eligible"] = False
    z["feed_reason"] = "NOT_CURRENT"
    if len(z):
        candidate_mask = (
            z["entry_time"].eq(bucket)
            & z["status"].astype(str).eq("OPEN")
            & pd.to_numeric(z["risk_bps"], errors="coerce").between(p2.RISK_MIN_BPS, 3000.0)
        )
        for idx in z.index[candidate_mask]:
            ok, reason = _still_executable(z.loc[idx], checked_at)
            z.at[idx, "feed_eligible"] = bool(ok)
            z.at[idx, "feed_reason"] = reason
        z["publish_delay_sec"] = (checked_at - z["entry_time"]).dt.total_seconds()
    else:
        z["publish_delay_sec"] = pd.Series(dtype=float)

    active = z[z["feed_eligible"].eq(True)].copy() if len(z) else z.copy()
    _write_csv(z, out / "signals.csv")
    _write_csv(active, out / "active_signals.csv")
    _write_csv(pd.DataFrame(sync_status), out / "coverage.csv")

    summary = {
        "cutoff": cutoff.isoformat(),
        "scan_started": scan_started.isoformat(),
        "published_at": checked_at.isoformat(),
        "entry_bucket": bucket.isoformat(),
        "pairs_with_entry_stub": len(good),
        "signals_since_cutoff": int(len(z)),
        "active_for_freqtrade": int(len(active)),
        "active_ids": active["signal_id"].astype(str).tolist() if len(active) else [],
        "publish_delay_sec": float((checked_at - bucket).total_seconds()),
    }
    (out / "snapshot.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FROZEN FAKEOUT EXECUTABLE FEED ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    a = parse_args()
    out = Path(a.outdir)
    cutoff = _cutoff(out, Path(a.cutoff_state))
    print("=== FROZEN FAKEOUT FREQTRADE SIGNAL FEED ===", flush=True)
    print(f"cutoff={cutoff.isoformat()} | paper/dry-run only", flush=True)

    if not a.loop:
        run_once(a, cutoff)
        return 0

    last_bucket = None
    while True:
        now = utc_now()
        bucket = now.floor("5min")
        # 20 seconds gives Binance time to expose the immutable next-candle open.
        # Freqtrade still receives most of the same 5m candle for real execution.
        if now.second >= 20 and bucket != last_bucket:
            try:
                run_once(a, cutoff)
                last_bucket = bucket
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"cycle ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(max(2, int(a.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
