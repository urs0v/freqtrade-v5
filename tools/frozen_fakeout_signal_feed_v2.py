#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import breakout_retest_profit_v1 as v1
import digash_v3_common as dc
import frozen_fakeout_incremental as inc
import prospective_fakeout_v2 as p2
from frozen_fakeout_signal_feed import (
    _cutoff,
    _load_cached_combined,
    _still_executable,
    _sync_pair,
    _write_csv,
    utc_now,
)

DEFAULT_OUT = "/freqtrade/user_data/frozen_fakeout_feed"
DEFAULT_CUTOFF_STATE = "/freqtrade/user_data/prospective_fakeout_v2/state.json"
DEFAULT_REFERENCE = "/freqtrade/user_data/breakout_retest_profit_v16/causal_selected.csv"


def parse_args():
    p = argparse.ArgumentParser(description="Persistent incremental feed for FrozenFakeoutV1")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--cutoff-state", default=DEFAULT_CUTOFF_STATE)
    p.add_argument("--reference-csv", default=DEFAULT_REFERENCE)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--sync-workers", type=int, default=20)
    p.add_argument("--loop", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=5)
    return p.parse_args()


def _checkpoint_path(out: Path, pair: str) -> Path:
    return out / "incremental_state" / f"{p2.symbol(pair)}.pkl"


def _read_pickle(path: Path):
    try:
        with path.open("rb") as f:
            return pickle.load(f)
    except Exception:
        return None


def _write_pickle(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def _prepare(pair: str, cfg: dict, datadir: Path, outdir: Path, now: pd.Timestamp):
    warm_start = p2.HISTORY_START - pd.Timedelta(days=p2.WARMUP_DAYS)
    raw5 = _load_cached_combined(cfg, datadir, outdir, pair, "5m")
    raw15 = _load_cached_combined(cfg, datadir, outdir, pair, "15m")
    raw5 = raw5[(raw5.date >= warm_start) & (raw5.date < now)].reset_index(drop=True)
    raw15 = raw15[(raw15.date >= warm_start) & (raw15.date < now)].reset_index(drop=True)
    if raw5.empty or raw15.empty:
        return None

    x15 = dc.prep_ohlcv(raw15, 15)
    x5 = v1._prep_exec(raw5)
    x5 = v1._add_activity(x5, v1._activity15(x15))
    tfs = {
        "15m": x15,
        "1h": dc.resample_from_15(x15, "1h", 60),
        "4h": dc.resample_from_15(x15, "4h", 240),
    }
    levels = []
    lid = 0
    for tf in p2.LEVEL_TFS:
        for period in p2.LEVEL_PERIODS:
            zz = dc.build_levels(tfs[tf], tf, period, lid)
            levels.extend(zz)
            lid += len(zz)
    return x5, levels


def _rows_from_events(pair: str, events, x5: pd.DataFrame, cutoff: pd.Timestamp) -> list[dict]:
    out = []
    for e in events:
        if e.setup != "H_FAKEOUT":
            continue
        outcome = p2._event_outcome(x5, e)
        if outcome is None or pd.Timestamp(outcome["entry_time"]) < cutoff:
            continue
        si = int(e.signal_idx)
        activity = float(x5.iloc[si].get("activity_score", np.nan))
        if not np.isfinite(activity) or activity < p2.THRESH:
            continue
        if float(outcome["risk_bps"]) < p2.RISK_MIN_BPS:
            continue
        d = {
            "pair": pair,
            "tf": str(e.tf),
            "period": int(e.period),
            "level_price": float(e.level_price),
            "level_kind": str(e.level_kind),
            "approach_no": int(e.approach_no),
            "confluence_tfs": int(e.confluence_tfs),
            "touch_error_pct": float(e.touch_error_pct),
            "activity_score": activity,
            "natr_ratio30d": float(x5.iloc[si].get("natr_ratio30d", np.nan)),
            "qvol24_ratio30d": float(x5.iloc[si].get("qvol24_ratio30d", np.nan)),
            "stop_source": str(e.stop_source),
            **outcome,
        }
        d["signal_id"] = (
            f"{p2.symbol(pair)}|{pd.Timestamp(d['signal_time']).isoformat()}|{d['side']}|"
            f"{d['tf']}|{d['period']}|{d['level_price']:.10g}"
        )
        out.append(d)
    return out


def _event_key(row) -> tuple:
    return (
        pd.Timestamp(row["entry_time"]).isoformat(),
        int(float(row["side"])),
        str(row["tf"]),
        int(float(row["period"])),
        f"{float(row['level_price']):.10g}",
    )


def _bootstrap_parity(pair: str, rows: list[dict], reference_path: Path) -> dict:
    if not reference_path.exists():
        return {"pass": False, "reason": "REFERENCE_MISSING"}
    try:
        ref = pd.read_csv(reference_path)
        ref = ref[ref["pair"].astype(str).eq(pair)].copy()
        ref["entry_time"] = pd.to_datetime(ref["entry_time"], utc=True)
        risk = pd.to_numeric(ref["risk_bps"], errors="coerce")
        ref = ref[risk.between(p2.RISK_MIN_BPS, 3000.0)].copy()
        if ref.empty:
            return {"pass": False, "reason": "REFERENCE_EXEC_EMPTY"}
        lo, hi = ref.entry_time.min(), ref.entry_time.max()
        got = pd.DataFrame(rows)
        if got.empty:
            got = pd.DataFrame(columns=ref.columns)
        else:
            got["entry_time"] = pd.to_datetime(got["entry_time"], utc=True)
            got = got[(got.entry_time >= lo) & (got.entry_time <= hi)].copy()
        rk = {_event_key(r) for _, r in ref.iterrows()}
        gk = {_event_key(r) for _, r in got.iterrows()}
        return {
            "pass": rk == gk,
            "reason": "OK" if rk == gk else "KEY_MISMATCH",
            "ref": len(rk),
            "got": len(gk),
            "ref_only": len(rk - gk),
            "got_only": len(gk - rk),
        }
    except Exception as e:
        return {"pass": False, "reason": f"REFERENCE_ERROR:{type(e).__name__}:{e}"}


def _fingerprint(x5: pd.DataFrame, idx: int) -> tuple:
    r = x5.iloc[int(idx)]
    return (
        pd.Timestamp(r["signal_time"]).isoformat(),
        round(float(r["open"]), 12),
        round(float(r["high"]), 12),
        round(float(r["low"]), 12),
        round(float(r["close"]), 12),
    )


def _bootstrap(pair: str, x5: pd.DataFrame, levels: list, cutoff: pd.Timestamp, reference: Path):
    events, level_state, end_i = inc.detect_events_incremental(x5, levels, start_i=1)
    selected, seen = inc.causal_dedup_incremental(events)
    all_rows = _rows_from_events(pair, selected, x5, p2.HISTORY_START)
    parity = _bootstrap_parity(pair, all_rows, reference)
    if not parity.get("pass"):
        raise RuntimeError(f"INCREMENTAL_BOOTSTRAP_PARITY_FAIL {parity}")
    live_rows = [r for r in all_rows if pd.Timestamp(r["entry_time"]) >= cutoff]
    cp = {
        "version": inc.STATE_VERSION,
        "pair": pair,
        "processed_end_idx": int(end_i),
        "processed_signal_time": pd.Timestamp(x5.iloc[end_i]["signal_time"]).isoformat(),
        "processed_fingerprint": _fingerprint(x5, end_i),
        "level_state": level_state,
        "dedup_seen": seen,
        "parity": parity,
        "created_at": utc_now().isoformat(),
        "updated_at": utc_now().isoformat(),
    }
    return live_rows, cp, parity


def _worker(pair: str, config: str, datadir_s: str, outdir_s: str, cutoff_s: str, now_s: str, reference_s: str):
    t0 = time.monotonic()
    cfg = json.loads(Path(config).read_text())
    datadir = Path(datadir_s)
    outdir = Path(outdir_s)
    cutoff = pd.Timestamp(cutoff_s)
    now = pd.Timestamp(now_s)
    prepared = _prepare(pair, cfg, datadir, outdir, now)
    if prepared is None:
        raise RuntimeError("NO_PREPARED_DATA")
    x5, levels = prepared
    cp_path = _checkpoint_path(outdir, pair)
    cp = _read_pickle(cp_path)
    mode = "INCREMENTAL"

    def full_bootstrap(reason: str):
        rows, fresh, parity = _bootstrap(pair, x5, levels, cutoff, Path(reference_s))
        fresh["bootstrap_reason"] = reason
        _write_pickle(fresh, cp_path)
        return rows, fresh, parity, "BOOTSTRAP"

    if not cp or int(cp.get("version", -1)) != inc.STATE_VERSION or cp.get("pair") != pair:
        rows, cp, parity, mode = full_bootstrap("NO_VALID_CHECKPOINT")
    else:
        try:
            prev = int(cp["processed_end_idx"])
            if prev < 1 or prev >= len(x5) - 1:
                raise RuntimeError("CHECKPOINT_INDEX_INVALID")
            if tuple(cp.get("processed_fingerprint", ())) != _fingerprint(x5, prev):
                raise RuntimeError("CHECKPOINT_HISTORY_CHANGED")
            current_keys = {inc.level_key(lv) for lv in levels}
            missing = set(cp.get("level_state", {})) - current_keys
            if missing:
                raise RuntimeError(f"CHECKPOINT_LEVELS_DISAPPEARED n={len(missing)}")
            events, level_state, end_i = inc.detect_events_incremental(
                x5,
                levels,
                start_i=prev + 1,
                initial_state=cp.get("level_state", {}),
                prior_signal_time=pd.Timestamp(cp["processed_signal_time"]),
            )
            selected, seen = inc.causal_dedup_incremental(events, set(cp.get("dedup_seen", set())))
            rows = _rows_from_events(pair, selected, x5, cutoff)
            cp.update({
                "processed_end_idx": int(end_i),
                "processed_signal_time": pd.Timestamp(x5.iloc[end_i]["signal_time"]).isoformat(),
                "processed_fingerprint": _fingerprint(x5, end_i),
                "level_state": level_state,
                "dedup_seen": seen,
                "updated_at": utc_now().isoformat(),
            })
            parity = dict(cp.get("parity", {}))
            if not parity.get("pass"):
                raise RuntimeError("CHECKPOINT_PARITY_NOT_PASS")
            _write_pickle(cp, cp_path)
        except Exception as e:
            rows, cp, parity, mode = full_bootstrap(f"REBOOTSTRAP:{type(e).__name__}:{e}")

    return rows, {
        "pair": pair,
        "mode": mode,
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "new_rows": len(rows),
        "processed_end_idx": int(cp["processed_end_idx"]),
        "parity_pass": bool(parity.get("pass")),
        "parity": parity,
    }


def run_once(a, cutoff: pd.Timestamp):
    out = Path(a.outdir)
    cfg = json.loads(Path(a.config).read_text())
    datadir = Path(a.datadir)
    scan_started = utc_now().floor("s")
    bucket = scan_started.floor("5min")
    cycle_t0 = time.monotonic()

    sync_t0 = time.monotonic()
    sync_status = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(a.sync_workers), len(p2.FROZEN_PAIRS)))) as ex:
        futs = {
            ex.submit(_sync_pair, pair, cfg, str(datadir), str(out), scan_started.isoformat()): pair
            for pair in p2.FROZEN_PAIRS
        }
        for fut in as_completed(futs):
            pair = futs[fut]
            try:
                r = fut.result()
            except Exception as e:
                r = {"pair": pair, "status": f"ERROR:{type(e).__name__}:{e}", "has_entry_stub": False, "elapsed_sec": 0.0}
            sync_status.append(r)
    sync_duration = time.monotonic() - sync_t0
    sync_status.sort(key=lambda r: p2.FROZEN_PAIRS.index(r["pair"]))
    good = [r["pair"] for r in sync_status if r["status"] == "OK" and r["has_entry_stub"]]

    worker_t0 = time.monotonic()
    new_rows = []
    worker_status = []
    with ProcessPoolExecutor(max_workers=max(1, min(int(a.workers), len(good) or 1))) as ex:
        futs = {
            ex.submit(
                _worker,
                pair,
                a.config,
                a.datadir,
                a.outdir,
                cutoff.isoformat(),
                scan_started.isoformat(),
                a.reference_csv,
            ): pair for pair in good
        }
        for fut in as_completed(futs):
            pair = futs[fut]
            try:
                rr, meta = fut.result()
                new_rows.extend(rr)
                worker_status.append(meta)
                print(f"state {pair:24s} {meta['mode']:11s} t={meta['elapsed_sec']:6.2f}s new={meta['new_rows']}", flush=True)
            except Exception as e:
                worker_status.append({"pair": pair, "mode": "ERROR", "elapsed_sec": 0.0, "new_rows": 0, "parity_pass": False, "error": f"{type(e).__name__}:{e}"})
                print(f"state {pair:24s} ERROR {type(e).__name__}: {e}", flush=True)
    worker_duration = time.monotonic() - worker_t0

    old_path = out / "signals.csv"
    try:
        old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
    except Exception:
        old = pd.DataFrame()
    add = pd.DataFrame(new_rows)
    parts = [z for z in (old, add) if z is not None and not z.empty]
    if parts:
        z = pd.concat(parts, ignore_index=True, sort=False)
        z["signal_time"] = pd.to_datetime(z["signal_time"], utc=True)
        z["entry_time"] = pd.to_datetime(z["entry_time"], utc=True)
        z = z.drop_duplicates("signal_id", keep="last").sort_values(["entry_time", "pair"]).reset_index(drop=True)
    else:
        z = pd.DataFrame(columns=[
            "signal_id", "pair", "signal_time", "entry_time", "entry_price", "side",
            "stop_price", "target_price", "risk_bps", "activity_score", "status", "exit_reason",
        ])

    checked_at = utc_now().floor("s")
    z["published_at"] = checked_at.isoformat()
    z["feed_eligible"] = False
    z["feed_reason"] = "NOT_CURRENT"
    bootstrap_pairs = sum(1 for r in worker_status if r.get("mode") == "BOOTSTRAP")
    all_parity = len(worker_status) == len(good) and all(bool(r.get("parity_pass")) for r in worker_status)

    # Bootstrap is deliberately shadow-only. It proves exact detector parity and
    # seeds lifecycle state, but we never enter a trade after spending minutes on
    # the bootstrap replay. The next 5m cycle is incremental and live-feasible.
    if len(z) and bootstrap_pairs == 0 and all_parity:
        candidate_mask = (
            z["entry_time"].eq(bucket)
            & z["status"].astype(str).eq("OPEN")
            & pd.to_numeric(z["risk_bps"], errors="coerce").between(p2.RISK_MIN_BPS, 3000.0)
        )
        for idx in z.index[candidate_mask]:
            ok, reason = _still_executable(z.loc[idx], checked_at)
            z.at[idx, "feed_eligible"] = bool(ok)
            z.at[idx, "feed_reason"] = reason
        live_idx = z.index[z["feed_eligible"].eq(True)]
        if len(live_idx):
            counts = z.loc[live_idx].groupby(["pair", "entry_time"])["signal_id"].transform("size")
            conflicts = counts[counts > 1].index
            if len(conflicts):
                z.loc[conflicts, "feed_eligible"] = False
                z.loc[conflicts, "feed_reason"] = "PAIR_SIGNAL_CONFLICT"
    elif len(z) and bootstrap_pairs:
        current = z["entry_time"].eq(bucket)
        z.loc[current, "feed_reason"] = "BOOTSTRAP_SHADOW_ONLY"
    elif len(z) and not all_parity:
        current = z["entry_time"].eq(bucket)
        z.loc[current, "feed_reason"] = "INCREMENTAL_PARITY_NOT_PASS"

    active = z[z["feed_eligible"].eq(True)].copy() if len(z) else z.copy()
    _write_csv(z, out / "signals.csv")
    _write_csv(active, out / "active_signals.csv")
    _write_csv(pd.DataFrame(sync_status), out / "coverage.csv")
    _write_csv(pd.DataFrame(worker_status), out / "incremental_coverage.csv")

    cycle_duration = time.monotonic() - cycle_t0
    summary = {
        "engine": "persistent_incremental_v1",
        "cutoff": cutoff.isoformat(),
        "scan_started": scan_started.isoformat(),
        "published_at": checked_at.isoformat(),
        "entry_bucket": bucket.isoformat(),
        "pairs_with_entry_stub": len(good),
        "signals_since_cutoff": int(len(z)),
        "active_for_freqtrade": int(len(active)),
        "active_ids": active["signal_id"].astype(str).tolist() if len(active) else [],
        "publish_delay_sec": float((checked_at - bucket).total_seconds()),
        "sync_duration_sec": round(float(sync_duration), 3),
        "state_duration_sec": round(float(worker_duration), 3),
        "cycle_duration_sec": round(float(cycle_duration), 3),
        "bootstrap_pairs": int(bootstrap_pairs),
        "incremental_pairs": int(sum(1 for r in worker_status if r.get("mode") == "INCREMENTAL")),
        "incremental_parity_pass": bool(all_parity),
        "within_entry_candle": bool(checked_at.floor("5min") == bucket),
    }
    (out / "snapshot.json").write_text(json.dumps(summary, indent=2))
    print("\n=== FROZEN FAKEOUT INCREMENTAL FEED ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    a = parse_args()
    out = Path(a.outdir)
    cutoff = _cutoff(out, Path(a.cutoff_state))
    print("=== FROZEN FAKEOUT PERSISTENT INCREMENTAL FEED ===", flush=True)
    print(f"cutoff={cutoff.isoformat()} | paper/dry-run only", flush=True)
    if not a.loop:
        run_once(a, cutoff)
        return 0

    last_bucket = None
    while True:
        now = utc_now()
        bucket = now.floor("5min")
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
