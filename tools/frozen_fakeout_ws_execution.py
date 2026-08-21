#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import time

import pandas as pd

import prospective_fakeout_v2 as p2
import frozen_fakeout_ws_detector_shadow as shadow
from frozen_fakeout_signal_feed import _still_executable, _sync_pair, _write_csv


DEFAULT_OUT = "/freqtrade/user_data/frozen_fakeout_ws_detector"
DEFAULT_REFERENCE = "/freqtrade/user_data/breakout_retest_profit_v16/causal_selected.csv"
DEFAULT_CACHE = "/freqtrade/user_data/frozen_fakeout_feed"


def parse_args():
    p = argparse.ArgumentParser(description="Websocket-driven frozen FAKEOUT execution feed")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--feed-cache", default=DEFAULT_CACHE)
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--reference-csv", default=DEFAULT_REFERENCE)
    p.add_argument("--bootstrap-workers", type=int, default=16)
    return p.parse_args()


class ExecutionService(shadow.ShadowService):
    """Promotes the proven websocket shadow into the Freqtrade dry-run feed.

    Alpha rules are unchanged. Execution eligibility is enabled only after all
    frozen pairs pass bootstrap parity. The old REST/current-candle feasibility
    recheck and same-pair conflict rule are retained for the rare actual signal.
    """

    def __init__(self, a):
        super().__init__(a)
        self.cache_sync: list[dict] = []
        self.cache_sync_ok = False

    def execution_ready(self) -> bool:
        return bool(
            len(self.states) == len(p2.FROZEN_PAIRS)
            and len(self.bootstrap_meta) == len(p2.FROZEN_PAIRS)
            and all(bool(m.get("parity", {}).get("pass")) for m in self.bootstrap_meta.values())
        )

    async def _sync_bootstrap_cache(self) -> None:
        cfg = json.loads(Path(self.a.config).read_text())
        now = shadow.utc_now()
        tasks = [
            asyncio.to_thread(
                _sync_pair,
                pair,
                cfg,
                self.a.datadir,
                self.a.feed_cache,
                now.isoformat(),
            )
            for pair in p2.FROZEN_PAIRS
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for pair, result in zip(p2.FROZEN_PAIRS, results):
            if isinstance(result, Exception):
                out.append({
                    "pair": pair,
                    "status": f"ERROR:{type(result).__name__}:{result}",
                    "has_entry_stub": False,
                })
            else:
                out.append(dict(result))
        self.cache_sync = out
        self.cache_sync_ok = bool(
            len(out) == len(p2.FROZEN_PAIRS)
            and all(r.get("status") == "OK" and bool(r.get("has_entry_stub")) for r in out)
        )
        self.snapshot()
        if not self.cache_sync_ok:
            bad = [r for r in out if r.get("status") != "OK" or not bool(r.get("has_entry_stub"))]
            raise RuntimeError(f"BOOTSTRAP_CACHE_SYNC_FAILED {bad}")

    async def bootstrap(self) -> None:
        # One lightweight concurrent REST catch-up makes the persistent market
        # cache current before the expensive one-time parity bootstrap. After
        # bootstrap, all live advancement is websocket-only.
        await self._sync_bootstrap_cache()
        await super().bootstrap()
        if not self.execution_ready():
            raise RuntimeError("EXECUTION_PARITY_GATE_FAILED")
        self.snapshot()

    async def _mark_execution_rows(self, rows: list[dict], boundary: pd.Timestamp) -> None:
        if not rows:
            return

        for r in rows:
            r["feed_eligible"] = False
            r["feed_reason"] = "BOOTSTRAP_NOT_READY"
            r["feed_source"] = "ws_stateful_v1"
            r["published_at"] = shadow.utc_now().isoformat()
            r["publish_delay_sec"] = float((shadow.utc_now() - boundary).total_seconds())

        if not self.execution_ready():
            return

        eligible_idx = []
        for idx, r in enumerate(rows):
            checked_at = shadow.utc_now()
            ok, reason = await asyncio.to_thread(_still_executable, pd.Series(r), checked_at)
            published = shadow.utc_now()
            r["feed_eligible"] = bool(ok)
            r["feed_reason"] = str(reason)
            r["published_at"] = published.isoformat()
            r["publish_delay_sec"] = float((published - boundary).total_seconds())
            if ok:
                eligible_idx.append(idx)

        # Preserve the predeclared execution rule from the original feed: if
        # more than one executable signal exists for the same pair/boundary,
        # reject all instead of inventing a prospective tie-breaker.
        if len(eligible_idx) > 1:
            for idx in eligible_idx:
                rows[idx]["feed_eligible"] = False
                rows[idx]["feed_reason"] = "PAIR_SIGNAL_CONFLICT"

    async def drain_pair(self, pair: str) -> None:
        async with self.locks[pair]:
            state = self.states.get(pair)
            if state is None:
                return
            while True:
                boundary = pd.Timestamp(state.prior_signal_time) + pd.Timedelta(minutes=5)
                ready = self.book.ready(pair, boundary)
                if ready is None:
                    break
                c5, o5, c15, received = ready
                try:
                    cycle_t0 = time.monotonic()
                    rows, meta = await asyncio.to_thread(
                        state.process_boundary, c5, o5, closed15=c15, received_at=received,
                    )
                    await self._mark_execution_rows(rows, boundary)
                    finished = shadow.utc_now()
                    meta["execution_check_ms"] = round(
                        max(0.0, (time.monotonic() - cycle_t0) * 1000.0 - float(meta.get("compute_ms", 0.0))),
                        3,
                    )
                    meta["end_to_end_ms"] = round(float((finished - boundary).total_seconds() * 1000.0), 3)
                    meta["finished_at"] = finished.isoformat()
                    meta["execution_ready"] = self.execution_ready()
                    meta["feed_eligible"] = sum(bool(r.get("feed_eligible")) for r in rows)
                    self.pair_meta[pair] = meta
                    if rows:
                        self.signals.extend(rows)
                    print(
                        f"live {pair:24s} {boundary.isoformat()} transport={meta['transport_to_ready_ms']:.0f}ms "
                        f"compute={meta['compute_ms']:.0f}ms end2end={meta['end_to_end_ms']:.0f}ms "
                        f"signals={len(rows)} eligible={meta['feed_eligible']}",
                        flush=True,
                    )
                    self.book.pop(pair, boundary)
                except Exception as e:
                    err = {
                        "at": shadow.utc_now().isoformat(),
                        "pair": pair,
                        "phase": "live",
                        "boundary": boundary.isoformat(),
                        "error": f"{type(e).__name__}:{e}",
                    }
                    self.errors.append(err)
                    print(f"live {pair} ERROR {err['error']}", flush=True)
                    break
                self.snapshot()
            self.book.trim(shadow.utc_now() - pd.Timedelta(minutes=30))

    def snapshot(self) -> None:
        super().snapshot()

        # Always materialize the execution feed, including an empty header-only
        # file, so Freqtrade never falls back to stale data after a restart.
        if self.signals:
            z = pd.DataFrame(self.signals).copy()
            z = z.drop_duplicates("signal_id", keep="last").sort_values(["entry_time", "pair"])
        else:
            z = pd.DataFrame(columns=[
                "signal_id", "pair", "signal_time", "entry_time", "entry_price", "side",
                "stop_price", "target_price", "risk_bps", "activity_score", "status",
                "feed_eligible", "feed_reason", "feed_source", "published_at", "publish_delay_sec",
            ])
        _write_csv(z, self.out / "signals.csv")

        if len(z) and "feed_eligible" in z:
            eligible = z["feed_eligible"].astype(str).str.lower().isin({"true", "1", "yes"})
            active = z[eligible].copy()
        else:
            active = z.iloc[0:0].copy()
        _write_csv(active, self.out / "active_signals.csv")

        snap_path = self.out / "snapshot.json"
        try:
            obj = json.loads(snap_path.read_text())
        except Exception:
            obj = {}
        latest_boundary = obj.get("latest_boundary")
        current_active = active
        if latest_boundary and len(active) and "entry_time" in active:
            et = pd.to_datetime(active["entry_time"], utc=True, errors="coerce")
            current_active = active[et.eq(pd.Timestamp(latest_boundary))]
        obj.update({
            "engine": "frozen_fakeout_ws_stateful_execution_v1",
            "paper_shadow_only": False,
            "dry_run_execution_feed": True,
            "execution_ready": self.execution_ready(),
            "execution_feed_path": str(self.out / "signals.csv"),
            "cache_sync_ok": bool(self.cache_sync_ok),
            "cache_sync": self.cache_sync,
            "active_for_freqtrade": int(len(current_active)),
            "active_ids": current_active["signal_id"].astype(str).tolist() if len(current_active) else [],
            "published_at": shadow.utc_now().isoformat(),
        })
        shadow._write_json(obj, snap_path)


async def _run(a) -> None:
    service = ExecutionService(a)
    await service.run()


def main() -> int:
    a = parse_args()
    try:
        asyncio.run(_run(a))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
