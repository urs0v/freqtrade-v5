#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import ProcessPoolExecutor
import json
import os
import pickle
from pathlib import Path
import statistics
import time

import aiohttp
import pandas as pd

import prospective_fakeout_v2 as p2
from frozen_fakeout_live_core import LivePairState
from frozen_fakeout_signal_feed import _load_cached_combined


DEFAULT_OUT = "/freqtrade/user_data/frozen_fakeout_ws_detector"
DEFAULT_REFERENCE = "/freqtrade/user_data/breakout_retest_profit_v16/causal_selected.csv"
WS_BASE = "wss://fstream.binance.com/market/stream?streams="


def parse_args():
    p = argparse.ArgumentParser(description="Websocket-driven frozen FAKEOUT detector shadow")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--feed-cache", default="/freqtrade/user_data/frozen_fakeout_feed")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--reference-csv", default=DEFAULT_REFERENCE)
    p.add_argument("--bootstrap-workers", type=int, default=16)
    return p.parse_args()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _write_json(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    os.replace(tmp, path)


def _write_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    z = sorted(float(v) for v in values)
    p95 = z[max(0, min(len(z) - 1, int(round(0.95 * (len(z) - 1)))))]
    return {
        "n": len(z),
        "p50_ms": round(float(statistics.median(z)), 3),
        "p95_ms": round(float(p95), 3),
        "max_ms": round(float(max(z)), 3),
    }


def _pair_map() -> dict[str, str]:
    return {p2.symbol(pair): pair for pair in p2.FROZEN_PAIRS}


def _bootstrap_worker(
    pair: str,
    config_path: str,
    datadir_s: str,
    cache_s: str,
    reference_s: str,
    state_path_s: str,
) -> dict:
    t0 = time.monotonic()
    cfg = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    cache = Path(cache_s)
    now = utc_now()
    warm_start = p2.HISTORY_START - pd.Timedelta(days=p2.WARMUP_DAYS)

    raw5 = _load_cached_combined(cfg, datadir, cache, pair, "5m")
    raw15 = _load_cached_combined(cfg, datadir, cache, pair, "15m")
    raw5 = raw5[(raw5.date >= warm_start) & (raw5.date < now)].reset_index(drop=True)
    raw15 = raw15[(raw15.date >= warm_start) & (raw15.date < now)].reset_index(drop=True)
    state, meta = LivePairState.bootstrap(pair, raw5, raw15, reference_s)
    _write_pickle(state, Path(state_path_s))
    return {
        "pair": pair,
        "status": "OK",
        "rows5": meta.rows5,
        "rows15": meta.rows15,
        "levels": meta.levels,
        "parity": meta.parity,
        "elapsed_sec": round(time.monotonic() - t0, 3),
        "last_processed_signal_time": state.prior_signal_time.isoformat(),
    }


class BoundaryBook:
    def __init__(self):
        self.closed5: dict[tuple[str, pd.Timestamp], tuple[dict, pd.Timestamp]] = {}
        self.open5: dict[tuple[str, pd.Timestamp], tuple[dict, pd.Timestamp]] = {}
        self.closed15: dict[tuple[str, pd.Timestamp], tuple[dict, pd.Timestamp]] = {}
        self.first_open_seen: set[tuple[str, pd.Timestamp]] = set()

    @staticmethod
    def _row(k: dict) -> dict:
        return {
            "date": pd.Timestamp(int(k["t"]), unit="ms", tz="UTC"),
            "open": float(k["o"]),
            "high": float(k["h"]),
            "low": float(k["l"]),
            "close": float(k["c"]),
            "volume": float(k["v"]),
        }

    def ingest(self, pair: str, k: dict, received_at: pd.Timestamp) -> set[pd.Timestamp]:
        tf = str(k.get("i"))
        start = pd.Timestamp(int(k["t"]), unit="ms", tz="UTC")
        changed: set[pd.Timestamp] = set()
        if tf == "5m":
            open_key = (pair, start)
            if open_key not in self.first_open_seen:
                self.first_open_seen.add(open_key)
                self.open5[open_key] = (self._row(k), received_at)
                changed.add(start)
            if bool(k.get("x")):
                boundary = pd.Timestamp(int(k["T"]) + 1, unit="ms", tz="UTC")
                self.closed5[(pair, boundary)] = (self._row(k), received_at)
                changed.add(boundary)
        elif tf == "15m" and bool(k.get("x")):
            boundary = pd.Timestamp(int(k["T"]) + 1, unit="ms", tz="UTC")
            self.closed15[(pair, boundary)] = (self._row(k), received_at)
            changed.add(boundary)
        return changed

    def ready(self, pair: str, boundary: pd.Timestamp):
        c5 = self.closed5.get((pair, boundary))
        o5 = self.open5.get((pair, boundary))
        c15 = self.closed15.get((pair, boundary)) if boundary.minute % 15 == 0 else None
        if c5 is None or o5 is None:
            return None
        if boundary.minute % 15 == 0 and c15 is None:
            return None
        received = max(c5[1], o5[1], c15[1] if c15 is not None else c5[1])
        return c5[0], o5[0], c15[0] if c15 is not None else None, received

    def pop(self, pair: str, boundary: pd.Timestamp) -> None:
        self.closed5.pop((pair, boundary), None)
        self.open5.pop((pair, boundary), None)
        self.closed15.pop((pair, boundary), None)

    def trim(self, floor: pd.Timestamp) -> None:
        for store in (self.closed5, self.open5, self.closed15):
            for key in list(store):
                if key[1] < floor:
                    store.pop(key, None)
        self.first_open_seen = {k for k in self.first_open_seen if k[1] >= floor}


class ShadowService:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.outdir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.book = BoundaryBook()
        self.states: dict[str, LivePairState] = {}
        self.bootstrap_meta: dict[str, dict] = {}
        self.pair_meta: dict[str, dict] = {}
        self.errors: list[dict] = []
        self.signals: list[dict] = []
        self.started_at = utc_now()
        self.connected_at: pd.Timestamp | None = None
        self.total_messages = 0
        self.reconnects = 0
        self.locks = {pair: asyncio.Lock() for pair in p2.FROZEN_PAIRS}
        self.tasks: set[asyncio.Task] = set()

    def snapshot(self) -> None:
        metas = list(self.pair_meta.values())
        transport = [m["transport_to_ready_ms"] for m in metas if "transport_to_ready_ms" in m]
        compute = [m["compute_ms"] for m in metas if "compute_ms" in m]
        end2end = [m["end_to_end_ms"] for m in metas if "end_to_end_ms" in m]
        boundaries = [pd.Timestamp(m["boundary"]) for m in metas if m.get("boundary")]
        obj = {
            "engine": "frozen_fakeout_ws_stateful_shadow_v1",
            "paper_shadow_only": True,
            "started_at": self.started_at.isoformat(),
            "connected_at": self.connected_at.isoformat() if self.connected_at is not None else None,
            "total_messages": int(self.total_messages),
            "reconnects": int(self.reconnects),
            "bootstrapped_pairs": len(self.states),
            "bootstrap_parity_pass": bool(
                len(self.states) == len(p2.FROZEN_PAIRS)
                and all(bool(m.get("parity", {}).get("pass")) for m in self.bootstrap_meta.values())
            ),
            "latest_boundary": max(boundaries).isoformat() if boundaries else None,
            "latest_pairs_processed": sum(1 for m in metas if boundaries and pd.Timestamp(m.get("boundary")) == max(boundaries)),
            "latency": {
                "transport_ready": _stats(transport),
                "compute": _stats(compute),
                "end_to_end": _stats(end2end),
            },
            "signals_total": len(self.signals),
            "signal_ids": [str(r["signal_id"]) for r in self.signals[-20:]],
            "errors": self.errors[-20:],
            "bootstrap": self.bootstrap_meta,
            "pair_latest": self.pair_meta,
            "published_at": utc_now().isoformat(),
        }
        _write_json(obj, self.out / "snapshot.json")
        if self.signals:
            z = pd.DataFrame(self.signals)
            tmp = self.out / "signals.csv.tmp"
            z.to_csv(tmp, index=False)
            os.replace(tmp, self.out / "signals.csv")

    async def bootstrap(self) -> None:
        loop = asyncio.get_running_loop()
        state_dir = self.out / "bootstrap_state"
        state_dir.mkdir(parents=True, exist_ok=True)
        workers = max(1, min(int(self.a.bootstrap_workers), len(p2.FROZEN_PAIRS)))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {}
            for pair in p2.FROZEN_PAIRS:
                path = state_dir / f"{p2.symbol(pair)}.pkl"
                fut = loop.run_in_executor(
                    ex, _bootstrap_worker, pair, self.a.config, self.a.datadir,
                    self.a.feed_cache, self.a.reference_csv, str(path),
                )
                futs[pair] = (fut, path)
            for pair, (fut, path) in futs.items():
                try:
                    meta = await fut
                    with path.open("rb") as f:
                        state = pickle.load(f)
                    self.states[pair] = state
                    self.bootstrap_meta[pair] = meta
                    print(
                        f"bootstrap {pair:24s} parity=1 levels={meta['levels']} t={meta['elapsed_sec']:.1f}s",
                        flush=True,
                    )
                except Exception as e:
                    err = {"at": utc_now().isoformat(), "pair": pair, "phase": "bootstrap", "error": f"{type(e).__name__}:{e}"}
                    self.errors.append(err)
                    print(f"bootstrap {pair} ERROR {err['error']}", flush=True)
                self.snapshot()

    def schedule_drain(self, pair: str) -> None:
        if pair not in self.states:
            return
        task = asyncio.create_task(self.drain_pair(pair))
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

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
                    rows, meta = await asyncio.to_thread(
                        state.process_boundary, c5, o5, closed15=c15, received_at=received,
                    )
                    finished = utc_now()
                    meta["end_to_end_ms"] = round(float((finished - boundary).total_seconds() * 1000.0), 3)
                    meta["finished_at"] = finished.isoformat()
                    self.pair_meta[pair] = meta
                    if rows:
                        self.signals.extend(rows)
                    print(
                        f"live {pair:24s} {boundary.isoformat()} transport={meta['transport_to_ready_ms']:.0f}ms "
                        f"compute={meta['compute_ms']:.0f}ms end2end={meta['end_to_end_ms']:.0f}ms signals={len(rows)}",
                        flush=True,
                    )
                    self.book.pop(pair, boundary)
                except Exception as e:
                    err = {"at": utc_now().isoformat(), "pair": pair, "phase": "live", "boundary": boundary.isoformat(), "error": f"{type(e).__name__}:{e}"}
                    self.errors.append(err)
                    print(f"live {pair} ERROR {err['error']}", flush=True)
                    break
                self.snapshot()
            self.book.trim(utc_now() - pd.Timedelta(minutes=30))

    async def receiver(self) -> None:
        symbol_map = _pair_map()
        streams = [f"{p2.symbol(pair).lower()}@kline_{tf}" for pair in p2.FROZEN_PAIRS for tf in ("5m", "15m")]
        url = WS_BASE + "/".join(streams)
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=90)
        headers = {"User-Agent": "rmv5-frozen-ws-detector/1.0"}
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            while True:
                try:
                    async with session.ws_connect(url, heartbeat=30, autoping=True, max_msg_size=2**20) as ws:
                        self.connected_at = utc_now()
                        self.snapshot()
                        print(f"detector WS connected streams={len(streams)}", flush=True)
                        async for msg in ws:
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                self.total_messages += 1
                                received = utc_now()
                                try:
                                    payload = json.loads(msg.data)
                                    data = payload.get("data", payload)
                                    if data.get("e") != "kline":
                                        continue
                                    k = data.get("k") or {}
                                    pair = symbol_map.get(str(k.get("s", data.get("s", ""))).upper())
                                    if pair is None:
                                        continue
                                    changed = self.book.ingest(pair, k, received)
                                    if changed:
                                        self.schedule_drain(pair)
                                except Exception as e:
                                    self.errors.append({"at": utc_now().isoformat(), "phase": "message", "error": f"{type(e).__name__}:{e}"})
                            elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                                break
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    self.reconnects += 1
                    self.errors.append({"at": utc_now().isoformat(), "phase": "ws", "error": f"{type(e).__name__}:{e}"})
                    self.snapshot()
                    await asyncio.sleep(2.0)

    async def run(self) -> None:
        recv = asyncio.create_task(self.receiver())
        try:
            await self.bootstrap()
            if len(self.states) != len(p2.FROZEN_PAIRS):
                raise RuntimeError(f"BOOTSTRAP_INCOMPLETE {len(self.states)}/{len(p2.FROZEN_PAIRS)}")
            for pair in p2.FROZEN_PAIRS:
                self.schedule_drain(pair)
            self.snapshot()
            await recv
        finally:
            recv.cancel()


def main() -> int:
    a = parse_args()
    svc = ShadowService(a)
    try:
        asyncio.run(svc.run())
    except KeyboardInterrupt:
        return 130
    except Exception as e:
        svc.errors.append({"at": utc_now().isoformat(), "phase": "fatal", "error": f"{type(e).__name__}:{e}"})
        svc.snapshot()
        print(f"FATAL {type(e).__name__}: {e}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
