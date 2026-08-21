#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import time
from pathlib import Path

import aiohttp
import pandas as pd

import prospective_fakeout_v2 as p2


DEFAULT_OUT = "/freqtrade/user_data/frozen_fakeout_ws_shadow"
WS_BASE = "wss://fstream.binance.com/stream?streams="
INTERVALS = ("5m", "15m", "1h", "4h")


def parse_args():
    p = argparse.ArgumentParser(description="Binance USD-M websocket latency shadow for FrozenFakeout")
    p.add_argument("--outdir", default=DEFAULT_OUT)
    p.add_argument("--reconnect-seconds", type=float, default=2.0)
    return p.parse_args()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def _write_json_atomic(obj: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=False))
    os.replace(tmp, path)


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "p50_ms": None, "p95_ms": None, "max_ms": None}
    z = sorted(float(v) for v in values)
    p95_i = max(0, min(len(z) - 1, int(round(0.95 * (len(z) - 1)))))
    return {
        "n": len(z),
        "p50_ms": round(float(statistics.median(z)), 3),
        "p95_ms": round(float(z[p95_i]), 3),
        "max_ms": round(float(max(z)), 3),
    }


def _pair_from_symbol(symbol: str) -> str | None:
    symbol = str(symbol).upper()
    for pair in p2.FROZEN_PAIRS:
        if p2.symbol(pair) == symbol:
            return pair
    return None


class Shadow:
    def __init__(self, outdir: Path):
        self.outdir = outdir
        self.started_at = utc_now()
        self.connected_at: pd.Timestamp | None = None
        self.reconnects = 0
        self.last_event_at: pd.Timestamp | None = None
        self.closed: dict[str, dict[int, dict[str, float]]] = {tf: {} for tf in INTERVALS}
        self.opens: dict[str, dict[int, dict[str, float]]] = {tf: {} for tf in INTERVALS}
        self.first_open_seen: set[tuple[str, int, str]] = set()
        self.total_messages = 0
        self.total_closed_messages = 0

    def _trim(self) -> None:
        for store in (self.closed, self.opens):
            for tf in INTERVALS:
                keys = sorted(store[tf])
                for k in keys[:-12]:
                    del store[tf][k]

    def on_kline(self, data: dict, received_ms: int) -> None:
        k = data.get("k") or {}
        tf = str(k.get("i", ""))
        if tf not in INTERVALS:
            return
        pair = _pair_from_symbol(str(k.get("s", data.get("s", ""))))
        if pair is None:
            return
        self.total_messages += 1
        self.last_event_at = pd.Timestamp(received_ms, unit="ms", tz="UTC")

        start_ms = int(k["t"])
        close_boundary_ms = int(k["T"]) + 1
        is_closed = bool(k.get("x"))

        open_key = (tf, start_ms, pair)
        if open_key not in self.first_open_seen:
            self.first_open_seen.add(open_key)
            self.opens[tf].setdefault(start_ms, {})[pair] = float(received_ms - start_ms)

        if is_closed:
            self.total_closed_messages += 1
            self.closed[tf].setdefault(close_boundary_ms, {})[pair] = float(received_ms - close_boundary_ms)

        self._trim()
        self.publish()

    def _latest_bucket(self, store: dict[int, dict[str, float]]) -> tuple[int | None, dict[str, float]]:
        if not store:
            return None, {}
        k = max(store)
        return k, store[k]

    def publish(self) -> None:
        intervals = {}
        for tf in INTERVALS:
            cb, cv = self._latest_bucket(self.closed[tf])
            ob, ov = self._latest_bucket(self.opens[tf])
            intervals[tf] = {
                "latest_close_boundary": pd.Timestamp(cb, unit="ms", tz="UTC").isoformat() if cb is not None else None,
                "close_pairs": len(cv),
                "close_latency": _stats(list(cv.values())),
                "latest_open_start": pd.Timestamp(ob, unit="ms", tz="UTC").isoformat() if ob is not None else None,
                "open_pairs": len(ov),
                "first_open_latency": _stats(list(ov.values())),
            }

        five = intervals["5m"]
        snapshot = {
            "engine": "binance_usdm_ws_shadow_v1",
            "paper_shadow_only": True,
            "started_at": self.started_at.isoformat(),
            "connected_at": self.connected_at.isoformat() if self.connected_at is not None else None,
            "last_event_at": self.last_event_at.isoformat() if self.last_event_at is not None else None,
            "reconnects": int(self.reconnects),
            "total_messages": int(self.total_messages),
            "total_closed_messages": int(self.total_closed_messages),
            "frozen_pairs": len(p2.FROZEN_PAIRS),
            "latest_5m_close_all_pairs": bool(five["close_pairs"] == len(p2.FROZEN_PAIRS)),
            "latest_5m_open_all_pairs": bool(five["open_pairs"] == len(p2.FROZEN_PAIRS)),
            "intervals": intervals,
            "published_at": utc_now().isoformat(),
        }
        _write_json_atomic(snapshot, self.outdir / "snapshot.json")


async def run(a) -> None:
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    streams = [f"{p2.symbol(pair).lower()}@kline_{tf}" for pair in p2.FROZEN_PAIRS for tf in INTERVALS]
    url = WS_BASE + "/".join(streams)
    shadow = Shadow(outdir)

    timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=90)
    headers = {"User-Agent": "rmv5-frozen-ws-shadow/1.0"}
    async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
        while True:
            try:
                async with session.ws_connect(url, heartbeat=30, autoping=True, max_msg_size=2**20) as ws:
                    shadow.connected_at = utc_now()
                    shadow.publish()
                    print(
                        f"WS connected streams={len(streams)} pairs={len(p2.FROZEN_PAIRS)} intervals={','.join(INTERVALS)}",
                        flush=True,
                    )
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            received_ms = int(time.time() * 1000)
                            try:
                                payload = json.loads(msg.data)
                                data = payload.get("data", payload)
                                if data.get("e") == "kline":
                                    shadow.on_kline(data, received_ms)
                            except Exception as e:
                                print(f"WS message error {type(e).__name__}: {e}", flush=True)
                        elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.ERROR):
                            break
            except asyncio.CancelledError:
                raise
            except Exception as e:
                shadow.reconnects += 1
                shadow.publish()
                print(f"WS reconnect after {type(e).__name__}: {e}", flush=True)
                await asyncio.sleep(max(0.5, float(a.reconnect_seconds)))


def main() -> int:
    a = parse_args()
    try:
        asyncio.run(run(a))
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
