#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import json
import math
import signal
import sqlite3
import statistics
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

import aiohttp

FAPI = "https://fapi.binance.com"
FSTREAM = "wss://fstream.binance.com/stream"


def now_ms() -> int:
    return int(time.time() * 1000)


def bps(a: float, b: float) -> float:
    return (a / b - 1.0) * 10_000.0 if b else 0.0


def floor_tick(px: float, tick: float) -> float:
    return math.floor(px / tick + 1e-12) * tick


def ceil_tick(px: float, tick: float) -> float:
    return math.ceil(px / tick - 1e-12) * tick


@dataclass
class Quote:
    side: str
    price: float
    qty: float
    remaining: float
    queue_ahead: float
    queue_initial: float
    created_ms: int


class DepthBook:
    def __init__(self) -> None:
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}
        self.snapshot_id = 0
        self.last_u = 0
        self.synced = False

    def reset_snapshot(self, snap: dict) -> None:
        self.bids = {float(p): float(q) for p, q in snap.get("bids", []) if float(q) > 0}
        self.asks = {float(p): float(q) for p, q in snap.get("asks", []) if float(q) > 0}
        self.snapshot_id = int(snap["lastUpdateId"])
        self.last_u = self.snapshot_id
        self.synced = False

    def apply(self, d: dict) -> bool:
        U, u = int(d.get("U", 0)), int(d.get("u", 0))
        pu = int(d.get("pu", 0))
        if not self.synced:
            if u < self.snapshot_id:
                return True
            if not (U <= self.snapshot_id <= u or pu <= self.snapshot_id <= u):
                return True
            self.synced = True
        else:
            if pu and pu != self.last_u:
                return False
        for p, q in d.get("b", []):
            p, q = float(p), float(q)
            if q == 0:
                self.bids.pop(p, None)
            else:
                self.bids[p] = q
        for p, q in d.get("a", []):
            p, q = float(p), float(q)
            if q == 0:
                self.asks.pop(p, None)
            else:
                self.asks[p] = q
        self.last_u = u
        return True

    def best(self) -> tuple[float, float]:
        if not self.bids or not self.asks:
            return 0.0, 0.0
        return max(self.bids), min(self.asks)

    def qty_at(self, side: str, price: float) -> float:
        return self.bids.get(price, 0.0) if side == "BUY" else self.asks.get(price, 0.0)

    def imbalance_top5(self) -> float:
        if not self.bids or not self.asks:
            return 0.0
        bids = sorted(self.bids.items(), reverse=True)[:5]
        asks = sorted(self.asks.items())[:5]
        bq = sum(q for _, q in bids)
        aq = sum(q for _, q in asks)
        den = bq + aq
        return (bq - aq) / den if den else 0.0


class ShadowMM:
    def __init__(self, cfg: argparse.Namespace) -> None:
        self.cfg = cfg
        self.symbol = cfg.symbol.upper()
        self.stream_symbol = self.symbol.lower()
        self.book = DepthBook()
        self.tick = 0.1
        self.best_bid = 0.0
        self.best_ask = 0.0
        self.mid = 0.0
        self.bid_quote: Quote | None = None
        self.ask_quote: Quote | None = None
        self.inventory = 0.0
        self.cash_trade = 0.0
        self.fees = 0.0
        self.funding_pnl = 0.0
        self.funding_rate = 0.0
        self.next_funding_time = 0
        self.last_mark_price = 0.0
        self.fill_seq = 0
        self.pending_markouts: list[dict] = []
        self.mid_samples: Deque[tuple[int, float]] = deque(maxlen=20_000)
        self.vol_samples: Deque[float] = deque(maxlen=600)
        self.last_mid_sample_ms = 0
        self.last_quote_ms = 0
        self.last_snapshot_ms = 0
        self.last_status_ms = 0
        self.last_commit_ms = 0
        self.last_trade_bucket_ms = 0
        self.buy_qty_bucket = 0.0
        self.sell_qty_bucket = 0.0
        self.stop = asyncio.Event()
        self.started_ms = now_ms()
        self.out = Path(cfg.output_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.db_path = self.out / "mm_shadow.sqlite"
        self.con = sqlite3.connect(self.db_path)
        self._init_db()

    def _init_db(self) -> None:
        c = self.con.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT)")
        c.execute("""
        CREATE TABLE IF NOT EXISTS snapshots(
          ts INTEGER PRIMARY KEY,bid REAL,ask REAL,mid REAL,spread_bps REAL,rv10_bps REAL,
          vol_gate INTEGER,depth_imbalance5 REAL,quote_bid REAL,quote_ask REAL,
          bid_queue REAL,ask_queue REAL,inventory_qty REAL,inventory_notional REAL,
          gross_equity REAL,fees REAL,funding_pnl REAL,net_equity REAL,
          buy_agg_qty REAL,sell_agg_qty REAL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS fills(
          id INTEGER PRIMARY KEY,ts INTEGER,side TEXT,price REAL,qty REAL,notional REAL,
          mid_at_fill REAL,capture_bps REAL,net_capture_bps REAL,fee REAL,
          queue_initial REAL,quote_age_ms INTEGER,inventory_after REAL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS markouts(
          fill_id INTEGER,horizon_ms INTEGER,ts INTEGER,mid REAL,markout_bps REAL,net_markout_bps REAL,
          PRIMARY KEY(fill_id,horizon_ms)
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS funding(
          ts INTEGER PRIMARY KEY,rate REAL,mark_price REAL,inventory_qty REAL,pnl REAL
        )""")
        params = {
            "symbol": self.symbol,
            "virtual_capital": self.cfg.virtual_capital,
            "quote_notional": self.cfg.quote_notional,
            "maker_fee_bps": self.cfg.maker_fee_bps,
            "min_halfspread_bps": self.cfg.min_halfspread_bps,
            "vol_mult": self.cfg.vol_mult,
            "vol_gate_mult": self.cfg.vol_gate_mult,
            "inventory_skew_bps": self.cfg.inventory_skew_bps,
            "max_inventory_notional": self.cfg.max_inventory_notional,
            "quote_refresh_ms": self.cfg.quote_refresh_ms,
            "started_ms": self.started_ms,
        }
        for k, v in params.items():
            c.execute("INSERT OR REPLACE INTO meta(key,value) VALUES(?,?)", (k, json.dumps(v)))
        self.con.commit()

    async def load_specs(self, session: aiohttp.ClientSession) -> None:
        async with session.get(f"{FAPI}/fapi/v1/exchangeInfo", timeout=20) as r:
            r.raise_for_status()
            data = await r.json()
        s = next(x for x in data["symbols"] if x["symbol"] == self.symbol)
        for f in s["filters"]:
            if f["filterType"] == "PRICE_FILTER":
                self.tick = float(f["tickSize"])
                break
        print(f"Symbol={self.symbol} tick={self.tick:g}", flush=True)

    async def resync_depth(self, session: aiohttp.ClientSession) -> None:
        async with session.get(
            f"{FAPI}/fapi/v1/depth", params={"symbol": self.symbol, "limit": 1000}, timeout=20
        ) as r:
            r.raise_for_status()
            snap = await r.json()
        self.book.reset_snapshot(snap)
        b, a = self.book.best()
        if b and a:
            self.best_bid, self.best_ask, self.mid = b, a, (b + a) / 2
        print(f"Depth snapshot id={self.book.snapshot_id:,} levels={len(self.book.bids)}/{len(self.book.asks)}", flush=True)

    def sample_mid(self, ts: int) -> None:
        if self.mid <= 0 or ts - self.last_mid_sample_ms < 250:
            return
        self.mid_samples.append((ts, self.mid))
        self.last_mid_sample_ms = ts
        cutoff = ts - 15_000
        while self.mid_samples and self.mid_samples[0][0] < cutoff:
            self.mid_samples.popleft()

    def rv10_bps(self, ts: int) -> float:
        vals = [p for t, p in self.mid_samples if t >= ts - 10_000]
        if len(vals) < 3:
            return 0.0
        ss = 0.0
        prev = vals[0]
        for p in vals[1:]:
            if p > 0 and prev > 0:
                x = math.log(p / prev)
                ss += x * x
            prev = p
        return math.sqrt(ss) * 10_000.0

    def quote_gate(self, rv: float) -> bool:
        if len(self.vol_samples) < 60:
            return True
        med = statistics.median(self.vol_samples)
        if med <= 0:
            return True
        return rv <= self.cfg.vol_gate_mult * med

    def make_quote(self, side: str, px: float, ts: int) -> Quote:
        qty = self.cfg.quote_notional / px
        ahead = self.book.qty_at(side, px)
        return Quote(side, px, qty, qty, ahead, ahead, ts)

    def update_quotes(self, ts: int) -> None:
        if not self.book.synced or self.mid <= 0 or self.best_bid <= 0 or self.best_ask <= 0:
            self.bid_quote = self.ask_quote = None
            return
        rv = self.rv10_bps(ts)
        self.vol_samples.append(rv)
        gate = self.quote_gate(rv)
        if not gate:
            self.bid_quote = self.ask_quote = None
            return
        spread_bps = bps(self.best_ask, self.best_bid)
        half = max(
            self.cfg.min_halfspread_bps,
            spread_bps / 2.0 + 0.25,
            self.cfg.vol_mult * rv,
        )
        inv_notional = self.inventory * self.mid
        inv_frac = max(-1.0, min(1.0, inv_notional / self.cfg.max_inventory_notional))
        center = self.mid * (1.0 - inv_frac * self.cfg.inventory_skew_bps / 10_000.0)
        bid_px = floor_tick(center * (1.0 - half / 10_000.0), self.tick)
        ask_px = ceil_tick(center * (1.0 + half / 10_000.0), self.tick)
        # Guaranteed post-only geometry in a future live version.
        bid_px = min(bid_px, self.best_bid)
        ask_px = max(ask_px, self.best_ask)

        allow_bid = inv_notional < self.cfg.max_inventory_notional
        allow_ask = inv_notional > -self.cfg.max_inventory_notional

        if allow_bid:
            if self.bid_quote is None or abs(self.bid_quote.price - bid_px) >= self.tick / 2:
                self.bid_quote = self.make_quote("BUY", bid_px, ts)
        else:
            self.bid_quote = None
        if allow_ask:
            if self.ask_quote is None or abs(self.ask_quote.price - ask_px) >= self.tick / 2:
                self.ask_quote = self.make_quote("SELL", ask_px, ts)
        else:
            self.ask_quote = None

    def execute_fill(self, q: Quote, qty: float, ts: int) -> None:
        qty = min(qty, q.remaining)
        if qty <= 1e-12 or self.mid <= 0:
            return
        sign = 1.0 if q.side == "BUY" else -1.0
        notional = qty * q.price
        fee = notional * self.cfg.maker_fee_bps / 10_000.0
        if q.side == "BUY":
            self.inventory += qty
            self.cash_trade -= notional
        else:
            self.inventory -= qty
            self.cash_trade += notional
        self.fees += fee
        q.remaining -= qty
        self.fill_seq += 1
        capture = sign * (self.mid / q.price - 1.0) * 10_000.0
        net_capture = capture - self.cfg.maker_fee_bps
        self.con.execute(
            "INSERT INTO fills VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                self.fill_seq, ts, q.side, q.price, qty, notional, self.mid,
                capture, net_capture, fee, q.queue_initial, ts - q.created_ms, self.inventory,
            ),
        )
        for h in (1000, 5000, 30000):
            self.pending_markouts.append({"fill_id": self.fill_seq, "side": sign, "price": q.price, "due": ts + h, "h": h})
        if q.remaining <= 1e-12:
            if q.side == "BUY":
                self.bid_quote = None
            else:
                self.ask_quote = None

    def consume_trade(self, d: dict) -> None:
        ts = int(d.get("T", d.get("E", now_ms())))
        px = float(d["p"])
        qty = float(d["q"])
        maker_buyer = bool(d["m"])
        if maker_buyer:
            self.sell_qty_bucket += qty
            q = self.bid_quote
            if q is not None:
                if px < q.price - self.tick / 2:
                    self.execute_fill(q, q.remaining, ts)
                elif abs(px - q.price) < self.tick / 2:
                    use = qty
                    if q.queue_ahead > 0:
                        consumed = min(q.queue_ahead, use)
                        q.queue_ahead -= consumed
                        use -= consumed
                    if use > 0:
                        self.execute_fill(q, use, ts)
        else:
            self.buy_qty_bucket += qty
            q = self.ask_quote
            if q is not None:
                if px > q.price + self.tick / 2:
                    self.execute_fill(q, q.remaining, ts)
                elif abs(px - q.price) < self.tick / 2:
                    use = qty
                    if q.queue_ahead > 0:
                        consumed = min(q.queue_ahead, use)
                        q.queue_ahead -= consumed
                        use -= consumed
                    if use > 0:
                        self.execute_fill(q, use, ts)

    def process_markouts(self, ts: int) -> None:
        if self.mid <= 0 or not self.pending_markouts:
            return
        keep = []
        for x in self.pending_markouts:
            if ts < x["due"]:
                keep.append(x)
                continue
            markout = x["side"] * (self.mid / x["price"] - 1.0) * 10_000.0
            self.con.execute(
                "INSERT OR IGNORE INTO markouts VALUES(?,?,?,?,?,?)",
                (x["fill_id"], x["h"], ts, self.mid, markout, markout - self.cfg.maker_fee_bps),
            )
        self.pending_markouts = keep

    def process_mark_price(self, d: dict) -> None:
        ts = int(d.get("E", now_ms()))
        mark = float(d.get("p", 0) or 0)
        rate = float(d.get("r", 0) or 0)
        nxt = int(d.get("T", 0) or 0)
        if self.next_funding_time and nxt != self.next_funding_time and ts >= self.next_funding_time:
            ref = self.last_mark_price if self.last_mark_price > 0 else self.mid
            pnl = -self.inventory * ref * self.funding_rate
            self.funding_pnl += pnl
            self.con.execute(
                "INSERT OR REPLACE INTO funding VALUES(?,?,?,?,?)",
                (self.next_funding_time, self.funding_rate, ref, self.inventory, pnl),
            )
        self.funding_rate = rate
        self.next_funding_time = nxt
        if mark > 0:
            self.last_mark_price = mark

    def save_snapshot(self, ts: int) -> None:
        if self.mid <= 0:
            return
        rv = self.rv10_bps(ts)
        gate = int(self.quote_gate(rv))
        spread = bps(self.best_ask, self.best_bid) if self.best_bid else 0.0
        gross = self.cash_trade + self.inventory * self.mid
        net = gross - self.fees + self.funding_pnl
        self.con.execute(
            "INSERT OR REPLACE INTO snapshots VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                ts, self.best_bid, self.best_ask, self.mid, spread, rv, gate,
                self.book.imbalance_top5(),
                self.bid_quote.price if self.bid_quote else None,
                self.ask_quote.price if self.ask_quote else None,
                self.bid_quote.queue_ahead if self.bid_quote else None,
                self.ask_quote.queue_ahead if self.ask_quote else None,
                self.inventory, self.inventory * self.mid, gross, self.fees, self.funding_pnl, net,
                self.buy_qty_bucket, self.sell_qty_bucket,
            ),
        )
        self.buy_qty_bucket = self.sell_qty_bucket = 0.0
        if ts - self.last_commit_ms >= 10_000:
            self.con.commit()
            self.last_commit_ms = ts

    def status(self, ts: int) -> None:
        gross = self.cash_trade + self.inventory * self.mid if self.mid else 0.0
        net = gross - self.fees + self.funding_pnl
        fills = self.con.execute("SELECT COUNT(*) FROM fills").fetchone()[0]
        print(
            f"MM shadow {((ts-self.started_ms)/3600000):.2f}h | mid={self.mid:.2f} | fills={fills} | "
            f"inv=${self.inventory*self.mid:+.2f} | grossPnL=${gross:+.3f} fees=${self.fees:.3f} "
            f"funding=${self.funding_pnl:+.3f} net=${net:+.3f}", flush=True,
        )

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=20, sock_read=60)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            await self.load_specs(session)
            streams = "/".join([
                f"{self.stream_symbol}@aggTrade",
                f"{self.stream_symbol}@depth@100ms",
                f"{self.stream_symbol}@bookTicker",
                f"{self.stream_symbol}@markPrice@1s",
            ])
            url = f"{FSTREAM}?streams={streams}"
            deadline = self.started_ms + int(self.cfg.runtime_seconds * 1000) if self.cfg.runtime_seconds > 0 else 0
            while not self.stop.is_set():
                if deadline and now_ms() >= deadline:
                    break
                try:
                    async with session.ws_connect(url, heartbeat=20, receive_timeout=60) as ws:
                        await self.resync_depth(session)
                        print("WebSocket connected. SHADOW ONLY: no API keys, no orders sent.", flush=True)
                        async for msg in ws:
                            if self.stop.is_set():
                                break
                            if msg.type != aiohttp.WSMsgType.TEXT:
                                if msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                                    break
                                continue
                            payload = json.loads(msg.data)
                            d = payload.get("data", payload)
                            event = d.get("e", "")
                            ts = int(d.get("E", now_ms()))
                            if event == "depthUpdate":
                                if not self.book.apply(d):
                                    print("Depth sequence gap -> resync", flush=True)
                                    await self.resync_depth(session)
                                b, a = self.book.best()
                                if b and a:
                                    self.best_bid, self.best_ask, self.mid = b, a, (b + a) / 2
                            elif event == "bookTicker" or ("b" in d and "a" in d and "u" in d and "e" not in d):
                                self.best_bid = float(d["b"])
                                self.best_ask = float(d["a"])
                                self.mid = (self.best_bid + self.best_ask) / 2
                            elif event == "aggTrade":
                                self.consume_trade(d)
                            elif event == "markPriceUpdate":
                                self.process_mark_price(d)

                            self.sample_mid(ts)
                            self.process_markouts(ts)
                            if ts - self.last_quote_ms >= self.cfg.quote_refresh_ms:
                                self.update_quotes(ts)
                                self.last_quote_ms = ts
                            if ts - self.last_snapshot_ms >= 1000:
                                self.save_snapshot(ts)
                                self.last_snapshot_ms = ts
                            if ts - self.last_status_ms >= 60_000:
                                self.status(ts)
                                self.last_status_ms = ts
                            if deadline and ts >= deadline:
                                self.stop.set()
                                break
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    print(f"WS loop error: {type(exc).__name__}: {exc}; reconnecting in 2s", flush=True)
                    await asyncio.sleep(2)
        self.con.commit()
        self.status(now_ms())
        print(f"Saved: {self.db_path}", flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Live Binance BTCUSDT shadow market maker with conservative queue simulation")
    p.add_argument("--symbol", default="BTCUSDT")
    p.add_argument("--output-dir", default="/freqtrade/user_data/mm_shadow_btc")
    p.add_argument("--runtime-seconds", type=float, default=21_600.0)
    p.add_argument("--virtual-capital", type=float, default=100.0)
    p.add_argument("--quote-notional", type=float, default=10.0)
    p.add_argument("--max-inventory-notional", type=float, default=30.0)
    p.add_argument("--maker-fee-bps", type=float, default=2.0)
    p.add_argument("--min-halfspread-bps", type=float, default=2.5)
    p.add_argument("--vol-mult", type=float, default=0.75)
    p.add_argument("--vol-gate-mult", type=float, default=2.5)
    p.add_argument("--inventory-skew-bps", type=float, default=2.0)
    p.add_argument("--quote-refresh-ms", type=int, default=1000)
    return p.parse_args()


async def amain() -> int:
    cfg = parse_args()
    mm = ShadowMM(cfg)
    loop = asyncio.get_running_loop()
    for s in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(s, mm.stop.set)
        except NotImplementedError:
            pass
    await mm.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
