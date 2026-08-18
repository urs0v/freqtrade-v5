from __future__ import annotations

import asyncio
import json
import os
import sqlite3
import time
from pathlib import Path
from typing import Dict, Any

import aiohttp

REST = "https://fapi.binance.com"
WS = "wss://fstream.binance.com/market/ws/!forceOrder@arr"
DB = Path(os.getenv("RMV5_FEATURE_DB", "/freqtrade/user_data/v5/features.sqlite"))
UNIVERSE = Path(os.getenv("RMV5_UNIVERSE_FILE", "/freqtrade/user_data/v5/universe.json"))
POLL_SECONDS = int(os.getenv("RMV5_OI_POLL_SECONDS", "30"))

def bucket_ms(ts_ms: int, minutes: int = 15) -> int:
    size = minutes * 60 * 1000
    return ts_ms - (ts_ms % size)

def ensure_db() -> None:
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS features (
            bucket_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            oi REAL,
            funding_rate REAL DEFAULT 0,
            long_liq_usdt REAL DEFAULT 0,
            short_liq_usdt REAL DEFAULT 0,
            taker_ratio REAL DEFAULT 1,
            top_ls_ratio REAL DEFAULT 1,
            updated_ms INTEGER NOT NULL,
            PRIMARY KEY (bucket_ms, symbol)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON features(symbol, bucket_ms)")
    con.commit()
    con.close()

def load_symbols() -> list[str]:
    env = os.getenv("V5_SYMBOLS", "").strip()
    if env:
        return [x.strip().upper() for x in env.split(",") if x.strip()]
    if UNIVERSE.exists():
        data = json.loads(UNIVERSE.read_text())
        if isinstance(data, dict):
            return [x.upper() for x in data.get("symbols", [])]
        if isinstance(data, list):
            return [x.upper() for x in data]
    return ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

async def get_json(session: aiohttp.ClientSession, path: str, params=None):
    async with session.get(REST + path, params=params, timeout=aiohttp.ClientTimeout(total=10)) as r:
        r.raise_for_status()
        return await r.json()

def blank_row():
    return {
        "oi": None, "funding_rate": 0.0,
        "long_liq_usdt": 0.0, "short_liq_usdt": 0.0,
        "taker_ratio": 1.0, "top_ls_ratio": 1.0,
    }

async def poll_derivatives(state: Dict[Any, Any], symbols: list[str]) -> None:
    async with aiohttp.ClientSession() as session:
        last_ratio_minute = None
        while True:
            now = int(time.time() * 1000)
            try:
                premium = await get_json(session, "/fapi/v1/premiumIndex")
                if isinstance(premium, dict):
                    premium = [premium]
                pmap = {x.get("symbol"): x for x in premium}
            except Exception as e:
                print("premiumIndex error:", e, flush=True)
                pmap = {}

            async def one_symbol(sym: str):
                try:
                    oi = await get_json(session, "/fapi/v1/openInterest", {"symbol": sym})
                    return sym, float(oi["openInterest"])
                except Exception:
                    return sym, None

            results = await asyncio.gather(*(one_symbol(s) for s in symbols))
            for sym, oi in results:
                row = state.setdefault((bucket_ms(now), sym), blank_row())
                if oi is not None:
                    row["oi"] = oi
                if sym in pmap:
                    try:
                        row["funding_rate"] = float(pmap[sym].get("lastFundingRate", 0.0))
                    except Exception:
                        pass

            minute = now // 60_000
            if minute % 5 == 0 and minute != last_ratio_minute:
                last_ratio_minute = minute

                async def ratio_symbol(sym: str):
                    taker = top = None
                    try:
                        x = await get_json(session, "/futures/data/takerlongshortRatio",
                                           {"symbol": sym, "period": "15m", "limit": 1})
                        if x:
                            taker = float(x[-1].get("buySellRatio", 1.0))
                    except Exception:
                        pass
                    try:
                        x = await get_json(session, "/futures/data/topLongShortAccountRatio",
                                           {"symbol": sym, "period": "15m", "limit": 1})
                        if x:
                            top = float(x[-1].get("longShortRatio", 1.0))
                    except Exception:
                        pass
                    return sym, taker, top

                ratios = await asyncio.gather(*(ratio_symbol(s) for s in symbols))
                for sym, taker, top in ratios:
                    row = state.setdefault((bucket_ms(now), sym), blank_row())
                    if taker is not None:
                        row["taker_ratio"] = taker
                    if top is not None:
                        row["top_ls_ratio"] = top

            await asyncio.sleep(POLL_SECONDS)

async def liquidation_stream(state: Dict[Any, Any], symbols: list[str]) -> None:
    symbols_set = set(symbols)
    while True:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(WS, heartbeat=20, timeout=30) as ws:
                    print("Liquidation WS connected", flush=True)
                    async for msg in ws:
                        if msg.type != aiohttp.WSMsgType.TEXT:
                            continue
                        data = json.loads(msg.data)
                        events = data if isinstance(data, list) else [data]
                        for ev in events:
                            order = ev.get("o", ev)
                            sym = order.get("s")
                            if sym not in symbols_set:
                                continue
                            ts = int(ev.get("E") or order.get("T") or time.time() * 1000)
                            price = float(order.get("ap") or order.get("p") or 0)
                            qty = float(order.get("z") or order.get("q") or 0)
                            notional = price * qty
                            if notional <= 0:
                                continue
                            row = state.setdefault((bucket_ms(ts), sym), blank_row())
                            side = str(order.get("S", "")).upper()
                            if side == "SELL":
                                row["long_liq_usdt"] += notional
                            elif side == "BUY":
                                row["short_liq_usdt"] += notional
        except Exception as e:
            print("Liquidation WS error:", e, "reconnecting...", flush=True)
            await asyncio.sleep(3)

async def writer(state: Dict[Any, Any]) -> None:
    ensure_db()
    while True:
        await asyncio.sleep(10)
        if not state:
            continue
        now = int(time.time() * 1000)
        con = sqlite3.connect(DB, timeout=5)
        try:
            for (b, sym), row in list(state.items()):
                con.execute("""
                    INSERT INTO features
                    (bucket_ms, symbol, oi, funding_rate,
                     long_liq_usdt, short_liq_usdt,
                     taker_ratio, top_ls_ratio, updated_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(bucket_ms, symbol) DO UPDATE SET
                      oi=COALESCE(excluded.oi, features.oi),
                      funding_rate=excluded.funding_rate,
                      long_liq_usdt=excluded.long_liq_usdt,
                      short_liq_usdt=excluded.short_liq_usdt,
                      taker_ratio=excluded.taker_ratio,
                      top_ls_ratio=excluded.top_ls_ratio,
                      updated_ms=excluded.updated_ms
                """, (
                    b, sym, row.get("oi"), row.get("funding_rate", 0.0),
                    row.get("long_liq_usdt", 0.0), row.get("short_liq_usdt", 0.0),
                    row.get("taker_ratio", 1.0), row.get("top_ls_ratio", 1.0), now,
                ))
            con.commit()
        finally:
            con.close()

        cutoff = bucket_ms(now) - 15 * 60 * 1000
        for key in list(state):
            if key[0] < cutoff:
                state.pop(key, None)

async def main() -> None:
    symbols = load_symbols()
    print("RMV5 collector symbols:", ",".join(symbols), flush=True)
    state: Dict[Any, Any] = {}
    await asyncio.gather(
        poll_derivatives(state, symbols),
        liquidation_stream(state, symbols),
        writer(state),
    )

if __name__ == "__main__":
    asyncio.run(main())
