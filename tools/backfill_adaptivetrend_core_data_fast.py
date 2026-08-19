#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sqlite3
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import requests

import backfill_adaptivetrend_core_data as base


def list_existing_h6_stamps(session: requests.Session, symbol: str, start, end) -> list[str]:
    prefix = f"data/futures/um/monthly/klines/{symbol}/6h/{symbol}-6h-"
    token = None
    stamps: list[str] = []
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(base.S3_LIST, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for node in root.findall("s3:Contents/s3:Key", ns):
            key = node.text or ""
            if not key.endswith(".zip"):
                continue
            tail = key.rsplit("/", 1)[-1]
            marker = f"{symbol}-6h-"
            if not tail.startswith(marker):
                continue
            stamp = tail[len(marker):-4]
            try:
                m = datetime.strptime(stamp, "%Y-%m").date()
            except ValueError:
                continue
            if (m.year, m.month) < (start.year, start.month) or (m.year, m.month) > (end.year, end.month):
                continue
            stamps.append(stamp)
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=ns) or "false").lower() == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)
        if not token:
            break
    return sorted(set(stamps))


def main() -> int:
    ap = argparse.ArgumentParser(description="Fast archive-aware backfill for AdaptiveTrend replication")
    ap.add_argument("--db", default=base.DEFAULT_DB)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--symbols", default="")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    con = base.ensure_db(Path(args.db))

    session = requests.Session()
    session.headers["User-Agent"] = "freqtrade-v5-adaptivetrend-replication/1.1"

    if args.symbols.strip():
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    else:
        print("Enumerating historical Binance USDT perpetual archive...", flush=True)
        symbols = base.list_historical_symbols(session)
    print(f"Historical candidate symbols: {len(symbols)}", flush=True)

    print("Discovering only REAL H6 monthly archives (skips nonexistent symbol-month guesses)...", flush=True)
    discovered: dict[str, list[str]] = {}
    listing_errors: list[str] = []
    with ThreadPoolExecutor(max_workers=max(4, min(args.workers * 2, 32))) as ex:
        futs = {ex.submit(list_existing_h6_stamps, session, sym, start, end): sym for sym in symbols}
        for n, fut in enumerate(as_completed(futs), 1):
            sym = futs[fut]
            try:
                stamps = fut.result()
                if stamps:
                    discovered[sym] = stamps
            except Exception as exc:
                listing_errors.append(sym)
                print(f"Archive-list ERROR {sym}: {type(exc).__name__}: {exc}", flush=True)
            if n % 50 == 0 or n == len(futs):
                months = sum(len(v) for v in discovered.values())
                print(f"Archive discovery {n}/{len(futs)} | active_symbols={len(discovered)} | real_symbol_months={months:,}", flush=True)
    if listing_errors:
        raise RuntimeError(f"Archive listing failed for {len(listing_errors)} symbols; rerun safely: {listing_errors[:10]}")

    completed = {(r[0], r[1], r[2]) for r in con.execute("SELECT kind,symbol,stamp FROM ingest WHERE status IN ('ok','missing')")}
    jobs: list[base.Job] = []
    for sym, stamps in discovered.items():
        for stamp in stamps:
            if ("kline", sym, stamp) not in completed:
                jobs.append(base.Job("kline", sym, stamp, f"{base.DATA_BASE}/klines/{sym}/6h/{sym}-6h-{stamp}.zip"))
            # Funding is relevant only while the perp has H6 data. A missing funding archive is still recorded once.
            if ("funding", sym, stamp) not in completed:
                jobs.append(base.Job("funding", sym, stamp, f"{base.DATA_BASE}/fundingRate/{sym}/{sym}-fundingRate-{stamp}.zip"))

    print(
        f"Real active symbol-months={sum(len(v) for v in discovered.values()):,} | jobs remaining={len(jobs):,} (resume-aware) | workers={args.workers}",
        flush=True,
    )
    started = time.monotonic()
    counts = {"ok": 0, "missing": 0, "errors": 0}
    inserted_candles = 0
    inserted_funding = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(base.fetch_job, j, args.timeout) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            job, status, rows = fut.result()
            if status == "ok":
                if job.kind == "kline" and rows:
                    con.executemany(
                        """
                        INSERT OR REPLACE INTO candles
                        (symbol,open_time,open,high,low,close,volume,close_time,quote_volume,trades,taker_buy_base,taker_buy_quote)
                        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
                        """,
                        [(job.symbol, *r) for r in rows],
                    )
                    inserted_candles += len(rows)
                elif job.kind == "funding" and rows:
                    con.executemany(
                        "INSERT OR REPLACE INTO funding_events(symbol,event_time,rate) VALUES (?,?,?)",
                        [(job.symbol, t, rate) for t, rate in rows],
                    )
                    inserted_funding += len(rows)
                counts["ok"] += 1
            elif status == "missing":
                counts["missing"] += 1
            else:
                counts["errors"] += 1
            con.execute(
                "INSERT OR REPLACE INTO ingest(kind,symbol,stamp,status,rows,updated_at) VALUES (?,?,?,?,?,datetime('now'))",
                (job.kind, job.symbol, job.stamp, status, len(rows)),
            )
            if n % 50 == 0 or n == len(jobs):
                con.commit()
                db_candles = con.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
                db_symbols = con.execute("SELECT COUNT(DISTINCT symbol) FROM candles").fetchone()[0]
                elapsed = time.monotonic() - started
                print(
                    f"Progress {n:,}/{len(jobs):,} | ok={counts['ok']} missing={counts['missing']} errors={counts['errors']} "
                    f"| new_candles={inserted_candles:,} new_funding={inserted_funding:,} "
                    f"| db_candles={db_candles:,} symbols={db_symbols} | {elapsed:.0f}s",
                    flush=True,
                )

    con.commit()
    candle_count = con.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    symbol_count = con.execute("SELECT COUNT(DISTINCT symbol) FROM candles").fetchone()[0]
    funding_count = con.execute("SELECT COUNT(*) FROM funding_events").fetchone()[0]
    print("\n=== BINANCE CORE DATA DONE ===")
    print(f"candles={candle_count:,} | symbols_with_H6={symbol_count} | funding_events={funding_count:,} | errors={counts['errors']}")
    print(f"DB: {args.db}")
    return 0 if counts["errors"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
