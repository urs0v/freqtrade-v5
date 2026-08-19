#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import sqlite3
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
import xml.etree.ElementTree as ET

import pandas as pd
import requests

S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
DATA_BASE = "https://data.binance.vision/data/futures/um/monthly"
DEFAULT_DB = "/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
EXCLUDED_BASES = {"BTCDOM", "DEFI"}
LEVERAGED_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")


@dataclass(frozen=True)
class Job:
    kind: str
    symbol: str
    stamp: str
    url: str


def month_iter(start: date, end: date):
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        cur = date(cur.year + (cur.month == 12), 1 if cur.month == 12 else cur.month + 1, 1)


def ensure_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-262144")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol TEXT NOT NULL,
            open_time INTEGER NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            close_time INTEGER,
            quote_volume REAL,
            trades REAL,
            taker_buy_base REAL,
            taker_buy_quote REAL,
            PRIMARY KEY(symbol, open_time)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_candles_time ON candles(open_time)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS funding_events (
            symbol TEXT NOT NULL,
            event_time INTEGER NOT NULL,
            rate REAL NOT NULL,
            PRIMARY KEY(symbol, event_time)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_funding_time ON funding_events(event_time)")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest (
            kind TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stamp TEXT NOT NULL,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(kind, symbol, stamp)
        )
        """
    )
    con.commit()
    return con


def list_historical_symbols(session: requests.Session) -> list[str]:
    prefix = "data/futures/um/monthly/klines/"
    token = None
    symbols: list[str] = []
    while True:
        params = {"list-type": "2", "delimiter": "/", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(S3_LIST, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for node in root.findall("s3:CommonPrefixes/s3:Prefix", ns):
            p = node.text or ""
            if not p.startswith(prefix):
                continue
            sym = p[len(prefix):].strip("/").upper()
            if sym:
                symbols.append(sym)
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=ns) or "false").lower() == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)
        if not token:
            break
    out = []
    for sym in sorted(set(symbols)):
        if not sym.endswith("USDT"):
            continue
        base = sym[:-4]
        if base in EXCLUDED_BASES or base.endswith(LEVERAGED_SUFFIXES):
            continue
        out.append(sym)
    return out


def parse_kline_zip(data: bytes) -> list[tuple]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            return []
        with zf.open(names[0]) as f:
            df = pd.read_csv(f, header=None)
    if df.empty or df.shape[1] < 6:
        return []
    open_time = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    valid = open_time.notna()
    df = df.loc[valid].copy()
    open_time = open_time.loc[valid]
    if df.empty:
        return []
    # Binance switched some archives to microseconds in 2025; normalize defensively.
    if float(open_time.abs().median()) > 1e14:
        open_time = (open_time // 1000).astype("int64")
    else:
        open_time = open_time.astype("int64")
    close_time = pd.to_numeric(df.iloc[:, 6], errors="coerce") if df.shape[1] > 6 else pd.Series(index=df.index, dtype=float)
    if close_time.notna().any() and float(close_time.dropna().abs().median()) > 1e14:
        close_time = close_time // 1000
    cols = [pd.to_numeric(df.iloc[:, i], errors="coerce") for i in range(1, 6)]
    qv = pd.to_numeric(df.iloc[:, 7], errors="coerce") if df.shape[1] > 7 else pd.Series(index=df.index, dtype=float)
    trades = pd.to_numeric(df.iloc[:, 8], errors="coerce") if df.shape[1] > 8 else pd.Series(index=df.index, dtype=float)
    tbb = pd.to_numeric(df.iloc[:, 9], errors="coerce") if df.shape[1] > 9 else pd.Series(index=df.index, dtype=float)
    tbq = pd.to_numeric(df.iloc[:, 10], errors="coerce") if df.shape[1] > 10 else pd.Series(index=df.index, dtype=float)
    rows = []
    for j, idx in enumerate(df.index):
        vals = [c.loc[idx] for c in cols]
        if any(pd.isna(v) for v in vals):
            continue
        rows.append((
            int(open_time.loc[idx]), float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4]),
            int(close_time.loc[idx]) if pd.notna(close_time.loc[idx]) else None,
            float(qv.loc[idx]) if pd.notna(qv.loc[idx]) else None,
            float(trades.loc[idx]) if pd.notna(trades.loc[idx]) else None,
            float(tbb.loc[idx]) if pd.notna(tbb.loc[idx]) else None,
            float(tbq.loc[idx]) if pd.notna(tbq.loc[idx]) else None,
        ))
    return rows


def parse_funding_zip(data: bytes) -> list[tuple[int, float]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            return []
        with zf.open(names[0]) as f:
            df = pd.read_csv(f)
    if df.empty:
        return []
    cmap = {str(c).strip().lower(): c for c in df.columns}
    tc = next((cmap[x] for x in ("calc_time", "funding_time", "fundingtime", "timestamp", "time") if x in cmap), None)
    rc = next((cmap[x] for x in ("last_funding_rate", "funding_rate", "fundingrate") if x in cmap), None)
    if tc is None or rc is None:
        # Some old files have no header.
        raw = pd.read_csv(io.BytesIO(data), header=None)
        if raw.shape[1] < 3:
            return []
        t = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
        rate = pd.to_numeric(raw.iloc[:, -1], errors="coerce")
    else:
        t = pd.to_numeric(df[tc], errors="coerce")
        rate = pd.to_numeric(df[rc], errors="coerce")
    valid = t.notna() & rate.notna()
    t, rate = t[valid], rate[valid]
    if t.empty:
        return []
    if float(t.abs().median()) > 1e14:
        t = t // 1000
    return [(int(a), float(b)) for a, b in zip(t, rate)]


def fetch_job(job: Job, timeout: float) -> tuple[Job, str, list]:
    try:
        r = requests.get(job.url, timeout=timeout)
        if r.status_code == 404:
            return job, "missing", []
        r.raise_for_status()
        if not r.content.startswith(b"PK"):
            return job, "bad_content", []
        rows = parse_kline_zip(r.content) if job.kind == "kline" else parse_funding_zip(r.content)
        return job, "ok", rows
    except Exception as exc:
        return job, f"error:{type(exc).__name__}", []


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill broad historical Binance Futures H6 + funding for AdaptiveTrend replication")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2024-12-31")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--timeout", type=float, default=45.0)
    ap.add_argument("--symbols", default="", help="Optional comma-separated Binance symbols; otherwise enumerate historical archive")
    args = ap.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    con = ensure_db(Path(args.db))
    session = requests.Session()
    session.headers["User-Agent"] = "freqtrade-v5-adaptivetrend-replication/1.0"

    if args.symbols.strip():
        symbols = [x.strip().upper() for x in args.symbols.split(",") if x.strip()]
    else:
        print("Enumerating historical Binance USDT perpetual archive...", flush=True)
        symbols = list_historical_symbols(session)
    print(f"Historical candidate symbols: {len(symbols)}", flush=True)

    completed = {(r[0], r[1], r[2]) for r in con.execute("SELECT kind,symbol,stamp FROM ingest WHERE status IN ('ok','missing')")}
    jobs: list[Job] = []
    for sym in symbols:
        for m in month_iter(start, end):
            stamp = f"{m.year:04d}-{m.month:02d}"
            if ("kline", sym, stamp) not in completed:
                jobs.append(Job("kline", sym, stamp, f"{DATA_BASE}/klines/{sym}/6h/{sym}-6h-{stamp}.zip"))
            if ("funding", sym, stamp) not in completed:
                jobs.append(Job("funding", sym, stamp, f"{DATA_BASE}/fundingRate/{sym}/{sym}-fundingRate-{stamp}.zip"))

    print(f"Jobs to process: {len(jobs):,} (resume-aware) | workers={args.workers}", flush=True)
    started = time.monotonic()
    counts = {"ok": 0, "missing": 0, "errors": 0}
    inserted_candles = 0
    inserted_funding = 0

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = [ex.submit(fetch_job, j, args.timeout) for j in jobs]
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
