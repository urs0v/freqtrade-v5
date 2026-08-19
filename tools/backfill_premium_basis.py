#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
TF = "15m"
BAR_MS = 15 * 60 * 1000


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def day_iter(a: date, b: date):
    d = a
    while d <= b:
        yield d
        d += timedelta(days=1)


def month_iter(a: date, b: date):
    d = date(a.year, a.month, 1)
    last = date(b.year, b.month, 1)
    while d <= last:
        yield d
        d = date(d.year + (d.month == 12), 1 if d.month == 12 else d.month + 1, 1)


def pair_to_symbol(pair: str) -> str:
    return pair.split("/")[0] + "USDT"


def read_symbols(config: Path) -> list[str]:
    obj = json.loads(config.read_text())
    pairs = obj.get("exchange", {}).get("pair_whitelist", [])
    syms = [pair_to_symbol(str(p)) for p in pairs]
    if not syms:
        raise RuntimeError("No pair_whitelist in config")
    return list(dict.fromkeys(syms))


def ensure_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS premium_15m (
            bucket_ms INTEGER NOT NULL,
            available_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            PRIMARY KEY(bucket_ms, symbol)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_log (
            kind TEXT NOT NULL,
            symbol TEXT NOT NULL,
            stamp TEXT NOT NULL,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0,
            bytes INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY(kind, symbol, stamp)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_premium_symbol_time ON premium_15m(symbol, bucket_ms)")
    con.commit()
    return con


@dataclass(frozen=True)
class Job:
    kind: str
    symbol: str
    stamp: str
    url: str


def build_jobs(symbols: list[str], start: date, end: date) -> list[Job]:
    jobs: list[Job] = []
    end_month = date(end.year, end.month, 1)
    for sym in symbols:
        for m in month_iter(start, end):
            if m >= end_month:
                continue
            stamp = f"{m.year:04d}-{m.month:02d}"
            name = f"{sym}-{TF}-{stamp}.zip"
            jobs.append(Job("monthly", sym, stamp, f"{BASE}/monthly/premiumIndexKlines/{sym}/{TF}/{name}"))
        current_start = max(start, end_month)
        for d in day_iter(current_start, end):
            stamp = d.isoformat()
            name = f"{sym}-{TF}-{stamp}.zip"
            jobs.append(Job("daily", sym, stamp, f"{BASE}/daily/premiumIndexKlines/{sym}/{TF}/{name}"))
    return jobs


def parse_zip(data: bytes, symbol: str) -> list[tuple]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            return []
        raw = zf.read(names[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if df.empty:
        return []
    # Some archives include a header row; numeric coercion safely removes it.
    ts = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    op = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    hi = pd.to_numeric(df.iloc[:, 2], errors="coerce")
    lo = pd.to_numeric(df.iloc[:, 3], errors="coerce")
    cl = pd.to_numeric(df.iloc[:, 4], errors="coerce")
    valid = ts.notna() & cl.notna()
    rows = []
    for t, o, h, l, c in zip(ts[valid], op[valid], hi[valid], lo[valid], cl[valid]):
        t = int(t)
        if t > 10**14:  # microseconds in newer archives
            t //= 1000
        elif t < 10**11:
            t *= 1000
        rows.append((t, t + BAR_MS, symbol,
                     float(o) if pd.notna(o) else None,
                     float(h) if pd.notna(h) else None,
                     float(l) if pd.notna(l) else None,
                     float(c)))
    return rows


def fetch(job: Job, timeout: int = 90) -> tuple[Job, str, list[tuple], int]:
    req = urllib.request.Request(job.url, headers={"User-Agent": "rmv5-premium-lab/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if not data.startswith(b"PK"):
            return job, "bad-content", [], len(data)
        return job, "ok", parse_zip(data, job.symbol), len(data)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return job, "missing", [], 0
        return job, f"error:http{e.code}", [], 0
    except Exception as e:
        return job, f"error:{type(e).__name__}:{str(e)[:100]}", [], 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill Binance premiumIndexKlines 15m with point-in-time availability")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/premium_basis.sqlite")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-08-19")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    started = time.monotonic()
    db = Path(args.db)
    con = ensure_db(db)
    symbols = read_symbols(Path(args.config))
    jobs = build_jobs(symbols, parse_date(args.start), parse_date(args.end))
    done = {(r[0], r[1], r[2]) for r in con.execute("SELECT kind,symbol,stamp FROM ingest_log WHERE status IN ('ok','missing')")}
    jobs = [j for j in jobs if (j.kind, j.symbol, j.stamp) not in done]

    print("=== PREMIUM / BASIS BACKFILL ===")
    print(f"Symbols: {len(symbols)} | range {args.start} -> {args.end} | workers={args.workers}")
    print(f"Remaining jobs: {len(jobs)}")

    ok = missing = errors = rows_new = bytes_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(fetch, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            job, status, rows, size = fut.result()
            bytes_total += size
            if status == "ok":
                ok += 1
                if rows:
                    con.executemany(
                        """
                        INSERT INTO premium_15m(bucket_ms,available_ms,symbol,open,high,low,close)
                        VALUES(?,?,?,?,?,?,?)
                        ON CONFLICT(bucket_ms,symbol) DO UPDATE SET
                          available_ms=excluded.available_ms, open=excluded.open, high=excluded.high,
                          low=excluded.low, close=excluded.close
                        """,
                        rows,
                    )
                    rows_new += len(rows)
            elif status == "missing":
                missing += 1
            else:
                errors += 1
            con.execute(
                "INSERT OR REPLACE INTO ingest_log(kind,symbol,stamp,status,rows,bytes,updated_at) VALUES(?,?,?,?,?,?,?)",
                (job.kind, job.symbol, job.stamp, status, len(rows), size, int(time.time())),
            )
            if n % 50 == 0 or n == len(jobs):
                con.commit()
                db_rows = con.execute("SELECT COUNT(*) FROM premium_15m").fetchone()[0]
                print(f"Progress {n}/{len(jobs)} | ok={ok} missing={missing} errors={errors} new_rows={rows_new} db_rows={db_rows} downloaded={bytes_total/1024**2:.1f} MiB elapsed={time.monotonic()-started:.0f}s", flush=True)
    con.commit()
    db_rows = con.execute("SELECT COUNT(*) FROM premium_15m").fetchone()[0]
    by_symbol = dict(con.execute("SELECT symbol,COUNT(*) FROM premium_15m GROUP BY symbol"))
    con.close()
    print("DONE")
    print({"jobs": len(jobs), "ok": ok, "missing": missing, "errors": errors, "db_rows": db_rows, "rows_by_symbol": by_symbol, "db": str(db)})
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
