#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import os
import sqlite3
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT"]
QUARTER_MS = 15 * 60 * 1000
OPENING_MS = 10 * 1000


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def day_iter(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def ensure_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=60)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS qh_flow (
            bucket_ms INTEGER NOT NULL,
            available_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            signed_qty REAL NOT NULL,
            total_qty REAL NOT NULL,
            signed_notional REAL NOT NULL,
            total_notional REAL NOT NULL,
            trade_count INTEGER NOT NULL,
            first_price REAL,
            last_price REAL,
            imbalance_qty REAL,
            imbalance_notional REAL,
            PRIMARY KEY (bucket_ms, symbol)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS ingest_log (
            symbol TEXT NOT NULL,
            day TEXT NOT NULL,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL DEFAULT 0,
            bytes INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY (symbol, day)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_qh_flow_symbol_time ON qh_flow(symbol, bucket_ms)")
    con.commit()
    return con


@dataclass(frozen=True)
class Job:
    symbol: str
    day: date

    @property
    def stamp(self) -> str:
        return self.day.isoformat()

    @property
    def url(self) -> str:
        name = f"{self.symbol}-aggTrades-{self.stamp}.zip"
        return f"{BASE}/{self.symbol}/{name}"


def bool_buyer_maker(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.to_numpy(dtype=bool)
    x = s.astype(str).str.strip().str.lower()
    return x.isin(["true", "1", "t", "yes"]).to_numpy(dtype=bool)


def normalize_ms(s: pd.Series) -> np.ndarray:
    x = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    valid = x[np.isfinite(x)]
    med = float(np.median(np.abs(valid))) if len(valid) else 0.0
    if med > 1e14:  # microseconds
        x = np.floor(x / 1000.0)
    elif med < 1e11:  # seconds
        x = np.floor(x * 1000.0)
    return x.astype("int64", copy=False)


def detect_header(zf: zipfile.ZipFile, member: str) -> bool:
    with zf.open(member) as f:
        line = f.readline().decode("utf-8", errors="replace").strip()
    first = line.split(",", 1)[0].strip().lower()
    return any(ch.isalpha() for ch in first) or "price" in line.lower() or "agg_trade" in line.lower()


def column_map(df: pd.DataFrame, has_header: bool) -> tuple[str | int, str | int, str | int, str | int]:
    if not has_header:
        return 1, 2, 5, 6
    lower = {str(c).strip().lower(): c for c in df.columns}
    def pick(*names: str):
        for n in names:
            if n in lower:
                return lower[n]
        raise KeyError(f"Missing expected aggTrades column among {names}; got {list(df.columns)}")
    price = pick("price", "p")
    qty = pick("quantity", "qty", "q")
    ts = pick("transact_time", "transacttime", "timestamp", "time", "t")
    bm = pick("is_buyer_maker", "isbuyermaker", "m")
    return price, qty, ts, bm


def parse_zip(path: Path, symbol: str) -> list[tuple]:
    acc: dict[int, list[float]] = {}
    with zipfile.ZipFile(path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            return []
        member = members[0]
        has_header = detect_header(zf, member)
        read_kwargs = {"chunksize": 500_000, "low_memory": False}
        if not has_header:
            read_kwargs.update({"header": None})
        with zf.open(member) as f:
            for chunk in pd.read_csv(f, **read_kwargs):
                price_col, qty_col, ts_col, bm_col = column_map(chunk, has_header)
                price = pd.to_numeric(chunk[price_col], errors="coerce").to_numpy(dtype=float)
                qty = pd.to_numeric(chunk[qty_col], errors="coerce").to_numpy(dtype=float)
                ts = normalize_ms(chunk[ts_col])
                bm = bool_buyer_maker(chunk[bm_col])
                bucket = (ts // QUARTER_MS) * QUARTER_MS
                offset = ts - bucket
                mask = (
                    (offset >= 0)
                    & (offset < OPENING_MS)
                    & np.isfinite(price)
                    & np.isfinite(qty)
                    & (qty > 0)
                )
                if not np.any(mask):
                    continue
                p = price[mask]
                q = qty[mask]
                b = bucket[mask]
                taker_sign = np.where(bm[mask], -1.0, 1.0)
                signed_q = q * taker_sign
                notion = p * q
                signed_n = notion * taker_sign
                # Chunk rows are chronological in Binance archives. Keep first/last observed price.
                for buck in np.unique(b):
                    m = b == buck
                    vals = acc.get(int(buck))
                    sq = float(signed_q[m].sum())
                    tq = float(q[m].sum())
                    sn = float(signed_n[m].sum())
                    tn = float(notion[m].sum())
                    cnt = int(m.sum())
                    fp = float(p[m][0])
                    lp = float(p[m][-1])
                    if vals is None:
                        acc[int(buck)] = [sq, tq, sn, tn, float(cnt), fp, lp]
                    else:
                        vals[0] += sq
                        vals[1] += tq
                        vals[2] += sn
                        vals[3] += tn
                        vals[4] += cnt
                        vals[6] = lp
    rows = []
    for buck, v in sorted(acc.items()):
        sq, tq, sn, tn, cnt, fp, lp = v
        iq = sq / tq if tq > 0 else None
        inn = sn / tn if tn > 0 else None
        rows.append((
            buck,
            buck + OPENING_MS,
            symbol,
            sq,
            tq,
            sn,
            tn,
            int(cnt),
            fp,
            lp,
            iq,
            inn,
        ))
    return rows


def fetch_and_parse(job: Job, tmpdir: Path, timeout: int = 180) -> tuple[Job, str, list[tuple], int]:
    fd, raw_path = tempfile.mkstemp(prefix=f"{job.symbol}-{job.stamp}-", suffix=".zip", dir=tmpdir)
    os.close(fd)
    p = Path(raw_path)
    size = 0
    try:
        req = urllib.request.Request(job.url, headers={"User-Agent": "rmv5-orderflow-lab/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r, p.open("wb") as out:
                while True:
                    buf = r.read(1024 * 1024)
                    if not buf:
                        break
                    out.write(buf)
                    size += len(buf)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return job, "missing", [], 0
            raise
        if size < 4:
            return job, "empty", [], size
        with p.open("rb") as f:
            if f.read(2) != b"PK":
                return job, "bad-content", [], size
        rows = parse_zip(p, job.symbol)
        return job, "ok", rows, size
    except Exception as e:
        return job, f"error:{type(e).__name__}:{str(e)[:120]}", [], size
    finally:
        try:
            p.unlink(missing_ok=True)
        except Exception:
            pass


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill first-10s quarter-hour signed order flow from Binance aggTrades")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/qh_orderflow.sqlite")
    ap.add_argument("--tmpdir", default="/freqtrade/user_data/alpha_lab/qh_tmp")
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-08-19")
    ap.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS))
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    tmpdir = Path(args.tmpdir)
    tmpdir.mkdir(parents=True, exist_ok=True)
    con = ensure_db(Path(args.db))

    done = {
        (str(r[0]), str(r[1]))
        for r in con.execute("SELECT symbol, day FROM ingest_log WHERE status='ok'")
    }
    jobs = [Job(sym, d) for sym in symbols for d in day_iter(start, end) if (sym, d.isoformat()) not in done]

    print("=== QUARTER-HOUR RAW ORDER-FLOW BACKFILL ===", flush=True)
    print(f"Symbols: {','.join(symbols)}", flush=True)
    print(f"Range: {start} -> {end}", flush=True)
    print("Feature: signed aggTrade volume from the first 10 seconds of every 15-minute boundary", flush=True)
    print(f"Remaining jobs: {len(jobs)} | workers={args.workers}", flush=True)
    print("Raw zip files are processed then deleted; only compact quarter-hour rows are retained.", flush=True)

    if not jobs:
        n = con.execute("SELECT COUNT(*) FROM qh_flow").fetchone()[0]
        print(f"Nothing to do. DB rows: {n}", flush=True)
        return 0

    processed = ok = missing = errors = rows_total = bytes_total = 0
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futures = {ex.submit(fetch_and_parse, job, tmpdir): job for job in jobs}
        for fut in as_completed(futures):
            job, status, rows, nbytes = fut.result()
            processed += 1
            bytes_total += nbytes
            if status == "ok":
                ok += 1
                if rows:
                    con.executemany(
                        """
                        INSERT INTO qh_flow (
                            bucket_ms, available_ms, symbol, signed_qty, total_qty,
                            signed_notional, total_notional, trade_count,
                            first_price, last_price, imbalance_qty, imbalance_notional
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(bucket_ms, symbol) DO UPDATE SET
                            available_ms=excluded.available_ms,
                            signed_qty=excluded.signed_qty,
                            total_qty=excluded.total_qty,
                            signed_notional=excluded.signed_notional,
                            total_notional=excluded.total_notional,
                            trade_count=excluded.trade_count,
                            first_price=excluded.first_price,
                            last_price=excluded.last_price,
                            imbalance_qty=excluded.imbalance_qty,
                            imbalance_notional=excluded.imbalance_notional
                        """,
                        rows,
                    )
                    rows_total += len(rows)
            elif status == "missing":
                missing += 1
            else:
                errors += 1
            con.execute(
                """
                INSERT INTO ingest_log(symbol, day, status, rows, bytes, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(symbol, day) DO UPDATE SET
                    status=excluded.status, rows=excluded.rows, bytes=excluded.bytes, updated_at=excluded.updated_at
                """,
                (job.symbol, job.stamp, status, len(rows), nbytes, int(time.time())),
            )
            con.commit()
            if processed % 25 == 0 or processed == len(jobs):
                db_rows = con.execute("SELECT COUNT(*) FROM qh_flow").fetchone()[0]
                print(
                    f"Progress {processed}/{len(jobs)} | ok={ok} missing={missing} errors={errors} "
                    f"new_rows={rows_total} db_rows={db_rows} downloaded={bytes_total/1024**3:.2f} GiB "
                    f"elapsed={time.monotonic()-started:.0f}s",
                    flush=True,
                )

    db_rows = con.execute("SELECT COUNT(*) FROM qh_flow").fetchone()[0]
    sym_rows = dict(con.execute("SELECT symbol, COUNT(*) FROM qh_flow GROUP BY symbol ORDER BY symbol").fetchall())
    print("DONE", flush=True)
    print({
        "jobs": len(jobs),
        "ok": ok,
        "missing": missing,
        "errors": errors,
        "downloaded_gib": round(bytes_total / 1024**3, 3),
        "db_rows": db_rows,
        "rows_by_symbol": sym_rows,
        "db": args.db,
    }, flush=True)
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
