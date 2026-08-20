#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import re
import sqlite3
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import pandas as pd

BASE = "https://data.binance.vision/data"
TF = "8h"
STABLE_BASES = {"USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI", "USDE", "USDS", "USD1", "PYUSD"}


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def month_floor(d: date) -> date:
    return date(d.year, d.month, 1)


def month_iter(a: date, b: date):
    d = month_floor(a)
    last = month_floor(b)
    while d <= last:
        yield d
        d = date(d.year + (d.month == 12), 1 if d.month == 12 else d.month + 1, 1)


def normalize_ms(v: int) -> int:
    v = int(v)
    if v > 10**14:
        return v // 1000
    if v < 10**11:
        return v * 1000
    return v


def spot_candidates(futures_symbol: str) -> list[tuple[str, float]]:
    """Map multiplier perpetuals such as 1000SHIBUSDT to a tradable Binance spot."""
    ans: list[tuple[str, float]] = [(futures_symbol, 1.0)]
    if not futures_symbol.endswith("USDT"):
        return ans
    base = futures_symbol[:-4]
    m = re.match(r"^(1000|10000|100000|1000000)(.+)$", base)
    if m:
        ans.append((m.group(2) + "USDT", float(m.group(1))))
    m2 = re.match(r"^1M(.+)$", base)
    if m2:
        ans.append((m2.group(1) + "USDT", 1_000_000.0))
    out: list[tuple[str, float]] = []
    seen = set()
    for x in ans:
        if x[0] not in seen:
            seen.add(x[0]); out.append(x)
    return out


def parse_zip(data: bytes) -> list[tuple[int, float, float, float]]:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            return []
        raw = zf.read(names[0])
    df = pd.read_csv(io.BytesIO(raw), header=None)
    if df.empty or df.shape[1] < 8:
        return []
    ts = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    op = pd.to_numeric(df.iloc[:, 1], errors="coerce")
    cl = pd.to_numeric(df.iloc[:, 4], errors="coerce")
    qv = pd.to_numeric(df.iloc[:, 7], errors="coerce")
    ok = ts.notna() & op.notna() & cl.notna()
    rows = []
    for t, o, c, q in zip(ts[ok], op[ok], cl[ok], qv[ok]):
        rows.append((normalize_ms(int(t)), float(o), float(c), float(q) if pd.notna(q) else 0.0))
    return rows


def get_url(url: str, timeout: int = 60) -> tuple[str, bytes]:
    req = urllib.request.Request(url, headers={"User-Agent": "rmv5-logbasis/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
        if not data.startswith(b"PK"):
            return "bad-content", data
        return "ok", data
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return "missing", b""
        return f"error:http{e.code}", b""
    except Exception as e:
        return f"error:{type(e).__name__}:{str(e)[:80]}", b""


@dataclass(frozen=True)
class Job:
    symbol: str
    stamp: str


def ensure_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("""
        CREATE TABLE IF NOT EXISTS perp_8h(
            symbol TEXT NOT NULL,
            open_ms INTEGER NOT NULL,
            open REAL NOT NULL,
            close REAL NOT NULL,
            quote_volume REAL NOT NULL,
            PRIMARY KEY(symbol, open_ms)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS spot_8h(
            symbol TEXT NOT NULL,
            open_ms INTEGER NOT NULL,
            spot_symbol TEXT NOT NULL,
            multiplier REAL NOT NULL,
            open REAL NOT NULL,
            close REAL NOT NULL,
            PRIMARY KEY(symbol, open_ms)
        )
    """)
    con.execute("""
        CREATE TABLE IF NOT EXISTS ingest(
            symbol TEXT NOT NULL,
            stamp TEXT NOT NULL,
            status TEXT NOT NULL,
            perp_rows INTEGER NOT NULL DEFAULT 0,
            spot_rows INTEGER NOT NULL DEFAULT 0,
            spot_symbol TEXT,
            bytes INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            updated_at INTEGER NOT NULL,
            PRIMARY KEY(symbol, stamp)
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_perp_time ON perp_8h(open_ms)")
    con.execute("CREATE INDEX IF NOT EXISTS idx_spot_time ON spot_8h(open_ms)")
    con.commit()
    return con


def build_jobs(core: Path, start: date, end: date) -> list[Job]:
    con = sqlite3.connect(core, timeout=120)
    rows = con.execute("SELECT symbol, MIN(open_time), MAX(open_time) FROM candles GROUP BY symbol").fetchall()
    con.close()
    jobs: list[Job] = []
    for sym, lo, hi in rows:
        if not sym.endswith("USDT") or sym[:-4] in STABLE_BASES:
            continue
        fd = datetime.utcfromtimestamp(int(lo) / 1000).date()
        ld = datetime.utcfromtimestamp(int(hi) / 1000).date()
        a, b = max(start, fd), min(end, ld)
        if a > b:
            continue
        for m in month_iter(a, b):
            jobs.append(Job(sym, f"{m.year:04d}-{m.month:02d}"))
    return jobs


def process(job: Job) -> tuple[Job, str, list, list, str | None, int, str | None]:
    sym, stamp = job.symbol, job.stamp
    fn = f"{sym}-{TF}-{stamp}.zip"
    fut_url = f"{BASE}/futures/um/monthly/klines/{sym}/{TF}/{fn}"
    fs, fdata = get_url(fut_url)
    total_bytes = len(fdata)
    if fs != "ok":
        return job, fs, [], [], None, total_bytes, None if fs == "missing" else fs
    try:
        frows = parse_zip(fdata)
    except Exception as e:
        return job, "error:perp-parse", [], [], None, total_bytes, repr(e)[:200]
    if not frows:
        return job, "error:empty-perp", [], [], None, total_bytes, "empty futures archive"

    srows_out: list[tuple] = []
    used_spot = None
    for spot_sym, mult in spot_candidates(sym):
        sfn = f"{spot_sym}-{TF}-{stamp}.zip"
        surl = f"{BASE}/spot/monthly/klines/{spot_sym}/{TF}/{sfn}"
        ss, sdata = get_url(surl)
        total_bytes += len(sdata)
        if ss != "ok":
            continue
        try:
            sr = parse_zip(sdata)
        except Exception:
            continue
        if not sr:
            continue
        used_spot = spot_sym
        srows_out = [(sym, t, spot_sym, mult, o * mult, c * mult) for t, o, c, _ in sr]
        break

    frows_out = [(sym, t, o, c, q) for t, o, c, q in frows]
    status = "ok" if srows_out else "perp_only"
    return job, status, frows_out, srows_out, used_spot, total_bytes, None


def main() -> int:
    ap = argparse.ArgumentParser(description="Backfill synchronized Binance USD-M perp + Binance Spot 8h klines for causal log-basis research")
    ap.add_argument("--core", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    ap.add_argument("--db", default="/freqtrade/user_data/logbasis_8h/logbasis.sqlite")
    ap.add_argument("--start", default="2021-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    core = Path(args.core)
    if not core.exists():
        raise RuntimeError(f"Missing core DB: {core}")
    db = Path(args.db)
    con = ensure_db(db)
    jobs = build_jobs(core, parse_date(args.start), parse_date(args.end))
    done = {(s, m) for s, m in con.execute("SELECT symbol,stamp FROM ingest WHERE status IN ('ok','perp_only','missing')")}
    jobs = [j for j in jobs if (j.symbol, j.stamp) not in done]

    print("=== LOG-BASIS 8H DATA BACKFILL ===")
    print(f"Range: {args.start} -> {args.end} | workers={args.workers}")
    print("Source: official Binance Data Vision monthly USD-M klines + Spot klines")
    print("Direct spot symbol first; multiplier-token mapping only when needed.")
    print(f"Remaining symbol-month jobs: {len(jobs):,}", flush=True)

    started = time.monotonic()
    ok = perp_only = missing = errors = bytes_total = new_perp = new_spot = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(process, j) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            job, status, frows, srows, spot_sym, nbytes, err = fut.result()
            bytes_total += nbytes
            if frows:
                con.executemany("INSERT OR REPLACE INTO perp_8h(symbol,open_ms,open,close,quote_volume) VALUES(?,?,?,?,?)", frows)
                new_perp += len(frows)
            if srows:
                con.executemany("INSERT OR REPLACE INTO spot_8h(symbol,open_ms,spot_symbol,multiplier,open,close) VALUES(?,?,?,?,?,?)", srows)
                new_spot += len(srows)
            if status == "ok": ok += 1
            elif status == "perp_only": perp_only += 1
            elif status == "missing": missing += 1
            else: errors += 1
            con.execute(
                "INSERT OR REPLACE INTO ingest(symbol,stamp,status,perp_rows,spot_rows,spot_symbol,bytes,error,updated_at) VALUES(?,?,?,?,?,?,?,?,?)",
                (job.symbol, job.stamp, status, len(frows), len(srows), spot_sym, nbytes, err, int(time.time())),
            )
            if n % 100 == 0 or n == len(jobs):
                con.commit()
                pr = con.execute("SELECT COUNT(*) FROM perp_8h").fetchone()[0]
                sr = con.execute("SELECT COUNT(*) FROM spot_8h").fetchone()[0]
                print(
                    f"Progress {n:,}/{len(jobs):,} | ok={ok} perp_only={perp_only} missing={missing} errors={errors} "
                    f"db_perp={pr:,} db_spot={sr:,} downloaded={bytes_total/1024**2:.1f} MiB elapsed={time.monotonic()-started:.0f}s",
                    flush=True,
                )
    con.commit()
    pr = con.execute("SELECT COUNT(*) FROM perp_8h").fetchone()[0]
    sr = con.execute("SELECT COUNT(*) FROM spot_8h").fetchone()[0]
    syms_p = con.execute("SELECT COUNT(DISTINCT symbol) FROM perp_8h").fetchone()[0]
    syms_s = con.execute("SELECT COUNT(DISTINCT symbol) FROM spot_8h").fetchone()[0]
    con.close()
    print("DONE")
    print({"perp_rows": pr, "spot_rows": sr, "perp_symbols": syms_p, "spot_mapped_symbols": syms_s, "errors": errors, "db": str(db)})
    return 0 if errors == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
