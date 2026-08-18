from __future__ import annotations

import argparse
import asyncio
import io
import json
import sqlite3
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import aiohttp
import pandas as pd

BASE = "https://data.binance.vision/data/futures/um"
DEFAULT_DB = "/freqtrade/user_data/v5/features.sqlite"
DEFAULT_UNIVERSE = "/freqtrade/user_data/v5/universe.json"
DEFAULT_CACHE = "/freqtrade/user_data/v5/free-cache"


def parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def month_iter(start: date, end: date):
    cur = date(start.year, start.month, 1)
    last = date(end.year, end.month, 1)
    while cur <= last:
        yield cur
        cur = date(cur.year + (cur.month == 12), 1 if cur.month == 12 else cur.month + 1, 1)


def day_iter(start: date, end: date):
    cur = start
    while cur <= end:
        yield cur
        cur += timedelta(days=1)


def to_utc_series(series: pd.Series) -> pd.Series:
    num = pd.to_numeric(series, errors="coerce")
    if num.notna().mean() > 0.8:
        med = num.dropna().abs().median() if num.notna().any() else 0
        if med > 1e14:
            unit = "us"
        elif med > 1e11:
            unit = "ms"
        elif med > 1e9:
            unit = "s"
        else:
            unit = None
        if unit:
            return pd.to_datetime(num, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(series, utc=True, errors="coerce")


def ensure_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path, timeout=30)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS features (
            bucket_ms INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            oi REAL,
            funding_rate REAL,
            long_liq_usdt REAL,
            short_liq_usdt REAL,
            taker_ratio REAL,
            top_ls_ratio REAL,
            liq_observed INTEGER NOT NULL DEFAULT 0,
            updated_ms INTEGER NOT NULL,
            PRIMARY KEY (bucket_ms, symbol)
        )
        """
    )
    cols = {r[1] for r in con.execute("PRAGMA table_info(features)")}
    if "liq_observed" not in cols:
        con.execute("ALTER TABLE features ADD COLUMN liq_observed INTEGER NOT NULL DEFAULT 1")
    con.execute("CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON features(symbol, bucket_ms)")
    con.commit()
    return con


def read_symbols(raw: str | None, universe: Path) -> list[str]:
    if raw:
        syms = [x.strip().upper() for x in raw.split(",") if x.strip()]
    elif universe.exists():
        obj = json.loads(universe.read_text())
        syms = obj.get("symbols", []) if isinstance(obj, dict) else obj
        syms = [str(x).upper() for x in syms]
    else:
        syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT", "DOGEUSDT", "LINKUSDT"]
    return list(dict.fromkeys(syms))


@dataclass(frozen=True)
class Job:
    kind: str
    symbol: str
    stamp: str
    url: str
    cache: Path


def build_jobs(symbols: list[str], start: date, end: date, cache: Path) -> list[Job]:
    jobs: list[Job] = []
    for sym in symbols:
        for d in day_iter(start, end):
            stamp = d.isoformat()
            name = f"{sym}-metrics-{stamp}.zip"
            jobs.append(Job(
                "metrics", sym, stamp,
                f"{BASE}/daily/metrics/{sym}/{name}",
                cache / "metrics" / sym / name,
            ))
        for m in month_iter(start, end):
            stamp = f"{m.year:04d}-{m.month:02d}"
            name = f"{sym}-fundingRate-{stamp}.zip"
            jobs.append(Job(
                "funding", sym, stamp,
                f"{BASE}/monthly/fundingRate/{sym}/{name}",
                cache / "fundingRate" / sym / name,
            ))
    return jobs


async def fetch_job(session: aiohttp.ClientSession, sem: asyncio.Semaphore, job: Job, retries: int = 3):
    if job.cache.exists() and job.cache.stat().st_size > 0:
        return job, "cached", job.cache.read_bytes()
    job.cache.parent.mkdir(parents=True, exist_ok=True)
    async with sem:
        for attempt in range(retries):
            try:
                async with session.get(job.url) as r:
                    if r.status == 404:
                        return job, "missing", None
                    if r.status in {429, 500, 502, 503, 504}:
                        await asyncio.sleep(1.5 * (attempt + 1))
                        continue
                    r.raise_for_status()
                    data = await r.read()
                    if not data.startswith(b"PK"):
                        return job, f"bad-content-{r.status}", None
                    job.cache.write_bytes(data)
                    return job, "downloaded", data
            except Exception as e:
                if attempt + 1 == retries:
                    return job, f"error:{type(e).__name__}", None
                await asyncio.sleep(1.5 * (attempt + 1))
    return job, "error", None


def csv_from_zip(data: bytes) -> pd.DataFrame:
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        names = [n for n in z.namelist() if n.lower().endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with z.open(names[0]) as f:
            return pd.read_csv(f)


def col(df: pd.DataFrame, *names: str) -> str | None:
    lower = {str(c).lower(): str(c) for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


def parse_metrics(job: Job, data: bytes) -> list[tuple]:
    df = csv_from_zip(data)
    if df.empty:
        return []
    t = col(df, "create_time", "timestamp", "time")
    s = col(df, "symbol")
    oi = col(df, "sum_open_interest", "open_interest", "openInterest")
    taker = col(df, "sum_taker_long_short_vol_ratio", "taker_long_short_ratio", "buySellRatio")
    top = col(df, "sum_toptrader_long_short_ratio", "top_long_short_ratio", "longShortRatio")
    if t is None:
        return []
    df["_date"] = to_utc_series(df[t])
    df = df.dropna(subset=["_date"])
    if df.empty:
        return []
    df["_bucket"] = df["_date"].dt.floor("15min")
    df["_symbol"] = df[s].astype(str).str.upper() if s else job.symbol
    agg: dict[str, str] = {}
    if oi: agg[oi] = "last"
    if taker: agg[taker] = "last"
    if top: agg[top] = "last"
    if not agg:
        return []
    grouped = df.groupby(["_symbol", "_bucket"], as_index=False).agg(agg)
    rows = []
    for _, r in grouped.iterrows():
        b = int(r["_bucket"].timestamp() * 1000)
        rows.append((
            b,
            str(r["_symbol"]),
            float(r[oi]) if oi and pd.notna(r[oi]) else None,
            float(r[taker]) if taker and pd.notna(r[taker]) else None,
            float(r[top]) if top and pd.notna(r[top]) else None,
            b,
        ))
    return rows


def parse_funding(job: Job, data: bytes) -> list[tuple[int, str, float]]:
    df = csv_from_zip(data)
    if df.empty:
        return []
    t = col(df, "calc_time", "funding_time", "fundingTime", "timestamp", "time")
    rate = col(df, "funding_rate", "last_funding_rate", "fundingRate")
    if t is None or rate is None:
        return []
    df["_date"] = to_utc_series(df[t])
    df["_rate"] = pd.to_numeric(df[rate], errors="coerce")
    df = df.dropna(subset=["_date", "_rate"]).sort_values("_date")
    return [(int(r["_date"].timestamp() * 1000), job.symbol, float(r["_rate"])) for _, r in df.iterrows()]


def upsert_metrics(con: sqlite3.Connection, rows: list[tuple]) -> None:
    if not rows:
        return
    con.executemany(
        """
        INSERT INTO features
        (bucket_ms, symbol, oi, funding_rate, long_liq_usdt, short_liq_usdt,
         taker_ratio, top_ls_ratio, liq_observed, updated_ms)
        VALUES (?, ?, ?, NULL, NULL, NULL, ?, ?, 0, ?)
        ON CONFLICT(bucket_ms, symbol) DO UPDATE SET
          oi=COALESCE(excluded.oi, features.oi),
          taker_ratio=COALESCE(excluded.taker_ratio, features.taker_ratio),
          top_ls_ratio=COALESCE(excluded.top_ls_ratio, features.top_ls_ratio),
          updated_ms=MAX(features.updated_ms, excluded.updated_ms)
        """,
        rows,
    )


def apply_funding(con: sqlite3.Connection, symbol: str, events: list[tuple[int, str, float]], start_ms: int, end_ms: int) -> int:
    if not events:
        return 0
    events = sorted(events, key=lambda x: x[0])
    feature_rows = con.execute(
        "SELECT bucket_ms FROM features WHERE symbol=? AND bucket_ms BETWEEN ? AND ? ORDER BY bucket_ms",
        (symbol, start_ms, end_ms),
    ).fetchall()
    if not feature_rows:
        return 0
    i = 0
    last_rate = None
    updates = []
    for (b,) in feature_rows:
        while i < len(events) and events[i][0] <= b:
            last_rate = events[i][2]
            i += 1
        if last_rate is not None:
            updates.append((last_rate, symbol, b))
    con.executemany("UPDATE features SET funding_rate=? WHERE symbol=? AND bucket_ms=?", updates)
    return len(updates)


async def main_async(args) -> None:
    start = parse_date(args.start)
    end = parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")
    symbols = read_symbols(args.symbols, Path(args.universe))
    if not symbols:
        raise SystemExit("No symbols")

    db = Path(args.db)
    cache = Path(args.cache)
    con = ensure_db(db)

    jobs = build_jobs(symbols, start, end, cache)
    metrics_jobs = sum(j.kind == "metrics" for j in jobs)
    funding_jobs = len(jobs) - metrics_jobs
    print(f"Symbols ({len(symbols)}): {','.join(symbols)}")
    print(f"Range: {start} -> {end}")
    print(f"Jobs: metrics={metrics_jobs}, funding={funding_jobs}")
    print("Source: Binance public data archives only")

    timeout = aiohttp.ClientTimeout(total=60)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    sem = asyncio.Semaphore(args.concurrency)
    funding_events: dict[str, list[tuple[int, str, float]]] = {s: [] for s in symbols}
    stats = {"downloaded": 0, "cached": 0, "missing": 0, "error": 0, "rows": 0}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector, headers={"User-Agent": "RMV5-free-backfill/1.0"}) as session:
        pending = [asyncio.create_task(fetch_job(session, sem, j)) for j in jobs]
        done_count = 0
        for fut in asyncio.as_completed(pending):
            job, status, data = await fut
            done_count += 1
            if status in {"downloaded", "cached"}:
                stats[status] += 1
            elif status == "missing":
                stats["missing"] += 1
            else:
                stats["error"] += 1
                if args.verbose:
                    print(job.url, status)
            if data:
                try:
                    if job.kind == "metrics":
                        rows = parse_metrics(job, data)
                        upsert_metrics(con, rows)
                        stats["rows"] += len(rows)
                    else:
                        funding_events[job.symbol].extend(parse_funding(job, data))
                except Exception as e:
                    stats["error"] += 1
                    if args.verbose:
                        print("parse error", job.url, repr(e))
            if done_count % 250 == 0:
                con.commit()
                print(f"Progress {done_count}/{len(jobs)} | rows={stats['rows']} | missing={stats['missing']} | errors={stats['error']}")

    con.commit()
    start_ms = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    end_ms = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000) - 1
    funding_updates = 0
    for sym in symbols:
        funding_updates += apply_funding(con, sym, funding_events[sym], start_ms, end_ms)
    con.commit()

    count = con.execute("SELECT COUNT(*) FROM features WHERE bucket_ms BETWEEN ? AND ?", (start_ms, end_ms)).fetchone()[0]
    covered = con.execute(
        "SELECT COUNT(DISTINCT symbol) FROM features WHERE bucket_ms BETWEEN ? AND ?",
        (start_ms, end_ms),
    ).fetchone()[0]
    con.close()

    print("DONE")
    print(json.dumps({
        **stats,
        "funding_updates": funding_updates,
        "db_rows_in_range": count,
        "symbols_with_data": covered,
        "db": str(db),
        "cache": str(cache),
    }, indent=2))
    print("Historical liquidation archives are not required: RMV5 uses the OI/price/flow cascade proxy when liq_observed=0.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Backfill RMV5 from free Binance public archives (no paid/free-tier API dependency).")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default=(date.today() - timedelta(days=1)).isoformat())
    ap.add_argument("--symbols", help="Comma-separated Binance futures symbols. Default: current RMV5 universe.json")
    ap.add_argument("--universe", default=DEFAULT_UNIVERSE)
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--cache", default=DEFAULT_CACHE)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
