#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import io
import math
import os
import sqlite3
import time
import xml.etree.ElementTree as ET
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT"]
S3_LIST = "https://s3-ap-northeast-1.amazonaws.com/data.binance.vision"
QH_MS = 15 * 60 * 1000
TEN_S_MS = 10 * 1000
HORIZON_MS = 8 * 60 * 60 * 1000
ROUND_TRIP_COST_BPS = 14.0
PAPER_8H_BPS_PER_OI = {
    "BTCUSDT": 5.55,
    "ETHUSDT": 4.68,
    "XRPUSDT": 6.20,
    "SOLUSDT": 2.41,
    "DOGEUSDT": 8.49,
    "ADAUSDT": 5.39,
}


@dataclass(frozen=True)
class ArchiveFile:
    symbol: str
    stamp: str
    key: str
    size: int

    @property
    def url(self) -> str:
        return f"https://data.binance.vision/{self.key}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Pre-registered Binance quarter-hour first-10s order-flow persistence diagnostic"
    )
    p.add_argument("--db", default="/freqtrade/user_data/qh_orderflow_v0/qh.sqlite")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--workers", type=int, default=int(os.environ.get("QH_WORKERS", "32")))
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument("--output-dir", default="/freqtrade/user_data/qh_orderflow_v0")
    p.add_argument("--skip-backfill", action="store_true")
    return p.parse_args()


def ensure_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), timeout=120)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS qh_events (
            symbol TEXT NOT NULL,
            boundary_ms INTEGER NOT NULL,
            buy_qty REAL NOT NULL,
            sell_qty REAL NOT NULL,
            oi REAL NOT NULL,
            open_trade_count INTEGER NOT NULL,
            open_last_side INTEGER,
            next10_last_price REAL,
            PRIMARY KEY(symbol, boundary_ms)
        )
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS qh_ingest (
            symbol TEXT NOT NULL,
            stamp TEXT NOT NULL,
            status TEXT NOT NULL,
            rows INTEGER NOT NULL,
            bytes INTEGER NOT NULL,
            error TEXT,
            updated_at TEXT NOT NULL,
            PRIMARY KEY(symbol, stamp)
        )
        """
    )
    con.execute("CREATE INDEX IF NOT EXISTS idx_qh_events_time ON qh_events(boundary_ms)")
    con.commit()
    return con


def list_daily_archives(session: requests.Session, symbol: str, start: date, end: date) -> list[ArchiveFile]:
    prefix = f"data/futures/um/daily/aggTrades/{symbol}/{symbol}-aggTrades-"
    marker = f"{symbol}-aggTrades-"
    token = None
    out: list[ArchiveFile] = []
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        r = session.get(S3_LIST, params=params, timeout=60)
        r.raise_for_status()
        root = ET.fromstring(r.text)
        ns = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}
        for node in root.findall("s3:Contents", ns):
            key = node.findtext("s3:Key", default="", namespaces=ns) or ""
            if not key.endswith(".zip"):
                continue
            tail = key.rsplit("/", 1)[-1]
            if not tail.startswith(marker):
                continue
            stamp = tail[len(marker):-4]
            try:
                d = datetime.strptime(stamp, "%Y-%m-%d").date()
            except ValueError:
                continue
            if start <= d <= end:
                size = int(node.findtext("s3:Size", default="0", namespaces=ns) or 0)
                out.append(ArchiveFile(symbol, stamp, key, size))
        truncated = (root.findtext("s3:IsTruncated", default="false", namespaces=ns) or "false").lower() == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", default=None, namespaces=ns)
        if not token:
            break
    return sorted(out, key=lambda x: x.stamp)


def _normalize_ts(v: str) -> int:
    ts = int(v)
    if ts > 100_000_000_000_000:
        ts //= 1000
    return ts


def parse_daily_zip(path: Path, symbol: str) -> list[tuple]:
    # Only two 10-second bins are needed per quarter hour:
    # [00,10): current OI; [10,20): P_{t+1}, paper forward-return start price.
    events: dict[int, list] = {}
    with zipfile.ZipFile(path) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise RuntimeError(f"No CSV in {path.name}")
        with zf.open(names[0], "r") as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.reader(text)
            for row in reader:
                if len(row) < 7:
                    continue
                try:
                    price = float(row[1])
                    qty = float(row[2])
                    ts = _normalize_ts(row[5])
                except (ValueError, TypeError):
                    continue
                boundary = (ts // QH_MS) * QH_MS
                off = ts - boundary
                if off >= 2 * TEN_S_MS:
                    continue
                rec = events.get(boundary)
                if rec is None:
                    rec = [0.0, 0.0, 0, None, None]
                    events[boundary] = rec
                if off < TEN_S_MS:
                    maker = str(row[6]).strip().lower() in {"true", "1", "t"}
                    side = -1 if maker else 1
                    if side > 0:
                        rec[0] += qty
                    else:
                        rec[1] += qty
                    rec[2] += 1
                    rec[3] = side
                else:
                    rec[4] = price

    rows = []
    for boundary, rec in events.items():
        buy, sell, count, last_side, pnext = rec
        total = buy + sell
        if total <= 0:
            continue
        oi = (buy - sell) / total
        rows.append((symbol, int(boundary), float(buy), float(sell), float(oi), int(count), last_side, pnext))
    return rows


def fetch_and_parse(job: ArchiveFile, tmpdir: Path, timeout: float) -> tuple[ArchiveFile, str, list[tuple], str | None]:
    target = tmpdir / f"{job.symbol}-{job.stamp}.zip"
    part = target.with_suffix(".zip.part")
    try:
        with requests.get(job.url, stream=True, timeout=(20, timeout), headers={"User-Agent": "freqtrade-v5-qh-orderflow/1.0"}) as r:
            if r.status_code == 404:
                return job, "missing", [], None
            r.raise_for_status()
            with open(part, "wb") as f:
                for chunk in r.iter_content(chunk_size=4 * 1024 * 1024):
                    if chunk:
                        f.write(chunk)
        os.replace(part, target)
        rows = parse_daily_zip(target, job.symbol)
        return job, "ok", rows, None
    except Exception as exc:
        return job, "error", [], f"{type(exc).__name__}: {exc}"
    finally:
        for p in (part, target):
            try:
                p.unlink()
            except FileNotFoundError:
                pass


def backfill(con: sqlite3.Connection, start: date, end: date, workers: int, timeout: float, outdir: Path) -> None:
    archive_end = end + timedelta(days=1)
    session = requests.Session()
    session.headers["User-Agent"] = "freqtrade-v5-qh-orderflow/1.0"
    print(f"Discovering official Binance daily aggTrades: {start} -> {archive_end} ...", flush=True)
    found: dict[str, list[ArchiveFile]] = {}
    with ThreadPoolExecutor(max_workers=min(12, max(1, workers))) as ex:
        futs = {ex.submit(list_daily_archives, session, s, start, archive_end): s for s in SYMBOLS}
        for fut in as_completed(futs):
            s = futs[fut]
            found[s] = fut.result()
            print(f"Archive discovery {s}: {len(found[s])} daily files", flush=True)

    all_files = [x for s in SYMBOLS for x in found.get(s, [])]
    expected_days = (archive_end - start).days + 1
    for s in SYMBOLS:
        if len(found.get(s, [])) < expected_days - 2:
            print(f"WARNING: {s} has {len(found.get(s, []))}/{expected_days} archive days", flush=True)

    done = {(r[0], r[1]) for r in con.execute("SELECT symbol,stamp FROM qh_ingest WHERE status='ok'")}
    jobs = [x for x in all_files if (x.symbol, x.stamp) not in done]
    total_bytes = sum(x.size for x in all_files)
    remaining_bytes = sum(x.size for x in jobs)
    print(
        f"Archive files={len(all_files):,} | compressed={total_bytes/1e9:.1f} GB | "
        f"remaining={len(jobs):,} files / {remaining_bytes/1e9:.1f} GB | workers={workers}",
        flush=True,
    )
    if not jobs:
        print("QH backfill already complete; reusing SQLite features.", flush=True)
        return

    tmpdir = outdir / "tmp"
    tmpdir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    stats = {"ok": 0, "missing": 0, "error": 0, "rows": 0, "bytes": 0}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(fetch_and_parse, j, tmpdir, timeout) for j in jobs]
        for n, fut in enumerate(as_completed(futs), 1):
            job, status, rows, err = fut.result()
            if status == "ok":
                con.executemany(
                    """
                    INSERT OR REPLACE INTO qh_events
                    (symbol,boundary_ms,buy_qty,sell_qty,oi,open_trade_count,open_last_side,next10_last_price)
                    VALUES (?,?,?,?,?,?,?,?)
                    """,
                    rows,
                )
                stats["ok"] += 1
                stats["rows"] += len(rows)
                stats["bytes"] += job.size
            elif status == "missing":
                stats["missing"] += 1
            else:
                stats["error"] += 1
                print(f"ERROR {job.symbol} {job.stamp}: {err}", flush=True)
            con.execute(
                "INSERT OR REPLACE INTO qh_ingest(symbol,stamp,status,rows,bytes,error,updated_at) VALUES (?,?,?,?,?,?,datetime('now'))",
                (job.symbol, job.stamp, status, len(rows), job.size, err),
            )
            if n % 25 == 0 or n == len(jobs):
                con.commit()
                elapsed = max(time.monotonic() - started, 1e-6)
                gb = stats["bytes"] / 1e9
                rate = gb / elapsed * 60.0
                eta = (remaining_bytes / 1e9 - gb) / rate if rate > 0 else math.nan
                print(
                    f"QH backfill {n:,}/{len(jobs):,} | ok={stats['ok']} err={stats['error']} "
                    f"| qh_rows={stats['rows']:,} | processed={gb:.1f} GB | {rate:.2f} GB/min "
                    f"| ETA~{eta:.1f} min",
                    flush=True,
                )
    con.commit()
    if stats["error"]:
        raise RuntimeError(f"Backfill completed with {stats['error']} errors; rerun is resume-aware")


def panel_hac_beta_t(df: pd.DataFrame, ycol: str, xcol: str, lag: int = 32) -> tuple[float, float, int]:
    q = df[["symbol", ycol, xcol, "boundary_ms"]].dropna().sort_values(["symbol", "boundary_ms"])
    if len(q) < 100:
        return math.nan, math.nan, len(q)
    syms = sorted(q.symbol.unique())
    cols = [np.ones(len(q)), q[xcol].to_numpy(float)]
    for s in syms[1:]:
        cols.append((q.symbol.to_numpy() == s).astype(float))
    X = np.column_stack(cols)
    y = q[ycol].to_numpy(float)
    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]), dtype=float)
    symbols_arr = q.symbol.to_numpy()
    for s in syms:
        idx = np.flatnonzero(symbols_arr == s)
        xs = X[idx]
        es = resid[idx]
        z = xs * es[:, None]
        meat += z.T @ z
        L = min(lag, len(idx) - 1)
        for ell in range(1, L + 1):
            w = 1.0 - ell / (L + 1.0)
            g = z[ell:].T @ z[:-ell]
            meat += w * (g + g.T)
    cov = xtx_inv @ meat @ xtx_inv
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    b = float(beta[1])
    return b, (b / se if se > 0 else math.nan), len(q)


def hac_mean_t(df: pd.DataFrame, value_col: str, lag: int = 32) -> tuple[float, float, int]:
    q = df[["symbol", value_col, "boundary_ms"]].dropna().sort_values(["symbol", "boundary_ms"])
    if len(q) < 100:
        return math.nan, math.nan, len(q)
    x = q[value_col].to_numpy(float)
    mean = float(np.mean(x))
    centered = x - mean
    var_sum = 0.0
    symbols_arr = q.symbol.to_numpy()
    for s in q.symbol.unique():
        idx = np.flatnonzero(symbols_arr == s)
        e = centered[idx]
        var_sum += float(e @ e)
        L = min(lag, len(e) - 1)
        for ell in range(1, L + 1):
            w = 1.0 - ell / (L + 1.0)
            var_sum += 2.0 * w * float(e[ell:] @ e[:-ell])
    se = math.sqrt(max(var_sum, 0.0)) / len(x)
    return mean, (mean / se if se > 0 else math.nan), len(x)


def asset_slope(df: pd.DataFrame, lag: int = 32) -> tuple[float, float]:
    q = df[["fwd_bps", "oi"]].dropna()
    n = len(q)
    if n < 100:
        return math.nan, math.nan
    X = np.column_stack([np.ones(n), q.oi.to_numpy(float)])
    y = q.fwd_bps.to_numpy(float)
    inv = np.linalg.pinv(X.T @ X)
    b = inv @ (X.T @ y)
    e = y - X @ b
    z = X * e[:, None]
    meat = z.T @ z
    L = min(lag, n - 1)
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        g = z[ell:].T @ z[:-ell]
        meat += w * (g + g.T)
    cov = inv @ meat @ inv
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    slope = float(b[1])
    return slope, slope / se if se > 0 else math.nan


def analyse(con: sqlite3.Connection, start: date, end: date, outdir: Path) -> None:
    lo = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    hi_dt = datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc) + timedelta(hours=8)
    hi = int(hi_dt.timestamp() * 1000)
    ph = ",".join("?" for _ in SYMBOLS)
    rows = con.execute(
        f"""
        SELECT symbol,boundary_ms,oi,buy_qty,sell_qty,open_trade_count,open_last_side,next10_last_price
        FROM qh_events
        WHERE symbol IN ({ph}) AND boundary_ms>=? AND boundary_ms<=?
        ORDER BY symbol,boundary_ms
        """,
        (*SYMBOLS, lo, hi),
    ).fetchall()
    if not rows:
        raise RuntimeError("No qh_events found; run backfill first")
    df = pd.DataFrame(rows, columns=["symbol","boundary_ms","oi","buy_qty","sell_qty","open_trade_count","open_last_side","p_start"])
    exits = df[["symbol", "boundary_ms", "p_start"]].rename(columns={"boundary_ms": "exit_boundary_ms", "p_start": "p_exit"})
    core_end = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000) - 1
    core = df[(df.boundary_ms >= lo) & (df.boundary_ms <= core_end)].copy()
    core["exit_boundary_ms"] = core.boundary_ms + HORIZON_MS
    core = core.merge(exits, on=["symbol", "exit_boundary_ms"], how="left")
    core["timestamp"] = pd.to_datetime(core.boundary_ms, unit="ms", utc=True)
    core["year"] = core.timestamp.dt.year
    valid = core.p_start.notna() & core.p_exit.notna() & (core.p_start > 0) & (core.p_exit > 0)
    core["fwd_bps"] = np.nan
    core.loc[valid, "fwd_bps"] = 10_000.0 * np.log(core.loc[valid, "p_exit"] / core.loc[valid, "p_start"])
    core["signed_bps"] = np.sign(core.oi) * core.fwd_bps
    core["linear_pnl_bps"] = core.oi * core.fwd_bps

    n_total = len(core)
    n_valid = int(valid.sum())
    coverage = n_valid / n_total if n_total else 0.0
    expected_events = ((end - start).days + 1) * 96 * len(SYMBOLS)
    opening_coverage = n_total / expected_events if expected_events else 0.0
    q = core[valid].copy()
    if q.empty:
        raise RuntimeError("No valid 8h forward returns")

    abs_oi = float(q.oi.abs().mean())
    linear_edge = float(q.linear_pnl_bps.mean() / abs_oi) if abs_oi > 0 else math.nan
    sign_mean, sign_t, _ = hac_mean_t(q, "signed_bps", lag=32)
    pooled_slope, pooled_t, pooled_n = panel_hac_beta_t(q, "fwd_bps", "oi", lag=32)

    asset_rows = []
    for s in SYMBOLS:
        z = q[q.symbol == s].sort_values("boundary_ms")
        slope, t = asset_slope(z, lag=32)
        sign_edge = float(z.signed_bps.mean()) if len(z) else math.nan
        asset_rows.append({
            "symbol": s,
            "n": len(z),
            "oi_mean": float(z.oi.mean()) if len(z) else math.nan,
            "oi_sd": float(z.oi.std(ddof=1)) if len(z) > 1 else math.nan,
            "slope_bps_per_oi": slope,
            "hac_t": t,
            "sign_edge_bps": sign_edge,
            "paper_2021_2024_8h_slope": PAPER_8H_BPS_PER_OI[s],
        })
    adf = pd.DataFrame(asset_rows)

    year_rows = []
    for y in sorted(q.year.unique()):
        z = q[q.year == y]
        b, t, n = panel_hac_beta_t(z, "fwd_bps", "oi", lag=32)
        sm, st, _ = hac_mean_t(z, "signed_bps", lag=32)
        aoi = float(z.oi.abs().mean())
        ledge = float(z.linear_pnl_bps.mean() / aoi) if aoi > 0 else math.nan
        year_rows.append({"year": int(y), "n": n, "slope_bps_per_oi": b, "hac_t": t, "sign_edge_bps": sm, "sign_t": st, "linear_edge_bps": ledge})
    ydf = pd.DataFrame(year_rows)

    slopes_pos = int((adf.slope_bps_per_oi > 0).sum())
    y25 = ydf[ydf.year == 2025]
    y26 = ydf[ydf.year == 2026]
    gates = {
        "Pooled 8h OI slope > 0": pooled_slope > 0,
        "Pooled HAC t > 2": pooled_t > 2,
        "2025 slope > 0": len(y25) == 1 and float(y25.slope_bps_per_oi.iloc[0]) > 0,
        "2026 slope > 0": len(y26) == 1 and float(y26.slope_bps_per_oi.iloc[0]) > 0,
        "At least 4/6 assets slope > 0": slopes_pos >= 4,
        f"Linear gross edge > {ROUND_TRIP_COST_BPS:.0f} bps taker RT": np.isfinite(linear_edge) and linear_edge > ROUND_TRIP_COST_BPS,
        "QH opening coverage >= 99%": opening_coverage >= 0.99,
        "Forward-price coverage >= 99%": coverage >= 0.99,
    }

    print("\n=== QH ORDER FLOW V0 RESULT ===")
    print(f"Sample: {start} -> {end} | symbols=6 | horizon=8h | QH first-10s OI")
    print(f"Events: {n_total:,}/{expected_events:,} expected | opening coverage={opening_coverage:.2%}")
    print(f"Valid 8h returns: {n_valid:,} | forward-price coverage={coverage:.2%}")
    print(f"Pooled OI slope: {pooled_slope:+.3f} bps/unit | HAC t={pooled_t:+.2f} | N={pooled_n:,}")
    print(f"Sign(OI) directional edge: {sign_mean:+.3f} bps/trade | HAC t={sign_t:+.2f}")
    print(f"OI-proportional normalized gross edge: {linear_edge:+.3f} bps per unit avg notional")
    print(f"Taker round-trip benchmark: {ROUND_TRIP_COST_BPS:.1f} bps")
    print(f"Positive asset slopes: {slopes_pos}/6")

    print("\nASSET BREAKDOWN")
    ap = adf.copy()
    for c in ["oi_mean", "oi_sd", "slope_bps_per_oi", "hac_t", "sign_edge_bps", "paper_2021_2024_8h_slope"]:
        ap[c] = ap[c].map(lambda x: f"{x:+.3f}" if np.isfinite(x) else "nan")
    print(ap.to_string(index=False))

    print("\nYEAR BREAKDOWN")
    yp = ydf.copy()
    for c in ["slope_bps_per_oi", "hac_t", "sign_edge_bps", "sign_t", "linear_edge_bps"]:
        yp[c] = yp[c].map(lambda x: f"{x:+.3f}" if np.isfinite(x) else "nan")
    print(yp.to_string(index=False))

    print("\nPRE-REGISTERED V0 GATES")
    for name, ok in gates.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(gates.values())
    print(f"Overall: {'PASS' if overall else 'FAIL'}")
    print("\nV0 VERDICT")
    if pooled_slope <= 0 or pooled_t <= 2 or slopes_pos < 4:
        print("[CLOSE] The published quarter-hour OI relation does not robustly persist in the 2025-2026 Binance OOS sample.")
    elif linear_edge <= ROUND_TRIP_COST_BPS:
        print("[STATISTICAL ONLY] OI predicts 8h returns, but the pre-registered gross edge does not clear realistic taker round-trip costs.")
    else:
        print("[PROMOTE TO EXECUTION TEST] OI persistence is statistically robust and clears the taker-cost hurdle before portfolio construction.")

    outdir.mkdir(parents=True, exist_ok=True)
    core.to_csv(outdir / "qh_events_with_8h_returns.csv", index=False)
    adf.to_csv(outdir / "asset_breakdown.csv", index=False)
    ydf.to_csv(outdir / "year_breakdown.csv", index=False)
    pd.DataFrame([{"gate": k, "pass": bool(v)} for k, v in gates.items()]).to_csv(outdir / "gates.csv", index=False)
    print(f"\nSaved under: {outdir}")


def main() -> int:
    cfg = parse_args()
    start = datetime.strptime(cfg.start, "%Y-%m-%d").date()
    end = datetime.strptime(cfg.end, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("--end must be >= --start")
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    con = ensure_db(Path(cfg.db))

    print("=== BINANCE QH ORDER FLOW V0 ===")
    print(f"OOS evaluation: {start} -> {end}")
    print("Assets: BTC, ETH, XRP, SOL, DOGE, ADA USDT perpetuals")
    print("Signal: signed base-volume OI in first 10 seconds of each 15-minute boundary")
    print("Target: exact aggTrade-derived log return from next 10s bin to same phase +8h")
    print("No ML, no threshold search, no hyperopt. One pre-registered persistence/cost test.")
    print(f"Workers: {cfg.workers}\n")

    if cfg.skip_backfill:
        print("Backfill skipped; reusing qh.sqlite.", flush=True)
    else:
        backfill(con, start, end, cfg.workers, cfg.timeout, outdir)
    analyse(con, start, end, outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
