#!/usr/bin/env python3
import argparse
import csv
import json
import math
import tempfile
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BINS_PER_DAY = 24 * 60 * 6
HORIZONS_H = (4, 8, 12)
SYMBOLS_DEFAULT = ("BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT")


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def iter_days(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def cache_path(root, symbol, d):
    return root / "cache" / symbol / f"{d.isoformat()}.npz"


def empty_day():
    return {
        "count": np.zeros(BINS_PER_DAY, dtype=np.int32),
        "vol": np.zeros(BINS_PER_DAY, dtype=np.float64),
        "signed": np.zeros(BINS_PER_DAY, dtype=np.float64),
        "price": np.full(BINS_PER_DAY, np.nan, dtype=np.float64),
        "side": np.zeros(BINS_PER_DAY, dtype=np.int8),
    }


def is_numeric_token(x):
    try:
        float(x)
        return True
    except Exception:
        return False


def download(url, dest):
    req = urllib.request.Request(url, headers={"User-Agent": "rmv5-qh-edge-audit/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)


def process_zip(zip_path, d):
    out = empty_day()
    start_ms = int(datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp() * 1000)

    with zipfile.ZipFile(zip_path) as zf:
        members = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not members:
            raise RuntimeError(f"No CSV inside {zip_path}")
        member = members[0]

        with zf.open(member) as f:
            first = f.readline().decode("utf-8", "replace").strip()
        has_header = not is_numeric_token(first.split(",")[0])

        with zf.open(member) as f:
            reader = pd.read_csv(
                f,
                header=0 if has_header else None,
                usecols=list(range(7)),
                chunksize=750_000,
                low_memory=False,
            )
            for chunk in reader:
                if chunk.empty:
                    continue

                a = chunk.iloc[:, :7]
                price = pd.to_numeric(a.iloc[:, 1], errors="coerce").to_numpy(np.float64)
                qty = pd.to_numeric(a.iloc[:, 2], errors="coerce").to_numpy(np.float64)
                ts = pd.to_numeric(a.iloc[:, 5], errors="coerce").to_numpy(np.float64)
                maker_raw = a.iloc[:, 6].astype(str).str.lower().to_numpy()

                valid = np.isfinite(price) & np.isfinite(qty) & np.isfinite(ts)
                if not valid.any():
                    continue
                price, qty, ts, maker_raw = price[valid], qty[valid], ts[valid], maker_raw[valid]

                ts_ms = np.where(ts > 1e14, ts / 1000.0, ts).astype(np.int64)
                idx = ((ts_ms - start_ms) // 10_000).astype(np.int64)

                valid = (idx >= 0) & (idx < BINS_PER_DAY)
                if not valid.any():
                    continue
                idx, price, qty, maker_raw = idx[valid], price[valid], qty[valid], maker_raw[valid]

                maker = np.isin(maker_raw, ["true", "1", "t"])
                side = np.where(maker, -1, 1).astype(np.int8)

                out["count"] += np.bincount(idx, minlength=BINS_PER_DAY).astype(np.int32)
                out["vol"] += np.bincount(idx, weights=qty, minlength=BINS_PER_DAY)
                out["signed"] += np.bincount(idx, weights=qty * side, minlength=BINS_PER_DAY)

                rev = idx[::-1]
                uniq, rev_pos = np.unique(rev, return_index=True)
                orig_pos = len(idx) - 1 - rev_pos
                out["price"][uniq] = price[orig_pos]
                out["side"][uniq] = side[orig_pos]

    return out


def ensure_day(root, symbol, d):
    cp = cache_path(root, symbol, d)
    cp.parent.mkdir(parents=True, exist_ok=True)

    if cp.exists():
        z = np.load(cp)
        return {k: z[k] for k in ("count", "vol", "signed", "price", "side")}, "cache"

    url = (
        "https://data.binance.vision/data/futures/um/daily/aggTrades/"
        f"{symbol}/{symbol}-aggTrades-{d.isoformat()}.zip"
    )
    with tempfile.TemporaryDirectory(prefix="qh_edge_") as td:
        zp = Path(td) / f"{symbol}-{d}.zip"
        try:
            download(url, zp)
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return empty_day(), "missing"
            raise

        data = process_zip(zp, d)
        np.savez_compressed(cp, **data)
        return data, "download"


def limited_ffill(x, limit=6):
    return pd.Series(x).ffill(limit=limit).to_numpy(np.float64)


def nw_mean_t(x, maxlag):
    x = np.asarray(x, np.float64)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan

    mu = float(x.mean())
    u = x - mu
    lrv = float(np.dot(u, u) / n)
    L = min(int(maxlag), n - 2)

    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        gam = float(np.dot(u[lag:], u[:-lag]) / n)
        lrv += 2.0 * w * gam

    se = math.sqrt(max(lrv, 0.0) / n)
    return mu, (mu / se if se > 0 else np.nan)


def nw_ols(y, X, maxlag):
    y = np.asarray(y, np.float64)
    X = np.asarray(X, np.float64)
    good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    y, X = y[good], X[good]

    n, k = X.shape
    if n <= k + 5:
        return np.full(k, np.nan), np.full(k, np.nan), n

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    u = y - X @ beta
    xu = X * u[:, None]

    meat = xu.T @ xu
    L = min(int(maxlag), n - 2)
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        g = xu[lag:].T @ xu[:-lag]
        meat += w * (g + g.T)

    cov = xtx_inv @ meat @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    t = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return beta, t, n


def index_grid(n, shift_minutes=0):
    return np.arange(shift_minutes * 6, n, 90, dtype=np.int64)


def evaluate_symbol(symbol, bars):
    count = bars["count"]
    vol = bars["vol"]
    signed = bars["signed"]
    price = limited_ffill(bars["price"], 6)
    side = bars["side"]
    n = len(price)

    oi = np.divide(signed, vol, out=np.full(n, np.nan), where=vol > 0)

    first10 = np.arange(0, n, 6, dtype=np.int64)
    qh = index_grid(n, 0)
    non_qh = np.setdiff1d(first10, qh, assume_unique=True)
    burst_ratio = float(np.nanmean(count[qh]) / np.nanmean(count[non_qh]))

    regressions, extremes, placebos = [], [], []

    for h in HORIZONS_H:
        step = h * 360
        lag_qh = h * 4

        idx = qh[qh + step < n]
        y = np.log(price[idx + step] / price[idx]) * 10000.0

        prev_side = np.where(idx > 0, side[idx - 1], 0)
        cur_side = side[idx]
        eta_bs = ((prev_side == 1) & (cur_side == -1)).astype(float)
        eta_sb = ((prev_side == -1) & (cur_side == 1)).astype(float)

        X = np.column_stack([np.ones(len(idx)), oi[idx], eta_bs, eta_sb])
        beta, t, nobs = nw_ols(y, X, lag_qh)

        regressions.append({
            "symbol": symbol,
            "horizon_h": h,
            "n": nobs,
            "oi_beta_bps": float(beta[1]),
            "oi_t_nw": float(t[1]),
            "burst_ratio": burst_ratio,
        })

        for thr in (0.30, 0.50):
            mask = np.isfinite(y) & np.isfinite(oi[idx]) & (np.abs(oi[idx]) >= thr)
            tr = np.sign(oi[idx][mask]) * y[mask]
            mu, mt = nw_mean_t(tr, lag_qh)

            extremes.append({
                "symbol": symbol,
                "horizon_h": h,
                "threshold": thr,
                "n": int(len(tr)),
                "gross_mean_bps": float(mu) if np.isfinite(mu) else np.nan,
                "median_bps": float(np.nanmedian(tr)) if len(tr) else np.nan,
                "winrate": float(np.mean(tr > 0)) if len(tr) else np.nan,
                "mean_t_nw": float(mt) if np.isfinite(mt) else np.nan,
                "net_if_8bps": float(mu - 8.0) if np.isfinite(mu) else np.nan,
            })

        for shift in (2, 5, 7):
            pidx = index_grid(n, shift)
            pidx = pidx[pidx + step < n]
            py = np.log(price[pidx + step] / price[pidx]) * 10000.0

            pprev = np.where(pidx > 0, side[pidx - 1], 0)
            pcur = side[pidx]
            pX = np.column_stack([
                np.ones(len(pidx)),
                oi[pidx],
                ((pprev == 1) & (pcur == -1)).astype(float),
                ((pprev == -1) & (pcur == 1)).astype(float),
            ])

            pb, pt, pn = nw_ols(py, pX, lag_qh)
            placebos.append({
                "symbol": symbol,
                "horizon_h": h,
                "shift_min": shift,
                "n": pn,
                "oi_beta_bps": float(pb[1]),
                "oi_t_nw": float(pt[1]),
            })

    return regressions, extremes, placebos


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    keys = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-08-18")
    ap.add_argument("--symbols", nargs="+", default=list(SYMBOLS_DEFAULT))
    ap.add_argument("--root", default="/freqtrade/user_data/qh_edge")
    args = ap.parse_args()

    start, end = parse_date(args.start), parse_date(args.end)
    if end < start:
        raise SystemExit("--end must be >= --start")

    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    all_reg, all_ext, all_pl = [], [], []
    missing = []

    print("=== QUARTER-HOUR EDGE AUDIT ===", flush=True)
    print(f"Range: {start} .. {end}", flush=True)
    print(f"Symbols: {', '.join(args.symbols)}", flush=True)
    print("Daily aggTrades ZIPs are reduced to cached 10s bars and deleted.", flush=True)

    days = list(iter_days(start, end))

    for symbol in args.symbols:
        arrays = {k: [] for k in ("count", "vol", "signed", "price", "side")}
        print(f"\n--- {symbol} ---", flush=True)

        for i, d in enumerate(days, 1):
            dat, src = ensure_day(root, symbol, d)
            if src == "missing":
                missing.append(f"{symbol}:{d}")

            for k in arrays:
                arrays[k].append(dat[k])

            if i == 1 or i % 10 == 0 or i == len(days):
                print(f"{symbol}: {i}/{len(days)} days [{src}]", flush=True)

        bars = {k: np.concatenate(v) for k, v in arrays.items()}
        reg, ext, pl = evaluate_symbol(symbol, bars)
        all_reg.extend(reg)
        all_ext.extend(ext)
        all_pl.extend(pl)

        for r in reg:
            print(
                f"{symbol} {r['horizon_h']:>2}h: "
                f"beta={r['oi_beta_bps']:+.3f} bps/OI, "
                f"t={r['oi_t_nw']:+.2f}, "
                f"burst={r['burst_ratio']:.3f}"
            )

    report = root / "reports" / f"{start}_{end}"
    write_csv(report / "qh_regression.csv", all_reg)
    write_csv(report / "extreme_flow.csv", all_ext)
    write_csv(report / "placebo.csv", all_pl)

    med_beta = {}
    positive_assets = {}

    for h in (8, 12):
        vals = [
            r["oi_beta_bps"] for r in all_reg
            if r["horizon_h"] == h and np.isfinite(r["oi_beta_bps"])
        ]
        med_beta[h] = float(np.median(vals)) if vals else np.nan
        positive_assets[h] = int(sum(v > 0 for v in vals))

    replicated = (
        positive_assets.get(8, 0) >= 4
        and positive_assets.get(12, 0) >= 4
        and med_beta.get(8, -np.inf) > 0
        and med_beta.get(12, -np.inf) > 0
    )

    tradable_cells = [
        r for r in all_ext
        if r["threshold"] == 0.30
        and r["horizon_h"] in (8, 12)
        and r["n"] >= 100
        and np.isfinite(r["gross_mean_bps"])
    ]
    strong_cells = [
        r for r in tradable_cells
        if r["gross_mean_bps"] >= 10.0 and r["winrate"] >= 0.55
    ]
    trading_candidate = len(strong_cells) >= max(2, len(args.symbols) // 2)

    summary = {
        "range": [str(start), str(end)],
        "symbols": args.symbols,
        "missing_days": missing,
        "paper_edge_replicated": bool(replicated),
        "trading_candidate": bool(trading_candidate),
        "positive_assets_8h": positive_assets.get(8, 0),
        "positive_assets_12h": positive_assets.get(12, 0),
        "median_beta_8h_bps_per_oi": med_beta.get(8),
        "median_beta_12h_bps_per_oi": med_beta.get(12),
        "strong_extreme_cells": len(strong_cells),
        "gate_note": (
            "PAPER_EDGE_REPLICATED tests sign consistency of the published effect. "
            "TRADING_CANDIDATE additionally requires >=10 bps gross and >=55% winrate "
            "for |OI|>=0.30 in enough 8h/12h asset cells. The two gates are intentionally separate."
        ),
    }

    with open(report / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(summary, indent=2))
    print(f"\nReports: {report}")
    print("Do NOT add leverage or optimize thresholds before reading these files.")


if __name__ == "__main__":
    raise SystemExit(main())
