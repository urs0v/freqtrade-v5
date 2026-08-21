#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

BINS_PER_DAY = 24 * 60 * 6
HORIZONS_H = (4, 8, 12)


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def iter_days(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def cache_path(root, symbol, d):
    return root / "cache" / symbol / f"{d.isoformat()}.npz"


def load_cache(root, symbol, start, end):
    arrays = {k: [] for k in ("count", "vol", "signed", "price", "side")}
    missing = []
    for d in iter_days(start, end):
        cp = cache_path(root, symbol, d)
        if not cp.exists():
            missing.append(str(d))
            continue
        z = np.load(cp)
        for k in arrays:
            arrays[k].append(z[k])
    if missing:
        raise SystemExit("CACHE_MISSING: " + ",".join(missing))
    return {k: np.concatenate(v) for k, v in arrays.items()}


def limited_ffill(x, limit=6):
    return pd.Series(x).ffill(limit=limit).to_numpy(np.float64)


def next_pow2(n):
    return 1 << (int(n - 1).bit_length())


def scalar_hac_se(X, resid, xtx_inv, c, lag):
    a = xtx_inv @ c
    z = resid * (X @ a)
    n = len(z)
    lag = min(int(lag), n - 2)
    if lag < 1:
        var = float(np.dot(z, z))
    else:
        nfft = next_pow2(2 * n - 1)
        f = np.fft.rfft(z, n=nfft)
        ac = np.fft.irfft(f * np.conj(f), n=nfft)[:lag + 1]
        weights = 1.0 - np.arange(1, lag + 1, dtype=np.float64) / (lag + 1.0)
        var = float(ac[0] + 2.0 * np.dot(weights, ac[1:]))
    return math.sqrt(max(var, 0.0))


def fit_nested(oi, price, side, valid_scope, horizon_h, hac=True):
    n = len(price)
    step = horizon_h * 360
    i = np.arange(0, n - step - 1, dtype=np.int64)
    scope = valid_scope[i]

    p0 = price[i + 1]
    p1 = price[i + 1 + step]
    o = oi[i]
    good = scope & np.isfinite(p0) & np.isfinite(p1) & (p0 > 0) & (p1 > 0) & np.isfinite(o)
    i = i[good]
    o = o[good]
    y = np.log(price[i + 1 + step] / price[i + 1]) * 10000.0

    b1 = (i % 6 == 0).astype(np.float64)
    minute = (i // 6) % 60
    b5 = ((i % 6 == 0) & (minute % 5 == 0)).astype(np.float64)
    b15 = ((i % 6 == 0) & (minute % 15 == 0)).astype(np.float64)

    prev_side = np.where(i > 0, side[i - 1], 0)
    cur_side = side[i]
    eta_bs = ((prev_side == 1) & (cur_side == -1)).astype(np.float64)
    eta_sb = ((prev_side == -1) & (cur_side == 1)).astype(np.float64)

    X = np.column_stack([
        np.ones(len(i), dtype=np.float64),
        o,
        o * b1,
        o * b5,
        o * b15,
        eta_bs,
        eta_sb,
    ])

    xtx_inv = np.linalg.pinv(X.T @ X)
    beta = xtx_inv @ (X.T @ y)
    resid = y - X @ beta

    c_base = np.array([0, 1, 0, 0, 0, 0, 0], np.float64)
    c_1m = np.array([0, 1, 1, 0, 0, 0, 0], np.float64)
    c_5m = np.array([0, 1, 1, 1, 0, 0, 0], np.float64)
    c_15m = np.array([0, 1, 1, 1, 1, 0, 0], np.float64)

    out = {
        "n": int(len(y)),
        "base_bps": float(c_base @ beta),
        "cfe_1m_bps": float(c_1m @ beta),
        "cfe_5m_bps": float(c_5m @ beta),
        "cfe_15m_bps": float(c_15m @ beta),
    }

    if hac:
        se = scalar_hac_se(X, resid, xtx_inv, c_15m, step)
        out["cfe_15m_t_nw"] = float(out["cfe_15m_bps"] / se) if se > 0 else np.nan
    return out


def month_scope(start, n, year, month):
    base = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    ts = np.arange(n, dtype=np.int64) * 10
    m0 = datetime(year, month, 1, tzinfo=timezone.utc)
    if month == 12:
        m1 = datetime(year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        m1 = datetime(year, month + 1, 1, tzinfo=timezone.utc)
    lo = int((m0 - base).total_seconds())
    hi = int((m1 - base).total_seconds())
    return (ts >= lo) & (ts < hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-08-18")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--root", default="/freqtrade/user_data/qh_edge")
    args = ap.parse_args()

    start, end = parse_date(args.start), parse_date(args.end)
    bars = load_cache(Path(args.root), args.symbol, start, end)

    vol = bars["vol"].astype(np.float64)
    signed = bars["signed"].astype(np.float64)
    price = limited_ffill(bars["price"], 6)
    side = bars["side"].astype(np.int8)
    oi = np.divide(signed, vol, out=np.full(len(vol), np.nan), where=vol > 0)
    full_scope = np.ones(len(oi), dtype=bool)

    print("=== EXACT PAPER-SPEC BTC CACHE REPLICATION ===")
    print("Equation: full 10s nested OI regression (base + 1m + 5m + 15m) + reversal controls")
    print("Forward return: P[t+1+h] / P[t+1], matching paper eq. (3)")
    print("No downloads. Existing qh_edge cache only.")
    print()

    rows = []
    for h in HORIZONS_H:
        r = fit_nested(oi, price, side, full_scope, h, hac=True)
        rows.append({"scope": "FULL", "horizon_h": h, **r})
        print(
            f"FULL {h:>2}h: base={r['base_bps']:+.3f} "
            f"1m={r['cfe_1m_bps']:+.3f} 5m={r['cfe_5m_bps']:+.3f} "
            f"15m={r['cfe_15m_bps']:+.3f} bps/OI "
            f"NW-t={r['cfe_15m_t_nw']:+.2f} N={r['n']}"
        )

    print("\n=== MONTHLY CFE15m POINT ESTIMATES ===")
    for y, m in ((2026, 5), (2026, 6), (2026, 7), (2026, 8)):
        scope = month_scope(start, len(oi), y, m)
        label = f"{y}-{m:02d}"
        vals = []
        for h in HORIZONS_H:
            r = fit_nested(oi, price, side, scope, h, hac=False)
            rows.append({"scope": label, "horizon_h": h, **r})
            vals.append(f"{h}h={r['cfe_15m_bps']:+.3f}")
        print(label + ": " + " | ".join(vals))

    first10 = np.arange(0, len(oi), 6, dtype=np.int64)
    qh = np.arange(0, len(oi), 90, dtype=np.int64)
    non_qh = np.setdiff1d(first10, qh, assume_unique=True)
    burst = float(np.nanmean(bars["count"][qh]) / np.nanmean(bars["count"][non_qh]))
    print(f"\nQuarter-hour first-10s trade-count burst ratio: {burst:.3f}")

    outdir = Path(args.root) / "reports" / f"exact_{start}_{end}"
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / f"{args.symbol}_paper_spec.csv", index=False)

    full_8 = next(x for x in rows if x["scope"] == "FULL" and x["horizon_h"] == 8)
    full_12 = next(x for x in rows if x["scope"] == "FULL" and x["horizon_h"] == 12)
    same_sign = full_8["cfe_15m_bps"] > 0 and full_12["cfe_15m_bps"] > 0
    summary = {
        "symbol": args.symbol,
        "range": [str(start), str(end)],
        "quarter_hour_burst_ratio": burst,
        "paper_spec_positive_8h_12h": bool(same_sign),
        "full": [x for x in rows if x["scope"] == "FULL"],
        "note": "Positive medium-horizon CFE15m matches the paper's sign. Economic tradability still requires a separate cost-aware trading test."
    }
    with open(outdir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== RESULT ===")
    print(json.dumps(summary, indent=2))
    print(f"Reports: {outdir}")


if __name__ == "__main__":
    main()
