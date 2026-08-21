#!/usr/bin/env python3
import argparse
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS_H = (4, 8, 12)
QUANTILES = (0.90, 0.95)


def parse_date(s):
    return datetime.strptime(s, "%Y-%m-%d").date()


def iter_days(start, end):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def load_cache(root, symbol, start, end):
    arrays = {k: [] for k in ("vol", "signed", "price")}
    missing = []
    for d in iter_days(start, end):
        p = root / "cache" / symbol / f"{d.isoformat()}.npz"
        if not p.exists():
            missing.append(str(d))
            continue
        z = np.load(p)
        arrays["vol"].append(z["vol"])
        arrays["signed"].append(z["signed"])
        arrays["price"].append(z["price"])
    if missing:
        raise SystemExit("CACHE_MISSING: " + ",".join(missing))
    return {k: np.concatenate(v) for k, v in arrays.items()}


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


def slope_xy(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if len(x) < 20:
        return np.nan
    xc = x - x.mean()
    den = float(np.dot(xc, xc))
    if den <= 0:
        return np.nan
    return float(np.dot(xc, y - y.mean()) / den)


def summarize(rows, horizon_h, q):
    vals = np.array([
        r["gross_bps"] for r in rows
        if r["horizon_h"] == horizon_h and r["quantile"] == q
    ], dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return {
            "horizon_h": horizon_h,
            "quantile": q,
            "n": 0,
            "gross_mean_bps": np.nan,
            "net_mean_bps_after_8bps": np.nan,
            "median_bps": np.nan,
            "winrate": np.nan,
            "nw_t": np.nan,
        }
    mu, t = nw_mean_t(vals, maxlag=horizon_h * 4)
    return {
        "horizon_h": horizon_h,
        "quantile": q,
        "n": int(len(vals)),
        "gross_mean_bps": float(mu),
        "net_mean_bps_after_8bps": float(mu - 8.0),
        "median_bps": float(np.median(vals)),
        "winrate": float(np.mean(vals > 0)),
        "nw_t": float(t) if np.isfinite(t) else np.nan,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-05-01")
    ap.add_argument("--end", default="2026-08-18")
    ap.add_argument("--symbol", default="BTCUSDT")
    ap.add_argument("--root", default="/freqtrade/user_data/qh_edge")
    ap.add_argument("--lookback-days", type=int, default=30)
    args = ap.parse_args()

    start, end = parse_date(args.start), parse_date(args.end)
    root = Path(args.root)
    bars = load_cache(root, args.symbol, start, end)

    vol = bars["vol"]
    signed = bars["signed"]
    price = limited_ffill(bars["price"], limit=6)
    oi = np.divide(signed, vol, out=np.full_like(signed, np.nan, dtype=float), where=vol > 0)

    qh = np.arange(0, len(price), 90, dtype=np.int64)
    start_dt = datetime(start.year, start.month, start.day, tzinfo=timezone.utc)
    event_dt = np.array([start_dt + timedelta(seconds=int(i * 10)) for i in qh], dtype=object)

    print("=== CAUSAL QUARTER-HOUR REGIME AUDIT ===")
    print(f"Symbol: {args.symbol}")
    print(f"Range: {start} .. {end}")
    print(f"Lookback: {args.lookback_days} calendar days")
    print("Each test day uses ONLY fully realized events from the preceding lookback window.")
    print("Direction = sign of trailing quarter-hour OI->return edge.")
    print("Trade filters = trailing 90th / 95th percentile of |OI|.")
    print("Fee stress = fixed 8 bps round trip. No funding/slippage included.")
    print()

    rows = []
    sign_rows = []

    for h in HORIZONS_H:
        step = h * 360
        valid_event = qh + 1 + step < len(price)
        idx = qh[valid_event]
        dts = event_dt[valid_event]

        x = oi[idx]
        entry = price[idx + 1]
        exitp = price[idx + 1 + step]
        y = np.log(exitp / entry) * 10000.0
        exits = np.array([dt + timedelta(hours=h, seconds=10) for dt in dts], dtype=object)

        test_start = start + timedelta(days=args.lookback_days)
        d = test_start
        while d <= end:
            day_start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
            next_day = day_start + timedelta(days=1)
            train_start = day_start - timedelta(days=args.lookback_days)

            train_mask = (
                (dts >= train_start)
                & (exits < day_start)
                & np.isfinite(x)
                & np.isfinite(y)
            )
            test_mask = (
                (dts >= day_start)
                & (dts < next_day)
                & np.isfinite(x)
                & np.isfinite(y)
            )

            tx, ty = x[train_mask], y[train_mask]
            if len(tx) < 1000 or not np.any(test_mask):
                d += timedelta(days=1)
                continue

            beta = slope_xy(tx, ty)
            edge = float(np.mean(np.sign(tx) * ty))
            if not np.isfinite(beta) or not np.isfinite(edge) or edge == 0:
                d += timedelta(days=1)
                continue

            direction = 1.0 if edge > 0 else -1.0
            sign_rows.append({
                "date": str(d),
                "horizon_h": h,
                "train_n": int(len(tx)),
                "train_beta_bps_per_oi": float(beta),
                "train_signed_edge_bps": float(edge),
                "direction": "CONTINUATION" if direction > 0 else "REVERSAL",
            })

            abs_train = np.abs(tx[np.isfinite(tx)])
            test_indices = np.where(test_mask)[0]

            for q in QUANTILES:
                thr = float(np.quantile(abs_train, q))
                sel = test_indices[np.abs(x[test_indices]) >= thr]
                for j in sel:
                    gross = direction * np.sign(x[j]) * y[j]
                    rows.append({
                        "date": str(d),
                        "horizon_h": h,
                        "quantile": q,
                        "threshold_abs_oi": thr,
                        "oi": float(x[j]),
                        "forward_bps": float(y[j]),
                        "direction": "CONTINUATION" if direction > 0 else "REVERSAL",
                        "gross_bps": float(gross),
                        "net_bps_after_8bps": float(gross - 8.0),
                    })

            d += timedelta(days=1)

    print("=== FORWARD RESULTS ===")
    summary = []
    for h in HORIZONS_H:
        for q in QUANTILES:
            s = summarize(rows, h, q)
            summary.append(s)
            wr = s["winrate"] * 100 if np.isfinite(s["winrate"]) else np.nan
            print(
                f"{h:>2}h top-{int(round((1-q)*100)):>2}% | "
                f"N={s['n']:>4} gross={s['gross_mean_bps']:+.2f} bps "
                f"net8={s['net_mean_bps_after_8bps']:+.2f} "
                f"WR={wr:5.1f}% median={s['median_bps']:+.2f} "
                f"NW-t={s['nw_t']:+.2f}"
            )

    print("\n=== MONTHLY FORWARD RESULTS (TOP-5%) ===")
    for h in HORIZONS_H:
        sub = [r for r in rows if r["horizon_h"] == h and r["quantile"] == 0.95]
        months = sorted(set(r["date"][:7] for r in sub))
        for m in months:
            vals = np.array([r["gross_bps"] for r in sub if r["date"].startswith(m)], dtype=float)
            if len(vals):
                print(
                    f"{m} {h:>2}h: N={len(vals):>3} "
                    f"gross={vals.mean():+.2f} net8={vals.mean()-8.0:+.2f} "
                    f"WR={np.mean(vals > 0)*100:5.1f}%"
                )

    sign_df = pd.DataFrame(sign_rows)
    if not sign_df.empty:
        print("\n=== CAUSAL DIRECTION DAYS ===")
        for h in HORIZONS_H:
            s = sign_df[sign_df.horizon_h == h]
            cont = int((s.direction == "CONTINUATION").sum())
            rev = int((s.direction == "REVERSAL").sum())
            print(f"{h:>2}h: continuation_days={cont}, reversal_days={rev}")

    report = root / "reports" / f"causal_regime_{start}_{end}"
    report.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(report / "forward_trades.csv", index=False)
    pd.DataFrame(sign_rows).to_csv(report / "daily_direction.csv", index=False)
    with open(report / "summary.json", "w") as f:
        json.dump({
            "symbol": args.symbol,
            "range": [str(start), str(end)],
            "lookback_days": args.lookback_days,
            "fee_stress_bps": 8.0,
            "summary": summary,
            "interpretation_gate": (
                "Interesting only if the result survives fee stress with positive net mean, "
                "winrate materially above 50%, and similar sign across 8h/12h and months. "
                "This is a causal diagnostic, not a portfolio backtest."
            ),
        }, f, indent=2)

    print(f"\nReports: {report}")
    print("No new market data was downloaded.")


if __name__ == "__main__":
    main()
