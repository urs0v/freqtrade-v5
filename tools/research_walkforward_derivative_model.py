#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import audit_flow_funding_cashflows as cf
import audit_flow_funding_portfolio as p
import research_derivatives_alpha as r

HORIZON = "12h"
BARS = r.HORIZONS[HORIZON]
COST_BPS = 8.0
RIDGE_LAMBDA = 10.0
TRAIN_DAYS = 365
START_TEST = pd.Timestamp("2023-01-01", tz="UTC")
END_TEST = pd.Timestamp("2026-08-19", tz="UTC")
QS = (0.20, 0.25, 0.30)

# Focused state-conditioned extension of the only derivative signal family that
# showed prior robustness. Fixed before this walk-forward run.
FEATURES = [
    "taker_minus_funding",
    "ff_x_trend",
    "ff_x_funding",
    "ff_x_positioning",
    "ff_x_vol",
    "price_4h",
    "funding_z",
]

# Strong gate because this is the stage before any leverage study.
GATE_MIN_SHARPE = 1.00
GATE_MAX_DD_PCT = 15.0
GATE_MIN_POSITIVE_FULL_YEARS = 3  # 2023, 2024, 2025
GATE_REQUIRE_Q20_Q30_POSITIVE = True


def month_starts(a: pd.Timestamp, b: pd.Timestamp):
    cur = pd.Timestamp(a.year, a.month, 1, tz="UTC")
    while cur < b:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    x = df.copy()
    ff = pd.to_numeric(x["taker_minus_funding"], errors="coerce")
    trend = pd.to_numeric(x["price_4h"], errors="coerce")
    fz = pd.to_numeric(x["funding_z"], errors="coerce")
    pos = pd.to_numeric(x["top_ratio_z"], errors="coerce")
    ret15 = pd.to_numeric(x["close"], errors="coerce").pct_change()
    rv = ret15.rolling(96, min_periods=48).std()
    rv_z = r.robust_z(rv, 672, 96).clip(-5, 5)

    x["ff_x_trend"] = (ff * trend).clip(-12, 12)
    x["ff_x_funding"] = (ff * fz).clip(-20, 20)
    x["ff_x_positioning"] = (ff * pos).clip(-20, 20)
    x["ff_x_vol"] = (ff * rv_z).clip(-20, 20)
    return x


def load_all(config: dict, datadir: Path, db: Path) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[str]]:
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    rows = []
    events_by_pair: dict[str, pd.DataFrame] = {}

    for i, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        price = r.load_price(config, datadir, pair)
        if price.empty:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: NO PRICE", flush=True)
            continue
        price["date"] = r.as_ns(price["date"])
        start_ms = int((price["date"].min() - pd.Timedelta("2D")).timestamp() * 1000)
        end_ms = int((price["date"].max() + pd.Timedelta("2D")).timestamp() * 1000)
        deriv = r.load_derivatives(db, r.pair_to_symbol(pair), start_ms, end_ms)
        if not deriv.empty:
            deriv = deriv.copy()
            deriv["available_time"] = pd.to_datetime(deriv["available_time"], utc=True).astype("datetime64[ns, UTC]")
        feat, _ = r.build_features(price, deriv)
        feat = make_features(feat)
        feat["pair"] = pair
        feat["row_id"] = np.arange(len(feat), dtype=np.int64)
        feat["eligible"] = (feat["row_id"] % BARS) == 0
        rows.append(feat)
        events_by_pair[pair] = cf.load_funding_events(Path(args.funding_cache), r.pair_to_symbol(pair))
        print(f"  [{i:02d}/{len(pairs)}] {pair}: ok [{time.monotonic()-t0:.1f}s]", flush=True)

    if not rows:
        raise RuntimeError("No usable pairs")
    return pd.concat(rows, ignore_index=True), events_by_pair, pairs


def fit_ridge(train: pd.DataFrame) -> dict:
    target = f"y_{HORIZON}"
    z = train.dropna(subset=[target, *FEATURES]).copy()
    if len(z) < 1000:
        raise RuntimeError(f"Insufficient training rows: {len(z)}")

    mu = z[FEATURES].mean()
    sd = z[FEATURES].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    X = ((z[FEATURES] - mu) / sd).to_numpy(dtype=float)
    y = z[target].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(X)), X])
    pen = np.eye(A.shape[1]) * RIDGE_LAMBDA
    pen[0, 0] = 0.0
    beta = np.linalg.solve(A.T @ A + pen, A.T @ y)
    pred = A @ beta
    center = float(np.median(pred))
    return {"mu": mu, "sd": sd, "beta": beta, "center": center, "n": len(z)}


def score(df: pd.DataFrame, model: dict) -> pd.DataFrame:
    x = df.dropna(subset=FEATURES).copy()
    if x.empty:
        x["wf_score"] = np.nan
        return x
    X = ((x[FEATURES] - model["mu"]) / model["sd"]).to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(X)), X])
    x["wf_score"] = A @ model["beta"] - model["center"]
    return x


def thresholds(train_scored: pd.DataFrame, q: float) -> tuple[float, float]:
    s = train_scored["wf_score"].dropna().to_numpy(dtype=float)
    return tuple(float(v) for v in np.quantile(s, [q, 1.0 - q]))


def select_trades(df: pd.DataFrame, qlo: float, qhi: float) -> pd.DataFrame:
    target = f"y_{HORIZON}"
    x = df[["signal_time", "pair", "wf_score", target]].dropna().copy()
    if x.empty:
        return pd.DataFrame(columns=["signal_time", "pair", "side", "gross_ret"])
    s = x["wf_score"].to_numpy(dtype=float)
    side = np.where(s >= qhi, 1.0, np.where(s <= qlo, -1.0, 0.0))
    chosen = side != 0
    x = x.loc[chosen].copy()
    side = side[chosen]
    x["side"] = side
    x["gross_ret"] = side * x[target].to_numpy(dtype=float)
    return x[["signal_time", "pair", "side", "gross_ret"]]


def portfolio_stats(all_rows: pd.DataFrame, trades: pd.DataFrame, n_pairs: int) -> dict:
    z = trades.copy()
    z["net_ret"] = z["economic_ret"]
    return p.portfolio_stats(all_rows, z, n_pairs)


def year_return(trades: pd.DataFrame, all_rows: pd.DataFrame, n_pairs: int, year: int) -> float:
    a = pd.Timestamp(f"{year}-01-01", tz="UTC")
    b = pd.Timestamp(f"{year+1}-01-01", tz="UTC")
    ar = all_rows[(all_rows["signal_time"] >= a) & (all_rows["signal_time"] < b)]
    tr = trades[(trades["signal_time"] >= a) & (trades["signal_time"] < b)]
    if ar.empty:
        return np.nan
    return float(portfolio_stats(ar, tr, n_pairs)["total_return_pct"])


def run_q(all_df: pd.DataFrame, events_by_pair: dict[str, pd.DataFrame], pairs: list[str], q: float) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    fold_rows = []
    all_test_rows = []
    all_trades = []

    for fold, test_start in enumerate(month_starts(START_TEST, END_TEST), 1):
        test_end = min(test_start + pd.offsets.MonthBegin(1), END_TEST)
        train_start = test_start - pd.Timedelta(days=TRAIN_DAYS)

        train = all_df[
            (all_df["eligible"])
            & (all_df["signal_time"] >= train_start)
            & (all_df[f"exit_time_{HORIZON}"] < test_start)
        ].copy()
        test = all_df[
            (all_df["eligible"])
            & (all_df["signal_time"] >= test_start)
            & (all_df["signal_time"] < test_end)
            & (all_df[f"exit_time_{HORIZON}"] < test_end)
        ].copy()
        if train.empty or test.empty:
            continue

        model = fit_ridge(train)
        train_scored = score(train, model)
        test_scored = score(test, model)
        qlo, qhi = thresholds(train_scored, q)
        trades = select_trades(test_scored, qlo, qhi)
        funded = cf.add_funding_cashflows(trades, events_by_pair)

        all_test_rows.append(test_scored)
        all_trades.append(funded)
        fold_rows.append({
            "fold": fold,
            "month": test_start.strftime("%Y-%m"),
            "train_n": model["n"],
            "test_rows": len(test_scored),
            "trades": len(funded),
            "qlo": qlo,
            "qhi": qhi,
            "avg_trade_bps": float(funded["economic_ret"].mean() * 10000.0) if len(funded) else np.nan,
        })

    return (
        pd.concat(all_test_rows, ignore_index=True) if all_test_rows else pd.DataFrame(),
        pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame(),
        fold_rows,
    )


def main() -> int:
    global args
    ap = argparse.ArgumentParser(description="Monthly walk-forward state-conditioned derivatives model")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite")
    ap.add_argument("--funding-cache", default="/freqtrade/user_data/v5/free-cache")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/walkforward_derivatives")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    print("=== WALK-FORWARD STATE-CONDITIONED DERIVATIVES MODEL ===")
    print("Trailing 365d fit -> next month; monthly refit; 12h non-overlapping; actual funding; 8bps.")
    print("Fixed ridge lambda=10. No leverage. Historical periods have been inspected before, so this is walk-forward robustness, not a pristine holdout.")

    all_df, events_by_pair, pairs = load_all(config, Path(args.datadir), Path(args.db))

    results = []
    exports = []
    fold_exports = []
    print("\n=== WALK-FORWARD RESULTS ===")
    for q in QS:
        test_rows, trades, folds = run_q(all_df, events_by_pair, pairs, q)
        if test_rows.empty:
            continue
        stats = portfolio_stats(test_rows, trades, len(pairs))
        years = {y: year_return(trades, test_rows, len(pairs), y) for y in (2023, 2024, 2025, 2026)}
        pos_full = sum(np.isfinite(years[y]) and years[y] > 0 for y in (2023, 2024, 2025))
        results.append({"q": q, **stats, **{f"ret_{y}_pct": years[y] for y in years}, "positive_full_years": pos_full})
        tx = trades.copy(); tx["q"] = q; exports.append(tx)
        ff = pd.DataFrame(folds); ff["q"] = q; fold_exports.append(ff)
        print(
            f"q{int(q*100):02d}: ret={stats['total_return_pct']:+7.2f}% Sharpe={stats['sharpe']:+.2f} "
            f"DD={stats['max_drawdown_pct']:+.2f}% avg={stats['avg_trade_net_bps']:+.2f}bps | "
            f"2023={years[2023]:+.2f}% 2024={years[2024]:+.2f}% 2025={years[2025]:+.2f}% 2026YTD={years[2026]:+.2f}%",
            flush=True,
        )

    out = pd.DataFrame(results)
    can = out[np.isclose(out["q"], 0.25)].iloc[0]
    q20 = out[np.isclose(out["q"], 0.20)].iloc[0]
    q30 = out[np.isclose(out["q"], 0.30)].iloc[0]
    gate = bool(
        can["total_return_pct"] > 0
        and can["sharpe"] >= GATE_MIN_SHARPE
        and can["max_drawdown_pct"] >= -GATE_MAX_DD_PCT
        and int(can["positive_full_years"]) >= GATE_MIN_POSITIVE_FULL_YEARS
        and (not GATE_REQUIRE_Q20_Q30_POSITIVE or (q20["total_return_pct"] > 0 and q30["total_return_pct"] > 0))
    )

    print("\n=== GATE SUMMARY ===")
    print("PASS requires q25: total return>0, Sharpe>=1.00, maxDD<=15%, all 2023-2025 positive, q20/q30 total returns>0.")
    print("GATE:", "PASS" if gate else "FAIL")
    print("Only after PASS do we test risk sizing / leverage.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "walkforward_summary.csv", index=False)
    if exports:
        pd.concat(exports, ignore_index=True).to_csv(outdir / "walkforward_trades.csv", index=False)
    if fold_exports:
        pd.concat(fold_exports, ignore_index=True).to_csv(outdir / "walkforward_folds.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps({
        "pass": gate,
        "features": FEATURES,
        "ridge_lambda": RIDGE_LAMBDA,
        "train_days": TRAIN_DAYS,
        "cost_bps": COST_BPS,
        "horizon": HORIZON,
        "criteria": {
            "q25_sharpe_gte": GATE_MIN_SHARPE,
            "q25_max_dd_abs_pct_lte": GATE_MAX_DD_PCT,
            "q25_positive_full_years_gte": GATE_MIN_POSITIVE_FULL_YEARS,
            "q20_q30_total_return_positive": GATE_REQUIRE_Q20_Q30_POSITIVE,
        },
    }, indent=2))
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
