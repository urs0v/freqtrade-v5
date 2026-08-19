#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import audit_flow_funding_cashflows as c
import audit_flow_funding_portfolio as p
import research_derivatives_alpha as r

FACTOR = "taker_minus_funding"
HORIZON = "12h"
BARS = r.HORIZONS[HORIZON]
Q = 0.25
COST_BPS = 8.0

YEAR_RANGES = {
    "2022": ("2022-01-01", "2023-01-01"),
    "2023": ("2023-01-01", "2024-01-01"),
    "2024": ("2024-01-01", "2025-01-01"),
    "2025": ("2025-01-01", "2026-01-01"),
    "2026YTD": ("2026-01-01", "2026-08-19"),
}
PRE2026_YEARS = ["2022", "2023", "2024", "2025"]

# This is a robustness/diagnostic audit, not a fresh OOS gate: side asymmetry has
# already been observed in aggregate 2025/2026 diagnostics. Criteria are fixed
# before looking at the year-by-year side results produced by this script.
MIN_POSITIVE_YEARS = 3
MIN_POOLED_SHARPE = 0.50
MAX_WORST_YEAR_DD_ABS_PCT = 25.0


def side_stats(all_rows: pd.DataFrame, funded: pd.DataFrame, n_pairs: int, side_name: str) -> dict:
    sign = 1.0 if side_name == "long" else -1.0
    z = funded[funded["side"] == sign].copy()
    if not z.empty:
        z["net_ret"] = z["economic_ret"]
    st = p.portfolio_stats(all_rows, z, n_pairs)
    if not st:
        return {
            "trades": 0,
            "avg_net_bps": np.nan,
            "return_pct": 0.0,
            "sharpe": 0.0,
            "dd_pct": 0.0,
            "funding_bps": np.nan,
        }
    funding_bps = float(z["funding_ret"].mean() * 10000.0) if len(z) else np.nan
    return {
        "trades": int(st["trades"]),
        "avg_net_bps": float(st["avg_trade_net_bps"]),
        "return_pct": float(st["total_return_pct"]),
        "sharpe": float(st["sharpe"]),
        "dd_pct": float(st["max_drawdown_pct"]),
        "funding_bps": funding_bps,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Long/short stability audit for taker-minus-funding including actual funding cashflows")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite")
    ap.add_argument("--funding-cache", default="/freqtrade/user_data/v5/free-cache")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/side_stability")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist in config")

    db = Path(args.db)
    if not db.exists():
        raise RuntimeError(f"Missing derivatives DB: {db}")

    chunks = {k: [] for k in YEAR_RANGES}
    events_by_pair: dict[str, pd.DataFrame] = {}

    print("=== FLOW-FUNDING SIDE STABILITY AUDIT ===")
    print("Canonical q25 / 12h / 8bps / 1x + actual archived funding cashflows.")
    print("Threshold/orientation trained on 2022-2024 only. 2026YTD remains diagnostic.")
    print("This is robustness diagnostics, not a fresh OOS gate, because side asymmetry was already observed in aggregate results.")

    for i, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        price = r.load_price(config, Path(args.datadir), pair)
        if price.empty:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: NO PRICE", flush=True)
            continue
        price["date"] = r.as_ns(price["date"])
        start_ms = int((price["date"].min() - pd.Timedelta("1D")).timestamp() * 1000)
        end_ms = int((price["date"].max() + pd.Timedelta("1D")).timestamp() * 1000)
        deriv = r.load_derivatives(db, r.pair_to_symbol(pair), start_ms, end_ms)
        if not deriv.empty:
            deriv = deriv.copy()
            deriv["available_time"] = pd.to_datetime(deriv["available_time"], utc=True).astype("datetime64[ns, UTC]")
        feat, _ = r.build_features(price, deriv)
        for label, (a, b) in YEAR_RANGES.items():
            x = r.slice_horizon(feat, a, b, HORIZON, BARS)
            if not x.empty:
                x["pair"] = pair
                chunks[label].append(x)
        events_by_pair[pair] = c.load_funding_events(Path(args.funding_cache), r.pair_to_symbol(pair))
        print(f"  [{i:02d}/{len(pairs)}] {pair}: ok [{time.monotonic()-t0:.1f}s]", flush=True)

    data = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in chunks.items()}
    train = pd.concat([data[y] for y in ("2022", "2023", "2024") if not data[y].empty], ignore_index=True)
    if train.empty:
        raise RuntimeError("No 2022-2024 training rows")
    orient, qlo, qhi = p.thresholds_from_train(train, Q)

    funded_by_year: dict[str, pd.DataFrame] = {}
    rows = []
    print("\n=== YEAR-BY-YEAR SIDE RESULTS ===")
    for year, all_rows in data.items():
        base = p.select_trades(all_rows, orient, qlo, qhi, 0.0)
        funded = c.add_funding_cashflows(base, events_by_pair)
        funded_by_year[year] = funded
        line = [year]
        for side in ("long", "short"):
            st = side_stats(all_rows, funded, len(pairs), side)
            rows.append({"year": year, "side": side, **st})
            line.append(
                f"{side}: ret={st['return_pct']:+6.2f}% avg={st['avg_net_bps']:+6.2f}bps "
                f"Sh={st['sharpe']:+.2f} DD={st['dd_pct']:+6.2f}% fund={st['funding_bps']:+.2f}bps"
            )
        print(" | ".join(line), flush=True)

    print("\n=== POOLED 2022-2025 SIDE RESULTS ===")
    pooled_rows = pd.concat([data[y] for y in PRE2026_YEARS if not data[y].empty], ignore_index=True)
    pooled_trades = pd.concat([funded_by_year[y] for y in PRE2026_YEARS if not funded_by_year[y].empty], ignore_index=True)
    pooled_summary = []
    for side in ("long", "short"):
        st = side_stats(pooled_rows, pooled_trades, len(pairs), side)
        pre = pd.DataFrame(rows)
        z = pre[(pre["side"] == side) & (pre["year"].isin(PRE2026_YEARS))]
        pos_ret_years = int((z["return_pct"] > 0).sum())
        pos_bps_years = int((z["avg_net_bps"] > 0).sum())
        worst_dd_abs = float((-z["dd_pct"]).max()) if len(z) else np.nan
        stable = bool(
            pos_ret_years >= MIN_POSITIVE_YEARS
            and pos_bps_years >= MIN_POSITIVE_YEARS
            and st["sharpe"] >= MIN_POOLED_SHARPE
            and worst_dd_abs <= MAX_WORST_YEAR_DD_ABS_PCT
        )
        pooled_summary.append({
            "side": side,
            **st,
            "positive_return_years": pos_ret_years,
            "positive_bps_years": pos_bps_years,
            "worst_year_dd_abs_pct": worst_dd_abs,
            "stable_candidate": stable,
        })
        print(
            f"{side:<5}: ret={st['return_pct']:+7.2f}% avg={st['avg_net_bps']:+6.2f}bps "
            f"Sharpe={st['sharpe']:+.2f} DD={st['dd_pct']:+6.2f}% | "
            f"positive years ret={pos_ret_years}/4 bps={pos_bps_years}/4 worstDD={worst_dd_abs:.2f}% "
            f"=> {'STABLE-CANDIDATE' if stable else 'NOT-STABLE'}",
            flush=True,
        )

    print("\nCriteria: >=3/4 positive pre-2026 years by return and avg net bps, pooled Sharpe>=0.50, worst yearly DD<=25%.")
    print("2026YTD is diagnostic only and never enters the stability criteria.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "side_yearly.csv", index=False)
    pd.DataFrame(pooled_summary).to_csv(outdir / "side_pooled_2022_2025.csv", index=False)
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
