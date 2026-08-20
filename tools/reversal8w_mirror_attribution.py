#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attribution audit for the exact 8-week momentum mirror")
    p.add_argument("--weekly", default="/freqtrade/user_data/reversal8w_perp/weekly_results.csv")
    p.add_argument("--assets", default="/freqtrade/user_data/reversal8w_perp/asset_contributions.csv")
    p.add_argument("--output-dir", default="/freqtrade/user_data/reversal8w_mirror_attribution")
    return p.parse_args()


def fmt_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100.0*x:+.2f}%"


def main() -> int:
    cfg = parse_args()
    wp = Path(cfg.weekly)
    ap = Path(cfg.assets)
    if not wp.exists() or not ap.exists():
        raise RuntimeError("Missing reversal source CSVs. Run run_reversal8w_perp_test.sh first.")

    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    w = pd.read_csv(wp, parse_dates=["date"])
    a = pd.read_csv(ap, parse_dates=["date"])
    required_w = {"strategy", "date", "gross_return", "funding_return", "cost_return", "net_return"}
    required_a = {
        "strategy", "date", "symbol", "weight", "formation_return_8w", "fwd_return",
        "gross_contribution", "funding_contribution"
    }
    if not required_w.issubset(w.columns):
        raise RuntimeError(f"weekly_results.csv missing columns: {sorted(required_w - set(w.columns))}")
    if not required_a.issubset(a.columns):
        raise RuntimeError(f"asset_contributions.csv missing columns: {sorted(required_a - set(a.columns))}")

    # Exact mirror. Signal/universe/weights are unchanged except sign.
    w = w.copy()
    w["mirror_price"] = -w["gross_return"].astype(float)
    w["mirror_funding"] = -w["funding_return"].astype(float)
    w["mirror_cost"] = w["cost_return"].astype(float)
    w["mirror_net"] = w["mirror_price"] + w["mirror_funding"] - w["mirror_cost"]
    w["year"] = w["date"].dt.year

    a = a.copy()
    a["mirror_weight"] = -a["weight"].astype(float)
    a["mirror_price_contribution"] = -a["gross_contribution"].astype(float)
    a["mirror_funding_contribution"] = -a["funding_contribution"].astype(float)
    a["mirror_pre_cost_contribution"] = a["mirror_price_contribution"] + a["mirror_funding_contribution"]
    # Original reversal shorts prior winners and longs prior losers.
    # Therefore exact mirror LONG_WINNERS are original negative weights.
    a["mirror_leg"] = np.where(a["weight"].astype(float) < 0, "LONG_WINNERS", "SHORT_LOSERS")
    a["year"] = a["date"].dt.year

    rows = []
    years = []
    legs = []
    for strat, g in w.groupby("strategy", sort=True):
        g = g.sort_values("date")
        mp = float(g.mirror_price.mean())
        mf = float(g.mirror_funding.mean())
        mc = float(g.mirror_cost.mean())
        mn = float(g.mirror_net.mean())
        pre = mp + mf
        rows.append({
            "strategy": strat,
            "weeks": len(g),
            "avg_price": mp,
            "avg_funding": mf,
            "avg_cost": mc,
            "avg_net": mn,
            "annualized_arith_price": 52.0 * mp,
            "annualized_arith_funding": 52.0 * mf,
            "annualized_arith_cost": 52.0 * mc,
            "annualized_arith_net": 52.0 * mn,
            "funding_share_of_positive_pre_cost": (mf / pre) if pre > 0 else np.nan,
            "price_share_of_positive_pre_cost": (mp / pre) if pre > 0 else np.nan,
        })
        for year, yy in g.groupby("year", sort=True):
            years.append({
                "strategy": strat,
                "year": int(year),
                "weeks": len(yy),
                "avg_price": float(yy.mirror_price.mean()),
                "avg_funding": float(yy.mirror_funding.mean()),
                "avg_cost": float(yy.mirror_cost.mean()),
                "avg_net": float(yy.mirror_net.mean()),
                "sum_price": float(yy.mirror_price.sum()),
                "sum_funding": float(yy.mirror_funding.sum()),
                "sum_cost": float(yy.mirror_cost.sum()),
                "sum_net_arith": float(yy.mirror_net.sum()),
            })

    for (strat, year, leg), g in a.groupby(["strategy", "year", "mirror_leg"], sort=True):
        dates = max(int(g.date.nunique()), 1)
        legs.append({
            "strategy": strat,
            "year": int(year),
            "leg": leg,
            "weeks": dates,
            "avg_price_per_week": float(g.mirror_price_contribution.sum() / dates),
            "avg_funding_per_week": float(g.mirror_funding_contribution.sum() / dates),
            "avg_pre_cost_per_week": float(g.mirror_pre_cost_contribution.sum() / dates),
            "total_price": float(g.mirror_price_contribution.sum()),
            "total_funding": float(g.mirror_funding_contribution.sum()),
            "total_pre_cost": float(g.mirror_pre_cost_contribution.sum()),
        })

    summary = pd.DataFrame(rows)
    year_df = pd.DataFrame(years)
    leg_df = pd.DataFrame(legs)

    print("=== 8-WEEK MIRROR ATTRIBUTION AUDIT ===")
    print("No new signal, filter, date choice, or optimization.")
    print("Decomposes the already-saved exact mirror into price, funding, cost, and long/short legs.\n")

    sp = summary.copy()
    for c in ["avg_price", "avg_funding", "avg_cost", "avg_net", "annualized_arith_price", "annualized_arith_funding", "annualized_arith_cost", "annualized_arith_net"]:
        sp[c] = sp[c].map(fmt_pct)
    for c in ["funding_share_of_positive_pre_cost", "price_share_of_positive_pre_cost"]:
        sp[c] = sp[c].map(lambda x: "nan" if not np.isfinite(x) else f"{100*x:.1f}%")
    print("OVERALL ATTRIBUTION")
    print(sp.to_string(index=False))

    print("\nYEAR ATTRIBUTION")
    yp = year_df.copy()
    for c in ["avg_price", "avg_funding", "avg_cost", "avg_net", "sum_price", "sum_funding", "sum_cost", "sum_net_arith"]:
        yp[c] = yp[c].map(fmt_pct)
    print(yp.to_string(index=False))

    print("\nLEG ATTRIBUTION")
    lp = leg_df.copy()
    for c in ["avg_price_per_week", "avg_funding_per_week", "avg_pre_cost_per_week", "total_price", "total_funding", "total_pre_cost"]:
        lp[c] = lp[c].map(fmt_pct)
    print(lp.to_string(index=False))

    hv = summary[summary.strategy == "HIGH_VOL_REVERSAL"]
    if not hv.empty:
        r = hv.iloc[0]
        print("\nHIGH_VOL DIAGNOSTIC")
        print(f"Average weekly price contribution:   {fmt_pct(float(r.avg_price))}")
        print(f"Average weekly funding contribution: {fmt_pct(float(r.avg_funding))}")
        print(f"Average weekly turnover cost:        {fmt_pct(float(r.avg_cost))}")
        print(f"Average weekly exact-mirror net:     {fmt_pct(float(r.avg_net))}")
        pre = float(r.avg_price + r.avg_funding)
        if pre > 0:
            print(f"Funding share of positive pre-cost expectancy: {100*float(r.avg_funding)/pre:.1f}%")

        hy = year_df[year_df.strategy == "HIGH_VOL_REVERSAL"].set_index("year")
        if 2022 in hy.index:
            rr = hy.loc[2022]
            print(
                "2022 failure attribution: "
                f"price={fmt_pct(float(rr.avg_price))}/week | "
                f"funding={fmt_pct(float(rr.avg_funding))}/week | "
                f"cost={fmt_pct(float(rr.avg_cost))}/week | "
                f"net={fmt_pct(float(rr.avg_net))}/week"
            )
        for y in [2025, 2026]:
            if y in hy.index:
                rr = hy.loc[y]
                print(
                    f"{y} attribution: "
                    f"price={fmt_pct(float(rr.avg_price))}/week | "
                    f"funding={fmt_pct(float(rr.avg_funding))}/week | "
                    f"cost={fmt_pct(float(rr.avg_cost))}/week | "
                    f"net={fmt_pct(float(rr.avg_net))}/week"
                )

    summary.to_csv(out / "overall_attribution.csv", index=False)
    year_df.to_csv(out / "year_attribution.csv", index=False)
    leg_df.to_csv(out / "leg_attribution.csv", index=False)
    w.to_csv(out / "weekly_mirror_attribution.csv", index=False)
    a.to_csv(out / "asset_mirror_attribution.csv", index=False)
    print(f"\nSaved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
