#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TOOLS = Path(__file__).resolve().parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))

import audit_level_edge as a


def main() -> int:
    ap = argparse.ArgumentParser(description="Aggregate per-pair level-edge worker outputs")
    ap.add_argument("--parts", required=True)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    parts = Path(args.parts)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    trade_files = sorted(parts.glob("*/trades.csv"))
    coverage_files = sorted(parts.glob("*/coverage.csv"))
    if not trade_files:
        raise RuntimeError(f"No trades.csv found under {parts}")

    trades = pd.concat([pd.read_csv(p) for p in trade_files], ignore_index=True)
    coverage = pd.concat([pd.read_csv(p) for p in coverage_files], ignore_index=True) if coverage_files else pd.DataFrame()

    for c in ("entry_time", "signal_time", "level_time"):
        trades[c] = pd.to_datetime(trades[c], utc=True)
    trades["active"] = trades["active"].astype(str).str.lower().map({"true": True, "false": False}).fillna(trades["active"].astype(bool))
    trades["compression"] = trades["compression"].astype(str).str.lower().map({"true": True, "false": False}).fillna(trades["compression"].astype(bool))
    trades["year"] = trades.entry_time.dt.year
    trades["month"] = trades.entry_time.dt.strftime("%Y-%m")

    trades.to_csv(outdir / "trades.csv", index=False)
    if not coverage.empty:
        coverage.to_csv(outdir / "coverage.csv", index=False)

    rows = []
    rows += a.summarize(trades, "ALL", np.ones(len(trades), dtype=bool))
    rows += a.summarize(trades, "ACTIVE", trades.active)
    rows += a.summarize(trades, "ACTIVE_COMP", trades.active & ((trades.setup == "BOUNCE") | trades.compression))
    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "summary.csv", index=False)

    yr = []
    for setup in ["BOUNCE", "BREAK_RETEST"]:
        for rt in a.R_TARGETS:
            z = trades[(trades.active) & (trades.setup == setup) & (trades.target_r == rt)]
            for y, g in z.groupby("year"):
                for c in a.COSTS_BPS:
                    col = f"net{int(c)}_r"
                    yr.append({
                        "setup": setup,
                        "target_r": rt,
                        "year": int(y),
                        "cost_bps": c,
                        "n": len(g),
                        "win_pct": float((g[col] > 0).mean() * 100),
                        "mean_net_r": float(g[col].mean()),
                        "mean_net_bps": float((g.gross_ret.mean() - c / 10000.0) * 10000.0),
                    })
    yearly = pd.DataFrame(yr)
    yearly.to_csv(outdir / "yearly_active.csv", index=False)

    print("\n=== PARALLEL AGGREGATE ===", flush=True)
    print(f"workers completed: {len(trade_files)} pair result files", flush=True)
    print(f"trade rows (3 targets each): {len(trades):,}", flush=True)
    print(f"unique pairs with trades: {trades.pair.nunique()}", flush=True)

    print("\n=== SUMMARY: ACTIVE SUBSET / 8 BPS ===", flush=True)
    v = summary[(summary.subset == "ACTIVE") & (summary.cost_bps == 8.0)]
    for r0 in v.itertuples(index=False):
        print(
            f"{r0.setup:12s} {r0.target_r:>3.1f}R | N={r0.n:5d} "
            f"WR={r0.win_pct:5.1f}% gross={r0.mean_gross_bps:+7.2f}bps "
            f"net={r0.mean_net_r:+.3f}R PF={r0.profit_factor_r:.2f} "
            f"positive_years={r0.positive_years}/{r0.years}",
            flush=True,
        )

    print("\n=== SUMMARY: ACTIVE + COMPRESSION FOR BREAKOUTS / 8 BPS ===", flush=True)
    v = summary[(summary.subset == "ACTIVE_COMP") & (summary.cost_bps == 8.0)]
    for r0 in v.itertuples(index=False):
        print(
            f"{r0.setup:12s} {r0.target_r:>3.1f}R | N={r0.n:5d} "
            f"WR={r0.win_pct:5.1f}% gross={r0.mean_gross_bps:+7.2f}bps "
            f"net={r0.mean_net_r:+.3f}R PF={r0.profit_factor_r:.2f} "
            f"positive_years={r0.positive_years}/{r0.years}",
            flush=True,
        )

    print("\n=== YEARLY ACTIVE / 1.5R / 8 BPS ===", flush=True)
    yv = yearly[(yearly.target_r == 1.5) & (yearly.cost_bps == 8.0)]
    for r0 in yv.itertuples(index=False):
        print(
            f"{r0.setup:12s} {r0.year}: N={r0.n:4d} WR={r0.win_pct:5.1f}% "
            f"net={r0.mean_net_r:+.3f}R ({r0.mean_net_bps:+.2f}bps)",
            flush=True,
        )

    print(f"\nReports: {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
