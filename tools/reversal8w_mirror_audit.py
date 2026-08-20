#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Exact post-hoc mirror of saved 8-week reversal weekly PnL")
    p.add_argument("--input", default="/freqtrade/user_data/reversal8w_perp/weekly_results.csv")
    p.add_argument("--output-dir", default="/freqtrade/user_data/reversal8w_mirror")
    return p.parse_args()


def perf(r: pd.Series) -> dict[str, float]:
    r = r.fillna(0.0).astype(float)
    if (r <= -1.0).any():
        first = int(np.flatnonzero((r <= -1.0).to_numpy())[0])
        rr = r.iloc[: first + 1]
        eq = (1.0 + rr).cumprod().clip(lower=0.0)
        equity = 0.0
        total = -1.0
        mdd = -1.0
    else:
        eq = (1.0 + r).cumprod()
        equity = float(eq.iloc[-1])
        total = equity - 1.0
        mdd = float((eq / eq.cummax() - 1.0).min())
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    sd = float(r.std(ddof=1))
    downside = float(r[r < 0].std(ddof=1)) if (r < 0).sum() > 1 else math.nan
    years = len(r) / 52.0
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 and years > 0 else -1.0
    return {
        "weeks": len(r),
        "ending_equity": 100.0 * equity,
        "total_return": total,
        "cagr": cagr,
        "weekly_wr": float((r > 0).mean()),
        "profit_factor": pos / neg if neg > 0 else math.inf,
        "sharpe": math.sqrt(52.0) * float(r.mean()) / sd if sd > 0 else math.nan,
        "sortino": math.sqrt(52.0) * float(r.mean()) / downside if np.isfinite(downside) and downside > 0 else math.nan,
        "max_drawdown": mdd,
    }


def fmt_pct(v: float) -> str:
    return "nan" if not np.isfinite(v) else f"{100*v:.2f}%"


def main() -> int:
    cfg = args()
    src = Path(cfg.input)
    if not src.exists():
        raise RuntimeError(f"Missing source weekly results: {src}")
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src)
    need = {"strategy", "date", "gross_return", "funding_return", "cost_return", "net_return"}
    miss = need - set(df.columns)
    if miss:
        raise RuntimeError(f"Missing columns: {sorted(miss)}")
    df["date"] = pd.to_datetime(df["date"], utc=True)

    # Exact mirrored portfolio accounting: all position weights change sign.
    # Gross and funding contributions invert; turnover/cost are unchanged.
    out = df.copy()
    out["source_strategy"] = out["strategy"]
    out["strategy"] = out["strategy"].replace({
        "BASELINE_REVERSAL": "BASELINE_MOMENTUM_MIRROR",
        "HIGH_VOL_REVERSAL": "HIGH_VOL_MOMENTUM_MIRROR",
    })
    out["gross_return"] = -out["gross_return"]
    out["funding_return"] = -out["funding_return"]
    out["net_return"] = out["gross_return"] + out["funding_return"] - out["cost_return"]

    print("=== 8-WEEK EXACT MIRROR DIAGNOSTIC ===")
    print("POST-HOC diagnostic only: direction was chosen after seeing reversal fail.")
    print("No parameter changes. Same 56d formation, universe, high-vol filter, rebalance and costs.")
    print("Mirror = long prior winners / short prior losers. Gross and funding signs invert; turnover cost does not.\n")

    summaries = []
    years = []
    for strat, q in out.groupby("strategy"):
        q = q.sort_values("date")
        m = perf(q.net_return)
        m.update({
            "strategy": strat,
            "avg_gross": float(q.gross_return.mean()),
            "avg_funding": float(q.funding_return.mean()),
            "avg_cost": float(q.cost_return.mean()),
            "avg_net": float(q.net_return.mean()),
            "avg_turnover": float(q.turnover.mean()) if "turnover" in q.columns else math.nan,
        })
        summaries.append(m)
        for year, yy in q.groupby(q.date.dt.year):
            mm = perf(yy.net_return)
            years.append({"strategy": strat, "year": int(year), **mm})

    sdf = pd.DataFrame(summaries)
    ydf = pd.DataFrame(years)

    disp = sdf.copy()
    for c in ["total_return", "cagr", "weekly_wr", "max_drawdown", "avg_gross", "avg_funding", "avg_cost", "avg_net"]:
        disp[c] = disp[c].map(fmt_pct)
    print("=== MIRROR RESULT ===")
    print(disp[["strategy","weeks","ending_equity","total_return","cagr","weekly_wr","profit_factor","sharpe","sortino","max_drawdown","avg_gross","avg_funding","avg_cost","avg_net"]].to_string(index=False))

    yd = ydf.copy()
    for c in ["total_return", "weekly_wr", "max_drawdown"]:
        yd[c] = yd[c].map(fmt_pct)
    print("\nYEAR BREAKDOWN")
    print(yd[["strategy","year","total_return","profit_factor","sharpe","max_drawdown","weekly_wr"]].to_string(index=False))

    hv = out[out.strategy == "HIGH_VOL_MOMENTUM_MIRROR"].sort_values("date")
    post = hv[(hv.date >= pd.Timestamp("2026-04-01", tz="UTC")) & (hv.date <= pd.Timestamp("2026-07-31", tz="UTC"))]
    pm = perf(post.net_return) if len(post) else None

    hvm = sdf.set_index("strategy").loc["HIGH_VOL_MOMENTUM_MIRROR"]
    yhv = ydf[ydf.strategy == "HIGH_VOL_MOMENTUM_MIRROR"].set_index("year")
    gates = [
        ("Mirror net total > 0", hvm.total_return > 0),
        ("Mirror PF > 1.30", hvm.profit_factor > 1.30),
        ("Mirror Sharpe > 1.00", hvm.sharpe > 1.00),
        ("2024 positive", 2024 in yhv.index and yhv.loc[2024, "total_return"] > 0),
        ("2025 positive", 2025 in yhv.index and yhv.loc[2025, "total_return"] > 0),
        ("2026 positive", 2026 in yhv.index and yhv.loc[2026, "total_return"] > 0),
        ("MDD better than -50%", hvm.max_drawdown > -0.50),
    ]
    print("\nPOST-HOC MIRROR QUALITY GATES")
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if pm:
        print(f"Apr-Jul 2026 mirror: return={fmt_pct(pm['total_return'])} PF={pm['profit_factor']:.3f} Sharpe={pm['sharpe']:.3f}")

    print("\nINTERPRETATION")
    if all(ok for _, ok in gates):
        print("[CANDIDATE] Exact mirror is economically strong enough to justify a genuinely new validation test. This result itself is NOT OOS proof.")
    else:
        print("[WEAK/UNSTABLE] Even the exact mirror does not clear basic quality gates; do not build on it.")

    out.to_csv(outdir / "mirror_weekly_results.csv", index=False)
    sdf.to_csv(outdir / "summary.csv", index=False)
    ydf.to_csv(outdir / "year_breakdown.csv", index=False)
    pd.DataFrame([{"gate": g, "pass": bool(v)} for g, v in gates]).to_csv(outdir / "gates.csv", index=False)
    print(f"\nSaved under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
