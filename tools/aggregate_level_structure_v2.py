#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SETUPS = (
    "LEVEL_BREAKOUT",
    "LEVEL_BREAK_RETEST",
    "CONSOLIDATION_BREAKOUT",
    "CONFIRMED_BOUNCE",
    "SWEEP_RECLAIM",
    "STRUCTURE_BREAK_RETEST",
)

def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Level/Structure Edge V2 worker events")
    p.add_argument("--parts", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()

def strict_mask(df: pd.DataFrame, setup: str) -> pd.Series:
    core = df["active_top10"] & df["has_rr3"]
    impulse = (df["volume_z"] >= 1.0) | (df["range_z"] >= 1.0)
    if setup == "LEVEL_BREAKOUT":
        return core & (df["level_touches"] >= 3) & (df["compression_score"] >= 2) & impulse
    if setup == "LEVEL_BREAK_RETEST":
        return core & (df["level_touches"] >= 2) & (df["reclaim_bars"] <= 6)
    if setup == "CONSOLIDATION_BREAKOUT":
        return core & impulse
    if setup == "CONFIRMED_BOUNCE":
        return core & (df["level_touches"] >= 3) & (df["interaction_no"] <= 2)
    if setup == "SWEEP_RECLAIM":
        return core & (df["level_touches"] >= 2) & (df["reclaim_bars"] <= 2) & impulse
    if setup == "STRUCTURE_BREAK_RETEST":
        return core & df["structure_subtype"].isin(["REVERSAL", "CONTINUATION"]) & (df["reclaim_bars"] <= 6)
    return core

def summarize(z: pd.DataFrame) -> dict:
    if z.empty:
        return {
            "n": 0, "hit1_pct": np.nan, "hit2_pct": np.nan, "hit3_pct": np.nan,
            "mean_net3_r": np.nan, "median_net3_r": np.nan, "pf3_r": np.nan,
            "mean_stress_net3_r": np.nan, "stress_pf3_r": np.nan,
            "mean_mfe_r": np.nan, "mean_mae_r": np.nan, "mean_risk_pct": np.nan,
            "mean_rr_available": np.nan, "positive_years": 0, "years": 0,
            "positive_months": 0, "months": 0,
        }
    pos = z["net_3r"].clip(lower=0).sum()
    neg = -z["net_3r"].clip(upper=0).sum()
    spos = z["stress_net_3r"].clip(lower=0).sum()
    sneg = -z["stress_net_3r"].clip(upper=0).sum()
    ym = z.assign(year=z["entry_time"].dt.year).groupby("year")["net_3r"].mean()
    mm = z.assign(month=z["entry_time"].dt.strftime("%Y-%m")).groupby("month")["net_3r"].mean()
    return {
        "n": len(z),
        "hit1_pct": z["hit_1r"].mean()*100,
        "hit2_pct": z["hit_2r"].mean()*100,
        "hit3_pct": z["hit_3r"].mean()*100,
        "mean_net3_r": z["net_3r"].mean(),
        "median_net3_r": z["net_3r"].median(),
        "pf3_r": pos/neg if neg > 0 else math.inf,
        "mean_stress_net3_r": z["stress_net_3r"].mean(),
        "stress_pf3_r": spos/sneg if sneg > 0 else math.inf,
        "mean_mfe_r": z["mfe_r"].mean(),
        "mean_mae_r": z["mae_r"].mean(),
        "mean_risk_pct": z["risk_pct"].mean(),
        "mean_rr_available": z["rr_available"].mean(),
        "positive_years": int((ym > 0).sum()),
        "years": int(len(ym)),
        "positive_months": int((mm > 0).sum()),
        "months": int(len(mm)),
    }

def main() -> int:
    a = parse_args()
    parts = Path(a.parts)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(parts.glob("*/events.csv"))
    covs = sorted(parts.glob("*/coverage.csv"))
    if not files:
        raise RuntimeError(f"No events.csv under {parts}")

    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    for b in ["active_top5","active_top10","has_rr3","tight_stop","hit_1r","hit_2r","hit_3r"]:
        if df[b].dtype != bool:
            df[b] = df[b].astype(str).str.lower().eq("true")
    df.to_csv(outdir / "events.csv", index=False)
    if covs:
        pd.concat([pd.read_csv(p) for p in covs], ignore_index=True).to_csv(outdir / "coverage.csv", index=False)

    rows = []
    yearly_rows = []
    split_rows = []
    for setup in SETUPS:
        s = df[df["setup"] == setup].copy()
        variants = {
            "ALL": pd.Series(True, index=s.index),
            "TOP10": s["active_top10"],
            "RR3": s["has_rr3"],
            "CORE": s["active_top10"] & s["has_rr3"],
            "STRICT": strict_mask(s, setup),
        }
        for variant, mask in variants.items():
            z = s.loc[mask].copy()
            row = {"setup": setup, "variant": variant, **summarize(z)}
            rows.append(row)
            if variant in ("CORE", "STRICT") and not z.empty:
                for year, g in z.groupby(z["entry_time"].dt.year):
                    yearly_rows.append({
                        "setup": setup, "variant": variant, "year": int(year), **summarize(g)
                    })
                splits = {
                    "2022-2024": z[(z.entry_time >= "2022-01-01") & (z.entry_time < "2025-01-01")],
                    "2025": z[(z.entry_time >= "2025-01-01") & (z.entry_time < "2026-01-01")],
                    "2026": z[z.entry_time >= "2026-01-01"],
                }
                for name, g in splits.items():
                    split_rows.append({"setup": setup, "variant": variant, "split": name, **summarize(g)})

    summary = pd.DataFrame(rows)
    yearly = pd.DataFrame(yearly_rows)
    splits = pd.DataFrame(split_rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    yearly.to_csv(outdir / "yearly.csv", index=False)
    splits.to_csv(outdir / "splits.csv", index=False)

    print("\n=== LEVEL/STRUCTURE EDGE V2 AGGREGATE ===", flush=True)
    print(f"pairs with events: {df.pair.nunique()} | events: {len(df):,}", flush=True)
    print("Execution metric: fixed +3R vs structural -1R stop, 8 bps round-trip cost; 12 bps is also reported as stress.", flush=True)
    print("CORE = top-10 activity + structural room >=3R.", flush=True)
    print("STRICT adds setup-specific facts declared before seeing results.", flush=True)

    for variant in ("CORE", "STRICT"):
        print(f"\n=== {variant} ===", flush=True)
        v = summary[summary.variant == variant]
        for r in v.itertuples(index=False):
            print(
                f"{r.setup:27s} N={r.n:6d} "
                f"H1={r.hit1_pct:5.1f}% H2={r.hit2_pct:5.1f}% H3={r.hit3_pct:5.1f}% "
                f"net3={r.mean_net3_r:+.3f}R PF={r.pf3_r:.2f} "
                f"stress12={r.mean_stress_net3_r:+.3f}R/PF={r.stress_pf3_r:.2f} "
                f"MFE={r.mean_mfe_r:.2f}R MAE={r.mean_mae_r:.2f}R "
                f"years={r.positive_years}/{r.years}",
                flush=True,
            )

    print("\n=== CORE SPLITS ===", flush=True)
    cs = splits[splits.variant == "CORE"]
    for setup in SETUPS:
        z = cs[cs.setup == setup]
        if z.empty:
            continue
        vals = []
        for r in z.itertuples(index=False):
            vals.append(f"{r.split}: N={r.n} net3={r.mean_net3_r:+.3f}R PF={r.pf3_r:.2f}")
        print(f"{setup:27s} | " + " | ".join(vals), flush=True)

    print("\n=== PREDECLARED CANDIDATE GATE ===", flush=True)
    print("Candidate only if CORE: N>=200, PF>=1.05, mean net3>0, >=4/5 positive years,", flush=True)
    print("and both 2025 and 2026 splits have positive mean net3. This is a research gate, not deployment approval.", flush=True)
    core = summary[summary.variant == "CORE"].set_index("setup")
    for setup in SETUPS:
        if setup not in core.index:
            print(f"{setup:27s} FAIL (no CORE sample)", flush=True)
            continue
        r = core.loc[setup]
        zs = cs[cs.setup == setup].set_index("split")
        val25 = float(zs.loc["2025","mean_net3_r"]) if "2025" in zs.index else np.nan
        val26 = float(zs.loc["2026","mean_net3_r"]) if "2026" in zs.index else np.nan
        ok = (
            r["n"] >= 200 and r["pf3_r"] >= 1.05 and r["mean_net3_r"] > 0
            and r["positive_years"] >= min(4, r["years"])
            and np.isfinite(val25) and val25 > 0
            and np.isfinite(val26) and val26 > 0
        )
        print(f"{setup:27s} {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\nReports: {outdir}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
