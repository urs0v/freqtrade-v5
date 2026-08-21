#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SETUPS = ("H_BREAK", "H_RETEST", "H_BOUNCE", "H_FAKEOUT")


def parse_args():
    p = argparse.ArgumentParser(description="Aggregate Digash replication V3")
    p.add_argument("--parts", required=True)
    p.add_argument("--outdir", required=True)
    return p.parse_args()


def mask_variants(s: pd.DataFrame) -> dict[str, pd.Series]:
    allm = pd.Series(True, index=s.index)
    return {
        "ALL": allm,
        "ACTIVE": s.active_any,
        "ACTIVE_STRICT": s.active_strict,
        "ACTIVE_PROTO": s.active_any & s.protor_proxy,
        "ACTIVE_PROTO_RR3": s.active_any & s.protor_proxy & s.has_rr3,
        "ACTIVE_RR3": s.active_any & s.has_rr3,
    }


def stats(z: pd.DataFrame) -> dict:
    if z.empty:
        return {k: np.nan for k in [
            "hit1_pct", "hit2_pct", "hit3_pct", "mean_net3_r", "median_net3_r", "pf3_r",
            "stress_net3_r", "stress_pf3_r", "mean_mfe_r", "mean_mae_r", "mean_risk_pct",
            "mean_rr_available", "events_per_week", "median_week_events"
        ]} | {"n": 0, "positive_years": 0, "years": 0}
    pos = z.net_3r.clip(lower=0).sum()
    neg = -z.net_3r.clip(upper=0).sum()
    spos = z.stress_net_3r.clip(lower=0).sum()
    sneg = -z.stress_net_3r.clip(upper=0).sum()
    years = z.assign(year=z.entry_time.dt.year).groupby("year").net_3r.mean()
    weeks = z.assign(week=z.entry_time.dt.to_period("W").astype(str)).groupby("week").size()
    span_days = max((z.entry_time.max() - z.entry_time.min()).total_seconds() / 86400.0, 7.0)
    return {
        "n": len(z),
        "hit1_pct": z.hit_1r.mean() * 100,
        "hit2_pct": z.hit_2r.mean() * 100,
        "hit3_pct": z.hit_3r.mean() * 100,
        "mean_net3_r": z.net_3r.mean(),
        "median_net3_r": z.net_3r.median(),
        "pf3_r": pos / neg if neg > 0 else math.inf,
        "stress_net3_r": z.stress_net_3r.mean(),
        "stress_pf3_r": spos / sneg if sneg > 0 else math.inf,
        "mean_mfe_r": z.mfe_r.mean(),
        "mean_mae_r": z.mae_r.mean(),
        "mean_risk_pct": z.risk_pct.mean(),
        "mean_rr_available": z.rr_available.replace([np.inf, -np.inf], np.nan).mean(),
        "positive_years": int((years > 0).sum()),
        "years": int(len(years)),
        "events_per_week": len(z) / (span_days / 7.0),
        "median_week_events": float(weeks.median()) if len(weeks) else np.nan,
    }


def main() -> int:
    a = parse_args()
    parts = Path(a.parts)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(parts.glob("*/events.csv"))
    covs = sorted(parts.glob("*/coverage.csv"))
    levs = sorted(parts.glob("*/levels.csv"))
    if not files:
        raise RuntimeError(f"No events.csv under {parts}")
    df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
    for c in ["signal_time", "entry_time"]:
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in [
        "clean_between", "protor_proxy", "impulse_proxy", "has_rr3", "active_any", "active_strict",
        "top_growth", "top_decline", "top_volatility", "spike_alert", "hit_1r", "hit_2r", "hit_3r"
    ]:
        if c in df and df[c].dtype != bool:
            df[c] = df[c].astype(str).str.lower().eq("true")
    df.to_csv(outdir / "events.csv", index=False)
    if covs:
        pd.concat([pd.read_csv(p) for p in covs], ignore_index=True).to_csv(outdir / "coverage.csv", index=False)
    if levs:
        pd.concat([pd.read_csv(p) for p in levs], ignore_index=True).to_csv(outdir / "levels.csv", index=False)

    rows = []
    yearly = []
    splits = []
    bytf = []
    for setup in SETUPS:
        s = df[df.setup == setup].copy()
        for name, mask in mask_variants(s).items():
            z = s.loc[mask]
            rows.append({"setup": setup, "variant": name, **stats(z)})
            if name in ("ACTIVE", "ACTIVE_PROTO", "ACTIVE_PROTO_RR3") and not z.empty:
                for y, g in z.groupby(z.entry_time.dt.year):
                    yearly.append({"setup": setup, "variant": name, "year": int(y), **stats(g)})
                ranges = {
                    "2022-2024": z[(z.entry_time >= "2022-01-01") & (z.entry_time < "2025-01-01")],
                    "2025": z[(z.entry_time >= "2025-01-01") & (z.entry_time < "2026-01-01")],
                    "2026": z[z.entry_time >= "2026-01-01"],
                }
                for sn, g in ranges.items():
                    splits.append({"setup": setup, "variant": name, "split": sn, **stats(g)})
        for (tf, period), g in s[s.active_any].groupby(["tf", "period"]):
            bytf.append({"setup": setup, "tf": tf, "period": int(period), **stats(g)})

    summary = pd.DataFrame(rows)
    year = pd.DataFrame(yearly)
    spl = pd.DataFrame(splits)
    tfdf = pd.DataFrame(bytf)
    summary.to_csv(outdir / "summary.csv", index=False)
    year.to_csv(outdir / "yearly.csv", index=False)
    spl.to_csv(outdir / "splits.csv", index=False)
    tfdf.to_csv(outdir / "by_timeframe.csv", index=False)

    print("\n=== DIGASH REPLICATION V3 AGGREGATE ===", flush=True)
    print(f"pairs={df.pair.nunique()} | deduplicated events={len(df):,}", flush=True)
    print("Known fidelity limits: no historical trade-count ranking, no order-book densities, no closed trend-line algorithm.", flush=True)
    print("Touch tolerance setting is frozen as 1% from public/secondary walkthrough evidence; it is not optimized here.", flush=True)
    print("Frequency is diagnostic only; user recalls roughly 15 trades/week, no artificial cap is applied.", flush=True)

    for variant in ("ACTIVE", "ACTIVE_PROTO", "ACTIVE_PROTO_RR3"):
        print(f"\n=== {variant} ===", flush=True)
        v = summary[summary.variant == variant]
        for r in v.itertuples(index=False):
            print(
                f"{r.setup:10s} N={r.n:6d} wk={r.events_per_week:5.1f} "
                f"H1={r.hit1_pct:5.1f}% H2={r.hit2_pct:5.1f}% H3={r.hit3_pct:5.1f}% "
                f"net3={r.mean_net3_r:+.3f}R PF={r.pf3_r:.2f} "
                f"stress12={r.stress_net3_r:+.3f}R/PF={r.stress_pf3_r:.2f} "
                f"MFE={r.mean_mfe_r:.2f} MAE={r.mean_mae_r:.2f} years={r.positive_years}/{r.years}",
                flush=True,
            )

    print("\n=== ACTIVE BY TIMEFRAME / PERIOD ===", flush=True)
    for r in tfdf.sort_values(["setup", "tf", "period"]).itertuples(index=False):
        print(
            f"{r.setup:10s} {r.tf:3s} p{r.period}: N={r.n:5d} wk={r.events_per_week:4.1f} "
            f"net3={r.mean_net3_r:+.3f}R PF={r.pf3_r:.2f}",
            flush=True,
        )

    print("\n=== ACTIVE_PROTO_RR3 SPLITS ===", flush=True)
    ss = spl[spl.variant == "ACTIVE_PROTO_RR3"]
    for setup in SETUPS:
        z = ss[ss.setup == setup]
        if z.empty:
            continue
        vals = [
            f"{r.split}: N={r.n} net3={r.mean_net3_r:+.3f}R PF={r.pf3_r:.2f}"
            for r in z.itertuples(index=False)
        ]
        print(f"{setup:10s} | " + " | ".join(vals), flush=True)

    print("\n=== PREDECLARED RESEARCH GATE (ACTIVE_PROTO_RR3) ===", flush=True)
    print("PASS requires N>=100, PF>=1.05, mean net3>0, >=4/5 positive years, and positive 2025 + 2026 splits.", flush=True)
    vv = summary[summary.variant == "ACTIVE_PROTO_RR3"].set_index("setup")
    for setup in SETUPS:
        if setup not in vv.index:
            print(f"{setup:10s} FAIL (no sample)", flush=True)
            continue
        r = vv.loc[setup]
        z = ss[ss.setup == setup].set_index("split")
        v25 = float(z.loc["2025", "mean_net3_r"]) if "2025" in z.index else np.nan
        v26 = float(z.loc["2026", "mean_net3_r"]) if "2026" in z.index else np.nan
        ok = (
            r.n >= 100 and r.pf3_r >= 1.05 and r.mean_net3_r > 0
            and r.positive_years >= min(4, r.years)
            and np.isfinite(v25) and v25 > 0
            and np.isfinite(v26) and v26 > 0
        )
        print(f"{setup:10s} {'PASS' if ok else 'FAIL'}", flush=True)
    print(f"\nReports: {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
