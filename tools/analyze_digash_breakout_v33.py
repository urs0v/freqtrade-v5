#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260821


def parse_args():
    p = argparse.ArgumentParser(description="Digash breakout V3.3 repeated-approach robustness audit")
    p.add_argument("--events", default="/freqtrade/user_data/digash_breakout_v32/breakout_events_v32.csv")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_breakout_v33")
    p.add_argument("--bootstrap", type=int, default=5000)
    return p.parse_args()


def as_bool(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def alt_net(z: pd.DataFrame, bps: float) -> pd.Series:
    risk_frac = pd.to_numeric(z["risk_pct"], errors="coerce") / 100.0
    fee_r = (bps / 10000.0) / risk_frac.replace(0, np.nan)
    return pd.to_numeric(z["gross_3r"], errors="coerce") - fee_r


def pf(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    pos = x.clip(lower=0).sum()
    neg = -x.clip(upper=0).sum()
    return float(pos / neg) if neg > 0 else math.inf


def stats(z: pd.DataFrame) -> dict:
    if z.empty:
        return {"n": 0, "wk": np.nan, "gross": np.nan, "gpf": np.nan,
                "net8": np.nan, "pf8": np.nan, "net12": np.nan, "pf12": np.nan,
                "net16": np.nan, "pf16": np.nan, "risk": np.nan, "pos_years8": 0, "years": 0}
    span_days = max((z.entry_time.max() - z.entry_time.min()).total_seconds() / 86400.0, 7.0)
    n8, n12, n16 = alt_net(z, 8.0), alt_net(z, 12.0), alt_net(z, 16.0)
    yy = z.assign(_n8=n8).groupby(z.entry_time.dt.year)["_n8"].mean()
    return {
        "n": int(len(z)),
        "wk": float(len(z) / (span_days / 7.0)),
        "gross": float(pd.to_numeric(z.gross_3r, errors="coerce").mean()),
        "gpf": pf(z.gross_3r),
        "net8": float(n8.mean()), "pf8": pf(n8),
        "net12": float(n12.mean()), "pf12": pf(n12),
        "net16": float(n16.mean()), "pf16": pf(n16),
        "risk": float(pd.to_numeric(z.risk_pct, errors="coerce").median()),
        "pos_years8": int((yy > 0).sum()), "years": int(len(yy)),
    }


def fmt(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"{label:24s} N=0"
    return (
        f"{label:24s} N={s['n']:4d} wk={s['wk']:4.1f} "
        f"gross={s['gross']:+.3f}R/gPF={s['gpf']:.2f} "
        f"8b={s['net8']:+.3f}R/PF={s['pf8']:.2f} "
        f"12b={s['net12']:+.3f}R/PF={s['pf12']:.2f} "
        f"16b={s['net16']:+.3f}R/PF={s['pf16']:.2f} "
        f"risk={s['risk']:.3f}% years8={s['pos_years8']}/{s['years']}"
    )


def approach_masks(z: pd.DataFrame) -> dict[str, pd.Series]:
    a = pd.to_numeric(z.approach_no, errors="coerce")
    return {
        "APPROACH_1": a <= 1,
        "APPROACH_2": a == 2,
        "APPROACH_2PLUS": a >= 2,
        "APPROACH_3PLUS": a >= 3,
    }


def cohort_masks(z: pd.DataFrame) -> dict[str, pd.Series]:
    true = pd.Series(True, index=z.index)
    return {
        "FACT_ALL": true,
        "LOCAL_ACTIVE": z.local_active_any,
        "BROAD_TOP10": z.broad_active_top10,
        "BROAD_TOP5": z.broad_active_top5,
    }


def clustered_bootstrap(z: pd.DataFrame, bps: float, reps: int) -> tuple[float, float, float]:
    if z.empty:
        return np.nan, np.nan, np.nan
    x = z.copy()
    x["_net"] = alt_net(x, bps)
    x["_month"] = x.entry_time.dt.tz_localize(None).dt.to_period("M").astype(str)
    g = x.groupby("_month")["_net"].agg(["sum", "count"])
    if len(g) < 3:
        m = float(x._net.mean())
        return m, np.nan, np.nan
    rng = np.random.default_rng(SEED + int(bps * 10))
    sums = g["sum"].to_numpy(float)
    counts = g["count"].to_numpy(float)
    idx = rng.integers(0, len(g), size=(reps, len(g)))
    means = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return float(x._net.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def pair_loo(z: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for pair in sorted(z.pair.unique()):
        s = stats(z[z.pair != pair])
        rows.append({"excluded": pair, **s})
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "approach3plus_leave_one_pair_out.csv", index=False)
    return out


def year_loo(z: pd.DataFrame, outdir: Path) -> pd.DataFrame:
    rows = []
    for y in sorted(z.entry_time.dt.year.unique()):
        s = stats(z[z.entry_time.dt.year != y])
        rows.append({"excluded_year": int(y), **s})
    out = pd.DataFrame(rows)
    out.to_csv(outdir / "approach3plus_leave_one_year_out.csv", index=False)
    return out


def main() -> int:
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    p = Path(a.events)
    if not p.exists():
        raise RuntimeError(f"V3.2 events not found: {p}. Run V3.2 first.")

    df = pd.read_csv(p)
    df["signal_time"] = pd.to_datetime(df.signal_time, utc=True)
    df["entry_time"] = pd.to_datetime(df.entry_time, utc=True)
    for c in ["fact_proxy", "local_active_any", "broad_active_top10", "broad_active_top5", "impulse_proxy"]:
        if c in df:
            df[c] = as_bool(df[c])
    base = df[(df.setup == "H_BREAK") & (df.tf == "4h") & (df.period == 30) & df.fact_proxy].copy()
    if base.empty:
        raise RuntimeError("No 4h/p30 factual breakout sample")

    print("=== DIGASH BREAKOUT V3.3 — REPEATED APPROACH ROBUSTNESS ===", flush=True)
    print(f"4h/p30 factual breakout rows={len(base):,}", flush=True)
    print("This is an exploratory follow-up to the V3.2 3+ approach finding, NOT independent OOS evidence.", flush=True)
    print("APPROACH_2PLUS is the closest cached-OHLCV proxy to the public density guidance '2+ approaches may be traded for breakout'.", flush=True)
    print("APPROACH_3PLUS is explicitly post-selected from V3.2 and must survive stronger robustness checks before promotion.", flush=True)

    rows = []
    print("\n=== APPROACH TABLE BY ACTIVITY COHORT ===", flush=True)
    for cname, cmask in cohort_masks(base).items():
        zc = base.loc[cmask].copy()
        for aname, amask in approach_masks(zc).items():
            z = zc.loc[amask]
            s = stats(z)
            rows.append({"cohort": cname, "approach": aname, **s})
            print(fmt(f"{cname} {aname}", s), flush=True)
    pd.DataFrame(rows).to_csv(outdir / "approach_by_cohort.csv", index=False)

    # Keep V3.2's primary cohort for continuity. The broad universe was only modestly wider,
    # so LOCAL_ACTIVE is reported alongside it and no broad-ranking claim is upgraded here.
    primary = base[base.broad_active_top10].copy()
    a3 = pd.to_numeric(primary.approach_no, errors="coerce") >= 3
    cand = primary[a3].copy()
    a2p = primary[pd.to_numeric(primary.approach_no, errors="coerce") >= 2].copy()

    print("\n=== PRIMARY BROAD_TOP10: SOURCE-PROXY 2+ VS POST-HOC 3+ ===", flush=True)
    print(fmt("APPROACH_2PLUS", stats(a2p)), flush=True)
    print(fmt("APPROACH_3PLUS", stats(cand)), flush=True)

    print("\n=== APPROACH_3PLUS TIME ROBUSTNESS ===", flush=True)
    split_rows = []
    split_map = {
        "2022-2024": cand[(cand.entry_time >= "2022-01-01") & (cand.entry_time < "2025-01-01")],
        "2025": cand[(cand.entry_time >= "2025-01-01") & (cand.entry_time < "2026-01-01")],
        "2026": cand[cand.entry_time >= "2026-01-01"],
        "2025-2026": cand[cand.entry_time >= "2025-01-01"],
    }
    for name, z in split_map.items():
        s = stats(z); split_rows.append({"split": name, **s}); print(fmt(name, s), flush=True)
    for y, z in cand.groupby(cand.entry_time.dt.year):
        s = stats(z); split_rows.append({"split": str(int(y)), **s}); print(fmt(str(int(y)), s), flush=True)
    pd.DataFrame(split_rows).to_csv(outdir / "approach3plus_time_splits.csv", index=False)

    print("\n=== APPROACH_3PLUS SIDE ROBUSTNESS ===", flush=True)
    side_rows = []
    for side, z in cand.groupby("side"):
        name = "LONG" if int(side) > 0 else "SHORT"
        s = stats(z); side_rows.append({"side": name, **s}); print(fmt(name, s), flush=True)
    pd.DataFrame(side_rows).to_csv(outdir / "approach3plus_sides.csv", index=False)

    print("\n=== APPROACH_3PLUS PAIR ROBUSTNESS ===", flush=True)
    pair_rows = []
    for pair, z in cand.groupby("pair"):
        s = stats(z); pair_rows.append({"pair": pair, "sum_net8_r": float(alt_net(z, 8).sum()), **s})
    pairdf = pd.DataFrame(pair_rows).sort_values("sum_net8_r", ascending=False)
    pairdf.to_csv(outdir / "approach3plus_by_pair.csv", index=False)
    print(f"pairs with trades={len(pairdf)}", flush=True)
    for r in pairdf.head(10).itertuples(index=False):
        print(f"{r.pair:20s} N={r.n:3d} 8b={r.net8:+.3f}R PF={r.pf8:.2f} 12b={r.net12:+.3f}R/PF={r.pf12:.2f} sum8={r.sum_net8_r:+.2f}R", flush=True)

    loo = pair_loo(cand, outdir)
    yloo = year_loo(cand, outdir)
    if not loo.empty:
        print(f"pair-LOO 8bps PF range {loo.pf8.min():.2f} .. {loo.pf8.max():.2f}; net range {loo.net8.min():+.3f}R .. {loo.net8.max():+.3f}R", flush=True)
        print(f"pair-LOO 12bps PF range {loo.pf12.min():.2f} .. {loo.pf12.max():.2f}", flush=True)
    if not yloo.empty:
        print(f"year-LOO 8bps PF range {yloo.pf8.min():.2f} .. {yloo.pf8.max():.2f}", flush=True)

    print("\n=== MONTH-CLUSTERED BOOTSTRAP OF APPROACH_3PLUS MEAN R ===", flush=True)
    boot_rows = []
    for bps in (8.0, 12.0, 16.0):
        mean, lo, hi = clustered_bootstrap(cand, bps, a.bootstrap)
        boot_rows.append({"bps": bps, "mean_r": mean, "ci025": lo, "ci975": hi, "reps": a.bootstrap})
        print(f"{bps:>4.0f} bps: mean={mean:+.3f}R | month-cluster bootstrap 95% CI [{lo:+.3f}, {hi:+.3f}]", flush=True)
    pd.DataFrame(boot_rows).to_csv(outdir / "approach3plus_bootstrap.csv", index=False)

    print("\n=== PAIR×YEAR COMPOSITION-CONTROL DIAGNOSTIC ===", flush=True)
    x = primary.copy()
    x["year"] = x.entry_time.dt.year
    x["net8_calc"] = alt_net(x, 8.0)
    x["cell_mean"] = x.groupby(["pair", "year"])["net8_calc"].transform("mean")
    x["adj_net8"] = x.net8_calc - x.cell_mean
    x["approach_bucket"] = np.where(pd.to_numeric(x.approach_no, errors="coerce") >= 3, "3+", np.where(pd.to_numeric(x.approach_no, errors="coerce") == 2, "2", "1"))
    adj = x.groupby("approach_bucket", as_index=False).agg(n=("adj_net8", "size"), raw_net8=("net8_calc", "mean"), adjusted_net8=("adj_net8", "mean"))
    adj.to_csv(outdir / "approach_pair_year_adjusted.csv", index=False)
    for r in adj.itertuples(index=False):
        print(f"approach {r.approach_bucket}: N={r.n} raw={r.raw_net8:+.3f}R pair-year-adjusted={r.adjusted_net8:+.3f}R", flush=True)

    print("\n=== APPROACH_3PLUS AGE OVERLAP (DIAGNOSTIC ONLY) ===", flush=True)
    age = pd.to_numeric(cand.level_age_h, errors="coerce")
    cand = cand.copy()
    cand["age_bucket"] = pd.cut(age, [-1e-9, 24, 72, 168, np.inf], labels=["<1d", "1-3d", "3-7d", "7d+"], include_lowest=True)
    age_rows = []
    for bucket, z in cand.groupby("age_bucket", observed=True):
        s = stats(z); age_rows.append({"age": str(bucket), **s}); print(fmt(str(bucket), s), flush=True)
    pd.DataFrame(age_rows).to_csv(outdir / "approach3plus_age_overlap.csv", index=False)

    # Promotion gate is intentionally demanding because APPROACH_3PLUS was discovered post-hoc in V3.2.
    s3 = stats(cand)
    s25 = stats(split_map["2025"]); s26 = stats(split_map["2026"])
    loo12_min = float(loo.pf12.min()) if not loo.empty else np.nan
    promote = (
        s3["n"] >= 100 and s3["pf8"] >= 1.20 and s3["net8"] > 0
        and s3["pf12"] >= 1.10 and s3["net12"] > 0
        and s3["pos_years8"] >= 4
        and s25["net8"] > 0 and s26["net8"] > 0
        and np.isfinite(loo12_min) and loo12_min >= 1.00
    )
    print("\n=== POST-SELECTION PROMOTION GATE ===", flush=True)
    print("Requires: N>=100, PF8>=1.20, PF12>=1.10, positive 8/12bps expectancy, >=4 positive years, positive 2025 and 2026, pair-LOO min PF12>=1.00.", flush=True)
    print(f"APPROACH_3PLUS {'PROMOTE_TO_EXPLICIT_FORMATION_TEST' if promote else 'DO_NOT_PROMOTE_YET'}", flush=True)
    print("This gate is not OOS proof; it only decides whether the post-hoc V3.2 finding deserves a separately frozen next test.", flush=True)

    print(f"\nReports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
