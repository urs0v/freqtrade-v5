#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

THRESH = 1.5
RISK_MIN_BPS = 160.0


def parse_args():
    p = argparse.ArgumentParser(description="Finalize Profit V1.6 from already-produced CSVs after post-processing crash")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v16")
    return p.parse_args()


def metric(g: pd.DataFrame, col: str):
    r = pd.to_numeric(g[col], errors="coerce").dropna().astype(float)
    if r.empty:
        return {"N": 0, "PF": np.nan, "WR": np.nan, "EXP": np.nan, "DD": np.nan}
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    curve = r.cumsum()
    dd = float((curve.cummax() - curve).max()) if len(curve) else 0.0
    return {"N": int(len(r)), "PF": float(pf), "WR": float((r > 0).mean() * 100.0), "EXP": float(r.mean()), "DD": dd}


def fmt(m):
    return f"N={m['N']:4d} PF={m['PF']:5.2f} WR={m['WR']:5.1f}% EXP={m['EXP']:+.3f}R DD={m['DD']:6.1f}R"


def selected(g: pd.DataFrame):
    return g[
        (pd.to_numeric(g["activity_score"], errors="coerce") >= THRESH)
        & (pd.to_numeric(g["risk_bps"], errors="coerce") >= RISK_MIN_BPS)
    ].copy()


def trade_ids(g: pd.DataFrame):
    return set(
        (str(r.pair), pd.Timestamp(r.entry_time).value, int(r.side), int(r.level_id))
        for r in g.itertuples(index=False)
    )


def main():
    a = parse_args()
    out = Path(a.outdir)
    trades_path = out / "dedup_all_fakeouts.csv"
    coverage_path = out / "dedup_pair_coverage.csv"
    if not trades_path.exists():
        raise FileNotFoundError(f"Missing {trades_path}; the V1.6 scan must complete first")

    df = pd.read_csv(trades_path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    if "split" not in df.columns:
        raise RuntimeError("dedup_all_fakeouts.csv has no split column")

    legacy = selected(df[df["mode"].eq("LEGACY_DEDUP")].sort_values("entry_time"))
    causal = selected(df[df["mode"].eq("CAUSAL_DEDUP")].sort_values("entry_time"))
    legacy.to_csv(out / "legacy_selected.csv", index=False)
    causal.to_csv(out / "causal_selected.csv", index=False)

    print("=== BREAKOUT / RETEST PROFIT V1.6 — RECOVERY FINALIZER ===")
    print("Uses the completed V1.6 scan CSVs only; no market scan is repeated and no parameters are changed.")

    print("\n=== DEDUP LOOKAHEAD SANITY ===")
    if coverage_path.exists():
        md = pd.read_csv(coverage_path)
        okm = md[md["status"].eq("OK")].copy() if "status" in md.columns else md.iloc[0:0]
        future_total = int(pd.to_numeric(okm.get("legacy_fakeout_future_replacements", 0), errors="coerce").fillna(0).sum()) if len(okm) else 0
        legacy_f = int(pd.to_numeric(okm.get("legacy_fakeouts", 0), errors="coerce").fillna(0).sum()) if len(okm) else 0
        causal_f = int(pd.to_numeric(okm.get("causal_fakeouts", 0), errors="coerce").fillna(0).sum()) if len(okm) else 0
        print(f"legacy fakeouts={legacy_f:,} | causal fakeouts={causal_f:,} | legacy selections that came from a later bar in their bucket={future_total:,}")
        if legacy_f:
            print(f"future-replacement share of legacy fakeouts={future_total/legacy_f*100:.2f}%")
    else:
        print("coverage CSV missing; skipping aggregate future-replacement count")

    lid = trade_ids(legacy)
    cid = trade_ids(causal)
    inter = len(lid & cid)
    union = len(lid | cid)
    print(f"frozen selected overlap={inter} legacy-only={len(lid-cid)} causal-only={len(cid-lid)} Jaccard={(inter/union if union else np.nan):.3f}")

    print("\n=== FROZEN SIGNAL: LEGACY VS FULLY CAUSAL ===")
    report = []
    for split in ("TRAIN", "VALID", "HOLDOUT"):
        lg = legacy[legacy["split"].eq(split)]
        cg = causal[causal["split"].eq(split)]
        lm = metric(lg, "net8_r")
        ls = metric(lg, "stress12_r")
        cm = metric(cg, "net8_r")
        cs = metric(cg, "stress12_r")
        print(f"{split:7s} LEGACY {fmt(lm)} stressEXP={ls['EXP']:+.3f}R")
        print(f"{split:7s} CAUSAL {fmt(cm)} stressEXP={cs['EXP']:+.3f}R")
        report += [
            {"split": split, "mode": "LEGACY_DEDUP", **lm, "STRESS_EXP": ls["EXP"]},
            {"split": split, "mode": "CAUSAL_DEDUP", **cm, "STRESS_EXP": cs["EXP"]},
        ]
    pd.DataFrame(report).to_csv(out / "dedup_metrics.csv", index=False)

    print("\n=== FULLY CAUSAL CALENDAR STABILITY (8bps) ===")
    for year, g in causal.groupby(causal["entry_time"].dt.year):
        print(f"year {year}: {fmt(metric(g, 'net8_r'))}")
    qq = causal["entry_time"].dt.tz_localize(None).dt.to_period("Q").astype(str)
    for quarter, g in causal.groupby(qq):
        m = metric(g, "net8_r")
        print(f"quarter {quarter}: N={m['N']:4d} PF={m['PF']:5.2f} EXP={m['EXP']:+.3f}R")

    tr = metric(causal[causal["split"].eq("TRAIN")], "net8_r")
    va = metric(causal[causal["split"].eq("VALID")], "net8_r")
    ho = metric(causal[causal["split"].eq("HOLDOUT")], "net8_r")
    hs = metric(causal[causal["split"].eq("HOLDOUT")], "stress12_r")
    survives = (
        tr["N"] >= 50 and tr["EXP"] > 0
        and va["N"] >= 50 and va["EXP"] > 0
        and ho["N"] >= 50 and ho["PF"] > 1 and ho["EXP"] > 0
        and hs["EXP"] >= 0
    )

    print("\n=== V1.6 VERDICT ===")
    print("SURVIVES_CAUSAL_DEDUP_CORRECTION" if survives else "FAILS_CAUSAL_DEDUP_CORRECTION")
    print("This is a recovery finalizer over already-produced V1.6 data, not a new holdout or parameter search.")
    print(f"Reports: {out}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
