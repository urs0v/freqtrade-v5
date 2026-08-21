#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc
from breakout_retest_profit_v1 import _activity15

BASE_COST_BPS = 8.0
THRESH = 1.5
RISK_MIN_BPS = 160.0
NET_COL = "r_3p0_h48_b"
STRESS_COL = "r_3p0_h48_s"


def parse_args():
    p = argparse.ArgumentParser(description="Profit V1.5: audit the V1 activity gate for entry-time causality")
    p.add_argument("--v1dir", default="/freqtrade/user_data/breakout_retest_profit_v1")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v15")
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


def _ns_utc(s: pd.Series) -> pd.Series:
    """Force one exact timezone-aware datetime dtype for pandas merge_asof.

    Pandas 3 / Python 3.14 can preserve CSV timestamps at microsecond precision
    while OHLCV preprocessing returns nanosecond precision. merge_asof requires
    identical key dtypes, so normalize both sides explicitly.
    """
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def activity_lookup(a15: pd.DataFrame, query: pd.DataFrame, offset_min: int, value_name: str):
    q = query[["_row", "entry_time"]].copy()
    q["entry_time"] = _ns_utc(q["entry_time"])
    q["query_time"] = _ns_utc(q["entry_time"] + pd.Timedelta(minutes=offset_min))
    a = a15[["signal_time", "activity_score", "natr_ratio30d", "qvol24_ratio30d"]].copy()
    a["signal_time"] = _ns_utc(a["signal_time"])
    q = q.sort_values("query_time")
    a = a.sort_values("signal_time")
    m = pd.merge_asof(
        q, a, left_on="query_time", right_on="signal_time",
        direction="backward", tolerance=pd.Timedelta("30min")
    )
    m = m.set_index("_row")
    return pd.DataFrame({
        value_name: m.activity_score,
        value_name + "_source_time": m.signal_time,
        value_name + "_natr": m.natr_ratio30d,
        value_name + "_qvol": m.qvol24_ratio30d,
    })


def frozen_mask(df: pd.DataFrame, activity_col: str):
    return (
        df.setup.eq("FAKEOUT")
        & (pd.to_numeric(df.risk_bps, errors="coerce") >= RISK_MIN_BPS)
        & (pd.to_numeric(df[activity_col], errors="coerce") >= THRESH)
    )


def main():
    a = parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    src = Path(a.v1dir) / "base_trades.csv"
    df = pd.read_csv(src)
    df["entry_time"] = _ns_utc(df["entry_time"])
    df["_row"] = np.arange(len(df), dtype=int)
    cfg = json.loads(Path(a.config).read_text())
    datadir = Path(a.datadir)

    print("=== BREAKOUT / RETEST PROFIT V1.5 — CAUSAL ACTIVITY TIMING AUDIT ===")
    print("No parameter tuning. Audits one implementation detail in V1: the activity gate was stored from entry_idx, whose 5m signal_time is entry_time+5m.")
    print("LEGACY_REBUILD intentionally reproduces that lookup. CAUSAL uses only 15m information whose signal_time <= actual entry_time.")
    print(f"rows={len(df):,} pairs={df.pair.nunique()} threshold={THRESH} risk>={RISK_MIN_BPS:.0f}bps frozen setup=FAKEOUT RR=3 hold=48")

    pieces = []
    for n, (pair, g) in enumerate(df.groupby("pair"), 1):
        raw15 = dc.load_tf(cfg, datadir, pair, "15m")
        if raw15.empty:
            print(f"activity {n:2d}/{df.pair.nunique()} {pair:24s} NO_15M", flush=True)
            continue
        x15 = dc.prep_ohlcv(raw15, 15)
        a15 = _activity15(x15)
        q = g[["_row", "entry_time"]].copy()
        legacy = activity_lookup(a15, q, 5, "legacy_rebuilt_activity")
        causal = activity_lookup(a15, q, 0, "causal_activity")
        z = legacy.join(causal, how="outer")
        z["pair"] = pair
        pieces.append(z)
        print(f"activity {n:2d}/{df.pair.nunique()} {pair:24s} rows={len(g)}", flush=True)

    if not pieces:
        raise RuntimeError("No activity rows reconstructed")
    act = pd.concat(pieces).sort_index()
    df = df.set_index("_row").join(act.drop(columns=["pair"]), how="left").reset_index()

    stored = pd.to_numeric(df.activity_score, errors="coerce")
    legacy = pd.to_numeric(df.legacy_rebuilt_activity, errors="coerce")
    causal = pd.to_numeric(df.causal_activity, errors="coerce")
    d = (stored - legacy).abs()
    finite = d[np.isfinite(d)]
    changed = (legacy - causal).abs() > 1e-12
    print("\n=== ACTIVITY LOOKUP SANITY ===")
    print(f"stored-vs-legacy finite={len(finite):,} median_abs={finite.median():.6g} max_abs={finite.max():.6g}")
    print(f"legacy-vs-causal changed rows={int(changed.sum()):,}/{int((np.isfinite(legacy)&np.isfinite(causal)).sum()):,} ({changed.mean()*100:.1f}% of all rows)")
    new_info = pd.to_datetime(df.legacy_rebuilt_activity_source_time, utc=True) > pd.to_datetime(df.causal_activity_source_time, utc=True)
    print(f"legacy lookup uses a newer completed 15m observation than entry-time causal lookup on {int(new_info.sum()):,} rows ({new_info.mean()*100:.1f}%).")

    df["old_selected"] = frozen_mask(df, "activity_score")
    df["legacy_selected"] = frozen_mask(df, "legacy_rebuilt_activity")
    df["causal_selected"] = frozen_mask(df, "causal_activity")
    df.to_csv(out / "causal_activity_audit_rows.csv", index=False)

    print("\n=== FROZEN SIGNAL: LEGACY VS ENTRY-TIME CAUSAL ===")
    rows = []
    for split in ("TRAIN", "VALID", "HOLDOUT"):
        base = df[df.split.eq(split)].sort_values("entry_time")
        old = base[base.old_selected]
        ca = base[base.causal_selected]
        oldm = metric(old, NET_COL); cam = metric(ca, NET_COL)
        olds = metric(old, STRESS_COL); cas = metric(ca, STRESS_COL)
        old_ids = set(old._row.tolist()); ca_ids = set(ca._row.tolist())
        inter = len(old_ids & ca_ids); union = len(old_ids | ca_ids)
        dropped = len(old_ids - ca_ids); added = len(ca_ids - old_ids)
        jac = inter / union if union else np.nan
        print(f"{split:7s} LEGACY {fmt(oldm)} stressEXP={olds['EXP']:+.3f}R")
        print(f"{split:7s} CAUSAL {fmt(cam)} stressEXP={cas['EXP']:+.3f}R | overlap={inter} dropped={dropped} added={added} Jaccard={jac:.3f}")
        rows += [
            {"split": split, "mode": "LEGACY", **oldm, "STRESS_EXP": olds["EXP"]},
            {"split": split, "mode": "CAUSAL", **cam, "STRESS_EXP": cas["EXP"]},
        ]
    pd.DataFrame(rows).to_csv(out / "causal_activity_metrics.csv", index=False)

    ca_all = df[df.causal_selected].sort_values("entry_time")
    print("\n=== CAUSAL SIGNAL CALENDAR STABILITY (8bps) ===")
    for year, g in ca_all.groupby(ca_all.entry_time.dt.year):
        print(f"year {year}: {fmt(metric(g, NET_COL))}")
    q = ca_all.entry_time.dt.tz_localize(None).dt.to_period("Q").astype(str)
    for quarter, g in ca_all.groupby(q):
        m = metric(g, NET_COL)
        print(f"quarter {quarter}: N={m['N']:4d} PF={m['PF']:5.2f} EXP={m['EXP']:+.3f}R")

    hold = df[df.split.eq("HOLDOUT") & df.causal_selected].sort_values("entry_time")
    hm = metric(hold, NET_COL); hs = metric(hold, STRESS_COL)
    verdict = "SURVIVES_CAUSAL_CORRECTION" if hm["N"] >= 50 and hm["PF"] > 1 and hm["EXP"] > 0 and hs["EXP"] >= 0 else "FAILS_CAUSAL_CORRECTION"
    print("\n=== V1.5 VERDICT ===")
    print(verdict)
    print("This is an implementation-correction audit, not a fresh holdout: V1.3 HOLDOUT has already been observed. If the corrected signal survives, its next proof must be prospective dry-run/live-paper data.")
    print(f"Reports: {out}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
