#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_COST = 8.0
STRESS_COST = 12.0
RISK_PCT = 1.0
SEED = 731991
BOOT_REPS = 2000


def parse_args():
    p = argparse.ArgumentParser(description="Breakout/retest Profit V1.3: evaluate exactly one frozen VALID-selected hypothesis on untouched HOLDOUT")
    p.add_argument("--v1dir", default="/freqtrade/user_data/breakout_retest_profit_v1")
    p.add_argument("--v12dir", default="/freqtrade/user_data/breakout_retest_profit_v12")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v13")
    return p.parse_args()


def metric_from_r(r: pd.Series):
    r = pd.to_numeric(r, errors="coerce").dropna().astype(float)
    if r.empty:
        return {"N":0,"PF":np.nan,"WR":np.nan,"EXP":np.nan,"DD":np.nan,"ROI1":np.nan}
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    curve = r.cumsum()
    dd = float((curve.cummax() - curve).max()) if len(curve) else 0.0
    eq = 100.0
    for v in r:
        eq *= max(0.001, 1.0 + RISK_PCT / 100.0 * float(v))
    return {
        "N": int(len(r)),
        "PF": float(pf),
        "WR": float((r > 0).mean() * 100.0),
        "EXP": float(r.mean()),
        "DD": dd,
        "ROI1": float(eq - 100.0),
    }


def fmt(m):
    return f"N={m['N']:4d} PF={m['PF']:5.2f} WR={m['WR']:5.1f}% EXP={m['EXP']:+.3f}R DD={m['DD']:6.1f}R ROI1%={m['ROI1']:+7.1f}%"


def gate15(df):
    return pd.to_numeric(df.activity_score, errors="coerce") >= 1.5


def apply_frozen(df: pd.DataFrame, name: str):
    # These are the exact V1.2 hypotheses. No HOLDOUT-derived parameter is allowed here.
    specs = {
        "BREAK_4H": ("BREAK", 1.5, 144, lambda x: x.tf.eq("4h")),
        "BREAK_APPROACH3": ("BREAK", 1.5, 144, lambda x: pd.to_numeric(x.approach_no, errors="coerce") >= 3),
        "HOLD2_1H": ("HOLD2", 1.0, 144, lambda x: x.tf.eq("1h")),
        "HOLD2_4H": ("HOLD2", 1.0, 144, lambda x: x.tf.eq("4h")),
        "HOLD2_PERIOD30": ("HOLD2", 1.0, 144, lambda x: pd.to_numeric(x.period, errors="coerce").eq(30)),
        "HOLD2_APPROACH3": ("HOLD2", 1.0, 144, lambda x: pd.to_numeric(x.approach_no, errors="coerce") >= 3),
        "HOLD2_RISK80_160": ("HOLD2", 1.0, 144, lambda x: pd.to_numeric(x.risk_bps, errors="coerce").between(80, 160, inclusive="left")),
        "RETEST_RISK80_160": ("RETEST", 3.0, 144, lambda x: pd.to_numeric(x.risk_bps, errors="coerce").between(80, 160, inclusive="left")),
        "RETEST_RISK160P": ("RETEST", 3.0, 144, lambda x: pd.to_numeric(x.risk_bps, errors="coerce") >= 160),
        "FAKEOUT_RISK160P": ("FAKEOUT", 3.0, 48, lambda x: pd.to_numeric(x.risk_bps, errors="coerce") >= 160),
    }
    if name not in specs:
        raise RuntimeError(f"Unknown frozen hypothesis: {name}")
    setup, rr, hb, filt = specs[name]
    base = df[df.setup.eq(setup) & gate15(df)].copy()
    z = base[filt(base)].copy()
    rtag = str(rr).replace(".", "p")
    net_col = f"r_{rtag}_h{hb}_b"
    risk = pd.to_numeric(z.risk_bps, errors="coerce")
    net = pd.to_numeric(z[net_col], errors="coerce")
    z["net8_r"] = net
    z["gross_r"] = net + BASE_COST / risk
    z["stress12_r"] = net - (STRESS_COST - BASE_COST) / risk
    for cost in (0.0, 4.0, 8.0, 12.0, 16.0):
        z[f"net_{int(cost)}bps_r"] = z["gross_r"] - cost / risk
    return z.sort_values("entry_time").reset_index(drop=True), setup, rr, hb


def weekly_block_bootstrap(z: pd.DataFrame, col: str):
    x = z[["entry_time", col]].copy()
    x[col] = pd.to_numeric(x[col], errors="coerce")
    x = x.dropna()
    if len(x) < 20:
        return {"EXP_LO":np.nan,"EXP_HI":np.nan,"PF_LO":np.nan,"PF_HI":np.nan,"P_EXP_POS":np.nan}
    x["week"] = x.entry_time.dt.to_period("W").astype(str)
    blocks = [g[col].to_numpy(float) for _, g in x.groupby("week")]
    if len(blocks) < 4:
        return {"EXP_LO":np.nan,"EXP_HI":np.nan,"PF_LO":np.nan,"PF_HI":np.nan,"P_EXP_POS":np.nan}
    rng = np.random.default_rng(SEED)
    exps, pfs = [], []
    nb = len(blocks)
    for _ in range(BOOT_REPS):
        pick = rng.integers(0, nb, size=nb)
        r = np.concatenate([blocks[i] for i in pick])
        if len(r) == 0:
            continue
        exps.append(float(np.mean(r)))
        pos = float(r[r > 0].sum())
        neg = float(-r[r < 0].sum())
        pfs.append(pos / neg if neg > 0 else np.nan)
    exps = np.asarray(exps, float)
    pfs = np.asarray(pfs, float)
    pfs = pfs[np.isfinite(pfs)]
    return {
        "EXP_LO": float(np.quantile(exps, .025)),
        "EXP_HI": float(np.quantile(exps, .975)),
        "PF_LO": float(np.quantile(pfs, .025)) if len(pfs) else np.nan,
        "PF_HI": float(np.quantile(pfs, .975)) if len(pfs) else np.nan,
        "P_EXP_POS": float(np.mean(exps > 0)),
    }


def main():
    a = parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    sel_path = Path(a.v12dir) / "selected_for_holdout.csv"
    if not sel_path.exists():
        raise RuntimeError(f"Missing V1.2 frozen selection: {sel_path}")
    sel = pd.read_csv(sel_path)
    if len(sel) != 1 or "hypothesis" not in sel.columns:
        raise RuntimeError("selected_for_holdout.csv must contain exactly one selected hypothesis")
    name = str(sel.iloc[0].hypothesis)

    src = Path(a.v1dir) / "base_trades.csv"
    df = pd.read_csv(src)
    df["entry_time"] = pd.to_datetime(df.entry_time, utc=True)
    if "split" not in df.columns:
        raise RuntimeError("base_trades.csv has no split column")

    # This is the first script in the Profit V1 chain that reads HOLDOUT performance.
    hold = df[df.split.eq("HOLDOUT")].copy()
    z, setup, rr, hb = apply_frozen(hold, name)
    z.to_csv(out / "holdout_selected_trades.csv", index=False)

    print("=== BREAKOUT / RETEST PROFIT V1.3 — UNTOUCHED HOLDOUT ===")
    print("ONE frozen hypothesis only. No reselection, no new filter, no parameter tuning on HOLDOUT.")
    print(f"FROZEN={name} setup={setup} RR={rr} hold_bars={hb} activity>=1.5")
    print(f"HOLDOUT raw rows={len(hold):,} | selected trades={len(z):,} | time={hold.entry_time.min()} .. {hold.entry_time.max()}")

    print("\n=== PRIMARY HOLDOUT RESULT ===")
    gross = metric_from_r(z.gross_r)
    base = metric_from_r(z.net8_r)
    stress = metric_from_r(z.stress12_r)
    print(f"GROSS 0bps  | {fmt(gross)}")
    print(f"BASE  8bps  | {fmt(base)}")
    print(f"STRESS12bps | {fmt(stress)}")

    print("\n=== FIXED COST ROBUSTNESS ===")
    cost_rows = []
    for cost in (0, 4, 8, 12, 16):
        m = metric_from_r(z[f"net_{cost}bps_r"])
        cost_rows.append({"cost_bps": cost, **m})
        print(f"cost={cost:2d}bps | {fmt(m)}")
    pd.DataFrame(cost_rows).to_csv(out / "holdout_cost_robustness.csv", index=False)

    print("\n=== WEEKLY BLOCK BOOTSTRAP (8bps) ===")
    b = weekly_block_bootstrap(z, "net8_r")
    print(f"EXP 95%=[{b['EXP_LO']:+.3f}, {b['EXP_HI']:+.3f}]R | PF 95%=[{b['PF_LO']:.2f}, {b['PF_HI']:.2f}] | P(EXP>0)={b['P_EXP_POS']*100:.1f}%")

    print("\n=== CONCENTRATION / STABILITY (8bps) ===")
    pair_rows = []
    for pair, g in z.groupby("pair"):
        m = metric_from_r(g.net8_r)
        pair_rows.append({"pair": pair, **m})
    pp = pd.DataFrame(pair_rows).sort_values("N", ascending=False) if pair_rows else pd.DataFrame()
    if not pp.empty:
        pp.to_csv(out / "holdout_by_pair.csv", index=False)
        print("top pair counts: " + ", ".join(f"{r.pair}:{int(r.N)}" for r in pp.head(8).itertuples(index=False)))
        total = float(pp.N.sum())
        top3 = float(pp.head(3).N.sum()) / total * 100.0 if total else np.nan
        print(f"top3 pair trade-share={top3:.1f}% | pairs={len(pp)}")
        for r in pp.head(8).itertuples(index=False):
            print(f"pair {r.pair:22s} N={int(r.N):3d} PF={r.PF:5.2f} EXP={r.EXP:+.3f}R")
    for year, g in z.groupby(z.entry_time.dt.year):
        m = metric_from_r(g.net8_r)
        print(f"year {year}: {fmt(m)}")
    q = z.entry_time.dt.to_period("Q").astype(str)
    for quarter, g in z.groupby(q):
        m = metric_from_r(g.net8_r)
        print(f"quarter {quarter}: N={m['N']:3d} PF={m['PF']:5.2f} EXP={m['EXP']:+.3f}R")

    weeks = max((z.entry_time.max() - z.entry_time.min()).total_seconds() / (7*86400), 1.0) if len(z) > 1 else np.nan
    tpw = len(z) / weeks if np.isfinite(weeks) else np.nan
    print(f"trades/week={tpw:.2f}")

    verdict = "FAIL"
    if base["N"] >= 50 and base["PF"] > 1 and base["EXP"] > 0 and stress["EXP"] >= 0:
        verdict = "PASS"
    print("\n=== FROZEN HOLDOUT VERDICT ===")
    print(f"{verdict}: require N>=50, PF>1 and EXP>0 at 8bps, with 12bps expectancy >=0.")
    if verdict == "PASS":
        print("Next stage is portfolio realism / $100 capital / concurrency / leverage / funding; do not tune this holdout result.")
    else:
        print("Do not salvage this HOLDOUT by adding filters from its results. Return to TRAIN+VALID and revise mechanics, then reserve a new future holdout/prospective test.")
    print(f"Reports: {out}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
