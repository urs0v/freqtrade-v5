#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

BASE_COST = 8.0
STRESS_COST = 12.0


def parse_args():
    p = argparse.ArgumentParser(description="Breakout/retest Profit V1.2: validate TRAIN-derived structural hypotheses on VALID only")
    p.add_argument("--v1dir", default="/freqtrade/user_data/breakout_retest_profit_v1")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v12")
    return p.parse_args()


def gate15(df):
    return pd.to_numeric(df.activity_score, errors="coerce") >= 1.5


def metric(g: pd.DataFrame, col: str):
    r = pd.to_numeric(g[col], errors="coerce").dropna()
    if r.empty:
        return {"N":0,"PF":np.nan,"WR":np.nan,"EXP":np.nan,"DD":np.nan}
    pos = float(r[r>0].sum()); neg = float(-r[r<0].sum())
    pf = pos/neg if neg>0 else np.inf
    curve = r.cumsum(); dd = float((curve.cummax()-curve).max()) if len(r) else 0.0
    return {"N":int(len(r)),"PF":float(pf),"WR":float((r>0).mean()*100),"EXP":float(r.mean()),"DD":dd}


def gross_col(df: pd.DataFrame, net_col: str):
    risk = pd.to_numeric(df.risk_bps, errors="coerce")
    net = pd.to_numeric(df[net_col], errors="coerce")
    return net + BASE_COST / risk


def stress_col(df: pd.DataFrame, net_col: str):
    risk = pd.to_numeric(df.risk_bps, errors="coerce")
    net = pd.to_numeric(df[net_col], errors="coerce")
    return net - (STRESS_COST-BASE_COST) / risk


def setup_base(df, setup):
    return df[df.setup.eq(setup) & gate15(df)].copy()


def hypotheses(df):
    # All hypotheses below were declared from V1.1 TRAIN diagnostics before reading VALID.
    specs = [
        ("BREAK_4H", "BREAK", 1.5, 144, lambda x: x.tf.eq("4h")),
        ("BREAK_APPROACH3", "BREAK", 1.5, 144, lambda x: pd.to_numeric(x.approach_no, errors="coerce") >= 3),
        ("HOLD2_1H", "HOLD2", 1.0, 144, lambda x: x.tf.eq("1h")),
        ("HOLD2_4H", "HOLD2", 1.0, 144, lambda x: x.tf.eq("4h")),
        ("HOLD2_PERIOD30", "HOLD2", 1.0, 144, lambda x: pd.to_numeric(x.period, errors="coerce").eq(30)),
        ("HOLD2_APPROACH3", "HOLD2", 1.0, 144, lambda x: pd.to_numeric(x.approach_no, errors="coerce") >= 3),
        ("HOLD2_RISK80_160", "HOLD2", 1.0, 144, lambda x: pd.to_numeric(x.risk_bps, errors="coerce").between(80,160, inclusive="left")),
        ("RETEST_RISK80_160", "RETEST", 3.0, 144, lambda x: pd.to_numeric(x.risk_bps, errors="coerce").between(80,160, inclusive="left")),
        ("RETEST_RISK160P", "RETEST", 3.0, 144, lambda x: pd.to_numeric(x.risk_bps, errors="coerce") >= 160),
        ("FAKEOUT_RISK160P", "FAKEOUT", 3.0, 48, lambda x: pd.to_numeric(x.risk_bps, errors="coerce") >= 160),
    ]
    out=[]
    for name, setup, rr, hb, filt in specs:
        rtag=str(rr).replace(".","p")
        col=f"r_{rtag}_h{hb}_b"
        base=setup_base(df, setup)
        z=base[filt(base)].copy()
        z["gross_r"] = gross_col(z,col)
        z["stress_r"] = stress_col(z,col)
        out.append((name,setup,rr,hb,z,col))
    return out


def fmt(m):
    return f"N={m['N']:4d} PF={m['PF']:5.2f} WR={m['WR']:5.1f}% EXP={m['EXP']:+.3f}R DD={m['DD']:6.1f}R"


def main():
    a=parse_args(); out=Path(a.outdir); out.mkdir(parents=True,exist_ok=True)
    src=Path(a.v1dir)/"base_trades.csv"
    df=pd.read_csv(src)
    df["entry_time"]=pd.to_datetime(df.entry_time,utc=True)
    if "split" not in df.columns:
        raise RuntimeError("base_trades.csv has no split column")
    # HOLDOUT is never included in any dataframe used below.
    work=df[df.split.isin(["TRAIN","VALID"])].copy()
    print("=== BREAKOUT / RETEST PROFIT V1.2 — VALIDATION ===")
    print("TRAIN-derived hypotheses only. VALID is now used for confirmation. HOLDOUT is not read for performance.")
    print(f"TRAIN rows={sum(work.split.eq('TRAIN')):,} VALID rows={sum(work.split.eq('VALID')):,}")

    rows=[]
    print("\n=== PREDECLARED HYPOTHESES: TRAIN -> VALID ===")
    for name,setup,rr,hb,z,col in hypotheses(work):
        tr=z[z.split.eq("TRAIN")].sort_values("entry_time")
        va=z[z.split.eq("VALID")].sort_values("entry_time")
        for split,g in (("TRAIN",tr),("VALID",va)):
            mb=metric(g,col); mg=metric(g,"gross_r"); ms=metric(g,"stress_r")
            rows.append({"hypothesis":name,"setup":setup,"rr":rr,"hold_bars":hb,"split":split,
                         "N":mb["N"],"PF":mb["PF"],"WR":mb["WR"],"EXP":mb["EXP"],"DD":mb["DD"],
                         "GROSS_PF":mg["PF"],"GROSS_EXP":mg["EXP"],"STRESS_PF":ms["PF"],"STRESS_EXP":ms["EXP"]})
        mt=metric(tr,col); mv=metric(va,col); sv=metric(va,"stress_r")
        ok = mv["N"]>=50 and np.isfinite(mv["PF"]) and mv["PF"]>1.0 and mv["EXP"]>0 and mt["EXP"]>0
        print(f"{name:20s} TRAIN {fmt(mt)} | VALID {fmt(mv)} | stress EXP={sv['EXP']:+.3f}R | {'PASS' if ok else 'FAIL'}")

    rep=pd.DataFrame(rows); rep.to_csv(out/"validation_metrics.csv",index=False)
    v=rep[rep.split.eq("VALID")].copy()
    t=rep[rep.split.eq("TRAIN")][["hypothesis","EXP"]].rename(columns={"EXP":"TRAIN_EXP"})
    v=v.merge(t,on="hypothesis",how="left")
    v["pass"]=(v.N>=50)&(v.PF>1)&(v.EXP>0)&(v.TRAIN_EXP>0)
    # Selection is predeclared and uses VALID only; HOLDOUT remains untouched.
    cand=v[v["pass"]].copy()
    print("\n=== VALIDATION SELECTION ===")
    if cand.empty:
        print("NO_VALIDATED_HYPOTHESIS")
        print("None of the broad TRAIN mechanisms replicated on VALID after 8bps. Do not open HOLDOUT; revise mechanics first.")
    else:
        cand["score"]=cand.EXP*np.sqrt(np.minimum(cand.N,300))*np.minimum(cand.PF,2.0)
        cand=cand.sort_values(["score","EXP","N"],ascending=False)
        for r in cand.itertuples(index=False):
            print(f"PASS {r.hypothesis:20s} | N={int(r.N):4d} PF={r.PF:.2f} EXP={r.EXP:+.3f}R stressPF={r.STRESS_PF:.2f} stressEXP={r.STRESS_EXP:+.3f}R score={r.score:.3f}")
        ch=cand.iloc[0]
        print(f"SELECTED_FOR_HOLDOUT={ch.hypothesis}")
        pd.DataFrame([ch]).to_csv(out/"selected_for_holdout.csv",index=False)

    print("\n=== DECISION RULE ===")
    print("A mechanism must keep positive TRAIN and VALID expectancy after 8bps with VALID N>=50. No new filters are created from VALID.")
    print("Only a frozen VALID survivor may be evaluated on HOLDOUT next. If none survives, change the breakout/retest mechanics before touching HOLDOUT.")
    print(f"Reports: {out}")
    print("=== DONE ===")
    return 0

if __name__=="__main__":
    raise SystemExit(main())
