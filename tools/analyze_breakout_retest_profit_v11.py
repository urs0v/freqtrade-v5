#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

BASE_COST_BPS = 8.0
RRS = (1.0, 1.5, 2.0, 3.0)
HOLDS = (48, 144)
SETUPS = ("BREAK", "HOLD2", "RETEST", "FAKEOUT")
GATES = ("ALL", "ACTIVE_1P2", "ACTIVE_1P5")
COST_SWEEP = (0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0)


def parse_args():
    p = argparse.ArgumentParser(description="Breakout/retest Profit V1.1 diagnostics — TRAIN only")
    p.add_argument("--v1dir", default="/freqtrade/user_data/breakout_retest_profit_v1")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v11")
    return p.parse_args()


def gate_mask(df: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "ALL":
        return pd.Series(True, index=df.index)
    x = pd.to_numeric(df["activity_score"], errors="coerce")
    return x >= (1.2 if gate == "ACTIVE_1P2" else 1.5)


def metrics(r: pd.Series) -> dict:
    x = pd.to_numeric(r, errors="coerce").dropna()
    if x.empty:
        return {"N": 0, "PF": np.nan, "WR": np.nan, "EXP": np.nan, "DD": np.nan}
    pos = float(x[x > 0].sum())
    neg = float(-x[x < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    curve = x.cumsum()
    dd = float((curve.cummax() - curve).max()) if len(curve) else 0.0
    return {
        "N": int(len(x)),
        "PF": float(pf),
        "WR": float((x > 0).mean() * 100.0),
        "EXP": float(x.mean()),
        "DD": dd,
    }


def config_col(rr: float, hb: int) -> str:
    return f"r_{str(rr).replace('.', 'p')}_h{hb}_b"


def gross_r(df: pd.DataFrame, col: str) -> pd.Series:
    rb = pd.to_numeric(df[col], errors="coerce")
    risk = pd.to_numeric(df["risk_bps"], errors="coerce")
    return rb + BASE_COST_BPS / risk


def net_r_at_cost(df: pd.DataFrame, col: str, cost_bps: float) -> pd.Series:
    g = gross_r(df, col)
    risk = pd.to_numeric(df["risk_bps"], errors="coerce")
    return g - float(cost_bps) / risk


def breakeven_cost_bps(df: pd.DataFrame, col: str) -> float:
    g = gross_r(df, col)
    risk = pd.to_numeric(df["risk_bps"], errors="coerce")
    z = pd.DataFrame({"g": g, "inv": 1.0 / risk}).replace([np.inf, -np.inf], np.nan).dropna()
    if z.empty or float(z["inv"].mean()) <= 0:
        return np.nan
    return float(z["g"].mean() / z["inv"].mean())


def all_configs(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for setup in SETUPS:
        for gate in GATES:
            base = train[train.setup.eq(setup) & gate_mask(train, gate)]
            for rr in RRS:
                for hb in HOLDS:
                    col = config_col(rr, hb)
                    gm = metrics(gross_r(base, col))
                    b8 = metrics(net_r_at_cost(base, col, 8.0))
                    s12 = metrics(net_r_at_cost(base, col, 12.0))
                    rows.append({
                        "setup": setup, "gate": gate, "rr": rr, "hold_bars": hb,
                        "N": b8["N"],
                        "GROSS_PF": gm["PF"], "GROSS_EXP": gm["EXP"],
                        "BASE_PF": b8["PF"], "BASE_EXP": b8["EXP"],
                        "STRESS_PF": s12["PF"], "STRESS_EXP": s12["EXP"],
                        "BE_COST_BPS": breakeven_cost_bps(base, col),
                    })
    return pd.DataFrame(rows)


def fmt(m: dict) -> str:
    return f"N={m['N']:5d} PF={m['PF']:5.2f} WR={m['WR']:5.1f}% EXP={m['EXP']:+.3f}R DD={m['DD']:6.1f}R"


def print_cost_sweep(cfgs: pd.DataFrame, train: pd.DataFrame):
    print("\n=== TRAIN COST SENSITIVITY — BEST CONFIG AT EACH COST ===")
    for cost in COST_SWEEP:
        best = None
        for r in cfgs.itertuples(index=False):
            g = train[train.setup.eq(r.setup) & gate_mask(train, r.gate)]
            m = metrics(net_r_at_cost(g, config_col(float(r.rr), int(r.hold_bars)), cost))
            if m["N"] < 100 or not np.isfinite(m["EXP"]):
                continue
            key = (m["EXP"], m["PF"] if np.isfinite(m["PF"]) else 999.0)
            if best is None or key > best[0]:
                best = (key, r, m)
        if best is None:
            print(f"cost={cost:4.1f}bps | no N>=100 config")
            continue
        _, r, m = best
        print(f"cost={cost:4.1f}bps | {r.setup:7s} {r.gate:11s} RR={r.rr:<3} hold={int(r.hold_bars):3d} | {fmt(m)}")


def best_per_setup(cfgs: pd.DataFrame) -> dict[str, dict]:
    out = {}
    for setup in SETUPS:
        g = cfgs[(cfgs.setup.eq(setup)) & (cfgs.N >= 100)].sort_values(["BASE_EXP", "BASE_PF"], ascending=False)
        if len(g):
            out[setup] = g.iloc[0].to_dict()
    return out


def subset_report(train: pd.DataFrame, cfg: dict, key: str, values: list[tuple[str, pd.Series]]):
    setup = str(cfg["setup"])
    gate = str(cfg["gate"])
    rr = float(cfg["rr"])
    hb = int(cfg["hold_bars"])
    col = config_col(rr, hb)
    base = train[train.setup.eq(setup) & gate_mask(train, gate)]
    print(f"\n{setup} diagnostic slices using its best TRAIN config: gate={gate} RR={rr} hold={hb}")
    for label, mask in values:
        g = base[mask.reindex(base.index, fill_value=False)]
        if len(g) < 30:
            continue
        mg = metrics(gross_r(g, col))
        mn = metrics(net_r_at_cost(g, col, 8.0))
        print(f"  {key}={label:14s} | gross {fmt(mg)} | net8 {fmt(mn)}")


def main() -> int:
    a = parse_args()
    v1dir = Path(a.v1dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    path = v1dir / "base_trades.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}; run Profit V1 first")
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df.entry_time, utc=True, errors="coerce")
    train = df[df.split.eq("TRAIN")].copy()
    if train.empty:
        raise RuntimeError("No TRAIN rows in base_trades.csv")

    # Never inspect VALID/HOLDOUT in this diagnostic. Keep them available for later confirmation.
    print("=== BREAKOUT / RETEST PROFIT V1.1 — FAILURE DIAGNOSTICS ===")
    print("TRAIN ONLY. VALID and HOLDOUT are intentionally not read for performance diagnostics.")
    print("Purpose: determine whether V1 failed because the raw edge is absent, costs kill a small edge, or broad inherited mechanics mix good/bad regimes.")
    print(f"TRAIN rows={len(train):,} pairs={train.pair.nunique()} time={train.entry_time.min()} .. {train.entry_time.max()}")

    risk = pd.to_numeric(train.risk_bps, errors="coerce")
    print("\n=== RISK / DATA SANITY ===")
    print(f"finite risk={risk.notna().mean()*100:.1f}% | median={risk.median():.1f}bps p10={risk.quantile(.1):.1f} p90={risk.quantile(.9):.1f} p99={risk.quantile(.99):.1f}")
    for setup, g in train.groupby("setup"):
        r = pd.to_numeric(g.risk_bps, errors="coerce")
        print(f"{setup:7s} N={len(g):5d} risk med={r.median():6.1f}bps p90={r.quantile(.9):6.1f}")

    cfgs = all_configs(train)
    cfgs.to_csv(outdir / "train_config_diagnostics.csv", index=False)

    print("\n=== BEST TRAIN CONFIGS EVEN IF NEGATIVE ===")
    show = cfgs[cfgs.N >= 100].sort_values(["BASE_EXP", "BASE_PF"], ascending=False).head(12)
    for r in show.itertuples(index=False):
        print(
            f"{r.setup:7s} {r.gate:11s} RR={r.rr:<3} hold={int(r.hold_bars):3d} N={int(r.N):5d} | "
            f"gross PF={r.GROSS_PF:5.2f} EXP={r.GROSS_EXP:+.3f}R | "
            f"net8 PF={r.BASE_PF:5.2f} EXP={r.BASE_EXP:+.3f}R | "
            f"net12 PF={r.STRESS_PF:5.2f} EXP={r.STRESS_EXP:+.3f}R | break-even cost={r.BE_COST_BPS:+.2f}bps"
        )

    print("\n=== BEST CONFIG PER SETUP ===")
    bests = best_per_setup(cfgs)
    for setup in SETUPS:
        r = bests.get(setup)
        if not r:
            print(f"{setup:7s}: no N>=100 config")
            continue
        print(
            f"{setup:7s} {r['gate']:11s} RR={r['rr']:<3} hold={int(r['hold_bars']):3d} N={int(r['N']):5d} | "
            f"gross PF={r['GROSS_PF']:.2f} EXP={r['GROSS_EXP']:+.3f}R | "
            f"net8 PF={r['BASE_PF']:.2f} EXP={r['BASE_EXP']:+.3f}R | BE={r['BE_COST_BPS']:+.2f}bps"
        )

    print_cost_sweep(cfgs, train)

    # Broad, predeclared diagnostic slices only. These are not promoted as strategy filters here.
    tf_values = [(str(v), train.tf.astype(str).eq(str(v))) for v in sorted(train.tf.astype(str).dropna().unique())]
    period_values = [(str(int(v)), pd.to_numeric(train.period, errors="coerce").eq(v)) for v in sorted(pd.to_numeric(train.period, errors="coerce").dropna().unique())]
    side_values = [("LONG", pd.to_numeric(train.side, errors="coerce") > 0), ("SHORT", pd.to_numeric(train.side, errors="coerce") < 0)]
    app = pd.to_numeric(train.approach_no, errors="coerce")
    approach_values = [("1", app.eq(1)), ("2", app.eq(2)), ("3+", app.ge(3))]
    rb = pd.to_numeric(train.risk_bps, errors="coerce")
    risk_values = [("<40bps", rb.lt(40)), ("40-80bps", rb.ge(40) & rb.lt(80)), ("80-160bps", rb.ge(80) & rb.lt(160)), ("160+bps", rb.ge(160))]

    print("\n=== BROAD TRAIN SLICES — DIAGNOSTIC ONLY, NOT FILTER SELECTION ===")
    for setup, cfg in bests.items():
        subset_report(train, cfg, "TF", tf_values)
        subset_report(train, cfg, "PERIOD", period_values)
        subset_report(train, cfg, "SIDE", side_values)
        subset_report(train, cfg, "APPROACH", approach_values)
        subset_report(train, cfg, "RISK", risk_values)

    print("\n=== INTERPRETATION RULE ===")
    print("1) If even 0bps/gross best configs are negative, the current level+entry+exit mechanics have no raw TRAIN edge; fees are not the primary problem.")
    print("2) If gross is positive but break-even cost is well below realistic round-trip cost, the edge is too small/tight-stop sensitive and needs better entry quality, not leverage.")
    print("3) If one broad structural slice (TF/period/side/approach/risk) is materially positive with large N, that is a mechanism hypothesis to test on VALID — not a parameter to declare profitable from TRAIN.")
    print("4) Do not inspect HOLDOUT until a revised mechanism is frozen and survives VALID.")
    print(f"Reports: {outdir}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
