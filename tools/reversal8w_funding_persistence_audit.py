#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal funding-persistence audit for saved 8-week mirror selections")
    p.add_argument("--assets", default="/freqtrade/user_data/reversal8w_mirror_attribution/asset_mirror_attribution.csv")
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--output-dir", default="/freqtrade/user_data/reversal8w_funding_persistence")
    return p.parse_args()


def hac_mean_t(x: np.ndarray, lag: int = 8) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 20:
        return math.nan, math.nan
    mu = float(x.mean())
    e = x - mu
    gamma0 = float(np.dot(e, e) / n)
    s = gamma0
    L = min(lag, n - 1)
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        gam = float(np.dot(e[ell:], e[:-ell]) / n)
        s += 2.0 * w * gam
    se = math.sqrt(max(s, 0.0) / n)
    return mu, (mu / se if se > 0 else math.nan)


def fmt_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100*x:+.3f}%"


def funding_prefix(con: sqlite3.Connection, symbols: list[str]):
    out = {}
    for i, sym in enumerate(sorted(set(symbols)), 1):
        rows = con.execute(
            "SELECT event_time, rate FROM funding_events WHERE symbol=? ORDER BY event_time", (sym,)
        ).fetchall()
        if rows:
            t = np.array([r[0] for r in rows], dtype=np.int64)
            r = np.array([r[1] for r in rows], dtype=float)
            out[sym] = (t, np.concatenate([[0.0], np.cumsum(r)]))
        if i % 150 == 0 or i == len(set(symbols)):
            print(f"Funding histories: {i}/{len(set(symbols))}", flush=True)
    return out


def sum_window(pref, sym: str, start: pd.Timestamp, end_exclusive: pd.Timestamp) -> float:
    item = pref.get(sym)
    if item is None:
        return math.nan
    t, cs = item
    a = int(start.timestamp() * 1000)
    b = int(end_exclusive.timestamp() * 1000)
    i0 = int(np.searchsorted(t, a, side="left"))
    i1 = int(np.searchsorted(t, b, side="left"))
    return float(cs[i1] - cs[i0])


def main() -> int:
    cfg = parse_args()
    ap = Path(cfg.assets)
    dbp = Path(cfg.db)
    if not ap.exists():
        raise RuntimeError(f"Missing saved mirror attribution assets: {ap}")
    if not dbp.exists():
        raise RuntimeError(f"Missing funding DB: {dbp}")
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(ap, parse_dates=["date"])
    req = {"strategy","date","symbol","mirror_weight","mirror_leg","funding_sum","mirror_funding_contribution"}
    if not req.issubset(a.columns):
        raise RuntimeError(f"Asset CSV missing: {sorted(req-set(a.columns))}")
    a = a[a.strategy == "HIGH_VOL_REVERSAL"].copy()
    if a.empty:
        raise RuntimeError("No HIGH_VOL_REVERSAL rows")

    print("=== HIGH-VOL MIRROR FUNDING PERSISTENCE AUDIT ===")
    print("Causal diagnostic only. No strategy/filter optimization.")
    print("Predictor fixed ex ante: funding accumulated in the 7 days immediately BEFORE Sunday-close entry.")
    print("Target: funding accumulated during the following held week (Monday-Sunday).\n")

    con = sqlite3.connect(str(dbp), timeout=120)
    pref = funding_prefix(con, a.symbol.unique().tolist())

    pri = []
    fwd = []
    for r in a.itertuples(index=False):
        # date represents the completed Sunday daily close. The position starts immediately after it.
        entry_close = pd.Timestamp(r.date).tz_localize("UTC") if pd.Timestamp(r.date).tzinfo is None else pd.Timestamp(r.date).tz_convert("UTC")
        entry_exclusive = entry_close + pd.Timedelta(days=1)  # Monday 00:00, just after Sunday close proxy
        prior_start = entry_exclusive - pd.Timedelta(days=7)
        next_end = entry_exclusive + pd.Timedelta(days=7)
        pri.append(sum_window(pref, r.symbol, prior_start, entry_exclusive))
        fwd.append(sum_window(pref, r.symbol, entry_exclusive, next_end))
    a["prior7_funding"] = pri
    a["next7_funding_rebuilt"] = fwd
    a["year"] = a.date.dt.year
    a["funding_rebuild_error"] = a.next7_funding_rebuilt - a.funding_sum.astype(float)

    valid = a.dropna(subset=["prior7_funding","next7_funding_rebuilt"]).copy()
    # Asset-week persistence diagnostics.
    pearson = float(valid.prior7_funding.corr(valid.next7_funding_rebuilt, method="pearson"))
    spearman = float(valid.prior7_funding.corr(valid.next7_funding_rebuilt, method="spearman"))
    sign_hit = float((np.sign(valid.prior7_funding) == np.sign(valid.next7_funding_rebuilt)).mean())

    # Weekly cross-sectional rank IC, then HAC mean t across weeks.
    ics = []
    for dt, g in valid.groupby("date", sort=True):
        if len(g) >= 8:
            ic = g.prior7_funding.corr(g.next7_funding_rebuilt, method="spearman")
            if np.isfinite(ic):
                ics.append((dt, float(ic)))
    icdf = pd.DataFrame(ics, columns=["date","rank_ic"])
    ic_mean, ic_t = hac_mean_t(icdf.rank_ic.to_numpy(float), lag=8) if not icdf.empty else (math.nan, math.nan)

    # What a trader could know at entry: expected funding PnL if next week simply repeats prior week's funding.
    valid["predicted_carry_from_prior7"] = -valid.mirror_weight.astype(float) * valid.prior7_funding
    valid["realized_carry"] = -valid.mirror_weight.astype(float) * valid.next7_funding_rebuilt
    pred_corr = float(valid.predicted_carry_from_prior7.corr(valid.realized_carry, method="spearman"))

    overall = pd.DataFrame([{
        "asset_weeks": len(valid),
        "pearson_prior_to_next": pearson,
        "spearman_prior_to_next": spearman,
        "funding_sign_persistence": sign_hit,
        "weekly_rank_ic_mean": ic_mean,
        "weekly_rank_ic_hac_t": ic_t,
        "predicted_vs_realized_carry_spearman": pred_corr,
        "mean_abs_rebuild_error": float(valid.funding_rebuild_error.abs().mean()),
        "max_abs_rebuild_error": float(valid.funding_rebuild_error.abs().max()),
    }])

    leg_rows = []
    for (year, leg), g in valid.groupby(["year","mirror_leg"], sort=True):
        dates = max(int(g.date.nunique()), 1)
        leg_rows.append({
            "year": int(year), "leg": leg, "asset_weeks": len(g), "weeks": dates,
            "avg_prior7_funding_asset": float(g.prior7_funding.mean()),
            "avg_next7_funding_asset": float(g.next7_funding_rebuilt.mean()),
            "prior_negative_share": float((g.prior7_funding < 0).mean()),
            "next_negative_share": float((g.next7_funding_rebuilt < 0).mean()),
            "predicted_carry_per_week": float(g.predicted_carry_from_prior7.sum()/dates),
            "realized_carry_per_week": float(g.realized_carry.sum()/dates),
        })
    legs = pd.DataFrame(leg_rows)

    year_rows = []
    for year, g in valid.groupby("year", sort=True):
        dates = max(int(g.date.nunique()), 1)
        year_rows.append({
            "year": int(year), "asset_weeks": len(g), "weeks": dates,
            "predicted_carry_per_week": float(g.predicted_carry_from_prior7.sum()/dates),
            "realized_carry_per_week": float(g.realized_carry.sum()/dates),
            "prior_next_spearman": float(g.prior7_funding.corr(g.next7_funding_rebuilt, method="spearman")),
            "sign_persistence": float((np.sign(g.prior7_funding)==np.sign(g.next7_funding_rebuilt)).mean()),
        })
    years = pd.DataFrame(year_rows)

    # Concentration / spike audit on realized mirror carry.
    abs_c = valid.realized_carry.abs().sort_values(ascending=False)
    total_abs = float(abs_c.sum())
    n10 = max(1, int(math.ceil(len(abs_c)*0.10)))
    concentration = float(abs_c.iloc[:n10].sum()/total_abs) if total_abs > 0 else math.nan
    quant = valid.next7_funding_rebuilt.abs().quantile([0.5,0.9,0.95,0.99,0.999]).rename("abs_next7_funding").reset_index().rename(columns={"index":"quantile"})
    tops = valid.nlargest(20, "realized_carry")[["date","symbol","mirror_leg","mirror_weight","prior7_funding","next7_funding_rebuilt","predicted_carry_from_prior7","realized_carry"]]

    print("OVERALL PERSISTENCE")
    print(f"Asset-weeks: {len(valid):,}")
    print(f"Prior7 -> next7 funding Pearson:  {pearson:+.3f}")
    print(f"Prior7 -> next7 funding Spearman: {spearman:+.3f}")
    print(f"Funding sign persistence:         {100*sign_hit:.1f}%")
    print(f"Weekly cross-sectional Rank IC:   {ic_mean:+.3f} | HAC t={ic_t:+.2f}")
    print(f"Prior-predicted carry vs realized carry Spearman: {pred_corr:+.3f}")
    print(f"Funding rebuild error mean/max abs: {float(valid.funding_rebuild_error.abs().mean()):.3e} / {float(valid.funding_rebuild_error.abs().max()):.3e}")
    print(f"Top 10% asset-weeks share of ABS realized carry: {100*concentration:.1f}%")

    print("\nYEAR PERSISTENCE")
    yp = years.copy()
    for c in ["predicted_carry_per_week","realized_carry_per_week"]:
        yp[c] = yp[c].map(fmt_pct)
    yp["prior_next_spearman"] = yp.prior_next_spearman.map(lambda x:f"{x:+.3f}")
    yp["sign_persistence"] = yp.sign_persistence.map(lambda x:f"{100*x:.1f}%")
    print(yp.to_string(index=False))

    print("\nLEG PERSISTENCE")
    lp = legs.copy()
    for c in ["avg_prior7_funding_asset","avg_next7_funding_asset","predicted_carry_per_week","realized_carry_per_week"]:
        lp[c] = lp[c].map(fmt_pct)
    for c in ["prior_negative_share","next_negative_share"]:
        lp[c] = lp[c].map(lambda x:f"{100*x:.1f}%")
    print(lp.to_string(index=False))

    print("\nABS NEXT-WEEK FUNDING QUANTILES")
    qp = quant.copy(); qp["abs_next7_funding"] = qp.abs_next7_funding.map(fmt_pct)
    print(qp.to_string(index=False))

    print("\nTOP REALIZED CARRY ASSET-WEEKS")
    tp = tops.copy()
    for c in ["prior7_funding","next7_funding_rebuilt","predicted_carry_from_prior7","realized_carry"]:
        tp[c] = tp[c].map(fmt_pct)
    print(tp.to_string(index=False))

    print("\nINTERPRETATION GATES")
    gates = [
        ("Prior->next funding Spearman > 0.30", spearman > 0.30),
        ("Funding sign persistence > 60%", sign_hit > 0.60),
        ("Weekly funding Rank IC HAC t > 2", ic_t > 2.0),
        ("Prior-predicted vs realized carry Spearman > 0.30", pred_corr > 0.30),
        ("Top 10% asset-weeks < 60% of abs carry", concentration < 0.60),
    ]
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if all(ok for _,ok in gates):
        print("[KEEP CARRY MECHANISM] Funding component is materially persistent and observable before entry. Next test should be a causal carry-aware portfolio, not a fitted regime filter.")
    else:
        print("[DO NOT CREDIT FUNDING YET] Realized mirror funding is not sufficiently predictable/stable from the prior week; treat it as post-hoc until another causal predictor is justified.")

    overall.to_csv(out/"overall.csv", index=False)
    years.to_csv(out/"year_persistence.csv", index=False)
    legs.to_csv(out/"leg_persistence.csv", index=False)
    quant.to_csv(out/"funding_quantiles.csv", index=False)
    tops.to_csv(out/"top_realized_carry.csv", index=False)
    icdf.to_csv(out/"weekly_rank_ic.csv", index=False)
    valid.to_csv(out/"asset_week_audit.csv", index=False)
    print(f"\nSaved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
