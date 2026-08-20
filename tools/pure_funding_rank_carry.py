#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import reversal8w_perp_test as rev
import cttrend_research_v3 as v3


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal weekly cross-sectional funding-rank carry on Binance USD-M perps")
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--universe", type=int, default=70)
    p.add_argument("--side-cost-bps", type=float, default=7.0)
    p.add_argument("--min-cross-section", type=int, default=25)
    p.add_argument("--output-dir", default="/freqtrade/user_data/pure_funding_rank_carry")
    return p.parse_args()


def funding_range(pref, sym: str, start: pd.Timestamp, end_day: pd.Timestamp) -> tuple[float, bool]:
    item = pref.get(sym)
    if item is None:
        return 0.0, False
    t, cs = item
    a = int(start.timestamp() * 1000)
    b = int((end_day + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    i0 = int(np.searchsorted(t, a, side="left"))
    i1 = int(np.searchsorted(t, b, side="right"))
    return float(cs[i1] - cs[i0]), True


def rank_weights(prior: pd.Series) -> pd.Series:
    """Linear cross-sectional rank weights: long low funding, short high funding.

    Uses every eligible name, no quantile threshold. Ranks are centered so net
    exposure is zero, then normalized to 1x gross exposure.
    """
    n = len(prior)
    if n < 2:
        return pd.Series(0.0, index=prior.index)
    ranks = prior.rank(method="average")
    score = ranks - (n + 1.0) / 2.0
    raw = -score.astype(float)
    gross = float(raw.abs().sum())
    if gross <= 0:
        return pd.Series(0.0, index=prior.index)
    return raw / gross


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100.0*x:+.2f}%"


def main() -> int:
    cfg = parse_args()
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=== PURE CAUSAL FUNDING-RANK CARRY ===")
    print(f"Evaluation: {start.date()} -> {end.date()}")
    print(f"Universe: point-in-time top {cfg.universe} USD-M perps by lagged 30d quote volume")
    print("Predictor: funding accumulated in the 7 days known by Sunday-close entry")
    print("Portfolio: linear cross-sectional rank, long low funding / short high funding")
    print("No threshold, no momentum, no magnitude fit, 1x gross / ~0 net, weekly Sunday-close rebalance")
    print(f"Costs: {cfg.side_cost_bps:.1f} bps per changed notional side; realized funding included\n")

    con = sqlite3.connect(cfg.db, timeout=120)
    daily = rev.load_daily(con, start, end)
    sun = daily[daily.date.dt.dayofweek == 6].copy()
    sun = v3.attach_forward_exits(sun, daily)
    sun = sun[(sun.date >= start) & (sun.date <= end)]
    sun = sun.replace([np.inf, -np.inf], np.nan)
    sun = sun.dropna(subset=["liq_30d", "close"])
    sun = sun[(sun.history_days >= 60) & (sun.liq_30d > 0)]

    pref = rev.funding_prefix(con, daily.symbol.unique().tolist(), start, end)
    side_cost = cfg.side_cost_bps / 10_000.0
    prev: dict[str, float] = {}
    weeks: list[dict] = []
    assets: list[dict] = []

    dates = sorted(pd.Timestamp(x) for x in sun.date.unique())
    for j, dt in enumerate(dates, 1):
        cross = sun[sun.date == dt].copy()
        if len(cross) < cfg.min_cross_section:
            continue
        cross = cross.nlargest(min(cfg.universe, len(cross)), "liq_30d").copy()
        if len(cross) < cfg.min_cross_section:
            continue

        prior_vals = []
        prior_known = []
        for r in cross.itertuples(index=False):
            # Entry at complete Sunday close. Prior window is Monday-Sunday and fully known.
            fr, ok = funding_range(pref, r.symbol, dt - pd.Timedelta(days=6), dt)
            prior_vals.append(fr)
            prior_known.append(ok)
        cross["prior7_funding"] = prior_vals
        cross["prior_known"] = prior_known
        cross = cross[cross.prior_known].copy()
        if len(cross) < cfg.min_cross_section:
            continue

        cross["weight"] = rank_weights(cross.prior7_funding)
        cross = cross[cross.weight.abs() > 1e-15].copy()
        if len(cross) < cfg.min_cross_section:
            continue

        target = dict(zip(cross.symbol, cross.weight.astype(float)))
        turnover = sum(abs(target.get(s, 0.0) - prev.get(s, 0.0)) for s in set(target) | set(prev))
        price = 0.0
        funding = 0.0
        forced_notional = 0.0
        known_next = 0

        for r in cross.itertuples(index=False):
            if not np.isfinite(r.fwd_ret):
                raise RuntimeError(f"Selected asset lacks unbiased weekly exit at {dt.date()}: {r.symbol}")
            actual = pd.Timestamp(r.actual_exit_date)
            # Held position starts after Sunday close: first held funding day is Monday.
            next_fr, ok = funding_range(pref, r.symbol, dt + pd.Timedelta(days=1), actual)
            known_next += int(ok)
            w = float(r.weight)
            pc = w * float(r.fwd_ret)
            fc = -w * next_fr
            price += pc
            funding += fc
            forced = bool(r.forced_exit)
            if forced:
                forced_notional += abs(w)
            assets.append({
                "date": dt,
                "symbol": r.symbol,
                "weight": w,
                "prior7_funding": float(r.prior7_funding),
                "next7_funding": float(next_fr),
                "price_contribution": pc,
                "funding_contribution": fc,
                "pre_cost_contribution": pc + fc,
                "forced_exit": forced,
            })

        if forced_notional > 0:
            turnover += forced_notional
        cost = turnover * side_cost
        net = price + funding - cost
        weeks.append({
            "date": dt,
            "positions": len(target),
            "gross_exposure": sum(abs(w) for w in target.values()),
            "net_exposure": sum(target.values()),
            "turnover": turnover,
            "price_return": price,
            "funding_return": funding,
            "cost_return": cost,
            "net_return": net,
            "next_funding_coverage": known_next / len(target),
        })
        forced_syms = set(cross.loc[cross.forced_exit.astype(bool), "symbol"])
        prev = {s: w for s, w in target.items() if s not in forced_syms}
        if j % 25 == 0 or j == len(dates):
            print(f"Weekly carry pass: {j}/{len(dates)} | {dt.date()}", flush=True)

    wdf = pd.DataFrame(weeks).sort_values("date").reset_index(drop=True)
    adf = pd.DataFrame(assets)
    if wdf.empty:
        raise RuntimeError("No carry portfolio weeks")

    if prev:
        terminal = sum(abs(w) for w in prev.values())
        i = wdf.index[-1]
        wdf.loc[i, "turnover"] += terminal
        wdf.loc[i, "cost_return"] += terminal * side_cost
        wdf.loc[i, "net_return"] -= terminal * side_cost

    m = rev.perf(wdf.net_return)
    m.update({
        "weeks": len(wdf),
        "avg_positions": float(wdf.positions.mean()),
        "avg_gross_exposure": float(wdf.gross_exposure.mean()),
        "avg_turnover": float(wdf.turnover.mean()),
        "avg_price": float(wdf.price_return.mean()),
        "avg_funding": float(wdf.funding_return.mean()),
        "avg_cost": float(wdf.cost_return.mean()),
        "avg_net": float(wdf.net_return.mean()),
        "funding_coverage": float(wdf.next_funding_coverage.mean()),
    })

    year_rows = []
    for year, g in wdf.groupby(wdf.date.dt.year):
        mm = rev.perf(g.net_return)
        year_rows.append({
            "year": int(year), **mm,
            "avg_price": float(g.price_return.mean()),
            "avg_funding": float(g.funding_return.mean()),
            "avg_cost": float(g.cost_return.mean()),
        })
    ydf = pd.DataFrame(year_rows)

    post = wdf[(wdf.date >= pd.Timestamp("2026-04-01", tz="UTC")) & (wdf.date <= pd.Timestamp("2026-07-31", tz="UTC"))]
    postm = rev.perf(post.net_return) if len(post) else {}

    # Severe tail stress: erase realized funding PnL from the 10% largest absolute
    # funding-contribution asset-weeks. Price and transaction costs are untouched.
    cutoff = math.nan
    stress = None
    sw = pd.DataFrame()
    if not adf.empty:
        cutoff = float(adf.funding_contribution.abs().quantile(0.90))
        adf["tail10"] = adf.funding_contribution.abs() >= cutoff
        sf = adf.assign(stress_funding=np.where(adf.tail10, 0.0, adf.funding_contribution)) \
                .groupby("date", as_index=False).stress_funding.sum()
        sw = wdf.merge(sf, on="date", how="left")
        sw["stress_funding"] = sw.stress_funding.fillna(0.0)
        sw["stress_net"] = sw.price_return + sw.stress_funding - sw.cost_return
        stress = rev.perf(sw.stress_net)

    # Concentration diagnostic across symbols.
    top_share = math.nan
    if not adf.empty:
        sym = adf.groupby("symbol", as_index=False).pre_cost_contribution.sum()
        pos = sym[sym.pre_cost_contribution > 0].pre_cost_contribution
        if len(pos) and float(pos.sum()) > 0:
            top_share = float(pos.max() / pos.sum())

    print("\nMAIN RESULT")
    print(f"Weeks={len(wdf)} | avg positions={wdf.positions.mean():.1f} | avg gross={wdf.gross_exposure.mean():.3f}x | avg turnover={wdf.turnover.mean():.3f}x")
    print(f"$100 -> ${m['ending_equity']:.2f} | total={pct(m['total_return'])} | CAGR={pct(m['cagr'])}")
    print(f"WR={100*m['weekly_wr']:.2f}% | PF={m['profit_factor']:.3f} | Sharpe={m['sharpe']:.3f} | Sortino={m['sortino']:.3f} | MDD={pct(m['max_drawdown'])}")
    print(f"Avg/week price={pct(m['avg_price'])} funding={pct(m['avg_funding'])} cost={pct(m['avg_cost'])} net={pct(m['avg_net'])}")

    print("\nYEAR BREAKDOWN")
    yp = ydf.copy()
    for c in ["total_return", "cagr", "max_drawdown", "weekly_wr", "avg_price", "avg_funding", "avg_cost"]:
        yp[c] = yp[c].map(pct)
    for c in ["profit_factor", "sharpe", "sortino"]:
        yp[c] = yp[c].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "nan")
    print(yp.to_string(index=False))

    print("\nPOST-PAPER APR-JUL 2026")
    if postm:
        print(f"return={pct(postm['total_return'])} PF={postm['profit_factor']:.3f} Sharpe={postm['sharpe']:.3f} MDD={pct(postm['max_drawdown'])}")
    else:
        print("no observations")

    print("\nTAIL-CONCENTRATION STRESS")
    print(f"Asset-week funding-contribution |90th percentile| cutoff: {pct(cutoff)}")
    if stress:
        print("Top 10% absolute realized funding contributions are set to ZERO; price and costs unchanged.")
        print(f"$100 -> ${stress['ending_equity']:.2f} | total={pct(stress['total_return'])} | CAGR={pct(stress['cagr'])} | PF={stress['profit_factor']:.3f} | Sharpe={stress['sharpe']:.3f} | MDD={pct(stress['max_drawdown'])}")
    print(f"Top positive symbol share of positive pre-cost contribution: {100*top_share:.2f}%" if np.isfinite(top_share) else "Top positive symbol share: nan")

    yr = ydf.set_index("year") if not ydf.empty else pd.DataFrame()
    years_ok = all(y in yr.index and float(yr.loc[y, "total_return"]) > 0 for y in [2022, 2023, 2024, 2025, 2026])
    gates = [
        ("Net total > 0", m["total_return"] > 0),
        ("PF > 1.30", m["profit_factor"] > 1.30),
        ("Sharpe > 1.00", m["sharpe"] > 1.00),
        ("MDD better than -50%", m["max_drawdown"] > -0.50),
        ("Every year 2022-2026 positive", years_ok),
        ("Post-paper Apr-Jul 2026 positive", bool(postm) and postm["total_return"] > 0),
        ("Tail10-zeroed stress remains profitable", bool(stress) and stress["total_return"] > 0),
        ("Top positive symbol <25% of positive pre-cost contribution", np.isfinite(top_share) and top_share < 0.25),
    ]
    print("\nPRE-REGISTERED PURE-CARRY GATES")
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    print("\nVERDICT")
    if all(ok for _, ok in gates):
        print("[KEEP] Pure causal funding-rank carry clears stability, post-paper, concentration and tail-stress gates. Build execution/live-shadow implementation next; do not add parameters.")
    else:
        print("[CLOSE FUNDING/MOMENTUM BRANCH] Pure causal carry does not clear the fixed robustness gates. Do not rescue this family with thresholds or additional momentum filters.")

    pd.DataFrame([m]).to_csv(out / "summary.csv", index=False)
    ydf.to_csv(out / "year_breakdown.csv", index=False)
    wdf.to_csv(out / "weekly_results.csv", index=False)
    adf.to_csv(out / "asset_results.csv", index=False)
    if not sw.empty:
        sw.to_csv(out / "tail10_stress_weekly.csv", index=False)
    pd.DataFrame([{"gate": label, "pass": bool(ok)} for label, ok in gates]).to_csv(out / "gates.csv", index=False)
    print(f"\nSaved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
