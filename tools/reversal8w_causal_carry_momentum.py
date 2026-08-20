#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Causal funding-confirmed version of the saved high-vol 8-week momentum mirror")
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--assets", default="/freqtrade/user_data/reversal8w_perp/asset_contributions.csv")
    p.add_argument("--side-cost-bps", type=float, default=7.0)
    p.add_argument("--output-dir", default="/freqtrade/user_data/reversal8w_causal_carry_momentum")
    return p.parse_args()


def perf(r: pd.Series) -> dict[str, float]:
    r = r.fillna(0.0).astype(float)
    if (r <= -1.0).any():
        first = int(np.flatnonzero((r <= -1.0).to_numpy())[0])
        rr = r.iloc[: first + 1]
        eq = (1.0 + rr).cumprod().clip(lower=0.0)
        equity = 0.0
        total = -1.0
        mdd = -1.0
    else:
        eq = (1.0 + r).cumprod()
        equity = float(eq.iloc[-1]) if len(eq) else 1.0
        total = equity - 1.0
        mdd = float((eq / eq.cummax() - 1.0).min()) if len(eq) else 0.0
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    sd = float(r.std(ddof=1))
    downside = float(r[r < 0].std(ddof=1)) if (r < 0).sum() > 1 else math.nan
    years = len(r) / 52.0
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 and years > 0 else -1.0
    return {
        "ending_equity": 100.0 * equity,
        "total_return": total,
        "cagr": cagr,
        "profit_factor": pos / neg if neg > 0 else math.inf,
        "sharpe": math.sqrt(52.0) * float(r.mean()) / sd if sd > 0 else math.nan,
        "sortino": math.sqrt(52.0) * float(r.mean()) / downside if np.isfinite(downside) and downside > 0 else math.nan,
        "max_drawdown": mdd,
        "weekly_wr": float((r > 0).mean()),
    }


def funding_prefix(con: sqlite3.Connection, symbols: list[str], lo: pd.Timestamp, hi: pd.Timestamp):
    out = {}
    a = int((lo - pd.Timedelta(days=10)).timestamp() * 1000)
    b = int((hi + pd.Timedelta(days=10)).timestamp() * 1000)
    for i, sym in enumerate(sorted(set(symbols)), 1):
        rows = con.execute(
            "SELECT event_time,rate FROM funding_events WHERE symbol=? AND event_time BETWEEN ? AND ? ORDER BY event_time",
            (sym, a, b),
        ).fetchall()
        if rows:
            t = np.asarray([x[0] for x in rows], dtype=np.int64)
            rr = np.asarray([x[1] for x in rows], dtype=float)
            out[sym] = (t, np.concatenate([[0.0], np.cumsum(rr)]))
        if i % 150 == 0 or i == len(set(symbols)):
            print(f"Funding histories: {i}/{len(set(symbols))}", flush=True)
    return out


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


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100*x:+.2f}%"


def main() -> int:
    cfg = parse_args()
    src = Path(cfg.assets)
    if not src.exists():
        raise RuntimeError(f"Missing source asset CSV: {src}")
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    a = pd.read_csv(src, parse_dates=["date"])
    need = {"strategy", "date", "symbol", "weight", "gross_contribution", "funding_contribution", "forced_exit"}
    miss = need - set(a.columns)
    if miss:
        raise RuntimeError(f"asset_contributions.csv missing {sorted(miss)}")
    a = a[a.strategy == "HIGH_VOL_REVERSAL"].copy()
    if a.empty:
        raise RuntimeError("No HIGH_VOL_REVERSAL rows")
    a["date"] = pd.to_datetime(a.date, utc=True)
    a["mirror_base_weight"] = -a.weight.astype(float)
    a["mirror_base_price"] = -a.gross_contribution.astype(float)
    a["mirror_base_funding"] = -a.funding_contribution.astype(float)
    a["leg"] = np.where(a.mirror_base_weight > 0, "LONG_WINNERS", "SHORT_LOSERS")

    con = sqlite3.connect(cfg.db, timeout=120)
    pref = funding_prefix(con, a.symbol.unique().tolist(), a.date.min(), a.date.max())
    prior = []
    known = []
    for r in a.itertuples(index=False):
        # Entry is at complete Sunday close. Predictor uses the just-finished Monday-Sunday week.
        start = pd.Timestamp(r.date) - pd.Timedelta(days=6)
        fr, ok = funding_range(pref, r.symbol, start, pd.Timestamp(r.date))
        prior.append(fr)
        known.append(ok)
    a["prior7_funding"] = prior
    a["prior_known"] = known
    # Funding PnL of a position is -weight * funding rate. Keep only ex-ante favorable sign.
    a["predicted_carry"] = -a.mirror_base_weight * a.prior7_funding
    a["carry_favorable"] = a.prior_known & (a.predicted_carry > 0)

    side_cost = cfg.side_cost_bps / 10_000.0
    prev: dict[str, float] = {}
    weeks = []
    assets = []

    for dt, g0 in a.groupby("date", sort=True):
        g = g0[g0.carry_favorable].copy()
        longs = g[g.mirror_base_weight > 0].copy()
        shorts = g[g.mirror_base_weight < 0].copy()

        target: dict[str, float] = {}
        scale_long = 0.0
        scale_short = 0.0
        retained_l = float(longs.mirror_base_weight.abs().sum())
        retained_s = float(shorts.mirror_base_weight.abs().sum())
        # Conservative construction: never increase any name above its original mirror weight.
        # Scale both retained legs down to the smaller retained side, preserving dollar neutrality and leaving cash idle.
        side_gross = min(retained_l, retained_s)
        if side_gross > 0 and retained_l > 0 and retained_s > 0:
            scale_long = side_gross / retained_l
            scale_short = side_gross / retained_s
            for r in longs.itertuples(index=False):
                target[r.symbol] = float(r.mirror_base_weight) * scale_long
            for r in shorts.itertuples(index=False):
                target[r.symbol] = float(r.mirror_base_weight) * scale_short

        turnover = sum(abs(target.get(s, 0.0) - prev.get(s, 0.0)) for s in set(target) | set(prev))
        price = 0.0
        funding = 0.0
        forced_notional = 0.0
        n_long = 0
        n_short = 0
        for r in g.itertuples(index=False):
            w = target.get(r.symbol, 0.0)
            if abs(w) <= 1e-15:
                continue
            scale = w / float(r.mirror_base_weight)
            pc = float(r.mirror_base_price) * scale
            fc = float(r.mirror_base_funding) * scale
            price += pc
            funding += fc
            if w > 0:
                n_long += 1
            else:
                n_short += 1
            if bool(r.forced_exit):
                forced_notional += abs(w)
            assets.append({
                "date": dt, "symbol": r.symbol, "leg": r.leg,
                "weight": w, "scale_from_base": scale,
                "prior7_funding": float(r.prior7_funding),
                "predicted_carry": float(-w * r.prior7_funding),
                "price_contribution": pc, "funding_contribution": fc,
                "forced_exit": bool(r.forced_exit),
            })
        if forced_notional > 0:
            turnover += forced_notional
        cost = turnover * side_cost
        net = price + funding - cost
        weeks.append({
            "date": dt, "longs": n_long, "shorts": n_short,
            "gross_exposure": sum(abs(w) for w in target.values()),
            "net_exposure": sum(target.values()),
            "turnover": turnover, "price_return": price,
            "funding_return": funding, "cost_return": cost, "net_return": net,
        })
        forced_syms = set(g.loc[g.forced_exit.astype(bool), "symbol"])
        prev = {s: w for s, w in target.items() if s not in forced_syms}

    wdf = pd.DataFrame(weeks).sort_values("date").reset_index(drop=True)
    adf = pd.DataFrame(assets)
    if prev and not wdf.empty:
        terminal = sum(abs(x) for x in prev.values())
        wdf.loc[wdf.index[-1], "turnover"] += terminal
        wdf.loc[wdf.index[-1], "cost_return"] += terminal * side_cost
        wdf.loc[wdf.index[-1], "net_return"] -= terminal * side_cost

    m = perf(wdf.net_return)
    m.update({
        "weeks": len(wdf),
        "avg_longs": float(wdf.longs.mean()),
        "avg_shorts": float(wdf.shorts.mean()),
        "avg_gross_exposure": float(wdf.gross_exposure.mean()),
        "avg_price": float(wdf.price_return.mean()),
        "avg_funding": float(wdf.funding_return.mean()),
        "avg_cost": float(wdf.cost_return.mean()),
        "avg_net": float(wdf.net_return.mean()),
    })
    summary = pd.DataFrame([m])

    year_rows = []
    for year, g in wdf.groupby(wdf.date.dt.year):
        mm = perf(g.net_return)
        year_rows.append({"year": int(year), **mm,
                          "avg_gross_exposure": float(g.gross_exposure.mean()),
                          "avg_price": float(g.price_return.mean()),
                          "avg_funding": float(g.funding_return.mean()),
                          "avg_cost": float(g.cost_return.mean())})
    ydf = pd.DataFrame(year_rows)

    post = wdf[(wdf.date >= pd.Timestamp("2026-04-01", tz="UTC")) & (wdf.date <= pd.Timestamp("2026-07-31", tz="UTC"))]
    postm = perf(post.net_return) if len(post) else {}

    # Tail stress: remove all realized funding PnL from the 10% most extreme selected asset-weeks.
    stress = None
    if not adf.empty:
        cutoff = float(adf.funding_contribution.abs().quantile(0.90))
        adf["tail10"] = adf.funding_contribution.abs() >= cutoff
        kept_funding = adf.assign(stress_funding=np.where(adf.tail10, 0.0, adf.funding_contribution)) \
                          .groupby("date", as_index=False).stress_funding.sum()
        sw = wdf.merge(kept_funding, on="date", how="left")
        sw["stress_funding"] = sw.stress_funding.fillna(0.0)
        sw["stress_net"] = sw.price_return + sw.stress_funding - sw.cost_return
        stress = perf(sw.stress_net)
        sw.to_csv(out / "tail10_stress_weekly.csv", index=False)
    else:
        cutoff = math.nan

    print("=== CAUSAL FUNDING-CONFIRMED HIGH-VOL MOMENTUM ===")
    print("Post-hoc candidate, but every weekly funding decision is causal.")
    print("Rule fixed: keep a high-vol momentum position only when PRIOR 7d funding has favorable sign for that position.")
    print("No magnitude threshold. No parameter search. No position is increased above its original mirror weight; unmatched exposure stays cash.\n")

    print("MAIN RESULT")
    print(f"Weeks: {len(wdf)} | avg longs={wdf.longs.mean():.2f} shorts={wdf.shorts.mean():.2f} | avg gross={wdf.gross_exposure.mean():.3f}x")
    print(f"$100 -> ${m['ending_equity']:.2f} | total={pct(m['total_return'])} | CAGR={pct(m['cagr'])}")
    print(f"WR={100*m['weekly_wr']:.2f}% | PF={m['profit_factor']:.3f} | Sharpe={m['sharpe']:.3f} | Sortino={m['sortino']:.3f} | MDD={pct(m['max_drawdown'])}")
    print(f"Avg/week price={pct(m['avg_price'])} funding={pct(m['avg_funding'])} cost={pct(m['avg_cost'])} net={pct(m['avg_net'])}")

    print("\nYEAR BREAKDOWN")
    yp = ydf.copy()
    for c in ["total_return", "cagr", "max_drawdown", "weekly_wr", "avg_price", "avg_funding", "avg_cost"]:
        if c in yp.columns:
            yp[c] = yp[c].map(pct)
    for c in ["profit_factor", "sharpe", "sortino", "avg_gross_exposure"]:
        if c in yp.columns:
            yp[c] = yp[c].map(lambda x: f"{x:.3f}" if np.isfinite(x) else "nan")
    print(yp.to_string(index=False))

    print("\nPOST-PAPER APR-JUL 2026")
    if postm:
        print(f"return={pct(postm['total_return'])} PF={postm['profit_factor']:.3f} Sharpe={postm['sharpe']:.3f} MDD={pct(postm['max_drawdown'])}")
    else:
        print("no observations")

    print("\nTAIL-CONCENTRATION STRESS")
    print(f"Selected asset-week funding |90th percentile| cutoff: {pct(cutoff)}")
    if stress:
        print("Stress sets realized funding contribution of the most extreme 10% selected asset-weeks to ZERO; price and costs unchanged.")
        print(f"$100 -> ${stress['ending_equity']:.2f} | total={pct(stress['total_return'])} | CAGR={pct(stress['cagr'])} | PF={stress['profit_factor']:.3f} | Sharpe={stress['sharpe']:.3f} | MDD={pct(stress['max_drawdown'])}")

    gates = [
        ("Net total > 0", m["total_return"] > 0),
        ("PF > 1.30", m["profit_factor"] > 1.30),
        ("Sharpe > 1.00", m["sharpe"] > 1.00),
        ("MDD better than -50%", m["max_drawdown"] > -0.50),
        ("2024 positive", bool(len(ydf[ydf.year == 2024]) and ydf.loc[ydf.year == 2024, "total_return"].iloc[0] > 0)),
        ("2025 positive", bool(len(ydf[ydf.year == 2025]) and ydf.loc[ydf.year == 2025, "total_return"].iloc[0] > 0)),
        ("2026 positive", bool(len(ydf[ydf.year == 2026]) and ydf.loc[ydf.year == 2026, "total_return"].iloc[0] > 0)),
        ("Post-paper Apr-Jul 2026 positive", bool(postm and postm["total_return"] > 0)),
        ("Tail10-zeroed stress remains profitable", bool(stress and stress["total_return"] > 0)),
    ]
    print("\nCAUSAL CANDIDATE GATES")
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if all(ok for _, ok in gates):
        print("\nVERDICT\n[PROMOTE TO ROBUSTNESS] Causal prior-funding confirmation survives return/risk/year/post-paper/tail stress gates. Next step is execution realism and regime robustness, not a new alpha search.")
    else:
        print("\nVERDICT\n[NOT READY] The causal sign-only funding confirmation does not clear all robustness gates. Inspect which component failed before changing any rule.")

    summary.to_csv(out / "summary.csv", index=False)
    ydf.to_csv(out / "year_breakdown.csv", index=False)
    wdf.to_csv(out / "weekly_results.csv", index=False)
    adf.to_csv(out / "asset_results.csv", index=False)
    pd.DataFrame([{"gate": x, "pass": bool(y)} for x, y in gates]).to_csv(out / "gates.csv", index=False)
    print(f"\nSaved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
