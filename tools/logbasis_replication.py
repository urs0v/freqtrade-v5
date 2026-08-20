#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

BAR_MS = 8 * 60 * 60 * 1000
PRIMARY_SIDE_COST_BPS = 7.0
STRESS_SIDE_COST_BPS = 10.0
VOL_FLOOR_USD = 1_000_000.0
MIN_CS = 10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Pre-registered causal Binance spot-perp log-basis replication")
    p.add_argument("--basis-db", default="/freqtrade/user_data/logbasis_8h/logbasis.sqlite")
    p.add_argument("--core-db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--rep-end", default="2023-12-31")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--side-cost-bps", type=float, default=PRIMARY_SIDE_COST_BPS)
    p.add_argument("--stress-side-cost-bps", type=float, default=STRESS_SIDE_COST_BPS)
    p.add_argument("--output-dir", default="/freqtrade/user_data/logbasis_replication")
    return p.parse_args()


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100*x:+.2f}%"


def perf(r: pd.Series, periods_per_year: float = 3.0 * 365.25) -> dict[str, float]:
    r = r.fillna(0.0).astype(float)
    if len(r) == 0:
        return {k: math.nan for k in ["ending_equity","total_return","cagr","profit_factor","sharpe","sortino","max_drawdown","win_rate"]}
    if (r <= -1.0).any():
        first = int(np.flatnonzero((r <= -1.0).to_numpy())[0])
        rr = r.iloc[:first+1]
        eq = (1.0 + rr).cumprod().clip(lower=0.0)
        equity = 0.0
        total = -1.0
        mdd = -1.0
    else:
        eq = (1.0 + r).cumprod()
        equity = float(eq.iloc[-1])
        total = equity - 1.0
        mdd = float((eq / eq.cummax() - 1.0).min())
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    sd = float(r.std(ddof=1))
    dn = r[r < 0]
    dsd = float(dn.std(ddof=1)) if len(dn) > 1 else math.nan
    years = len(r) / periods_per_year
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 and years > 0 else -1.0
    return {
        "ending_equity": 100.0 * equity,
        "total_return": total,
        "cagr": cagr,
        "profit_factor": pos / neg if neg > 0 else math.inf,
        "sharpe": math.sqrt(periods_per_year) * float(r.mean()) / sd if sd > 0 else math.nan,
        "sortino": math.sqrt(periods_per_year) * float(r.mean()) / dsd if np.isfinite(dsd) and dsd > 0 else math.nan,
        "max_drawdown": mdd,
        "win_rate": float((r > 0).mean()),
    }


def funding_prefix(core: Path, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp):
    con = sqlite3.connect(core, timeout=120)
    lo = int((start - pd.Timedelta(days=2)).timestamp() * 1000)
    hi = int((end + pd.Timedelta(days=2)).timestamp() * 1000)
    out = {}
    for i, sym in enumerate(sorted(set(symbols)), 1):
        rows = con.execute(
            "SELECT event_time,rate FROM funding_events WHERE symbol=? AND event_time BETWEEN ? AND ? ORDER BY event_time",
            (sym, lo, hi),
        ).fetchall()
        if rows:
            t = np.asarray([x[0] for x in rows], dtype=np.int64)
            rr = np.asarray([x[1] for x in rows], dtype=float)
            out[sym] = (t, np.concatenate([[0.0], np.cumsum(rr)]))
        if i % 150 == 0 or i == len(symbols):
            print(f"Funding prefix: {i}/{len(symbols)}", flush=True)
    con.close()
    return out


def funding_between(pref, sym: str, t0_ms: int, t1_ms: int) -> tuple[float, bool]:
    item = pref.get(sym)
    if item is None:
        return 0.0, False
    t, cs = item
    # Position starts immediately AFTER the settlement at t0, and is held through t1.
    i0 = int(np.searchsorted(t, t0_ms, side="right"))
    i1 = int(np.searchsorted(t, t1_ms, side="right"))
    return float(cs[i1] - cs[i0]), True


def load_panel(db: Path, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    con = sqlite3.connect(db, timeout=120)
    lo = int((start - pd.Timedelta(days=2)).timestamp() * 1000)
    hi = int((end + pd.Timedelta(days=1)).timestamp() * 1000)
    q = pd.read_sql_query(
        """
        SELECT p.symbol,p.open_ms,p.open AS perp_open,p.close AS perp_close,p.quote_volume,
               s.spot_symbol,s.multiplier,s.open AS spot_open,s.close AS spot_close
        FROM perp_8h p
        LEFT JOIN spot_8h s ON s.symbol=p.symbol AND s.open_ms=p.open_ms
        WHERE p.open_ms BETWEEN ? AND ?
        ORDER BY p.symbol,p.open_ms
        """,
        con,
        params=(lo, hi),
    )
    con.close()
    if q.empty:
        raise RuntimeError("No log-basis 8h data. Run backfill_logbasis_8h.py first.")
    q["date"] = pd.to_datetime(q.open_ms, unit="ms", utc=True)
    q = q.sort_values(["symbol","open_ms"]).reset_index(drop=True)
    g = q.groupby("symbol", sort=False)
    q["prior24_quote_volume"] = g.quote_volume.transform(lambda x: x.shift(1).rolling(3, min_periods=3).sum())
    q["next_ms"] = g.open_ms.shift(-1)
    q["next_perp_open"] = g.perp_open.shift(-1)
    q["next_spot_open"] = g.spot_open.shift(-1)
    q["contiguous_next"] = (q.next_ms - q.open_ms) == BAR_MS
    q["exit_perp"] = np.where(q.contiguous_next, q.next_perp_open, q.perp_close)
    q["exit_spot"] = np.where(q.contiguous_next & q.next_spot_open.notna(), q.next_spot_open, q.spot_close)
    q["forced_exit"] = ~q.contiguous_next
    q["basis"] = np.log(q.perp_open / q.spot_open)
    q["fut_ret"] = q.exit_perp / q.perp_open - 1.0
    q["spot_log_ret"] = np.log(q.exit_spot / q.spot_open)
    q["exit_basis"] = np.log(q.exit_perp / q.exit_spot)
    q["term_log_ret"] = q.exit_basis - q.basis
    q = q[(q.date >= start) & (q.date <= end)]
    return q


def rank_ic(x: pd.Series, y: pd.Series) -> float:
    z = pd.concat([x,y], axis=1).dropna()
    return float(z.iloc[:,0].corr(z.iloc[:,1], method="spearman")) if len(z) >= 5 else math.nan


def weekly_compound(df: pd.DataFrame, col: str) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=float)
    x = df[["date",col]].copy()
    x["week"] = x.date.dt.tz_convert(None).dt.to_period("W-SUN")
    return x.groupby("week", sort=True)[col].apply(lambda r: float(np.prod(1.0 + r) - 1.0))


def main() -> int:
    cfg = parse_args()
    start = pd.Timestamp(cfg.start, tz="UTC")
    rep_end = pd.Timestamp(cfg.rep_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    end = pd.Timestamp(cfg.end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    out = Path(cfg.output_dir); out.mkdir(parents=True, exist_ok=True)

    print("=== BINANCE LOG-BASIS CAUSAL REPLICATION ===")
    print("Paper rule: log-basis = ln(perpetual/spot), quintile sort, long lowest / short highest basis.")
    print("Frozen implementation: common 8h funding boundaries; entry at 8h bar OPEN immediately after settlement; hold 8h.")
    print("Causal liquidity rule: PRIOR completed 24h perp quote-volume > $1m (paper says daily price-volume > $1m).")
    print("Portfolio: equal-weight tails, 50% long + 50% short = 1x gross / ~0 net.")
    print(f"Costs: {cfg.side_cost_bps:.1f} bps per changed notional side; stress={cfg.stress_side_cost_bps:.1f} bps. Real funding included.")
    print(f"Replication-era diagnostic: {start.date()} -> {cfg.rep_end}; frozen OOS: 2024-01-01 -> {cfg.end}")
    print("No parameter search. Missing spot at decision time means the contract is not observable for this direct-spot signal.\n")

    panel = load_panel(Path(cfg.basis_db), start, end)
    print(f"Perp 8h rows: {len(panel):,} | symbols={panel.symbol.nunique()} | mapped spot symbols={panel.loc[panel.spot_open.notna(),'symbol'].nunique()}")
    pref = funding_prefix(Path(cfg.core_db), panel.symbol.unique().tolist(), start, end)

    side_cost = cfg.side_cost_bps / 10_000.0
    rows = []
    assets = []
    prev: dict[str,float] = {}
    dates = sorted(pd.Timestamp(x) for x in panel.date.unique())

    for j, dt in enumerate(dates, 1):
        all_now = panel[panel.date == dt].copy()
        eligible = all_now[(all_now.prior24_quote_volume > VOL_FLOOR_USD) & np.isfinite(all_now.perp_open)]
        observable = eligible[np.isfinite(eligible.basis)].copy()
        denom = len(eligible)
        if len(observable) < MIN_CS:
            continue
        observable = observable.sort_values(["basis","symbol"]).reset_index(drop=True)
        groups = np.array_split(np.arange(len(observable)), 5)
        if len(groups[0]) == 0 or len(groups[-1]) == 0:
            continue
        q1 = observable.iloc[groups[0]].copy()
        q5 = observable.iloc[groups[-1]].copy()

        target: dict[str,float] = {}
        for s in q1.symbol:
            target[s] = target.get(s, 0.0) + 0.5 / len(q1)
        for s in q5.symbol:
            target[s] = target.get(s, 0.0) - 0.5 / len(q5)
        turnover = sum(abs(target.get(s,0.0) - prev.get(s,0.0)) for s in set(target) | set(prev))

        price = funding = spot_comp = term_comp = 0.0
        known = 0
        forced_notional = 0.0
        qtot = []
        for qi, inds in enumerate(groups, 1):
            gg = observable.iloc[inds]
            vals = []
            for r in gg.itertuples(index=False):
                if not np.isfinite(r.fut_ret):
                    continue
                fr, ok = funding_between(pref, r.symbol, int(r.open_ms), int(r.open_ms + BAR_MS))
                vals.append(float(r.fut_ret) - fr)
            qtot.append(float(np.mean(vals)) if vals else math.nan)

        selected = pd.concat([q1,q5], ignore_index=True)
        for r in selected.itertuples(index=False):
            w = target[r.symbol]
            if not np.isfinite(r.fut_ret):
                # Selection never conditions on future availability. If even the current bar lacks a usable close, treat as full loss.
                rr = -1.0
            else:
                rr = float(r.fut_ret)
            fr, ok = funding_between(pref, r.symbol, int(r.open_ms), int(r.open_ms + BAR_MS))
            price += w * rr
            funding += -w * fr
            known += int(ok)
            if np.isfinite(r.spot_log_ret): spot_comp += w * float(r.spot_log_ret)
            if np.isfinite(r.term_log_ret): term_comp += w * float(r.term_log_ret)
            if bool(r.forced_exit): forced_notional += abs(w)
            assets.append({
                "date": dt, "symbol": r.symbol, "weight": w, "basis": float(r.basis),
                "prior24_quote_volume": float(r.prior24_quote_volume), "fut_ret": rr,
                "funding_rate_sum": fr, "funding_contribution": -w*fr,
                "price_contribution": w*rr, "spot_log_contribution": w*float(r.spot_log_ret) if np.isfinite(r.spot_log_ret) else np.nan,
                "term_log_contribution": w*float(r.term_log_ret) if np.isfinite(r.term_log_ret) else np.nan,
                "forced_exit": bool(r.forced_exit),
            })
        if forced_notional > 0:
            turnover += forced_notional
        cost = turnover * side_cost
        gross = price + funding
        net = gross - cost
        ic = rank_ic(observable.basis, observable.fut_ret)
        rows.append({
            "date": dt, "eligible_perps": denom, "observable_spot": len(observable),
            "spot_coverage": len(observable)/denom if denom else np.nan,
            "positions": len(target), "turnover": turnover,
            "price_return": price, "funding_return": funding, "gross_return": gross,
            "cost_return": cost, "net_return": net,
            "spot_log_component": spot_comp, "term_log_component": term_comp,
            "rank_ic_basis_to_next_fut": ic, "funding_coverage": known/len(target) if target else np.nan,
            "q1_total": qtot[0], "q2_total": qtot[1], "q3_total": qtot[2], "q4_total": qtot[3], "q5_total": qtot[4],
        })
        forced_syms = set(selected.loc[selected.forced_exit.astype(bool), "symbol"])
        prev = {s:w for s,w in target.items() if s not in forced_syms}
        if j % 250 == 0 or j == len(dates):
            print(f"Portfolio pass: {j}/{len(dates)} | {dt}", flush=True)

    w = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    a = pd.DataFrame(assets)
    if w.empty:
        raise RuntimeError("No eligible log-basis portfolio periods")
    if prev:
        terminal = sum(abs(x) for x in prev.values())
        w.loc[w.index[-1], "turnover"] += terminal
        w.loc[w.index[-1], "cost_return"] += terminal * side_cost
        w.loc[w.index[-1], "net_return"] -= terminal * side_cost

    w["stress_cost_return"] = w.turnover * (cfg.stress_side_cost_bps / 10_000.0)
    w["stress_net_return"] = w.gross_return - w.stress_cost_return
    rep = w[w.date <= rep_end].copy()
    oos = w[w.date > rep_end].copy()

    def summarize(label: str, x: pd.DataFrame) -> dict:
        m = perf(x.net_return)
        wg = weekly_compound(x, "gross_return")
        wn = weekly_compound(x, "net_return")
        m.update({
            "split": label, "periods": len(x),
            "mean_weekly_gross": float(wg.mean()) if len(wg) else np.nan,
            "mean_weekly_net": float(wn.mean()) if len(wn) else np.nan,
            "avg_turnover": float(x.turnover.mean()),
            "avg_spot_component_8h": float(x.spot_log_component.mean()),
            "avg_term_component_8h": float(x.term_log_component.mean()),
            "avg_funding_8h": float(x.funding_return.mean()),
            "avg_rank_ic": float(x.rank_ic_basis_to_next_fut.mean()),
            "median_spot_coverage": float(x.spot_coverage.median()),
            "median_observable": float(x.observable_spot.median()),
            "funding_coverage": float(x.funding_coverage.mean()),
        })
        return m

    summary = pd.DataFrame([summarize("REPLICATION_2021_2023", rep), summarize("OOS_2024_2026", oos)])
    year_rows = []
    for year, g in w.groupby(w.date.dt.year):
        mm = perf(g.net_return)
        year_rows.append({"year": int(year), **mm, "mean_weekly_net": float(weekly_compound(g,"net_return").mean()), "avg_turnover": float(g.turnover.mean())})
    years = pd.DataFrame(year_rows)

    quintile_rows = []
    for label, x in [("REPLICATION_2021_2023",rep),("OOS_2024_2026",oos)]:
        qr = {"split": label}
        for i in range(1,6):
            ws = weekly_compound(x, f"q{i}_total")
            qr[f"q{i}_mean_weekly"] = float(ws.mean()) if len(ws) else np.nan
        qr["q1_minus_q5_scaled_1x"] = 0.5*(qr["q1_mean_weekly"] - qr["q5_mean_weekly"])
        quintile_rows.append(qr)
    quint = pd.DataFrame(quintile_rows)

    stress = perf(oos.stress_net_return) if len(oos) else {}
    repm = summary.iloc[0]; oosm = summary.iloc[1]
    yr = years.set_index("year") if not years.empty else pd.DataFrame()
    qrep = quint.iloc[0]

    gates = [
        ("Replication Q1 weekly > Q5 weekly", bool(qrep.q1_mean_weekly > qrep.q5_mean_weekly)),
        ("Replication mean weekly gross > 0", bool(repm.mean_weekly_gross > 0)),
        ("Replication net total > 0 after 7bps-side turnover costs", bool(repm.total_return > 0)),
        ("Replication basis rank IC < 0", bool(repm.avg_rank_ic < 0)),
        ("OOS net total > 0", bool(oosm.total_return > 0)),
        ("OOS PF > 1.30", bool(oosm.profit_factor > 1.30)),
        ("OOS Sharpe > 1.00", bool(oosm.sharpe > 1.00)),
        ("OOS MDD better than -50%", bool(oosm.max_drawdown > -0.50)),
        ("2024 positive", bool(2024 in yr.index and yr.loc[2024,"total_return"] > 0)),
        ("2025 positive", bool(2025 in yr.index and yr.loc[2025,"total_return"] > 0)),
        ("2026 positive", bool(2026 in yr.index and yr.loc[2026,"total_return"] > 0)),
        ("OOS remains profitable at 10bps per changed side", bool(stress and stress["total_return"] > 0)),
        ("Median direct-spot coverage >= 60%", bool(oosm.median_spot_coverage >= 0.60)),
    ]

    print("\n=== LOG-BASIS RESULT ===")
    sp = summary.copy()
    for c in ["total_return","cagr","max_drawdown","win_rate","mean_weekly_gross","mean_weekly_net","avg_spot_component_8h","avg_term_component_8h","avg_funding_8h","median_spot_coverage","funding_coverage"]:
        sp[c] = sp[c].map(pct)
    for c in ["ending_equity","profit_factor","sharpe","sortino","avg_turnover","avg_rank_ic","median_observable"]:
        sp[c] = sp[c].map(lambda z: f"{z:.3f}" if np.isfinite(z) else "nan")
    print(sp.to_string(index=False))

    print("\nQUINTILE MONOTONICITY (weekly long-only total returns, including funding)")
    qp = quint.copy()
    for c in qp.columns[1:]: qp[c] = qp[c].map(pct)
    print(qp.to_string(index=False))

    print("\nYEAR BREAKDOWN")
    yp = years.copy()
    for c in ["total_return","cagr","max_drawdown","win_rate","mean_weekly_net"]: yp[c] = yp[c].map(pct)
    for c in ["ending_equity","profit_factor","sharpe","sortino","avg_turnover"]: yp[c] = yp[c].map(lambda z: f"{z:.3f}" if np.isfinite(z) else "nan")
    print(yp.to_string(index=False))

    print("\nOOS 10BPS-SIDE COST STRESS")
    if stress:
        print(f"$100 -> ${stress['ending_equity']:.2f} | total={pct(stress['total_return'])} | PF={stress['profit_factor']:.3f} | Sharpe={stress['sharpe']:.3f} | MDD={pct(stress['max_drawdown'])}")

    print("\nPRE-REGISTERED LOG-BASIS GATES")
    for name, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    overall = all(ok for _,ok in gates)
    print("Overall:", "PASS" if overall else "FAIL")
    if overall:
        print("VERDICT: [SURVIVES REPLICATION + OOS] Basis family merits execution-level development; do not optimize yet.")
    else:
        print("VERDICT: [NO RESCUE FITTING] The frozen direct-spot log-basis implementation does not clear the full replication/OOS/cost hurdle.")

    w.to_csv(out / "period_results.csv", index=False)
    a.to_csv(out / "asset_results.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    years.to_csv(out / "year_breakdown.csv", index=False)
    quint.to_csv(out / "quintiles.csv", index=False)
    pd.DataFrame(gates, columns=["gate","pass"]).to_csv(out / "gates.csv", index=False)
    print(f"\nSaved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
