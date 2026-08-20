#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import cttrend_research_v3 as v3

STABLE_BASES = {
    "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI",
    "USDE", "USDS", "USD1", "PYUSD",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Paper-grounded 8-week cross-sectional reversal audit on Binance USD-M perpetuals"
    )
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--start", default="2021-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--universe", type=int, default=70)
    p.add_argument("--tail-frac", type=float, default=0.20)
    p.add_argument("--high-vol-frac", type=float, default=0.50)
    p.add_argument("--side-cost-bps", type=float, default=7.0)
    p.add_argument("--min-cross-section", type=int, default=25)
    p.add_argument("--output-dir", default="/freqtrade/user_data/reversal8w_perp")
    return p.parse_args()


def load_daily(con: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    warm = start - pd.Timedelta(days=120)
    lo = int(warm.timestamp() * 1000)
    hi = int((end + pd.Timedelta(days=8)).timestamp() * 1000) - 1
    syms = [
        r[0] for r in con.execute(
            "SELECT DISTINCT symbol FROM candles WHERE open_time BETWEEN ? AND ? ORDER BY symbol", (lo, hi)
        )
        if r[0].endswith("USDT") and r[0][:-4] not in STABLE_BASES
    ]
    print(f"Historical non-stable USDT perps: {len(syms)}", flush=True)
    parts = []
    for i, sym in enumerate(syms, 1):
        rows = con.execute(
            """
            SELECT open_time, close, quote_volume
            FROM candles WHERE symbol=? AND open_time BETWEEN ? AND ? ORDER BY open_time
            """, (sym, lo, hi)
        ).fetchall()
        if not rows:
            continue
        x = pd.DataFrame(rows, columns=["open_time", "close", "quote_volume"])
        x["date"] = pd.to_datetime(x.open_time, unit="ms", utc=True).dt.floor("D")
        d = (x.groupby("date", sort=True)
             .agg(close=("close", "last"), quote_volume=("quote_volume", "sum"), bars=("open_time", "count"))
             .reset_index())
        d = d[d.bars == 4].drop(columns="bars")
        if len(d) < 60:
            continue
        d["symbol"] = sym
        d["logret"] = np.log(d.close / d.close.shift(1))
        d["ret_8w"] = d.close / d.close.shift(56) - 1.0
        d["vol_8w"] = d.logret.rolling(56, min_periods=42).std(ddof=1)
        d["liq_30d"] = d.quote_volume.rolling(30, min_periods=25).mean()
        d["history_days"] = np.arange(1, len(d) + 1)
        parts.append(d)
        if i % 100 == 0 or i == len(syms):
            print(f"Daily reversal features: {i}/{len(syms)}", flush=True)
    if not parts:
        raise RuntimeError("No daily data built")
    daily = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    print(
        f"Daily panel: {len(daily):,} rows | {daily.symbol.nunique()} symbols | "
        f"{daily.date.min().date()} -> {daily.date.max().date()}", flush=True
    )
    return daily


def funding_prefix(con: sqlite3.Connection, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp):
    lo = int((start - pd.Timedelta(days=14)).timestamp() * 1000)
    hi = int((end + pd.Timedelta(days=14)).timestamp() * 1000)
    out = {}
    for i, sym in enumerate(sorted(set(symbols)), 1):
        rows = con.execute(
            "SELECT event_time, rate FROM funding_events WHERE symbol=? AND event_time BETWEEN ? AND ? ORDER BY event_time",
            (sym, lo, hi),
        ).fetchall()
        if rows:
            t = np.array([r[0] for r in rows], dtype=np.int64)
            r = np.array([r[1] for r in rows], dtype=float)
            out[sym] = (t, np.concatenate([[0.0], np.cumsum(r)]))
        if i % 150 == 0 or i == len(symbols):
            print(f"Funding prefix: {i}/{len(symbols)}", flush=True)
    return out


def funding_sum(pref, sym: str, entry: pd.Timestamp, exit_: pd.Timestamp):
    item = pref.get(sym)
    if item is None:
        return 0.0, False
    t, cs = item
    a = int(entry.timestamp() * 1000)
    b = int((exit_ + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    i0 = int(np.searchsorted(t, a, side="right"))
    i1 = int(np.searchsorted(t, b, side="right"))
    return float(cs[i1] - cs[i0]), True


def perf(r: pd.Series) -> dict[str, float]:
    r = r.fillna(0.0).astype(float)
    # A weekly loss <= -100% means portfolio ruin under the stated 1x gross construction.
    if (r <= -1.0).any():
        first = int(np.flatnonzero((r <= -1.0).to_numpy())[0])
        rr = r.iloc[: first + 1]
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
    downside = float(r[r < 0].std(ddof=1)) if (r < 0).sum() > 1 else math.nan
    years = len(r) / 52.0
    cagr = equity ** (1.0 / years) - 1.0 if equity > 0 and years > 0 else -1.0
    return {
        "ending_equity": 100.0 * equity,
        "total_return": total,
        "cagr": cagr,
        "profit_factor": pos / neg if neg > 0 else math.inf,
        "sharpe": math.sqrt(52.0) * float(r.mean()) / sd if sd > 0 else math.nan,
        "sortino": math.sqrt(52.0) * float(r.mean()) / downside if downside and np.isfinite(downside) and downside > 0 else math.nan,
        "max_drawdown": mdd,
        "weekly_wr": float((r > 0).mean()),
    }


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100*x:.2f}%"


def select_weights(elig: pd.DataFrame, strategy: str, tail_frac: float, high_vol_frac: float) -> dict[str, float]:
    q = elig.copy()
    if strategy == "HIGH_VOL_REVERSAL":
        n_keep = max(10, int(math.ceil(len(q) * high_vol_frac)))
        q = q.nlargest(min(n_keep, len(q)), "vol_8w")
    if len(q) < 10:
        return {}
    k = max(1, int(math.ceil(len(q) * tail_frac)))
    losers = q.nsmallest(k, "ret_8w")
    winners = q.nlargest(k, "ret_8w")
    lw = 0.5 / len(losers)
    sw = -0.5 / len(winners)
    out = {s: lw for s in losers.symbol}
    for s in winners.symbol:
        out[s] = out.get(s, 0.0) + sw
    return {s: w for s, w in out.items() if abs(w) > 1e-15}


def main() -> int:
    cfg = parse_args()
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    outdir = Path(cfg.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(cfg.db, timeout=120)

    print("=== 8-WEEK CRYPTO REVERSAL -> BINANCE PERP AUDIT ===")
    print(f"Evaluation: {start.date()} -> {end.date()}")
    print("External premise: Binance spot 2021-2026 paper reports 8-10w cross-sectional reversal;")
    print("baseline ~39.6% annualized / Sharpe ~0.96; high-vol stronger and robust to large costs.")
    print("This is a FUTURES TRANSFER TEST, not a claimed exact replication of the spot paper.")
    print(f"Point-in-time universe: top {cfg.universe} USD-M perps by lagged 30d quote volume")
    print("Signal: rank own trailing 56d return; long bottom quintile, short top quintile")
    print("BASELINE_REVERSAL = all liquid universe; HIGH_VOL_REVERSAL = top half by trailing 56d realized vol")
    print(f"Portfolio: 50% long + 50% short = 1x gross, ~0 net; costs {cfg.side_cost_bps:.1f} bps per changed notional side + real funding")
    print("No leverage, no SL/TP, no parameter optimization. Weekly Sunday UTC rebalance.\n")

    daily = load_daily(con, start, end)
    sun = daily[daily.date.dt.dayofweek == 6].copy()
    sun = v3.attach_forward_exits(sun, daily)
    sun = sun[(sun.date >= start) & (sun.date <= end)]
    sun = sun.replace([np.inf, -np.inf], np.nan)
    sun = sun.dropna(subset=["ret_8w", "vol_8w", "liq_30d", "close"])
    sun = sun[(sun.history_days >= 60) & (sun.liq_30d > 0)]
    pref = funding_prefix(con, daily.symbol.unique().tolist(), start, end)

    strategies = ["BASELINE_REVERSAL", "HIGH_VOL_REVERSAL"]
    prev = {s: {} for s in strategies}
    rows = []
    assets = []
    side_cost = cfg.side_cost_bps / 10_000.0

    dates = sorted(pd.Timestamp(x) for x in sun.date.unique())
    for j, dt in enumerate(dates, 1):
        cross = sun[sun.date == dt].copy()
        if len(cross) < cfg.min_cross_section:
            continue
        cross = cross.nlargest(min(cfg.universe, len(cross)), "liq_30d")
        if len(cross) < cfg.min_cross_section:
            continue
        bysym = cross.set_index("symbol")
        for strat in strategies:
            weights = select_weights(cross, strat, cfg.tail_frac, cfg.high_vol_frac)
            if not weights:
                continue
            missing = [s for s in weights if s not in bysym.index or not np.isfinite(bysym.loc[s, "fwd_ret"])]
            if missing:
                raise RuntimeError(f"Selected assets lack unbiased weekly exit at {dt.date()} {strat}: {missing[:10]}")

            turnover = sum(abs(weights.get(s, 0.0) - prev[strat].get(s, 0.0)) for s in set(weights) | set(prev[strat]))
            gross = 0.0; funding = 0.0; known = 0; forced_notional = 0.0
            for sym, w in weights.items():
                rr = bysym.loc[sym]
                r = float(rr.fwd_ret)
                actual = pd.Timestamp(rr.actual_exit_date)
                fr, ok = funding_sum(pref, sym, dt, actual)
                gp = w * r
                fp = -w * fr
                gross += gp; funding += fp; known += int(ok)
                forced = bool(rr.forced_exit)
                if forced:
                    forced_notional += abs(w)
                assets.append({
                    "strategy": strat, "date": dt, "symbol": sym, "weight": w,
                    "formation_return_8w": float(rr.ret_8w), "vol_8w": float(rr.vol_8w),
                    "fwd_return": r, "gross_contribution": gp, "funding_sum": fr,
                    "funding_contribution": fp, "forced_exit": forced,
                })
            if forced_notional > 0:
                turnover += forced_notional
            cost = turnover * side_cost
            net = gross + funding - cost
            rows.append({
                "strategy": strat, "date": dt, "positions": len(weights),
                "gross_exposure": sum(abs(w) for w in weights.values()),
                "net_exposure": sum(weights.values()), "turnover": turnover,
                "gross_return": gross, "funding_return": funding, "cost_return": cost,
                "net_return": net, "funding_coverage": known / len(weights),
            })
            forced_syms = {s for s in weights if bool(bysym.loc[s, "forced_exit"])}
            prev[strat] = {s: w for s, w in weights.items() if s not in forced_syms}
        if j % 25 == 0 or j == len(dates):
            print(f"Weekly portfolio pass: {j}/{len(dates)} | {dt.date()}", flush=True)

    rdf = pd.DataFrame(rows).sort_values(["strategy", "date"]).reset_index(drop=True)
    adf = pd.DataFrame(assets)
    if rdf.empty:
        raise RuntimeError("No reversal portfolio weeks")

    # Terminal closing cost for still-open target weights.
    for strat in strategies:
        if prev[strat]:
            idxs = rdf.index[rdf.strategy == strat]
            if len(idxs):
                i = idxs[-1]
                terminal = sum(abs(w) for w in prev[strat].values())
                rdf.loc[i, "turnover"] += terminal
                rdf.loc[i, "cost_return"] += terminal * side_cost
                rdf.loc[i, "net_return"] -= terminal * side_cost

    summary_rows = []
    year_rows = []
    for strat, q in rdf.groupby("strategy"):
        q = q.sort_values("date")
        m = perf(q.net_return)
        m.update({
            "strategy": strat,
            "weeks": len(q),
            "avg_positions": float(q.positions.mean()),
            "avg_turnover": float(q.turnover.mean()),
            "avg_gross": float(q.gross_return.mean()),
            "avg_funding": float(q.funding_return.mean()),
            "avg_cost": float(q.cost_return.mean()),
            "funding_coverage": float(q.funding_coverage.mean()),
        })
        summary_rows.append(m)
        for year, yy in q.groupby(q.date.dt.year):
            mm = perf(yy.net_return)
            year_rows.append({"strategy": strat, "year": int(year), **mm})

    sdf = pd.DataFrame(summary_rows)
    ydf = pd.DataFrame(year_rows)

    # Contribution concentration uses gross + funding before shared turnover costs.
    contrib = pd.DataFrame()
    if not adf.empty:
        adf["pre_cost_contribution"] = adf.gross_contribution + adf.funding_contribution
        contrib = adf.groupby(["strategy", "symbol"], as_index=False).pre_cost_contribution.sum()

    print("\n=== 8-WEEK REVERSAL PERP RESULT ===")
    pp = sdf.copy()
    for c in ["total_return", "cagr", "max_drawdown", "weekly_wr", "avg_gross", "avg_funding", "avg_cost"]:
        pp[c] = pp[c].map(pct)
    print(pp[["strategy","weeks","ending_equity","total_return","cagr","weekly_wr","profit_factor","sharpe","sortino","max_drawdown","avg_positions","avg_turnover","avg_gross","avg_funding","avg_cost","funding_coverage"]].to_string(index=False))

    print("\nYEAR BREAKDOWN")
    yp = ydf.copy()
    for c in ["total_return", "cagr", "max_drawdown", "weekly_wr"]:
        yp[c] = yp[c].map(pct)
    print(yp[["strategy","year","total_return","profit_factor","sharpe","max_drawdown","weekly_wr"]].to_string(index=False))

    hv = sdf.set_index("strategy").loc["HIGH_VOL_REVERSAL"]
    hy = ydf[ydf.strategy == "HIGH_VOL_REVERSAL"].set_index("year")
    post = rdf[(rdf.strategy == "HIGH_VOL_REVERSAL") & (rdf.date >= pd.Timestamp("2026-04-01", tz="UTC"))]
    post_m = perf(post.net_return) if len(post) else None
    top_share = math.nan
    if not contrib.empty:
        cc = contrib[contrib.strategy == "HIGH_VOL_REVERSAL"]
        pos = cc[cc.pre_cost_contribution > 0].pre_cost_contribution
        if len(pos) and pos.sum() > 0:
            top_share = float(pos.max() / pos.sum())

    gates = [
        ("HIGH_VOL net total > 0", hv.total_return > 0),
        ("HIGH_VOL PF > 1.30", hv.profit_factor > 1.30),
        ("HIGH_VOL Sharpe > 1.00", hv.sharpe > 1.00),
        ("2024 positive", 2024 in hy.index and hy.loc[2024, "total_return"] > 0),
        ("2025 positive", 2025 in hy.index and hy.loc[2025, "total_return"] > 0),
        ("2026 positive", 2026 in hy.index and hy.loc[2026, "total_return"] > 0),
        ("Post-paper Apr-Jul 2026 positive", post_m is not None and post_m["total_return"] > 0),
        ("MDD better than -50%", hv.max_drawdown > -0.50),
        ("Top positive contributor <35%", np.isfinite(top_share) and top_share < 0.35),
    ]
    print("\nPRE-REGISTERED FUTURES-TRANSFER GATES")
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    print(f"Top positive contributor share: {pct(top_share)}")
    if post_m is not None:
        print(f"Post-paper Apr-Jul 2026: return={pct(post_m['total_return'])} PF={post_m['profit_factor']:.3f} Sharpe={post_m['sharpe']:.3f}")
    overall = all(ok for _, ok in gates)
    print("Overall:", "PASS" if overall else "FAIL")

    print("\nVERDICT")
    if overall:
        print("[KEEP] Paper-grounded 8-week reversal transfers cleanly to Binance perps at 1x after funding/costs. Next step: robustness, then controlled leverage.")
    elif hv.profit_factor > 1.0 and hv.sharpe > 0:
        print("[WEAK TRANSFER] Reversal remains positive but misses production gates. Diagnose only paper-specified robustness; do not hyperopt.")
    else:
        print("[CLOSE TRANSFER] The documented Binance-spot reversal does not transfer to our USD-M perp implementation. Do not parameter-fit it.")

    rdf.to_csv(outdir / "weekly_results.csv", index=False)
    sdf.to_csv(outdir / "summary.csv", index=False)
    ydf.to_csv(outdir / "year_breakdown.csv", index=False)
    adf.to_csv(outdir / "asset_contributions.csv", index=False)
    if not contrib.empty:
        contrib.to_csv(outdir / "contribution_summary.csv", index=False)
    pd.DataFrame([{"gate": label, "pass": bool(ok)} for label, ok in gates]).to_csv(outdir / "gates.csv", index=False)
    print(f"\nSaved under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
