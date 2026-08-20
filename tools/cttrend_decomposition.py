#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import cttrend_research_v3 as v3

base = v3.base
STRATEGIES = ("CTREND_ONLY", "TSMOM_ONLY", "HYBRID")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CTREND decomposition: CTREND-only vs 28d TSMOM-only vs hybrid"
    )
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--universe", type=int, default=50)
    p.add_argument("--top-frac", type=float, default=0.20)
    p.add_argument("--train-weeks", type=int, default=52)
    p.add_argument("--min-history-days", type=int, default=210)
    p.add_argument("--min-cross-section", type=int, default=15)
    p.add_argument("--side-cost-bps", type=float, default=7.0)
    p.add_argument("--workers", type=int, default=int(os.environ.get("CTREND_WORKERS", "16")))
    p.add_argument("--output-dir", default="/freqtrade/user_data/cttrend_decomposition")
    return p.parse_args()


def fit_all_weeks(
    panel: pd.DataFrame,
    dates: list[pd.Timestamp],
    train_weeks: int,
    min_cs: int,
    workers: int,
) -> dict[pd.Timestamp, object | None]:
    fits: dict[pd.Timestamp, object | None] = {}

    def one(dt: pd.Timestamp):
        return dt, base.fit_week(panel, dt, train_weeks, min_cs)

    workers = max(1, workers)
    print(f"Model workers: {workers}", flush=True)
    if workers == 1:
        for n, dt in enumerate(dates, 1):
            _, fit = one(dt)
            fits[dt] = fit
            if n % 20 == 0 or n == len(dates):
                print(f"Model fits: {n}/{len(dates)}", flush=True)
        return fits

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, dt) for dt in dates]
        for n, fut in enumerate(as_completed(futs), 1):
            dt, fit = fut.result()
            fits[dt] = fit
            if n % 20 == 0 or n == len(dates):
                print(f"Model fits: {n}/{len(dates)}", flush=True)
    return fits


def weights_for(name: str, cur: pd.DataFrame, top_frac: float) -> dict[str, float]:
    if name == "CTREND_ONLY":
        x = cur.dropna(subset=["cttrend"])
        if x.empty:
            return {}
        k = max(1, int(math.ceil(len(x) * top_frac)))
        x = x.nlargest(k, "cttrend")
    elif name == "TSMOM_ONLY":
        x = cur[cur["ret_28d"] > 0]
    elif name == "HYBRID":
        x = cur.dropna(subset=["cttrend"])
        if x.empty:
            return {}
        k = max(1, int(math.ceil(len(x) * top_frac)))
        x = x.nlargest(k, "cttrend")
        x = x[x["ret_28d"] > 0]
    else:
        raise ValueError(name)

    if x.empty:
        return {}
    w = 1.0 / len(x)
    return {str(s): w for s in x["symbol"]}


def exact_funding_for_shortened_week(
    con: sqlite3.Connection,
    symbol: str,
    entry_date: pd.Timestamp,
    exit_date: pd.Timestamp,
) -> tuple[float, bool]:
    # Entry is Sunday's daily close; the holding interval starts Monday 00:00 UTC.
    lo = int((entry_date + pd.Timedelta(days=1)).timestamp() * 1000)
    # exit_date is also a complete daily close; include funding through that UTC day.
    hi = int((exit_date + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    row = con.execute(
        "SELECT SUM(rate), COUNT(*) FROM funding_events WHERE symbol=? AND event_time>=? AND event_time<=?",
        (symbol, lo, hi),
    ).fetchone()
    if row is None or int(row[1] or 0) == 0:
        return 0.0, False
    return float(row[0] or 0.0), True


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100.0 * x:.2f}%"


def run_decomposition(
    con: sqlite3.Connection,
    panel: pd.DataFrame,
    funding: dict[tuple[str, pd.Timestamp], float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_weeks: int,
    top_frac: float,
    min_cs: int,
    side_cost_bps: float,
    workers: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    all_dates = sorted(pd.Timestamp(d) for d in panel["date"].unique())
    dates: list[pd.Timestamp] = []
    for i, dt in enumerate(all_dates):
        if dt < start or dt > end - pd.Timedelta(days=7):
            continue
        if i < train_weeks:
            continue
        dates.append(dt)
    if not dates:
        raise RuntimeError("No eligible OOS dates")

    fits = fit_all_weeks(panel, dates, train_weeks, min_cs, workers)
    prev: dict[str, dict[str, float]] = {s: {} for s in STRATEGIES}
    rows: list[dict] = []
    ic_rows: list[dict] = []
    side_cost = side_cost_bps / 10_000.0

    for n, dt in enumerate(dates, 1):
        cur = panel[panel["date"] == dt].copy()
        if len(cur) < min_cs:
            continue

        fit = fits.get(dt)
        cur["cttrend"] = np.nan
        selected_features = 0
        alpha = np.nan
        if fit is not None:
            cur["cttrend"] = fit.score.reindex(cur.index)
            selected_features = len(fit.features)
            alpha = fit.alpha

        valid_ic = cur.dropna(subset=["cttrend", "fwd_ret"])
        if len(valid_ic) >= min_cs:
            ic = valid_ic["cttrend"].rank(method="average").corr(
                valid_ic["fwd_ret"].rank(method="average")
            )
        else:
            ic = np.nan
        ic_rows.append({
            "date": dt,
            "rank_ic": ic,
            "cross_section": len(valid_ic),
            "selected_features": selected_features,
            "enet_alpha": alpha,
        })

        bysym = cur.set_index("symbol")
        planned_week_end = dt + pd.Timedelta(days=7)

        for name in STRATEGIES:
            cur_weights = weights_for(name, cur, top_frac)
            old = prev[name]
            turnover = sum(
                abs(cur_weights.get(s, 0.0) - old.get(s, 0.0))
                for s in set(old) | set(cur_weights)
            )
            cost = side_cost * turnover
            gross = 0.0
            fund = 0.0
            fund_known = 0
            forced_notional = 0.0
            forced_syms: set[str] = set()
            shortened_exits = 0

            for sym, w in cur_weights.items():
                if sym not in bysym.index:
                    raise RuntimeError(f"{name}: selected symbol missing: {sym} {dt}")
                rr = bysym.loc[sym, "fwd_ret"]
                if pd.isna(rr):
                    raise RuntimeError(
                        f"{name}: selected {sym} on {dt.date()} has no unbiased exit price"
                    )
                gross += w * float(rr)

                actual_exit = pd.Timestamp(bysym.loc[sym, "actual_exit_date"])
                if actual_exit < planned_week_end:
                    fr, known = exact_funding_for_shortened_week(con, sym, dt, actual_exit)
                    shortened_exits += 1
                else:
                    key = (sym, planned_week_end)
                    known = key in funding
                    fr = float(funding.get(key, 0.0))
                if known:
                    fund_known += 1
                fund -= w * fr

                forced = bool(bysym.loc[sym, "forced_exit"])
                if forced:
                    forced_notional += abs(w)
                    forced_syms.add(sym)

            if forced_notional:
                turnover += forced_notional
                cost += side_cost * forced_notional

            net = gross + fund - cost
            rows.append({
                "date": dt,
                "strategy": name,
                "positions": len(cur_weights),
                "gross_return": gross,
                "funding_return": fund,
                "turnover": turnover,
                "cost_return": cost,
                "net_return": net,
                "funding_coverage": fund_known / len(cur_weights) if cur_weights else 1.0,
                "forced_exits": len(forced_syms),
                "shortened_exits": shortened_exits,
                "selected_features": selected_features,
            })
            prev[name] = {s: w for s, w in cur_weights.items() if s not in forced_syms}

        if n % 20 == 0 or n == len(dates):
            h = [r for r in rows if r["date"] == dt and r["strategy"] == "HYBRID"]
            hn = h[-1]["net_return"] if h else np.nan
            print(
                f"Portfolio pass: {n}/{len(dates)} | {dt.date()} | hybrid={hn:+.3%} | IC={ic:+.3f}",
                flush=True,
            )

    w = pd.DataFrame(rows).sort_values(["date", "strategy"]).reset_index(drop=True)

    # One explicit terminal close per strategy, same convention as V3.
    for name in STRATEGIES:
        final_notional = sum(abs(x) for x in prev[name].values())
        if final_notional <= 0:
            continue
        idxs = w.index[w["strategy"] == name]
        if len(idxs) == 0:
            continue
        idx = idxs[-1]
        extra = side_cost * final_notional
        w.loc[idx, "turnover"] += final_notional
        w.loc[idx, "cost_return"] += extra
        w.loc[idx, "net_return"] -= extra

    summaries: list[dict] = []
    years: list[dict] = []
    for name in STRATEGIES:
        x = w[w["strategy"] == name].sort_values("date").copy()
        x["equity"] = 100.0 * (1.0 + x["net_return"]).cumprod()
        w.loc[x.index, "equity"] = x["equity"]
        m = base.metrics(x["net_return"])
        active = x[(x["positions"] > 0) | (x["turnover"] > 1e-12)]
        invested = x[x["positions"] > 0]
        wr = float((active["net_return"] > 0).mean()) if len(active) else np.nan
        summaries.append({
            "strategy": name,
            "weeks": len(x),
            "active_weeks": len(active),
            "ending_equity": float(x["equity"].iloc[-1]),
            "total_return": m["total"],
            "cagr": m["cagr"],
            "profit_factor": m["pf"],
            "sharpe": m["sharpe"],
            "sortino": m["sortino"],
            "max_drawdown": m["mdd"],
            "active_weekly_wr": wr,
            "avg_positions": float(x["positions"].mean()),
            "avg_turnover": float(x["turnover"].mean()),
            "avg_gross": float(x["gross_return"].mean()),
            "avg_funding": float(x["funding_return"].mean()),
            "avg_cost": float(x["cost_return"].mean()),
            "funding_coverage": float(invested["funding_coverage"].mean()) if len(invested) else np.nan,
            "forced_exits": int(x["forced_exits"].sum()),
            "shortened_exits": int(x["shortened_exits"].sum()),
        })
        for year, y in x.groupby(x["date"].dt.year):
            ym = base.metrics(y["net_return"])
            ya = y[(y["positions"] > 0) | (y["turnover"] > 1e-12)]
            years.append({
                "strategy": name,
                "year": int(year),
                "return": float((1.0 + y["net_return"]).prod() - 1.0),
                "active_wr": float((ya["net_return"] > 0).mean()) if len(ya) else np.nan,
                "pf": ym["pf"],
                "sharpe": ym["sharpe"],
                "mdd": ym["mdd"],
            })

    return w, pd.DataFrame(summaries), pd.DataFrame(ic_rows), pd.DataFrame(years)


def report(summary: pd.DataFrame, ic: pd.DataFrame, yearly: pd.DataFrame) -> None:
    print("\n=== CTREND DECOMPOSITION AUDIT ===")
    z = summary.copy()
    for c in (
        "total_return", "cagr", "max_drawdown", "active_weekly_wr",
        "avg_gross", "avg_funding", "avg_cost",
    ):
        z[c] = z[c].map(pct)
    cols = [
        "strategy", "ending_equity", "total_return", "cagr", "active_weekly_wr",
        "profit_factor", "sharpe", "sortino", "max_drawdown", "avg_positions",
        "avg_turnover", "avg_gross", "avg_funding", "avg_cost",
        "forced_exits", "shortened_exits",
    ]
    print(z[cols].to_string(index=False))

    clean = ic["rank_ic"].dropna()
    mean_ic = float(clean.mean()) if len(clean) else np.nan
    median_ic = float(clean.median()) if len(clean) else np.nan
    positive_ic = float((clean > 0).mean()) if len(clean) else np.nan
    sd = float(clean.std(ddof=1)) if len(clean) > 1 else np.nan
    tstat = mean_ic / (sd / math.sqrt(len(clean))) if len(clean) > 1 and sd > 0 else np.nan

    print("\nCTREND NEXT-WEEK CROSS-SECTIONAL RANK IC")
    print(
        f"Weeks={len(clean)} | mean={mean_ic:+.4f} | median={median_ic:+.4f} | "
        f"positive={positive_ic:.2%} | t-stat={tstat:+.2f}"
    )

    print("\nYEAR BREAKDOWN")
    y = yearly.copy()
    y["return"] = y["return"].map(pct)
    y["active_wr"] = y["active_wr"].map(pct)
    y["mdd"] = y["mdd"].map(pct)
    print(y.to_string(index=False))

    by = summary.set_index("strategy")
    cp = by.loc["CTREND_ONLY"]
    tp = by.loc["TSMOM_ONLY"]
    hp = by.loc["HYBRID"]
    c_ok = bool(mean_ic > 0 and cp["profit_factor"] > 1.0 and cp["sharpe"] > 0)
    t_ok = bool(tp["profit_factor"] > 1.0 and tp["sharpe"] > 0)

    print("\nDIAGNOSTIC VERDICT")
    if not np.isfinite(mean_ic) or mean_ic <= 0:
        print("[CLOSE CTREND] CTREND rank IC <= 0 on Binance perpetual OOS universe.")
    elif c_ok and not t_ok:
        print("[KEEP CTREND / DROP TSMOM] CTREND survives; standalone 28d TSMOM does not.")
    elif t_ok and not c_ok:
        print("[KEEP TSMOM / DROP CTREND] 28d TSMOM survives; CTREND implementation does not.")
    elif c_ok and t_ok:
        print("[BOTH HAVE EDGE] Both components survive independently; hybrid failure is interaction/portfolio construction.")
    else:
        print("[CLOSE BOTH] Neither component shows positive standalone net expectancy.")

    print(
        f"Hybrid reference: PF={hp['profit_factor']:.3f}, Sharpe={hp['sharpe']:.3f}, "
        f"Total={pct(float(hp['total_return']))}"
    )


def main() -> int:
    cfg = parse_args()
    if cfg.workers < 1:
        raise ValueError("--workers must be >= 1")
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    db = Path(cfg.db)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not db.exists():
        raise FileNotFoundError(db)

    print("=== CTREND COMPONENT DECOMPOSITION ===")
    print(f"Evaluation: {start.date()} -> {end.date()}")
    print(f"Universe: same point-in-time top {cfg.universe} Binance USDT perps")
    print(f"CTREND-only: top {cfg.top_frac:.0%}, no TSMOM gate")
    print("TSMOM-only: all universe members with own 28d return > 0")
    print(f"Hybrid: top {cfg.top_frac:.0%} CTREND AND own 28d return > 0")
    print(f"Costs/funding: same {cfg.side_cost_bps:.1f} bps per changed side + archived funding")
    print("No parameter search. Post-mortem decomposition of the failed V3 test.\n")

    con = sqlite3.connect(str(db), timeout=120)
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-262144")

    daily = base.load_daily(con, start, end)
    panel = base.weekly_panel(
        daily, end, cfg.universe, cfg.min_history_days, cfg.min_cross_section
    )
    funding = base.funding_by_week(con, panel.symbol.unique(), start, end)

    weekly, summary, ic, yearly = run_decomposition(
        con, panel, funding, start, end, cfg.train_weeks, cfg.top_frac,
        cfg.min_cross_section, cfg.side_cost_bps, cfg.workers,
    )

    weekly.to_csv(out / "weekly_decomposition.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    ic.to_csv(out / "rank_ic_by_week.csv", index=False)
    yearly.to_csv(out / "year_breakdown.csv", index=False)

    report(summary, ic, yearly)
    print(f"\nSaved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
