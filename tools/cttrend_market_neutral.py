#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

import cttrend_author_causal as ac

STRATEGIES = (
    "TOP_LONG",
    "BOTTOM_LONG",
    "BOTTOM_SHORT",
    "HL_50_50",
    "UNIVERSE_LONG",
    "TOP_VS_UNIVERSE",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Causal CTREND spread monetization audit using the exact author-derived signal"
    )
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--train-weeks", type=int, default=52)
    p.add_argument("--eval-universe", type=int, default=50)
    p.add_argument("--top-frac", type=float, default=0.20)
    p.add_argument("--min-cross-section", type=int, default=25)
    p.add_argument("--side-cost-bps", type=float, default=7.0)
    p.add_argument("--workers", type=int, default=int(os.environ.get("CTREND_WORKERS", "32")))
    p.add_argument("--output-dir", default="/freqtrade/user_data/cttrend_market_neutral")
    return p.parse_args()


def combine_weights(*legs: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for leg in legs:
        for sym, w in leg.items():
            out[sym] = out.get(sym, 0.0) + float(w)
    return {s: w for s, w in out.items() if abs(w) > 1e-15}


def equal_leg(symbols: list[str], gross: float, sign: float = 1.0) -> dict[str, float]:
    if not symbols:
        return {}
    w = sign * gross / len(symbols)
    return {s: w for s in symbols}


def evaluate_strategy(
    name: str,
    target: dict[str, float],
    prev: dict[str, float],
    rows_by_symbol: dict[str, object],
    raw_ret_map: dict[str, float],
    actual_map: dict[str, pd.Timestamp],
    forced_map: dict[str, bool],
    pref,
    side_cost: float,
    period_ord: int,
    week_key: int,
    period_end: pd.Timestamp,
) -> tuple[dict, list[dict], dict[str, float]]:
    missing = [s for s in target if s not in raw_ret_map]
    if missing:
        raise RuntimeError(
            f"{name}: selected assets lack causal exit in period {week_key}: {missing}"
        )

    turnover = sum(
        abs(target.get(s, 0.0) - prev.get(s, 0.0)) for s in set(target) | set(prev)
    )
    cost = side_cost * turnover
    gross = 0.0
    funding = 0.0
    known_notional = 0.0
    gross_notional = sum(abs(w) for w in target.values())
    forced_notional = 0.0
    assets: list[dict] = []

    for sym, w in target.items():
        rr = rows_by_symbol[sym]
        r = float(raw_ret_map[sym])
        actual = actual_map[sym]
        forced = bool(forced_map[sym])
        fr, known = ac.funding_sum(pref, sym, pd.Timestamp(rr.lag_formation_date), actual)
        gp = w * r
        fp = -w * fr
        gross += gp
        funding += fp
        if known:
            known_notional += abs(w)
        if forced:
            forced_notional += abs(w)
        assets.append({
            "strategy": name,
            "period_ord": period_ord,
            "week_key": week_key,
            "symbol": sym,
            "weight": w,
            "cttrend": float(rr.cttrend),
            "raw_return": r,
            "funding_sum": fr,
            "gross_contribution": gp,
            "funding_contribution": fp,
            "forced_exit": forced,
            "entry_date": rr.lag_formation_date,
            "exit_date": actual,
        })

    # A contract that disappears during the holding period must be closed then;
    # charge that side and do not carry it into the next rebalance state.
    if forced_notional:
        turnover += forced_notional
        cost += side_cost * forced_notional

    net = gross + funding - cost
    carried = {
        s: w for s, w in target.items()
        if not forced_map.get(s, False)
    }
    return ({
        "strategy": name,
        "period_ord": period_ord,
        "week_key": week_key,
        "period_end": period_end,
        "positions": len(target),
        "gross_exposure": gross_notional,
        "net_exposure": sum(target.values()),
        "gross_return": gross,
        "funding_return": funding,
        "turnover": turnover,
        "cost_return": cost,
        "net_return": net,
        "funding_coverage": known_notional / gross_notional if gross_notional > 0 else 1.0,
    }, assets, carried)


def summarize(w: pd.DataFrame) -> pd.DataFrame:
    out = []
    for name, q in w.groupby("strategy", sort=False):
        q = q.sort_values("period_ord")
        m = ac.perf(q.net_return)
        active = q[(q.gross_exposure > 1e-12) | (q.turnover > 1e-12)]
        out.append({
            "strategy": name,
            "ending_equity": m["equity"],
            "total_return": m["total"],
            "cagr": m["cagr"],
            "active_wr": float((active.net_return > 0).mean()) if len(active) else np.nan,
            "profit_factor": m["pf"],
            "sharpe": m["sharpe"],
            "max_drawdown": m["mdd"],
            "avg_gross_exposure": float(q.gross_exposure.mean()),
            "avg_net_exposure": float(q.net_exposure.mean()),
            "avg_turnover": float(q.turnover.mean()),
            "avg_gross": float(q.gross_return.mean()),
            "avg_funding": float(q.funding_return.mean()),
            "avg_cost": float(q.cost_return.mean()),
            "funding_coverage": float(q[q.gross_exposure > 0].funding_coverage.mean()),
        })
    return pd.DataFrame(out)


def year_breakdown(w: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (name, year), q in w.groupby(["strategy", w.period_end.dt.year], sort=False):
        m = ac.perf(q.net_return)
        out.append({
            "strategy": name,
            "year": int(year),
            "return": m["total"],
            "pf": m["pf"],
            "sharpe": m["sharpe"],
            "mdd": m["mdd"],
            "wr": float((q.net_return > 0).mean()),
        })
    return pd.DataFrame(out)


def main() -> int:
    cfg = parse_args()
    if not 0 < cfg.top_frac <= 0.5:
        raise ValueError("--top-frac must be in (0, 0.5]")
    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    outdir = Path(cfg.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    con = sqlite3.connect(cfg.db, timeout=120)
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-262144")

    print("=== CTREND MARKET-NEUTRAL SPREAD AUDIT ===")
    print(f"Evaluation: {start.date()} -> {end.date()}")
    print("Signal: unchanged author-code-derived causal CTREND")
    print(f"Universe: point-in-time top {cfg.eval_universe} Binance USDT perps by lagged 30d quote volume")
    print(f"Tails: top/bottom {cfg.top_frac:.0%}")
    print("Primary trade: 50% long TOP + 50% short BOTTOM (gross 1x, net ~0)")
    print("Diagnostics: TOP long, BOTTOM long/short, universe long, TOP-vs-universe hedge")
    print(f"Execution: {cfg.side_cost_bps:.1f} bps per changed side + archived funding; no stop/TP/hyperopt")
    print(f"Model workers: {cfg.workers}\n")

    raw = ac.raw_daily(con, start, end)
    rc = ac.apply_author_truncation(raw)
    weekly, raw_groups = ac.build_weekly(rc, cfg.workers)
    panel = ac.build_target_panel(weekly)
    period_meta = (
        weekly[["period_ord", "week_key", "period_end"]]
        .drop_duplicates()
        .sort_values("period_ord")
    )
    eval_ords = period_meta[
        (period_meta.period_end >= start) & (period_meta.period_end <= end)
    ].period_ord.astype(int).tolist()
    eval_ords = [o for o in eval_ords if o >= cfg.train_weeks]
    print(f"Evaluation periods with 52-week history: {len(eval_ords)}", flush=True)

    fits = ac.fit_all(panel, eval_ords, cfg.train_weeks, cfg.min_cross_section, cfg.workers)
    pref = ac.funding_prefix(con, panel.symbol.unique().tolist(), start, end)

    side_cost = cfg.side_cost_bps / 10_000.0
    prev = {name: {} for name in STRATEGIES}
    weekly_rows: list[dict] = []
    asset_rows: list[dict] = []
    signal_rows: list[dict] = []

    for i, o in enumerate(eval_ords, 1):
        fit = fits.get(o)
        cur = panel[(panel.period_ord == o) & panel.all_x].copy()
        if fit is None or cur.empty:
            continue
        cur["cttrend"] = fit.scores.reindex(cur.index)
        cur = cur.dropna(subset=["cttrend"])
        if len(cur) < cfg.min_cross_section:
            continue

        end_date = pd.Timestamp(cur.period_end.iloc[0])
        week_key = int(cur.week_key.iloc[0])
        raw_ret_map: dict[str, float] = {}
        actual_map: dict[str, pd.Timestamp] = {}
        forced_map: dict[str, bool] = {}

        for rr in cur.itertuples():
            fx = ac.forward_exit(raw_groups, rr.symbol, rr.lag_formation_date, end_date)
            if fx is None or not np.isfinite(rr.lag_formation_close):
                continue
            actual, exit_close, forced = fx
            raw_ret_map[rr.symbol] = exit_close / float(rr.lag_formation_close) - 1.0
            actual_map[rr.symbol] = actual
            forced_map[rr.symbol] = forced

        # Point-in-time universe is chosen without looking at exit availability.
        elig = cur.dropna(subset=["lag_liq_30d", "cttrend"])
        elig = elig.nlargest(min(cfg.eval_universe, len(elig)), "lag_liq_30d")
        if len(elig) < cfg.min_cross_section:
            continue
        k = max(1, int(math.ceil(len(elig) * cfg.top_frac)))
        top = elig.nlargest(k, "cttrend")
        bottom = elig.nsmallest(k, "cttrend")
        top_syms = top.symbol.tolist()
        bottom_syms = bottom.symbol.tolist()
        universe_syms = elig.symbol.tolist()

        # Abort rather than make the universe conditional on future data availability.
        needed = set(top_syms) | set(bottom_syms) | set(universe_syms)
        missing = sorted(s for s in needed if s not in raw_ret_map)
        if missing:
            raise RuntimeError(
                f"Period {week_key}: point-in-time evaluation universe has no causal exit for {missing}. "
                "Aborting instead of future-filtering the universe."
            )

        rows_by_symbol = {rr.symbol: rr for rr in elig.itertuples()}
        top_long = equal_leg(top_syms, 1.0, +1.0)
        bottom_long = equal_leg(bottom_syms, 1.0, +1.0)
        bottom_short = equal_leg(bottom_syms, 1.0, -1.0)
        hl = combine_weights(
            equal_leg(top_syms, 0.5, +1.0),
            equal_leg(bottom_syms, 0.5, -1.0),
        )
        universe_long = equal_leg(universe_syms, 1.0, +1.0)
        top_vs_universe = combine_weights(
            equal_leg(top_syms, 0.5, +1.0),
            equal_leg(universe_syms, 0.5, -1.0),
        )
        targets = {
            "TOP_LONG": top_long,
            "BOTTOM_LONG": bottom_long,
            "BOTTOM_SHORT": bottom_short,
            "HL_50_50": hl,
            "UNIVERSE_LONG": universe_long,
            "TOP_VS_UNIVERSE": top_vs_universe,
        }

        top_set = set(top_syms)
        bottom_set = set(bottom_syms)
        for rr in elig.itertuples():
            signal_rows.append({
                "period_ord": o,
                "week_key": week_key,
                "period_end": end_date,
                "symbol": rr.symbol,
                "cttrend": float(rr.cttrend),
                "lag_liq_30d": float(rr.lag_liq_30d),
                "raw_oos_ret": float(raw_ret_map[rr.symbol]),
                "is_top": rr.symbol in top_set,
                "is_bottom": rr.symbol in bottom_set,
            })

        for name in STRATEGIES:
            row, assets, carried = evaluate_strategy(
                name=name,
                target=targets[name],
                prev=prev[name],
                rows_by_symbol=rows_by_symbol,
                raw_ret_map=raw_ret_map,
                actual_map=actual_map,
                forced_map=forced_map,
                pref=pref,
                side_cost=side_cost,
                period_ord=o,
                week_key=week_key,
                period_end=end_date,
            )
            row["selected_features"] = len(fit.selected)
            weekly_rows.append(row)
            asset_rows.extend(assets)
            prev[name] = carried

        if i % 20 == 0 or i == len(eval_ords):
            last = [r for r in weekly_rows if r["period_ord"] == o and r["strategy"] == "HL_50_50"][-1]
            print(
                f"Spread pass: {i}/{len(eval_ords)} | {week_key} | "
                f"H-L net={last['net_return']:+.3%} | top={len(top_syms)} bottom={len(bottom_syms)}",
                flush=True,
            )

    if not weekly_rows:
        raise RuntimeError("No OOS spread periods produced")

    wdf = pd.DataFrame(weekly_rows).sort_values(["strategy", "period_ord"]).reset_index(drop=True)
    adf = pd.DataFrame(asset_rows)
    sdf = pd.DataFrame(signal_rows)

    # Explicit terminal close for every strategy.
    for name in STRATEGIES:
        if not prev[name]:
            continue
        terminal = sum(abs(v) for v in prev[name].values())
        idx = wdf.index[wdf.strategy == name][-1]
        wdf.loc[idx, "turnover"] += terminal
        wdf.loc[idx, "cost_return"] += terminal * side_cost
        wdf.loc[idx, "net_return"] -= terminal * side_cost

    summary = summarize(wdf)
    ydf = year_breakdown(wdf)

    print("\n=== CTREND SPREAD RESULT ===")
    show = summary.copy()
    for c in ["total_return", "cagr", "active_wr", "max_drawdown", "avg_gross", "avg_funding", "avg_cost"]:
        show[c] = show[c].map(ac.pct)
    print(show[[
        "strategy", "ending_equity", "total_return", "cagr", "active_wr",
        "profit_factor", "sharpe", "max_drawdown", "avg_gross_exposure",
        "avg_net_exposure", "avg_turnover", "avg_gross", "avg_funding", "avg_cost"
    ]].to_string(index=False))

    print("\nYEAR BREAKDOWN")
    yp = ydf.copy()
    for c in ["return", "mdd", "wr"]:
        yp[c] = yp[c].map(ac.pct)
    print(yp.to_string(index=False))

    # Cross-sectional monotonicity diagnostic using raw tradable returns.
    mono_rows = []
    for po, q in sdf.groupby("period_ord", sort=True):
        mono_rows.append({
            "period_ord": int(po),
            "top_raw": float(q.loc[q.is_top, "raw_oos_ret"].mean()),
            "universe_raw": float(q.raw_oos_ret.mean()),
            "bottom_raw": float(q.loc[q.is_bottom, "raw_oos_ret"].mean()),
        })
    mono = pd.DataFrame(mono_rows)
    mono["top_minus_bottom"] = mono.top_raw - mono.bottom_raw
    mono["top_minus_universe"] = mono.top_raw - mono.universe_raw
    print("\nRAW RANKING MONOTONICITY (before funding/costs)")
    print(f"Avg TOP:              {ac.pct(float(mono.top_raw.mean()))}")
    print(f"Avg UNIVERSE:         {ac.pct(float(mono.universe_raw.mean()))}")
    print(f"Avg BOTTOM:           {ac.pct(float(mono.bottom_raw.mean()))}")
    print(f"Avg TOP-BOTTOM spread:{ac.pct(float(mono.top_minus_bottom.mean()))}")
    print(f"TOP>BOTTOM periods:   {(mono.top_minus_bottom > 0).mean():.2%}")
    print(f"Avg TOP-UNIVERSE:     {ac.pct(float(mono.top_minus_universe.mean()))}")

    hl = summary.loc[summary.strategy == "HL_50_50"].iloc[0]
    hly = ydf[ydf.strategy == "HL_50_50"].set_index("year")
    gates = {
        "H-L total net > 0": float(hl.total_return) > 0,
        "H-L PF > 1.30": float(hl.profit_factor) > 1.30,
        "H-L Sharpe > 1.00": float(hl.sharpe) > 1.00,
        "2024 positive": 2024 in hly.index and float(hly.loc[2024, "return"]) > 0,
        "2025 positive": 2025 in hly.index and float(hly.loc[2025, "return"]) > 0,
        "2026 positive": 2026 in hly.index and float(hly.loc[2026, "return"]) > 0,
        "TOP raw > BOTTOM raw": float(mono.top_minus_bottom.mean()) > 0,
    }
    print("\nPRE-REGISTERED H-L GATES")
    for name, ok in gates.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"Overall: {'PASS' if all(gates.values()) else 'FAIL'}")

    print("\nSPREAD VERDICT")
    if all(gates.values()):
        print("[KEEP] CTREND ranking monetizes as a robust market-neutral Binance spread. Next step: finite robustness and risk scaling.")
    elif float(hl.total_return) > 0 and float(hl.sharpe) > 0:
        print("[WEAK SPREAD] Ranking has tradable spread value, but it misses the pre-registered quality gates. Do not leverage or hyperopt.")
    else:
        print("[CLOSE CTREND TRADING] Positive rank IC does not monetize into a usable top-minus-bottom spread after real funding/costs.")

    wdf.to_csv(outdir / "weekly_spreads.csv", index=False)
    adf.to_csv(outdir / "asset_spreads.csv", index=False)
    sdf.to_csv(outdir / "signal_panel.csv", index=False)
    summary.to_csv(outdir / "summary.csv", index=False)
    ydf.to_csv(outdir / "year_breakdown.csv", index=False)
    mono.to_csv(outdir / "ranking_monotonicity.csv", index=False)
    print(f"\nSaved under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
