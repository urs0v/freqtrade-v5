#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DB = "/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
DEFAULT_OUT = "/freqtrade/user_data/strategy_build/adaptivetrend/results"

# Paper-specified pieces.
TIMEFRAME_HOURS = 6
K_LONG = 15
K_SHORT = 15  # Paper leaves K_S unspecified; frozen symmetric documented-core assumption.
LONG_SHARPE_GATE = 1.3
SHORT_SHARPE_GATE = 1.7
LONG_ALLOC = 0.70
SHORT_ALLOC = 0.30
RISK_FREE_ANNUAL = 0.045
ANNUAL_PERIODS = 365.25 * 4.0
BUFFER_HOURS = 24

# Paper does not disclose the full optimization grid or ATR k. These values are frozen
# before seeing replication results and match the project's pre-existing AdaptiveTrend grid.
LOOKBACKS = [4, 6, 8, 10, 12, 16]
THRESHOLDS = [0.015, 0.025, 0.035, 0.050, 0.075]
ATR_MULTS = [2.0, 2.5, 3.0, 3.5]
ATR_PERIOD = 14
MIN_TRAIN_TRADES = 2
MIN_TRAIN_BARS = 80
MIN_MONTH_UNIVERSE = 75

# Cost is per fill. 4 bps is the Binance taker fee stated by the paper; 8/12 bps are
# deliberately harsher execution-cost stresses because the paper's exact slippage curve is undisclosed.
COST_SCENARIOS = [4.0, 8.0, 12.0]
BASE_COST_BPS = 4.0

# Predeclared replication gate. This is intentionally below the paper's reported SR=2.41,
# but high enough that a weak approximation is not promoted into a leveraged strategy.
GATE_MIN_SHARPE = 1.25
GATE_MAX_DD_PCT = 25.0
GATE_REQUIRE_ALL_YEARS_POSITIVE = True
GATE_MIN_COST8_SHARPE = 0.75


@dataclass
class Opt:
    sharpe: float
    lookback: int
    theta: float
    alpha: float
    trades: int


def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    while cur < end:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def load_symbol_frame(con: sqlite3.Connection, symbol: str) -> pd.DataFrame:
    df = pd.read_sql_query(
        "SELECT open_time,open,high,low,close,volume,close_time FROM candles WHERE symbol=? ORDER BY open_time",
        con,
        params=(symbol,),
    )
    if df.empty:
        return df
    for c in ("open", "high", "low", "close", "volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()
    df["time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True) + pd.Timedelta(hours=TIMEFRAME_HOURS)
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    df["atr"] = tr.ewm(alpha=1.0 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
    for lb in LOOKBACKS:
        df[f"mom_{lb}"] = df["close"].pct_change(lb)
    return df.reset_index(drop=True)


def load_funding(con: sqlite3.Connection, symbol: str) -> tuple[np.ndarray, np.ndarray]:
    x = pd.read_sql_query(
        "SELECT event_time,rate FROM funding_events WHERE symbol=? ORDER BY event_time",
        con,
        params=(symbol,),
    )
    if x.empty:
        return np.array([], dtype=np.int64), np.array([], dtype=float)
    return x["event_time"].to_numpy(dtype=np.int64), x["rate"].to_numpy(dtype=float)


def sum_funding(event_t: np.ndarray, event_r: np.ndarray, a: pd.Timestamp, b: pd.Timestamp) -> float:
    if len(event_t) == 0:
        return 0.0
    aa = int(a.timestamp() * 1000)
    bb = int(b.timestamp() * 1000)
    # Strict boundaries avoid assuming a fill before a funding snapshot at the same timestamp.
    li = int(np.searchsorted(event_t, aa, side="right"))
    ri = int(np.searchsorted(event_t, bb, side="left"))
    if ri <= li:
        return 0.0
    return float(event_r[li:ri].sum())


def simulate(
    df: pd.DataFrame,
    funding: tuple[np.ndarray, np.ndarray],
    start: pd.Timestamp,
    end: pd.Timestamp,
    side: str,
    lookback: int,
    theta: float,
    alpha: float,
    cost_bps: float,
) -> tuple[pd.Series, int]:
    warm = start - pd.Timedelta(days=8)
    x = df[(df["time"] >= warm) & (df["time"] < end)].copy().reset_index(drop=True)
    if len(x) < 10:
        return pd.Series(dtype=float), 0
    score_mask = (x["time"] >= start) & (x["time"] < end)
    close = x["close"].to_numpy(dtype=float)
    atr = x["atr"].to_numpy(dtype=float)
    mom = x[f"mom_{lookback}"].to_numpy(dtype=float)
    times = x["time"].tolist()
    sign = 1.0 if side == "long" else -1.0
    cost = cost_bps / 10000.0
    pos = False
    trail = np.nan
    trades = 0
    rets = np.zeros(len(x), dtype=float)
    event_t, event_r = funding

    for i in range(1, len(x)):
        if not score_mask.iloc[i]:
            continue
        was_pos = pos
        if was_pos:
            rets[i] += sign * (close[i] / close[i - 1] - 1.0)
            fsum = sum_funding(event_t, event_r, times[i - 1], times[i])
            rets[i] += -sign * fsum

            if np.isfinite(atr[i]) and atr[i] > 0:
                candidate = close[i] - alpha * atr[i] if side == "long" else close[i] + alpha * atr[i]
                trail = max(trail, candidate) if side == "long" else min(trail, candidate)
                crossed = close[i] < trail if side == "long" else close[i] > trail
                if crossed:
                    rets[i] -= cost
                    pos = False
                    trail = np.nan
                    continue

        if not pos and not was_pos and np.isfinite(mom[i]) and np.isfinite(atr[i]) and atr[i] > 0:
            fire = mom[i] > theta if side == "long" else mom[i] < -theta
            if fire:
                rets[i] -= cost
                pos = True
                trades += 1
                trail = close[i] - alpha * atr[i] if side == "long" else close[i] + alpha * atr[i]

    idx = np.flatnonzero(score_mask.to_numpy())
    if len(idx) == 0:
        return pd.Series(dtype=float), trades
    # Conservative monthly rebalance: close any surviving position on the last included H6 decision.
    if pos:
        rets[idx[-1]] -= cost
    out = pd.Series(rets[idx], index=pd.DatetimeIndex([times[i] for i in idx], tz="UTC"), dtype=float)
    return out, trades


def sharpe(r: pd.Series) -> float:
    if len(r) < 20:
        return float("-inf")
    arr = r.to_numpy(dtype=float)
    rf_step = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / ANNUAL_PERIODS) - 1.0
    ex = arr - rf_step
    sd = float(np.std(ex, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("-inf")
    return float(np.mean(ex) / sd * math.sqrt(ANNUAL_PERIODS))


def optimize(
    df: pd.DataFrame,
    funding: tuple[np.ndarray, np.ndarray],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    side: str,
) -> Opt | None:
    best: Opt | None = None
    for lb in LOOKBACKS:
        for theta in THRESHOLDS:
            for alpha in ATR_MULTS:
                ret, trades = simulate(df, funding, train_start, train_end, side, lb, theta, alpha, BASE_COST_BPS)
                if trades < MIN_TRAIN_TRADES:
                    continue
                sr = sharpe(ret)
                if not np.isfinite(sr):
                    continue
                cand = Opt(sr, lb, theta, alpha, trades)
                if best is None or cand.sharpe > best.sharpe:
                    best = cand
    return best


def market_caps_at(con: sqlite3.Connection, cutoff: pd.Timestamp) -> dict[str, float]:
    ts = int(cutoff.timestamp() * 1000)
    rows = con.execute(
        """
        SELECT m.symbol, m.market_cap
        FROM market_caps m
        JOIN (
          SELECT symbol, MAX(ts_ms) AS mx FROM market_caps WHERE ts_ms<=? GROUP BY symbol
        ) q ON q.symbol=m.symbol AND q.mx=m.ts_ms
        WHERE m.market_cap>0
        """,
        (ts,),
    ).fetchall()
    return {str(s): float(mc) for s, mc in rows}


def metrics(period_ret: pd.Series) -> dict:
    r = period_ret.fillna(0.0).astype(float)
    if r.empty:
        return {}
    equity = (1.0 + r).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0
    span_days = max((r.index[-1] - r.index[0]).total_seconds() / 86400.0, 1.0)
    total = float(equity.iloc[-1] - 1.0)
    cagr = float(equity.iloc[-1] ** (365.25 / span_days) - 1.0) if equity.iloc[-1] > 0 else -1.0
    sr = sharpe(r)
    monthly = (1.0 + r).groupby(r.index.to_period("M")).prod() - 1.0
    yearly = (1.0 + r).groupby(r.index.to_period("Y")).prod() - 1.0
    return {
        "total_return_pct": total * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": sr,
        "max_drawdown_pct": float(dd.min() * 100.0),
        "positive_months": int((monthly > 0).sum()),
        "months": int(len(monthly)),
        "avg_month_pct": float(monthly.mean() * 100.0),
        "median_month_pct": float(monthly.median() * 100.0),
        "best_month_pct": float(monthly.max() * 100.0),
        "worst_month_pct": float(monthly.min() * 100.0),
        "year_returns": {str(k): float(v * 100.0) for k, v in yearly.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="AdaptiveTrend documented-core 1x replication")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-01-01", help="exclusive")
    ap.add_argument("--outdir", default=DEFAULT_OUT)
    args = ap.parse_args()

    db = Path(args.db)
    if not db.exists():
        raise RuntimeError(f"Missing Strategy Build DB: {db}")
    con = sqlite3.connect(db, timeout=60)
    cap_symbols = con.execute("SELECT COUNT(DISTINCT symbol) FROM market_caps").fetchone()[0]
    candle_symbols = con.execute("SELECT COUNT(DISTINCT symbol) FROM candles").fetchone()[0]
    if cap_symbols < MIN_MONTH_UNIVERSE:
        raise RuntimeError(f"Insufficient CoinGecko coverage: {cap_symbols} symbols; need at least {MIN_MONTH_UNIVERSE}")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    print("=== STRATEGY BUILD: ADAPTIVETREND DOCUMENTED CORE ===")
    print("H6 | 1x | monthly previous-month optimization | 24h buffer | CoinGecko market-cap universe")
    print("Long top-15 / short bottom-15 assumption | SR gates 1.3/1.7 | 70/30 | exact archived funding")
    print("Frozen undisclosed grid: L=[4,6,8,10,12,16], theta=[1.5,2.5,3.5,5,7.5]%, ATR14 x [2,2.5,3,3.5]")
    print(f"DB coverage: H6 symbols={candle_symbols}, market-cap symbols={cap_symbols}")

    frame_cache: dict[str, pd.DataFrame] = {}
    funding_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def frame(sym: str) -> pd.DataFrame:
        if sym not in frame_cache:
            frame_cache[sym] = load_symbol_frame(con, sym)
        return frame_cache[sym]

    def funding(sym: str) -> tuple[np.ndarray, np.ndarray]:
        if sym not in funding_cache:
            funding_cache[sym] = load_funding(con, sym)
        return funding_cache[sym]

    # Selections and parameters are ALWAYS learned with the baseline 4-bps-per-fill model.
    selection_rows: list[dict] = []
    month_models: dict[str, dict[str, dict[str, Opt]]] = {}
    started = time.monotonic()

    for mi, m in enumerate(month_starts(start, end), 1):
        nxt = min(m + pd.offsets.MonthBegin(1), end)
        train_start = m - pd.offsets.MonthBegin(1)
        train_end = m - pd.Timedelta(hours=BUFFER_HOURS)
        caps = market_caps_at(con, train_end)
        eligible: list[tuple[str, float]] = []
        for sym, mc in caps.items():
            df = frame(sym)
            if df.empty:
                continue
            ntrain = int(((df["time"] >= train_start) & (df["time"] < train_end)).sum())
            if ntrain >= MIN_TRAIN_BARS:
                eligible.append((sym, mc))
        eligible.sort(key=lambda z: z[1], reverse=True)
        if len(eligible) < MIN_MONTH_UNIVERSE:
            print(f"{m:%Y-%m}: DATA QUALITY FAIL universe={len(eligible)} < {MIN_MONTH_UNIVERSE}", flush=True)
            month_models[m.strftime("%Y-%m")] = {"long": {}, "short": {}}
            continue

        long_candidates = [s for s, _ in eligible[:K_LONG]]
        short_candidates = [s for s, _ in eligible[-K_SHORT:]]
        selected_l: dict[str, Opt] = {}
        selected_s: dict[str, Opt] = {}

        for side, candidates, gate, selected in (
            ("long", long_candidates, LONG_SHARPE_GATE, selected_l),
            ("short", short_candidates, SHORT_SHARPE_GATE, selected_s),
        ):
            for sym in candidates:
                opt = optimize(frame(sym), funding(sym), train_start, train_end, side)
                if opt and opt.sharpe >= gate:
                    selected[sym] = opt
                    selection_rows.append({
                        "month": m.strftime("%Y-%m"), "side": side, "symbol": sym,
                        "market_cap": caps.get(sym), "train_sharpe": opt.sharpe,
                        "lookback": opt.lookback, "theta": opt.theta, "alpha": opt.alpha,
                        "train_trades": opt.trades,
                    })

        month_models[m.strftime("%Y-%m")] = {"long": selected_l, "short": selected_s}
        print(
            f"[{mi:02d}] {m:%Y-%m} universe={len(eligible):3d} candidates=15/15 "
            f"selected L/S={len(selected_l):2d}/{len(selected_s):2d} "
            f"elapsed={(time.monotonic()-started)/60:.1f}m",
            flush=True,
        )

    scenario_rows: list[dict] = []
    period_exports: list[pd.DataFrame] = []
    trade_counts: dict[float, int] = {}

    for cost_bps in COST_SCENARIOS:
        all_parts: list[pd.Series] = []
        total_trades = 0
        for m in month_starts(start, end):
            nxt = min(m + pd.offsets.MonthBegin(1), end)
            model = month_models.get(m.strftime("%Y-%m"), {"long": {}, "short": {}})
            month_index = pd.date_range(m + pd.Timedelta(hours=6), nxt - pd.Timedelta(seconds=1), freq="6h", tz="UTC")
            port = pd.Series(0.0, index=month_index, dtype=float)
            nl = len(model["long"])
            ns = len(model["short"])
            for side, selected, alloc in (("long", model["long"], LONG_ALLOC), ("short", model["short"], SHORT_ALLOC)):
                nsel = len(selected)
                if nsel == 0:
                    continue
                w = alloc / nsel
                for sym, opt in selected.items():
                    rr, ntr = simulate(frame(sym), funding(sym), m, nxt, side, opt.lookback, opt.theta, opt.alpha, cost_bps)
                    port = port.add(rr.reindex(port.index, fill_value=0.0) * w, fill_value=0.0)
                    total_trades += ntr
            all_parts.append(port)
        returns = pd.concat(all_parts).sort_index() if all_parts else pd.Series(dtype=float)
        met = metrics(returns)
        trade_counts[cost_bps] = total_trades
        scenario_rows.append({"cost_bps_per_fill": cost_bps, "trades": total_trades, **{k: v for k, v in met.items() if k != "year_returns"}, "year_returns": json.dumps(met.get("year_returns", {}))})
        ex = pd.DataFrame({"time": returns.index, "portfolio_return": returns.values})
        ex["cost_bps_per_fill"] = cost_bps
        period_exports.append(ex)
        yrs = " ".join(f"{k}={v:+.1f}%" for k, v in met.get("year_returns", {}).items())
        print(
            f"\nCOST {cost_bps:.0f}bps/fill | ret={met['total_return_pct']:+.2f}% CAGR={met['cagr_pct']:+.2f}% "
            f"Sharpe={met['sharpe']:+.2f} DD={met['max_drawdown_pct']:+.2f}% avg_month={met['avg_month_pct']:+.2f}% "
            f"trades={total_trades} | {yrs}",
            flush=True,
        )

    out = pd.DataFrame(scenario_rows)
    base = out[out["cost_bps_per_fill"] == BASE_COST_BPS].iloc[0]
    stress = out[out["cost_bps_per_fill"] == 8.0].iloc[0]
    base_years = json.loads(base["year_returns"])
    gate = bool(
        float(base["sharpe"]) >= GATE_MIN_SHARPE
        and float(base["max_drawdown_pct"]) >= -GATE_MAX_DD_PCT
        and (not GATE_REQUIRE_ALL_YEARS_POSITIVE or all(float(v) > 0 for v in base_years.values()))
        and float(stress["sharpe"]) >= GATE_MIN_COST8_SHARPE
        and float(stress["total_return_pct"]) > 0
    )

    print("\n=== DOCUMENTED-CORE REPLICATION GATE ===")
    print(
        f"PASS requires baseline 4bps/fill: Sharpe>={GATE_MIN_SHARPE:.2f}, DD<={GATE_MAX_DD_PCT:.0f}%, all 2022-24 years positive; "
        f"8bps/fill stress: Sharpe>={GATE_MIN_COST8_SHARPE:.2f} and total return>0."
    )
    print("GATE:", "PASS" if gate else "FAIL")
    print("No leverage was used. No 2025/2026 data was used in this replication.")
    print("If PASS: extend unchanged through 2025-2026, then run leverage ladder. If FAIL: reject this architecture without parameter mining.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(selection_rows).to_csv(outdir / "monthly_selections.csv", index=False)
    out.to_csv(outdir / "cost_scenarios.csv", index=False)
    if period_exports:
        pd.concat(period_exports, ignore_index=True).to_csv(outdir / "period_returns.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps({
        "pass": gate,
        "paper": "arXiv:2602.11708",
        "replication": "documented-core approximation",
        "paper_specified": {
            "timeframe": "6h", "K_long": K_LONG, "long_sharpe_gate": LONG_SHARPE_GATE,
            "short_sharpe_gate": SHORT_SHARPE_GATE, "long_allocation": LONG_ALLOC, "short_allocation": SHORT_ALLOC,
            "previous_month_optimization": True, "buffer_hours": BUFFER_HOURS,
        },
        "frozen_assumptions_not_disclosed_by_paper": {
            "K_short": K_SHORT, "lookbacks": LOOKBACKS, "thresholds": THRESHOLDS,
            "atr_multipliers": ATR_MULTS, "atr_period": ATR_PERIOD, "min_train_trades": MIN_TRAIN_TRADES,
            "slippage_model": "fixed execution-cost stress because paper's 5m-volume calibration coefficients are undisclosed",
        },
        "gate_criteria": {
            "baseline_cost_bps_per_fill": BASE_COST_BPS, "min_sharpe": GATE_MIN_SHARPE,
            "max_drawdown_abs_pct": GATE_MAX_DD_PCT, "all_years_positive": GATE_REQUIRE_ALL_YEARS_POSITIVE,
            "cost8_min_sharpe": GATE_MIN_COST8_SHARPE,
        },
    }, indent=2))
    print(f"Output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
