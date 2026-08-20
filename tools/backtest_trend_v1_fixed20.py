#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import audit_flow_funding_cashflows as funding_io
import research_derivatives_alpha as data_io

CORE20_PATH = Path("/opt/rmv5/tools/historical_universe.json")
BAR_HOURS = 6
VOL_BARS = 120          # 30 days on H6
RET_24_BARS = 4
RET_72_BARS = 12
RET_168_BARS = 28
MAX_PAIR_WEIGHT = 0.10
GROSS_CAP = 1.0
COST_SCENARIOS_BPS = [4.0, 8.0, 12.0]
CANONICAL_COST_BPS = 8.0
ANNUAL_PERIODS = 365.25 * 4.0

# Frozen before seeing this test.
GATE_MIN_SHARPE = 1.0
GATE_MAX_DD_PCT = 20.0
GATE_MAX_SYMBOL_SHARE = 0.25
GATE_MAX_YEAR_SHARE = 0.75


def to_pair(symbol: str) -> str:
    return f"{symbol[:-4]}/USDT:USDT"


def load_core20() -> list[str]:
    obj = json.loads(CORE20_PATH.read_text())
    symbols = [str(x).upper() for x in obj.get("symbols", [])]
    if len(symbols) != 20:
        raise RuntimeError(f"Expected exactly 20 frozen symbols, got {len(symbols)}")
    return symbols


def resample_h6(price15: pd.DataFrame) -> pd.DataFrame:
    if price15.empty:
        return pd.DataFrame()
    x = price15[["date", "open", "high", "low", "close", "volume"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values("date").drop_duplicates("date").set_index("date")

    rule = "6h"
    ohlc = x.resample(rule, label="left", closed="left", origin="epoch").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    )
    count = x["close"].resample(rule, label="left", closed="left", origin="epoch").count()
    ohlc["count15"] = count
    # A H6 bar is accepted only when all 24 constituent 15m candles exist.
    bad = ohlc["count15"] != 24
    ohlc.loc[bad, ["open", "high", "low", "close", "volume"]] = np.nan
    return ohlc


def build_pair_frame(price15: pd.DataFrame) -> pd.DataFrame:
    h6 = resample_h6(price15)
    if h6.empty:
        return h6

    close = h6["close"]
    valid = close.notna().astype(int)
    contiguous29 = valid.rolling(RET_168_BARS + 1, min_periods=RET_168_BARS + 1).sum() == (RET_168_BARS + 1)

    h6["r24"] = close.pct_change(RET_24_BARS, fill_method=None)
    h6["r72"] = close.pct_change(RET_72_BARS, fill_method=None)
    h6["r168"] = close.pct_change(RET_168_BARS, fill_method=None)
    h6.loc[~contiguous29, ["r24", "r72", "r168"]] = np.nan

    logret = np.log(close / close.shift(1))
    h6["vol30"] = logret.rolling(VOL_BARS, min_periods=VOL_BARS).std(ddof=1)

    # Signal from completed H6 bar j executes at the open of bar j+1.
    h6["exec_time"] = h6.index + pd.Timedelta(hours=BAR_HOURS)
    h6["exec_open"] = h6["open"].shift(-1)
    h6["next_open"] = h6["open"].shift(-2)
    h6["price_ret"] = h6["next_open"] / h6["exec_open"] - 1.0

    eligible = (
        h6["vol30"].notna()
        & (h6["vol30"] > 0)
        & h6["exec_open"].notna()
        & h6["next_open"].notna()
        & contiguous29
    )

    # A: long-only benchmark. B: simple 7d time-series trend. C: V1 confirmation.
    h6["side_A"] = np.where(eligible, 1.0, 0.0)
    h6["side_B"] = np.where(eligible & (h6["r168"] > 0), 1.0,
                            np.where(eligible & (h6["r168"] < 0), -1.0, 0.0))
    all_up = eligible & (h6["r24"] > 0) & (h6["r72"] > 0) & (h6["r168"] > 0)
    all_dn = eligible & (h6["r24"] < 0) & (h6["r72"] < 0) & (h6["r168"] < 0)
    h6["side_C"] = np.where(all_up, 1.0, np.where(all_dn, -1.0, 0.0))
    h6["invvol"] = np.where(eligible, 1.0 / h6["vol30"].clip(lower=1e-12), 0.0)

    out = h6.set_index("exec_time")[["side_A", "side_B", "side_C", "invvol", "price_ret"]].copy()
    out.index = pd.DatetimeIndex(out.index, tz="UTC")
    return out


def funding_sum_for_times(events: pd.DataFrame, times: pd.DatetimeIndex) -> pd.Series:
    if len(times) == 0:
        return pd.Series(dtype=float, index=times)
    if events.empty:
        return pd.Series(0.0, index=times)
    ev = events.copy()
    ev["time"] = pd.to_datetime(ev["time"], utc=True)
    et = ev["time"].astype("int64").to_numpy()
    rr = pd.to_numeric(ev["rate"], errors="coerce").fillna(0.0).to_numpy(dtype=float)
    cs = np.concatenate([[0.0], np.cumsum(rr)])
    a = pd.Series(times).astype("int64").to_numpy()
    b = (pd.Series(times) + pd.Timedelta(hours=BAR_HOURS)).astype("int64").to_numpy()
    # Same conservative convention used elsewhere in this project: events exactly at entry/exit excluded.
    li = np.searchsorted(et, a, side="right")
    ri = np.searchsorted(et, b, side="left")
    return pd.Series(cs[ri] - cs[li], index=times, dtype=float)


def capped_inverse_vol(sides: pd.Series, invvol: pd.Series) -> pd.Series:
    active = sides.ne(0) & invvol.gt(0) & np.isfinite(invvol)
    out = pd.Series(0.0, index=sides.index)
    if not active.any():
        return out

    names = list(sides.index[active])
    raw = invvol.loc[names].astype(float)
    remaining = set(names)
    remaining_gross = GROSS_CAP
    absw = pd.Series(0.0, index=names)

    # Water-fill inverse-vol weights with a hard 10% per-pair cap.
    while remaining and remaining_gross > 1e-12:
        idx = list(remaining)
        r = raw.loc[idx]
        denom = float(r.sum())
        if not np.isfinite(denom) or denom <= 0:
            break
        proposal = r / denom * remaining_gross
        capped = proposal[proposal > MAX_PAIR_WEIGHT + 1e-15]
        if capped.empty:
            absw.loc[idx] = proposal
            remaining_gross = 0.0
            break
        for name in capped.index:
            absw.loc[name] = MAX_PAIR_WEIGHT
            remaining_gross -= MAX_PAIR_WEIGHT
            remaining.remove(name)
        if remaining_gross <= 0:
            break

    out.loc[absw.index] = absw * sides.loc[absw.index].astype(float)
    return out


def build_matrices(pair_frames: dict[str, pd.DataFrame], funding_by_pair: dict[str, pd.DataFrame], start: pd.Timestamp, end: pd.Timestamp):
    all_times = sorted(set().union(*[set(df.index) for df in pair_frames.values() if not df.empty]))
    times = pd.DatetimeIndex([t for t in all_times if start <= t < end], tz="UTC")
    pairs = list(pair_frames)

    price = pd.DataFrame(index=times, columns=pairs, dtype=float)
    invvol = pd.DataFrame(index=times, columns=pairs, dtype=float)
    funding = pd.DataFrame(index=times, columns=pairs, dtype=float)
    sides = {k: pd.DataFrame(0.0, index=times, columns=pairs) for k in ("A", "B", "C")}

    for pair, df in pair_frames.items():
        z = df.reindex(times)
        price[pair] = z["price_ret"]
        invvol[pair] = z["invvol"]
        for k in ("A", "B", "C"):
            sides[k][pair] = z[f"side_{k}"].fillna(0.0)
        funding[pair] = funding_sum_for_times(funding_by_pair.get(pair, pd.DataFrame()), times)

    return times, pairs, price, invvol, funding, sides


def weights_from_sides(side_df: pd.DataFrame, invvol: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for t in side_df.index:
        rows.append(capped_inverse_vol(side_df.loc[t], invvol.loc[t]))
    return pd.DataFrame(rows, index=side_df.index, columns=side_df.columns, dtype=float)


def max_drawdown(r: pd.Series) -> float:
    eq = (1.0 + r.fillna(0.0)).cumprod()
    if eq.empty:
        return float("nan")
    return float((eq / eq.cummax() - 1.0).min())


def sharpe(r: pd.Series) -> float:
    x = r.dropna().to_numpy(dtype=float)
    if len(x) < 20:
        return float("nan")
    sd = float(np.std(x, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("nan")
    return float(np.mean(x) / sd * math.sqrt(ANNUAL_PERIODS))


def evaluate(name: str, weights: pd.DataFrame, sides: pd.DataFrame, price: pd.DataFrame, funding: pd.DataFrame, cost_bps: float) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prev = weights.shift(1).fillna(0.0)
    turnover_pair = (weights - prev).abs()
    cost_rate = cost_bps / 10000.0

    price_part = weights * price.fillna(0.0)
    funding_part = -weights * funding.fillna(0.0)
    cost_part = turnover_pair * cost_rate
    pair_ret = price_part + funding_part - cost_part

    # Close any surviving positions at the final interval end.
    if len(pair_ret):
        final_close = weights.iloc[-1].abs() * cost_rate
        pair_ret.iloc[-1] = pair_ret.iloc[-1] - final_close
        cost_part.iloc[-1] = cost_part.iloc[-1] + final_close

    port = pair_ret.sum(axis=1)
    equity = (1.0 + port).cumprod()
    total = float(equity.iloc[-1] - 1.0) if len(equity) else float("nan")
    span_days = max((port.index[-1] - port.index[0]).total_seconds() / 86400.0, 1.0) if len(port) else 1.0
    cagr = float(equity.iloc[-1] ** (365.25 / span_days) - 1.0) if len(equity) and equity.iloc[-1] > 0 else float("nan")
    dd = max_drawdown(port)
    sr = sharpe(port)

    side_prev = sides.shift(1).fillna(0.0)
    entries = ((sides != 0) & ((side_prev == 0) | (np.sign(sides) != np.sign(side_prev)))).sum(axis=1)
    monthly_entries = entries.groupby(entries.index.to_period("M")).sum()
    monthly_ret = (1.0 + port).groupby(port.index.to_period("M")).prod() - 1.0
    yearly_ret = (1.0 + port).groupby(port.index.to_period("Y")).prod() - 1.0

    contrib = pair_ret.sum(axis=0)
    positive = contrib.clip(lower=0.0)
    symbol_share = float(positive.max() / positive.sum()) if positive.sum() > 0 else 1.0
    year_simple = port.groupby(port.index.to_period("Y")).sum()
    positive_year = year_simple.clip(lower=0.0)
    year_share = float(positive_year.max() / positive_year.sum()) if positive_year.sum() > 0 else 1.0

    stats = {
        "strategy": name,
        "cost_bps": cost_bps,
        "total_return_pct": total * 100.0,
        "cagr_pct": cagr * 100.0 if np.isfinite(cagr) else np.nan,
        "sharpe": sr,
        "max_drawdown_pct": dd * 100.0,
        "avg_gross_exposure": float(weights.abs().sum(axis=1).mean()),
        "avg_net_exposure": float(weights.sum(axis=1).mean()),
        "turnover": float(turnover_pair.sum(axis=1).sum() + (weights.iloc[-1].abs().sum() if len(weights) else 0.0)),
        "entries": int(entries.sum()),
        "avg_entries_per_month": float(monthly_entries.mean()) if len(monthly_entries) else 0.0,
        "median_entries_per_month": float(monthly_entries.median()) if len(monthly_entries) else 0.0,
        "positive_months": int((monthly_ret > 0).sum()),
        "months": int(len(monthly_ret)),
        "avg_month_pct": float(monthly_ret.mean() * 100.0) if len(monthly_ret) else np.nan,
        "median_month_pct": float(monthly_ret.median() * 100.0) if len(monthly_ret) else np.nan,
        "max_positive_symbol_share": symbol_share,
        "max_positive_year_share": year_share,
    }

    ts = pd.DataFrame({"return": port, "equity": equity, "gross": weights.abs().sum(axis=1), "net": weights.sum(axis=1), "entries": entries})
    yr = pd.DataFrame({"year": yearly_ret.index.astype(str), "return_pct": yearly_ret.to_numpy() * 100.0})
    sy = pd.DataFrame({"pair": contrib.index, "simple_contribution": contrib.to_numpy()}).sort_values("simple_contribution", ascending=False)
    return stats, ts, yr, sy


def basic_gate(s: dict) -> bool:
    return bool(
        s["total_return_pct"] > 0
        and s["sharpe"] >= GATE_MIN_SHARPE
        and s["max_drawdown_pct"] >= -GATE_MAX_DD_PCT
        and s["max_positive_symbol_share"] <= GATE_MAX_SYMBOL_SHARE
        and s["max_positive_year_share"] <= GATE_MAX_YEAR_SHARE
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Frozen fixed-20 H6 time-series trend V1")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--funding-cache", default="/freqtrade/user_data/v5/free-cache")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-08-19", help="exclusive")
    ap.add_argument("--outdir", default="/freqtrade/user_data/strategy_build/trend_v1_fixed20")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    symbols = load_core20()
    pairs = [to_pair(s) for s in symbols]
    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")

    print("=== FROZEN TREND V1 / FIXED CORE-20 ===")
    print("No downloads. Existing 15m Binance futures data -> exact H6 bars.")
    print("A=long-only vol-scaled | B=sign(168h) | C=24h+72h+168h confirmation")
    print("Signal at completed H6 close -> execute next H6 open | 30d inverse-vol | max 10%/pair | gross<=1x")
    print("No ML, ATR, OI, regime filter, monthly optimization or parameter search.")

    frames: dict[str, pd.DataFrame] = {}
    funding_by_pair: dict[str, pd.DataFrame] = {}
    for i, (symbol, pair) in enumerate(zip(symbols, pairs), 1):
        t0 = time.monotonic()
        p15 = data_io.load_price(config, Path(args.datadir), pair)
        if p15.empty:
            raise RuntimeError(f"Missing existing price data for {pair}")
        frame = build_pair_frame(p15)
        frames[pair] = frame
        ev = funding_io.load_funding_events(Path(args.funding_cache), symbol)
        funding_by_pair[pair] = ev
        print(
            f"[{i:02d}/20] {pair:<18} H6={len(frame):,} "
            f"range={frame.index.min()}..{frame.index.max()} funding={len(ev):,} [{time.monotonic()-t0:.1f}s]",
            flush=True,
        )

    times, pairs, price, invvol, funding, sides = build_matrices(frames, funding_by_pair, start, end)
    if len(times) < 200:
        raise RuntimeError(f"Insufficient aligned H6 history: {len(times)} intervals")

    weights = {k: weights_from_sides(sides[k], invvol) for k in ("A", "B", "C")}
    names = {"A": "A_long_only", "B": "B_simple_168h", "C": "C_triple_trend_v1"}

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    canonical = {}

    print("\n=== RESULTS ===")
    for cost in COST_SCENARIOS_BPS:
        for k in ("A", "B", "C"):
            stats, ts, yr, sy = evaluate(names[k], weights[k], sides[k], price, funding, cost)
            summary_rows.append(stats)
            if cost == CANONICAL_COST_BPS:
                canonical[k] = stats
                ts.to_csv(outdir / f"timeseries_{k}_cost8.csv")
                yr.to_csv(outdir / f"yearly_{k}_cost8.csv", index=False)
                sy.to_csv(outdir / f"symbol_contrib_{k}_cost8.csv", index=False)
            print(
                f"cost{cost:>2.0f} {k}: ret={stats['total_return_pct']:+7.2f}% "
                f"Sharpe={stats['sharpe']:+.2f} DD={stats['max_drawdown_pct']:+.2f}% "
                f"entries={stats['entries']} avg/mo={stats['avg_entries_per_month']:.1f} "
                f"gross={stats['avg_gross_exposure']:.2f}",
                flush=True,
            )

    A, B, C = canonical["A"], canonical["B"], canonical["C"]
    b_basic = basic_gate(B)
    c_basic = basic_gate(C)
    b_beats_a = B["sharpe"] > A["sharpe"] and B["max_drawdown_pct"] >= A["max_drawdown_pct"]
    c_beats_a = C["sharpe"] > A["sharpe"] and C["max_drawdown_pct"] >= A["max_drawdown_pct"]
    c_wins_vs_b = sum([
        C["total_return_pct"] > B["total_return_pct"],
        C["sharpe"] > B["sharpe"],
        C["max_drawdown_pct"] > B["max_drawdown_pct"],
    ]) >= 2

    if c_basic and c_beats_a and c_wins_vs_b:
        decision = "PASS_V1"
        candidate = "C_triple_trend_v1"
    elif b_basic and b_beats_a:
        decision = "PASS_TREND_CLASS_SIMPLER_RULE_WINS"
        candidate = "B_simple_168h"
    else:
        decision = "FAIL_TREND_CLASS"
        candidate = None

    gate = {
        "decision": decision,
        "candidate": candidate,
        "canonical_cost_bps": CANONICAL_COST_BPS,
        "frozen_rules": {
            "universe": symbols,
            "signal_horizons_hours": [24, 72, 168],
            "execution": "next H6 open",
            "volatility": "120 H6 log-return std",
            "max_pair_weight": MAX_PAIR_WEIGHT,
            "gross_cap": GROSS_CAP,
        },
        "basic_gate": {
            "return_gt_0": True,
            "sharpe_gte": GATE_MIN_SHARPE,
            "max_drawdown_abs_lte_pct": GATE_MAX_DD_PCT,
            "max_positive_symbol_share_lte": GATE_MAX_SYMBOL_SHARE,
            "max_positive_year_share_lte": GATE_MAX_YEAR_SHARE,
        },
        "B_basic": b_basic,
        "C_basic": c_basic,
        "B_beats_A_risk": b_beats_a,
        "C_beats_A_risk": c_beats_a,
        "C_wins_2of3_vs_B": c_wins_vs_b,
    }

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps(gate, indent=2))

    print("\n=== FROZEN GATE ===")
    print(f"{decision} | candidate={candidate}")
    print("If FAIL_TREND_CLASS: do not tune 24/72/168 or add indicators; close this alpha class.")
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
