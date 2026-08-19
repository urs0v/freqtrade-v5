#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research_derivatives_alpha as r

FACTOR = "taker_minus_funding"
HORIZON = "12h"
BARS = r.HORIZONS[HORIZON]
CANONICAL_Q = 0.25
CANONICAL_COST_BPS = 8.0
SCENARIOS = [
    (0.20, 8.0, "q20_cost8"),
    (0.25, 8.0, "canonical"),
    (0.30, 8.0, "q30_cost8"),
    (0.25, 4.0, "q25_cost4"),
    (0.25, 12.0, "q25_cost12"),
]

# Predeclared before running this audit. 2026 is diagnostic only.
PASS_MIN_VAL_TOTAL_RETURN_PCT = 0.0
PASS_MIN_VAL_SHARPE = 0.50
PASS_MAX_VAL_DRAWDOWN_PCT = 20.0
PASS_REQUIRE_THRESHOLD_STRESS_POSITIVE = True

_original_load_derivatives = r.load_derivatives


def load_derivatives_ns(db: Path, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    df = _original_load_derivatives(db, symbol, start_ms, end_ms)
    if not df.empty:
        df = df.copy()
        df["available_time"] = pd.to_datetime(df["available_time"], utc=True).astype("datetime64[ns, UTC]")
    return df


r.load_derivatives = load_derivatives_ns


def thresholds_from_train(train: pd.DataFrame, q: float) -> tuple[float, float, float]:
    target = f"y_{HORIZON}"
    base = train[[FACTOR, target]].dropna()
    ic = r.spearman(base[FACTOR], base[target])
    orient = 1.0 if not np.isfinite(ic) or ic >= 0 else -1.0
    score = orient * base[FACTOR]
    qlo, qhi = np.quantile(score, [q, 1.0 - q])
    return orient, float(qlo), float(qhi)


def select_trades(df: pd.DataFrame, orient: float, qlo: float, qhi: float, cost_bps: float) -> pd.DataFrame:
    target = f"y_{HORIZON}"
    x = df[["signal_time", "pair", FACTOR, target]].dropna().copy()
    if x.empty:
        return pd.DataFrame(columns=["signal_time", "pair", "side", "gross_ret", "net_ret"])
    score = orient * x[FACTOR].to_numpy(dtype=float)
    side = np.where(score >= qhi, 1.0, np.where(score <= qlo, -1.0, 0.0))
    chosen = side != 0
    x = x.loc[chosen, ["signal_time", "pair", target]].copy()
    side = side[chosen]
    x["side"] = side
    x["gross_ret"] = side * x[target].to_numpy(dtype=float)
    x["net_ret"] = x["gross_ret"] - cost_bps / 10000.0
    return x.drop(columns=[target])


def portfolio_stats(all_rows: pd.DataFrame, trades: pd.DataFrame, n_pairs: int) -> dict:
    if all_rows.empty:
        return {}
    times = pd.Index(sorted(all_rows["signal_time"].dropna().unique()))
    if len(times) == 0:
        return {}

    if trades.empty:
        period_ret = pd.Series(0.0, index=times)
    else:
        realized = trades.groupby("signal_time")["net_ret"].sum() / float(n_pairs)
        period_ret = realized.reindex(times, fill_value=0.0).astype(float)

    equity = (1.0 + period_ret).cumprod()
    peak = equity.cummax()
    dd = equity / peak - 1.0

    total_ret = float(equity.iloc[-1] - 1.0)
    span_days = max((pd.Timestamp(times[-1]) - pd.Timestamp(times[0])).total_seconds() / 86400.0, 1.0)
    cagr = float(equity.iloc[-1] ** (365.25 / span_days) - 1.0) if equity.iloc[-1] > 0 else -1.0
    std = float(period_ret.std(ddof=1))
    sharpe = float(period_ret.mean() / std * np.sqrt(2.0 * 365.25)) if std > 0 else 0.0

    avg_trade_bps = float(trades["net_ret"].mean() * 10000.0) if not trades.empty else np.nan
    win_pct = float((trades["net_ret"] > 0).mean() * 100.0) if not trades.empty else np.nan
    exposure = float(len(trades) / (len(times) * n_pairs) * 100.0)

    long_rows = trades[trades["side"] > 0]
    short_rows = trades[trades["side"] < 0]
    long_bps = float(long_rows["net_ret"].mean() * 10000.0) if len(long_rows) else np.nan
    short_bps = float(short_rows["net_ret"].mean() * 10000.0) if len(short_rows) else np.nan

    return {
        "periods": int(len(times)),
        "trades": int(len(trades)),
        "exposure_pct": exposure,
        "avg_trade_net_bps": avg_trade_bps,
        "win_pct": win_pct,
        "long_trades": int(len(long_rows)),
        "long_avg_net_bps": long_bps,
        "short_trades": int(len(short_rows)),
        "short_avg_net_bps": short_bps,
        "total_return_pct": total_ret * 100.0,
        "cagr_pct": cagr * 100.0,
        "sharpe": sharpe,
        "max_drawdown_pct": float(dd.min() * 100.0),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Executable portfolio audit for robust flow-funding alpha")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/portfolio")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist in config")
    db = Path(args.db)
    if not db.exists():
        raise RuntimeError(f"Missing derivatives DB: {db}")

    ranges = {
        "train": ("2022-01-01", "2025-01-01"),
        "val": ("2025-01-01", "2026-01-01"),
        "test": ("2026-01-01", "2026-08-19"),
    }
    chunks = {k: [] for k in ranges}

    print(f"Loading {len(pairs)} pairs for executable {FACTOR}/{HORIZON} portfolio audit...", flush=True)
    for i, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        price = r.load_price(config, Path(args.datadir), pair)
        if price.empty:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: NO PRICE DATA", flush=True)
            continue
        price["date"] = r.as_ns(price["date"])
        start_ms = int((price["date"].min() - pd.Timedelta("1D")).timestamp() * 1000)
        end_ms = int((price["date"].max() + pd.Timedelta("1D")).timestamp() * 1000)
        deriv = r.load_derivatives(db, r.pair_to_symbol(pair), start_ms, end_ms)
        feat, _ = r.build_features(price, deriv)
        for split, (start, end) in ranges.items():
            x = r.slice_horizon(feat, start, end, HORIZON, BARS)
            if not x.empty:
                x["pair"] = pair
                chunks[split].append(x)
        print(f"  [{i:02d}/{len(pairs)}] {pair}: ok [{time.monotonic()-t0:.1f}s]", flush=True)

    data = {
        k: pd.concat(v, ignore_index=True) if v else pd.DataFrame()
        for k, v in chunks.items()
    }
    if data["train"].empty or data["val"].empty:
        raise RuntimeError("Missing train/validation data")

    rows = []
    trade_exports = []
    print("\n=== EXECUTABLE PORTFOLIO SCENARIOS ===")
    print("Capital model: 20 equal 5% pair slots, 1x, 12h fixed hold, cash when no signal.")
    print("Signal thresholds and orientation are learned only on 2022-2024. 2026 is diagnostic only.")

    for q, cost, label in SCENARIOS:
        orient, qlo, qhi = thresholds_from_train(data["train"], q)
        for split in ("train", "val", "test"):
            trades = select_trades(data[split], orient, qlo, qhi, cost)
            stats = portfolio_stats(data[split], trades, len(pairs))
            rows.append({
                "scenario": label,
                "q": q,
                "cost_bps": cost,
                "split": split,
                "orientation": orient,
                "qlo": qlo,
                "qhi": qhi,
                **stats,
            })
            if label == "canonical":
                tx = trades.copy()
                tx["split"] = split
                tx["scenario"] = label
                trade_exports.append(tx)

        val = rows[-2]
        test = rows[-1]
        print(
            f"{label:<12} val ret={val['total_return_pct']:+6.2f}% Sharpe={val['sharpe']:+.2f} "
            f"DD={val['max_drawdown_pct']:+.2f}% avg={val['avg_trade_net_bps']:+.2f}bps | "
            f"2026 ret={test['total_return_pct']:+6.2f}% Sharpe={test['sharpe']:+.2f}",
            flush=True,
        )

    out = pd.DataFrame(rows)
    canonical_val = out[(out["scenario"] == "canonical") & (out["split"] == "val")].iloc[0]
    q20_val = out[(out["scenario"] == "q20_cost8") & (out["split"] == "val")].iloc[0]
    q30_val = out[(out["scenario"] == "q30_cost8") & (out["split"] == "val")].iloc[0]

    gate = (
        canonical_val["total_return_pct"] > PASS_MIN_VAL_TOTAL_RETURN_PCT
        and canonical_val["sharpe"] >= PASS_MIN_VAL_SHARPE
        and canonical_val["max_drawdown_pct"] >= -PASS_MAX_VAL_DRAWDOWN_PCT
        and (
            not PASS_REQUIRE_THRESHOLD_STRESS_POSITIVE
            or (q20_val["total_return_pct"] > 0 and q30_val["total_return_pct"] > 0)
        )
    )

    print("\n=== EXECUTABLE PORTFOLIO GATE ===")
    print(
        f"PASS requires canonical 2025 return>0, Sharpe>={PASS_MIN_VAL_SHARPE:.2f}, "
        f"max DD<={PASS_MAX_VAL_DRAWDOWN_PCT:.1f}%, and q20/q30 stress returns>0."
    )
    print("GATE:", "PASS" if gate else "FAIL")

    can = out[out["scenario"] == "canonical"][
        ["split", "trades", "exposure_pct", "avg_trade_net_bps", "win_pct", "long_avg_net_bps", "short_avg_net_bps", "total_return_pct", "cagr_pct", "sharpe", "max_drawdown_pct"]
    ]
    print("\n=== CANONICAL q25 / 8bps ===")
    print(can.to_string(index=False))
    print("2026 is diagnostic only; it cannot rescue or fail the pre-2026 gate.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "portfolio_scenarios.csv", index=False)
    if trade_exports:
        pd.concat(trade_exports, ignore_index=True).to_csv(outdir / "canonical_trades.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps({
        "pass": bool(gate),
        "factor": FACTOR,
        "horizon": HORIZON,
        "canonical_q": CANONICAL_Q,
        "canonical_cost_bps": CANONICAL_COST_BPS,
        "criteria": {
            "val_total_return_pct_gt": PASS_MIN_VAL_TOTAL_RETURN_PCT,
            "val_sharpe_gte": PASS_MIN_VAL_SHARPE,
            "val_max_drawdown_abs_pct_lte": PASS_MAX_VAL_DRAWDOWN_PCT,
            "q20_q30_val_total_return_positive": PASS_REQUIRE_THRESHOLD_STRESS_POSITIVE,
        },
    }, indent=2))

    print(f"\nOutput: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
