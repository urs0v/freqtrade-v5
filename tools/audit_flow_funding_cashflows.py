#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

import audit_flow_funding_portfolio as p
import research_derivatives_alpha as r

FACTOR = "taker_minus_funding"
HORIZON = "12h"
BARS = r.HORIZONS[HORIZON]
Q = 0.25
COST_BPS = 8.0


def _find_col(df: pd.DataFrame, *names: str) -> str | None:
    m = {str(c).strip().lower(): str(c) for c in df.columns}
    for n in names:
        if n.lower() in m:
            return m[n.lower()]
    return None


def _to_utc(s: pd.Series) -> pd.Series:
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() > 0.8:
        med = float(num.dropna().abs().median()) if num.notna().any() else 0.0
        unit = "us" if med > 1e14 else "ms" if med > 1e11 else "s" if med > 1e9 else None
        if unit:
            return pd.to_datetime(num, unit=unit, utc=True, errors="coerce").astype("datetime64[ns, UTC]")
    return pd.to_datetime(s, utc=True, errors="coerce").astype("datetime64[ns, UTC]")


def load_funding_events(cache_root: Path, symbol: str) -> pd.DataFrame:
    root = cache_root / "fundingRate" / symbol
    rows = []
    if not root.exists():
        return pd.DataFrame(columns=["time", "rate"])
    for zpath in sorted(root.glob("*.zip")):
        try:
            with zipfile.ZipFile(zpath) as zf:
                names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
                if not names:
                    continue
                with zf.open(names[0]) as f:
                    df = pd.read_csv(f)
        except Exception:
            continue
        tc = _find_col(df, "calc_time", "funding_time", "fundingTime", "timestamp", "time")
        rc = _find_col(df, "last_funding_rate", "funding_rate", "fundingRate")
        if tc is None or rc is None:
            continue
        t = _to_utc(df[tc])
        rate = pd.to_numeric(df[rc], errors="coerce")
        x = pd.DataFrame({"time": t, "rate": rate}).dropna()
        if not x.empty:
            rows.append(x)
    if not rows:
        return pd.DataFrame(columns=["time", "rate"])
    out = pd.concat(rows, ignore_index=True).drop_duplicates("time").sort_values("time")
    return out


def add_funding_cashflows(trades: pd.DataFrame, events_by_pair: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if trades.empty:
        return trades.copy()
    out_parts = []
    for pair, g in trades.groupby("pair", sort=False):
        x = g.copy()
        x["entry_time"] = pd.to_datetime(x["signal_time"], utc=True).astype("datetime64[ns, UTC]")
        x["exit_time"] = x["entry_time"] + pd.Timedelta(hours=12)
        ev = events_by_pair.get(pair, pd.DataFrame())
        if ev.empty:
            x["funding_ret"] = 0.0
            x["funding_events"] = 0
        else:
            et = ev["time"].astype("int64").to_numpy()
            rr = ev["rate"].to_numpy(dtype=float)
            cs = np.concatenate([[0.0], np.cumsum(rr)])
            a = x["entry_time"].astype("int64").to_numpy()
            b = x["exit_time"].astype("int64").to_numpy()
            # Strict boundaries: an event exactly at entry/exit is excluded to avoid
            # assuming a fill before the funding snapshot at the same timestamp.
            li = np.searchsorted(et, a, side="right")
            ri = np.searchsorted(et, b, side="left")
            sums = cs[ri] - cs[li]
            x["funding_events"] = ri - li
            x["funding_ret"] = -x["side"].to_numpy(dtype=float) * sums
        x["economic_ret"] = x["gross_ret"] + x["funding_ret"] - COST_BPS / 10000.0
        out_parts.append(x)
    return pd.concat(out_parts, ignore_index=True)


def stats_with_col(all_rows: pd.DataFrame, trades: pd.DataFrame, n_pairs: int, col: str) -> dict:
    z = trades.copy()
    z["net_ret"] = z[col]
    return p.portfolio_stats(all_rows, z, n_pairs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Economic audit of taker-minus-funding including settled funding cashflows")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite")
    ap.add_argument("--funding-cache", default="/freqtrade/user_data/v5/free-cache")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/funding_cashflow")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    ranges = {
        "train": ("2022-01-01", "2025-01-01"),
        "val": ("2025-01-01", "2026-01-01"),
        "test": ("2026-01-01", "2026-08-19"),
    }
    chunks = {k: [] for k in ranges}
    events_by_pair: dict[str, pd.DataFrame] = {}

    print("=== FLOW-FUNDING CASHFLOW AUDIT ===")
    print("Canonical q25 / 12h / 8bps / 1x; adds actual archived funding settlements.")
    print("Funding cashflow uses -side * funding_rate, with strict entry/exit boundaries.")

    for i, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        price = r.load_price(config, Path(args.datadir), pair)
        if price.empty:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: NO PRICE")
            continue
        price["date"] = r.as_ns(price["date"])
        start_ms = int((price["date"].min() - pd.Timedelta("1D")).timestamp() * 1000)
        end_ms = int((price["date"].max() + pd.Timedelta("1D")).timestamp() * 1000)
        deriv = r.load_derivatives(Path(args.db), r.pair_to_symbol(pair), start_ms, end_ms)
        if not deriv.empty:
            deriv = deriv.copy()
            deriv["available_time"] = pd.to_datetime(deriv["available_time"], utc=True).astype("datetime64[ns, UTC]")
        feat, _ = r.build_features(price, deriv)
        for split, (a, b) in ranges.items():
            x = r.slice_horizon(feat, a, b, HORIZON, BARS)
            if not x.empty:
                x["pair"] = pair
                chunks[split].append(x)
        events = load_funding_events(Path(args.funding_cache), r.pair_to_symbol(pair))
        events_by_pair[pair] = events
        print(f"  [{i:02d}/{len(pairs)}] {pair}: funding_events={len(events):,} [{time.monotonic()-t0:.1f}s]", flush=True)

    data = {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame() for k, v in chunks.items()}
    orient, qlo, qhi = p.thresholds_from_train(data["train"], Q)

    rows = []
    exports = []
    print("\n=== ECONOMIC RESULTS ===")
    for split in ("train", "val", "test"):
        base = p.select_trades(data[split], orient, qlo, qhi, 0.0)
        funded = add_funding_cashflows(base, events_by_pair)
        funded["price_net_ret"] = funded["gross_ret"] - COST_BPS / 10000.0
        s_price = stats_with_col(data[split], funded, len(pairs), "price_net_ret")
        s_econ = stats_with_col(data[split], funded, len(pairs), "economic_ret")
        fr_bps = float(funded["funding_ret"].mean() * 10000.0) if len(funded) else np.nan
        long_fr = float(funded.loc[funded["side"] > 0, "funding_ret"].mean() * 10000.0) if (funded["side"] > 0).any() else np.nan
        short_fr = float(funded.loc[funded["side"] < 0, "funding_ret"].mean() * 10000.0) if (funded["side"] < 0).any() else np.nan
        rows.append({
            "split": split,
            "trades": len(funded),
            "avg_funding_bps": fr_bps,
            "long_funding_bps": long_fr,
            "short_funding_bps": short_fr,
            "price_only_return_pct": s_price["total_return_pct"],
            "economic_return_pct": s_econ["total_return_pct"],
            "price_only_sharpe": s_price["sharpe"],
            "economic_sharpe": s_econ["sharpe"],
            "price_only_dd_pct": s_price["max_drawdown_pct"],
            "economic_dd_pct": s_econ["max_drawdown_pct"],
        })
        print(
            f"{split:>5}: funding={fr_bps:+.3f}bps/trade | "
            f"ret {s_price['total_return_pct']:+.2f}% -> {s_econ['total_return_pct']:+.2f}% | "
            f"Sharpe {s_price['sharpe']:+.2f} -> {s_econ['sharpe']:+.2f} | "
            f"DD {s_price['max_drawdown_pct']:+.2f}% -> {s_econ['max_drawdown_pct']:+.2f}%",
            flush=True,
        )
        funded["split"] = split
        exports.append(funded)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "cashflow_summary.csv", index=False)
    if exports:
        pd.concat(exports, ignore_index=True).to_csv(outdir / "trades_with_funding.csv", index=False)
    print("\n2026 is diagnostic only. This is an economic-correction audit, not a new alpha gate.")
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
