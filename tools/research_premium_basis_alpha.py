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

FACTORS = ["premium_z", "premium_delta4h_z"]
HORIZONS = {"4h": 16, "8h": 32, "12h": 48}
COST_BPS = 8.0
TRAIN_START, TRAIN_END = "2022-01-01", "2025-01-01"
VAL_START, VAL_END = "2025-01-01", "2026-01-01"
TEST_START, TEST_END = "2026-01-01", "2026-08-19"
GATE_MIN_TRAIN_NET_BPS = 0.0
GATE_MIN_VAL_NET_BPS = 2.0
GATE_MIN_VAL_IC = 0.005
GATE_MIN_POSITIVE_MONTHS = 7
GATE_MIN_VAL_POSITIONS = 500


def load_premium(db: Path, symbol: str) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        x = pd.read_sql_query(
            "SELECT available_ms, close AS premium_close FROM premium_15m WHERE symbol=? ORDER BY available_ms",
            con,
            params=(symbol,),
        )
    if x.empty:
        return x
    x["available_time"] = pd.to_datetime(x["available_ms"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
    return x.drop(columns=["available_ms"])


def build_frame(config: dict, datadir: Path, db: Path, pair: str) -> pd.DataFrame:
    price = r.load_price(config, datadir, pair)
    if price.empty:
        return pd.DataFrame()
    price = price[["date", "open", "close"]].copy().sort_values("date")
    price["date"] = r.as_ns(price["date"])
    price["signal_time"] = price["date"] + pd.Timedelta(minutes=15)
    prem = load_premium(db, r.pair_to_symbol(pair))
    if prem.empty:
        return pd.DataFrame()
    x = pd.merge_asof(
        price.sort_values("signal_time"),
        prem.sort_values("available_time"),
        left_on="signal_time",
        right_on="available_time",
        direction="backward",
        tolerance=pd.Timedelta("1min"),
    )
    p = pd.to_numeric(x["premium_close"], errors="coerce")
    x["premium_z"] = r.robust_z(p, 2880, 672)
    x["premium_delta4h_z"] = r.robust_z(p.diff(16), 2880, 672)
    entry = x["open"].shift(-1)
    for name, bars in HORIZONS.items():
        x[f"y_{name}"] = x["open"].shift(-(1 + bars)) / entry - 1.0
        x[f"exit_time_{name}"] = x["signal_time"] + pd.Timedelta(minutes=15 * bars)
    x["pair"] = pair
    return x


def slice_horizon(df: pd.DataFrame, start: str, end: str, horizon: str, bars: int) -> pd.DataFrame:
    a, b = pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC")
    target, exit_col = f"y_{horizon}", f"exit_time_{horizon}"
    x = df.loc[(df["signal_time"] >= a) & (df["signal_time"] < b) & (df[exit_col] < b), ["signal_time", "pair", target, *FACTORS]].dropna(subset=[target]).copy()
    return x.iloc[::bars].copy() if bars > 1 and not x.empty else x


def learn(train: pd.DataFrame, factor: str, target: str) -> tuple[float, float, float]:
    z = train[[factor, target]].dropna()
    ic = r.spearman(z[factor], z[target])
    orient = 1.0 if not np.isfinite(ic) or ic >= 0 else -1.0
    score = orient * z[factor]
    qlo, qhi = np.quantile(score, [0.25, 0.75])
    return orient, float(qlo), float(qhi)


def stats(df: pd.DataFrame, factor: str, target: str, orient: float, qlo: float, qhi: float) -> dict:
    x = df[["signal_time", factor, target]].dropna().copy()
    if x.empty:
        return {"n": 0, "ic": np.nan, "net_bps": np.nan, "gross_bps": np.nan, "positive_months": 0, "months": 0}
    score = orient * x[factor]
    side = np.where(score >= qhi, 1.0, np.where(score <= qlo, -1.0, 0.0))
    chosen = side != 0
    if not chosen.any():
        return {"n": 0, "ic": np.nan, "net_bps": np.nan, "gross_bps": np.nan, "positive_months": 0, "months": 0}
    ret = side[chosen] * x.loc[chosen, target].to_numpy(dtype=float)
    gross = float(ret.mean() * 10000.0)
    t = x.loc[chosen, "signal_time"].reset_index(drop=True)
    monthly = pd.DataFrame({"month": t.dt.strftime("%Y-%m"), "ret": ret}).groupby("month")["ret"].mean() * 10000.0 - COST_BPS
    return {
        "n": int(len(ret)),
        "ic": r.spearman(score, x[target]),
        "gross_bps": gross,
        "net_bps": gross - COST_BPS,
        "positive_months": int((monthly > 0).sum()),
        "months": int(len(monthly)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Binance premium-index / basis dislocation alpha lab")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/premium_basis.sqlite")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/premium_basis_alpha")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    frames = []
    print("=== PREMIUM / BASIS ALPHA LAB ===")
    print("2022-2024 train | 2025 validation | 2026 diagnostic | 8bps | no leverage")
    for i, pair in enumerate(pairs, 1):
        f = build_frame(config, Path(args.datadir), Path(args.db), pair)
        if not f.empty:
            frames.append(f)
            cov = f["premium_close"].notna().mean() * 100.0
            print(f"  [{i:02d}/{len(pairs)}] {pair}: rows={len(f):,} premium_coverage={cov:.1f}%", flush=True)
        else:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: NO PREMIUM DATA", flush=True)
    if not frames:
        raise RuntimeError("No premium data")
    all_df = pd.concat(frames, ignore_index=True)

    rows = []
    print("\n=== PREMIUM FACTOR GATE ===")
    print("PASS: train net>0; 2025 net>2bps; val IC>0.005; >=7/12 positive months; n>=500.")
    for horizon, bars in HORIZONS.items():
        train_parts, val_parts, test_parts = [], [], []
        for _, g in all_df.groupby("pair", sort=True):
            g = g.sort_values("signal_time")
            train_parts.append(slice_horizon(g, TRAIN_START, TRAIN_END, horizon, bars))
            val_parts.append(slice_horizon(g, VAL_START, VAL_END, horizon, bars))
            test_parts.append(slice_horizon(g, TEST_START, TEST_END, horizon, bars))
        train = pd.concat(train_parts, ignore_index=True)
        val = pd.concat(val_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        target = f"y_{horizon}"
        for factor in FACTORS:
            orient, qlo, qhi = learn(train, factor, target)
            tr = stats(train, factor, target, orient, qlo, qhi)
            va = stats(val, factor, target, orient, qlo, qhi)
            te = stats(test, factor, target, orient, qlo, qhi)
            passed = bool(tr["net_bps"] > GATE_MIN_TRAIN_NET_BPS and va["net_bps"] > GATE_MIN_VAL_NET_BPS and va["ic"] > GATE_MIN_VAL_IC and va["positive_months"] >= GATE_MIN_POSITIVE_MONTHS and va["n"] >= GATE_MIN_VAL_POSITIONS)
            rows.append({"factor": factor, "horizon": horizon, "orientation": orient, "qlo": qlo, "qhi": qhi, "train_net_bps": tr["net_bps"], "val_net_bps": va["net_bps"], "val_ic": va["ic"], "val_positive_months": va["positive_months"], "val_n": va["n"], "test_net_bps": te["net_bps"], "test_ic": te["ic"], "pass": passed})
            print(f"{horizon:>3} {factor:<19} train={tr['net_bps']:+7.2f}bps val={va['net_bps']:+7.2f}bps IC={va['ic']:+.4f} months={va['positive_months']}/{va['months']} n={va['n']:5d} | 2026={te['net_bps']:+7.2f}bps IC={te['ic']:+.4f} => {'PASS' if passed else 'FAIL'}", flush=True)

    out = pd.DataFrame(rows)
    passed = out[out["pass"]]
    print("\n=== GATE SUMMARY ===")
    print("GATE:", "PASS" if len(passed) else "FAIL")
    if len(passed):
        print(passed[["factor", "horizon", "train_net_bps", "val_net_bps", "val_ic", "val_positive_months", "test_net_bps"]].to_string(index=False))
    print("2026 is diagnostic only. Funding cashflows are added only if a premium candidate survives this gate.")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "premium_alpha_grid.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps({"passes": int(len(passed)), "cost_bps": COST_BPS}, indent=2))
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
