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

SYMBOLS = ["BTCUSDT", "ETHUSDT", "XRPUSDT", "SOLUSDT", "DOGEUSDT", "ADAUSDT"]
FACTORS = ["imbalance_notional", "imbalance_qty"]
HORIZONS = {"4h": 16, "8h": 32, "12h": 48}  # quarter-hour bars
COST_BPS = 8.0
TRAIN_START, TRAIN_END = "2024-01-01", "2025-01-01"
VAL_START, VAL_END = "2025-01-01", "2026-01-01"
TEST_START, TEST_END = "2026-01-01", "2026-08-19"

# Predeclared before viewing this lab's 2025/2026 results.
GATE_MIN_TRAIN_NET_BPS = 0.0
GATE_MIN_VAL_NET_BPS = 2.0
GATE_MIN_VAL_IC = 0.005
GATE_MIN_POSITIVE_MONTHS = 7
GATE_MIN_VAL_POSITIONS = 500


def pair_for_symbol(symbol: str) -> str:
    if not symbol.endswith("USDT"):
        raise ValueError(symbol)
    return f"{symbol[:-4]}/USDT:USDT"


def load_qh(db: Path, symbol: str) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        x = pd.read_sql_query(
            """
            SELECT bucket_ms, available_ms, symbol,
                   signed_qty, total_qty, signed_notional, total_notional,
                   trade_count, first_price, last_price,
                   imbalance_qty, imbalance_notional
            FROM qh_flow
            WHERE symbol=?
            ORDER BY bucket_ms
            """,
            con,
            params=(symbol,),
        )
    if x.empty:
        return x
    x["bucket_time"] = pd.to_datetime(x["bucket_ms"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
    x["available_time"] = pd.to_datetime(x["available_ms"], unit="ms", utc=True).astype("datetime64[ns, UTC]")
    return x


def build_symbol_frame(config: dict, datadir: Path, db: Path, symbol: str) -> pd.DataFrame:
    qh = load_qh(db, symbol)
    if qh.empty:
        return qh

    pair = pair_for_symbol(symbol)
    px = r.load_price(config, datadir, pair)
    if px.empty:
        return pd.DataFrame()
    px = px[["date", "open"]].copy()
    px["date"] = r.as_ns(px["date"])
    open_map = px.set_index("date")["open"]

    qh = qh.copy()
    qh["pair"] = pair
    qh["entry_price"] = pd.to_numeric(qh["last_price"], errors="coerce")

    # The signal is known after the first 10 seconds of the quarter-hour boundary.
    # Entry proxy is the last aggTrade price observed in that 10-second peak window.
    # Exit is the exact future 15m boundary open, so horizons are ~3:59:50 / 7:59:50 / 11:59:50.
    for name, bars in HORIZONS.items():
        exit_time = qh["bucket_time"] + pd.Timedelta(minutes=15 * bars)
        qh[f"exit_time_{name}"] = exit_time
        qh[f"exit_price_{name}"] = exit_time.map(open_map)
        qh[f"y_{name}"] = qh[f"exit_price_{name}"] / qh["entry_price"] - 1.0

    for f in FACTORS:
        qh[f] = pd.to_numeric(qh[f], errors="coerce").clip(-1.0, 1.0)
    return qh


def split_horizon(df: pd.DataFrame, start: str, end: str, horizon: str, bars: int) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    target = f"y_{horizon}"
    exit_col = f"exit_time_{horizon}"
    cols = ["bucket_time", "available_time", "pair", target, *FACTORS]
    x = df.loc[
        (df["available_time"] >= start_ts)
        & (df["available_time"] < end_ts)
        & (df[exit_col] < end_ts),
        cols,
    ].dropna(subset=[target]).copy()
    # Conservative non-overlapping observations per pair/horizon.
    if bars > 1 and not x.empty:
        x = x.iloc[::bars].copy()
    return x


def spearman(x: pd.Series, y: pd.Series) -> float:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3:
        return float("nan")
    return float(z.iloc[:, 0].corr(z.iloc[:, 1], method="spearman"))


def learn_rule(train: pd.DataFrame, factor: str, target: str) -> tuple[float, float, float, float]:
    z = train[[factor, target]].dropna()
    if len(z) < 100:
        raise RuntimeError(f"Not enough training rows for {factor}/{target}: {len(z)}")
    raw_ic = spearman(z[factor], z[target])
    orient = 1.0 if not np.isfinite(raw_ic) or raw_ic >= 0 else -1.0
    score = orient * z[factor]
    qlo, qhi = np.quantile(score, [0.25, 0.75])
    return orient, float(qlo), float(qhi), float(raw_ic)


def stats(df: pd.DataFrame, factor: str, target: str, orient: float, qlo: float, qhi: float) -> dict:
    x = df[["bucket_time", factor, target]].dropna().copy()
    if x.empty:
        return {"n": 0, "ic": np.nan, "gross_bps": np.nan, "net_bps": np.nan, "win_pct": np.nan, "positive_months": 0, "months": 0}
    score = orient * x[factor]
    side = np.where(score >= qhi, 1.0, np.where(score <= qlo, -1.0, 0.0))
    chosen = side != 0
    if not chosen.any():
        return {"n": 0, "ic": np.nan, "gross_bps": np.nan, "net_bps": np.nan, "win_pct": np.nan, "positive_months": 0, "months": 0}
    ret = side[chosen] * x.loc[chosen, target].to_numpy(dtype=float)
    t = x.loc[chosen, "bucket_time"].reset_index(drop=True)
    monthly = pd.DataFrame({"month": t.dt.strftime("%Y-%m"), "ret": ret}).groupby("month")["ret"].mean() * 10000.0 - COST_BPS
    gross = float(np.mean(ret) * 10000.0)
    return {
        "n": int(len(ret)),
        "ic": spearman(score, x[target]),
        "gross_bps": gross,
        "net_bps": gross - COST_BPS,
        "win_pct": float(np.mean(ret > 0) * 100.0),
        "positive_months": int((monthly > 0).sum()),
        "months": int(len(monthly)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Quarter-hour first-10s raw order-flow alpha lab")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/qh_orderflow.sqlite")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/qh_alpha")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    db = Path(args.db)
    if not db.exists():
        raise RuntimeError(f"Missing quarter-hour DB: {db}")

    frames = []
    print(f"Loading {len(SYMBOLS)} quarter-hour order-flow symbols...", flush=True)
    for i, sym in enumerate(SYMBOLS, 1):
        t0 = time.monotonic()
        f = build_symbol_frame(config, Path(args.datadir), db, sym)
        if f.empty:
            print(f"  [{i}/{len(SYMBOLS)}] {sym}: NO DATA", flush=True)
            continue
        frames.append(f)
        print(
            f"  [{i}/{len(SYMBOLS)}] {sym}: {len(f):,} rows "
            f"{f['bucket_time'].min()} -> {f['bucket_time'].max()} [{time.monotonic()-t0:.1f}s]",
            flush=True,
        )
    if not frames:
        raise RuntimeError("No usable order-flow data")

    all_df = pd.concat(frames, ignore_index=True)
    rows = []
    print("\n=== QUARTER-HOUR ORDER-FLOW ALPHA GATE ===")
    print("2024 calibration only | 2025 validation | 2026 diagnostic only")
    print("Signal available at boundary+10s; entry proxy=last trade in first 10s; roundtrip cost=8bps.")
    print("PASS per factor/horizon: train net>0; val net>2bps; val IC>0.005; >=7/12 positive val months; n>=500.")

    for horizon, bars in HORIZONS.items():
        train_parts, val_parts, test_parts = [], [], []
        for pair, g in all_df.groupby("pair", sort=True):
            g = g.sort_values("bucket_time")
            train_parts.append(split_horizon(g, TRAIN_START, TRAIN_END, horizon, bars))
            val_parts.append(split_horizon(g, VAL_START, VAL_END, horizon, bars))
            test_parts.append(split_horizon(g, TEST_START, TEST_END, horizon, bars))
        train = pd.concat(train_parts, ignore_index=True)
        val = pd.concat(val_parts, ignore_index=True)
        test = pd.concat(test_parts, ignore_index=True)
        target = f"y_{horizon}"

        for factor in FACTORS:
            orient, qlo, qhi, train_raw_ic = learn_rule(train, factor, target)
            tr = stats(train, factor, target, orient, qlo, qhi)
            va = stats(val, factor, target, orient, qlo, qhi)
            te = stats(test, factor, target, orient, qlo, qhi)
            passed = bool(
                tr["net_bps"] > GATE_MIN_TRAIN_NET_BPS
                and va["net_bps"] > GATE_MIN_VAL_NET_BPS
                and va["ic"] > GATE_MIN_VAL_IC
                and va["positive_months"] >= GATE_MIN_POSITIVE_MONTHS
                and va["n"] >= GATE_MIN_VAL_POSITIONS
            )
            rows.append({
                "factor": factor,
                "horizon": horizon,
                "orientation": orient,
                "qlo": qlo,
                "qhi": qhi,
                "train_raw_ic": train_raw_ic,
                "train_n": tr["n"],
                "train_ic": tr["ic"],
                "train_net_bps": tr["net_bps"],
                "val_n": va["n"],
                "val_ic": va["ic"],
                "val_net_bps": va["net_bps"],
                "val_win_pct": va["win_pct"],
                "val_positive_months": va["positive_months"],
                "test_n": te["n"],
                "test_ic": te["ic"],
                "test_net_bps": te["net_bps"],
                "test_win_pct": te["win_pct"],
                "pass": passed,
            })
            print(
                f"{horizon:>3} {factor:<19} train={tr['net_bps']:+7.2f}bps "
                f"val={va['net_bps']:+7.2f}bps IC={va['ic']:+.4f} "
                f"months={va['positive_months']}/{va['months']} n={va['n']:5d} "
                f"| 2026={te['net_bps']:+7.2f}bps IC={te['ic']:+.4f} "
                f"=> {'PASS' if passed else 'FAIL'}",
                flush=True,
            )

    out = pd.DataFrame(rows)
    passed = out[out["pass"]].copy()
    print("\n=== GATE SUMMARY ===")
    if passed.empty:
        print("GATE: FAIL - no raw quarter-hour factor/horizon cleared the predeclared 2025 gate.")
    else:
        print(f"GATE: PASS - {len(passed)} factor/horizon combinations cleared the gate.")
        print(passed[["factor", "horizon", "train_net_bps", "val_net_bps", "val_ic", "val_positive_months", "test_net_bps", "test_ic"]].to_string(index=False))
    print("2026 is diagnostic only and was not used to choose orientation, thresholds, factor, or horizon.")
    print("Funding cashflows are NOT included in these price-return alpha results; add them only if a candidate survives robustness.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out.to_csv(outdir / "quarter_hour_alpha_grid.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps({
        "passes": int(len(passed)),
        "criteria": {
            "train_net_bps_gt": GATE_MIN_TRAIN_NET_BPS,
            "val_net_bps_gt": GATE_MIN_VAL_NET_BPS,
            "val_ic_gt": GATE_MIN_VAL_IC,
            "val_positive_months_gte": GATE_MIN_POSITIVE_MONTHS,
            "val_positions_gte": GATE_MIN_VAL_POSITIONS,
            "cost_bps": COST_BPS,
        },
    }, indent=2))
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
