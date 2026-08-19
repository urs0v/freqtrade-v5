#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

BAR_MINUTES = 15
HORIZONS = {"1h": 4, "4h": 16, "8h": 32, "12h": 48}
COST_BPS = 8.0

FACTOR_FAMILY = {
    "price_1h": "price_baseline",
    "price_4h": "price_baseline",
    "taker_now": "flow",
    "taker_1h": "flow",
    "funding_z": "funding",
    "top_ratio_z": "positioning",
    "price_x_oi_1h": "oi_price",
    "price_x_oi_4h": "oi_price",
    "taker_x_oi_1h": "flow_oi",
    "taker_minus_funding": "flow_funding",
    "top_minus_taker": "positioning_flow",
}
FACTORS = list(FACTOR_FAMILY)

GATE_MIN_VAL_NET_BPS = 2.0
GATE_MIN_VAL_IC = 0.005
GATE_MIN_POSITIVE_MONTHS = 7
GATE_MIN_VAL_POSITIONS = 500


def as_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def robust_z(s: pd.Series, window: int, min_periods: int) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    med = x.rolling(window, min_periods=min_periods).median()
    dev = (x - med).abs()
    mad = dev.rolling(window, min_periods=min_periods).median()
    return ((x - med) / (1.4826 * mad).replace(0, np.nan)).clip(-5.0, 5.0)


def pair_to_symbol(pair: str) -> str:
    return pair.split("/")[0] + "USDT"


def load_price(config: dict, datadir: Path, pair: str) -> pd.DataFrame:
    return load_pair_history(
        pair=pair,
        timeframe="15m",
        datadir=datadir,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"),
        candle_type=CandleType.FUTURES,
    )


def load_derivatives(db: Path, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    with sqlite3.connect(db) as con:
        df = pd.read_sql_query(
            """
            SELECT bucket_ms AS available_ms,
                   oi,
                   funding_rate,
                   taker_ratio,
                   top_ls_ratio
            FROM features
            WHERE symbol=? AND bucket_ms BETWEEN ? AND ?
            ORDER BY bucket_ms
            """,
            con,
            params=(symbol, start_ms, end_ms),
        )
    if df.empty:
        return df
    df["available_time"] = pd.to_datetime(df["available_ms"], unit="ms", utc=True)
    return df.drop(columns=["available_ms"])


def build_features(price: pd.DataFrame, deriv: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = price[["date", "open", "high", "low", "close", "volume"]].copy().sort_values("date")
    df["date"] = as_ns(df["date"])
    # Freqtrade candle date is the candle OPEN. The signal is usable at candle close.
    df["signal_time"] = df["date"] + pd.Timedelta(minutes=BAR_MINUTES)

    if deriv.empty:
        for c in ["oi", "funding_rate", "taker_ratio", "top_ls_ratio"]:
            df[c] = np.nan
    else:
        deriv = deriv.copy().sort_values("available_time")
        df = pd.merge_asof(
            df.sort_values("signal_time"),
            deriv,
            left_on="signal_time",
            right_on="available_time",
            direction="backward",
            tolerance=pd.Timedelta("30min"),
        )

    pre = df[df["signal_time"] < pd.Timestamp("2026-01-01", tz="UTC")]
    coverage = {
        "rows_pre2026": int(len(pre)),
        "oi_pct": float(pre["oi"].notna().mean() * 100.0) if len(pre) else 0.0,
        "taker_pct": float(pre["taker_ratio"].notna().mean() * 100.0) if len(pre) else 0.0,
        "funding_pct": float(pre["funding_rate"].notna().mean() * 100.0) if len(pre) else 0.0,
        "top_pct": float(pre["top_ls_ratio"].notna().mean() * 100.0) if len(pre) else 0.0,
    }

    eps = 1e-12
    ret15 = df["close"].pct_change()
    vol96 = ret15.rolling(96, min_periods=48).std().replace(0, np.nan)
    ret1h = df["close"].pct_change(4)
    ret4h = df["close"].pct_change(16)
    df["price_1h"] = np.tanh(ret1h / (vol96 * np.sqrt(4.0) + eps))
    df["price_4h"] = np.tanh(ret4h / (vol96 * np.sqrt(16.0) + eps))

    oi = pd.to_numeric(df["oi"], errors="coerce").where(lambda x: x > 0)
    log_oi = np.log(oi)
    oi1h = log_oi.diff(4)
    oi4h = log_oi.diff(16)
    oi1h_z = robust_z(oi1h, 672, 96)
    oi4h_z = robust_z(oi4h, 672, 96)

    taker = pd.to_numeric(df["taker_ratio"], errors="coerce").where(lambda x: x > 0)
    taker_imb = ((taker - 1.0) / (taker + 1.0)).clip(-0.95, 0.95)
    df["taker_now"] = robust_z(taker_imb, 672, 96)
    df["taker_1h"] = robust_z(taker_imb.rolling(4, min_periods=2).mean(), 672, 96)

    funding = pd.to_numeric(df["funding_rate"], errors="coerce")
    df["funding_z"] = robust_z(funding, 2880, 256)

    top = pd.to_numeric(df["top_ls_ratio"], errors="coerce").where(lambda x: x > 0)
    top_imb = ((top - 1.0) / (top + 1.0)).clip(-0.95, 0.95)
    df["top_ratio_z"] = robust_z(top_imb, 672, 96)

    # Predeclared interactions. No post-2025 result is used to choose their sign:
    # orientation is learned on 2022-2024 only and frozen for validation/test.
    df["price_x_oi_1h"] = (df["price_1h"] * oi1h_z).clip(-5, 5)
    df["price_x_oi_4h"] = (df["price_4h"] * oi4h_z).clip(-5, 5)
    df["taker_x_oi_1h"] = (df["taker_1h"] * (1.0 + oi1h_z.abs().clip(0, 3))).clip(-8, 8)
    df["taker_minus_funding"] = (df["taker_1h"] - 0.5 * df["funding_z"]).clip(-8, 8)
    df["top_minus_taker"] = (df["top_ratio_z"] - df["taker_1h"]).clip(-8, 8)

    entry = df["open"].shift(-1)
    for name, bars in HORIZONS.items():
        df[f"y_{name}"] = df["open"].shift(-(1 + bars)) / entry - 1.0
        df[f"exit_time_{name}"] = df["signal_time"] + pd.Timedelta(minutes=BAR_MINUTES * bars)

    return df, coverage


def slice_horizon(df: pd.DataFrame, start: str, end: str, horizon: str, bars: int) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    target = f"y_{horizon}"
    exit_col = f"exit_time_{horizon}"
    cols = ["signal_time", target, *FACTORS]
    x = df.loc[
        (df["signal_time"] >= start_ts)
        & (df["signal_time"] < end_ts)
        & (df[exit_col] < end_ts),
        cols,
    ].copy()
    x = x.dropna(subset=[target])
    # Non-overlapping observations per pair/horizon reduce serial dependence.
    if bars > 1:
        x = x.iloc[::bars].copy()
    return x


def spearman(x: pd.Series, y: pd.Series) -> float:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3:
        return float("nan")
    return float(z.iloc[:, 0].corr(z.iloc[:, 1], method="spearman"))


def position_stats(df: pd.DataFrame, factor: str, target: str, orient: float, qlo: float, qhi: float, cost_bps: float) -> dict:
    x = df[["signal_time", factor, target]].dropna().copy()
    if x.empty:
        return {"n": 0, "ic": np.nan, "gross_bps": np.nan, "net_bps": np.nan, "win_pct": np.nan, "positive_months": 0, "months": 0}

    score = orient * x[factor]
    side = np.where(score >= qhi, 1.0, np.where(score <= qlo, -1.0, 0.0))
    chosen = side != 0
    if not chosen.any():
        return {"n": 0, "ic": np.nan, "gross_bps": np.nan, "net_bps": np.nan, "win_pct": np.nan, "positive_months": 0, "months": 0}

    ret = side[chosen] * x.loc[chosen, target].to_numpy(dtype=float)
    times = x.loc[chosen, "signal_time"].reset_index(drop=True)
    gross_bps = float(np.mean(ret) * 10000.0)
    tmp = pd.DataFrame({"month": times.dt.strftime("%Y-%m"), "ret": ret})
    monthly_net = tmp.groupby("month")["ret"].mean() * 10000.0 - cost_bps
    return {
        "n": int(len(ret)),
        "ic": spearman(score, x[target]),
        "gross_bps": gross_bps,
        "net_bps": gross_bps - cost_bps,
        "win_pct": float(np.mean(ret > 0) * 100.0),
        "positive_months": int((monthly_net > 0).sum()),
        "months": int(len(monthly_net)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Point-in-time derivatives alpha lab")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/results")
    ap.add_argument("--train-start", default="2022-01-01")
    ap.add_argument("--train-end", default="2025-01-01")
    ap.add_argument("--val-start", default="2025-01-01")
    ap.add_argument("--val-end", default="2026-01-01")
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--test-end", default="2026-08-19")
    ap.add_argument("--cost-bps", type=float, default=COST_BPS)
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist in config")
    db = Path(args.db)
    if not db.exists():
        raise RuntimeError(f"Missing point-in-time derivatives DB: {db}. Run backfill_derivatives_lab.sh first.")

    datasets = {
        split: {h: [] for h in HORIZONS}
        for split in ("train", "val", "test")
    }
    ranges = {
        "train": (args.train_start, args.train_end),
        "val": (args.val_start, args.val_end),
        "test": (args.test_start, args.test_end),
    }
    coverage_rows = []

    print(f"Loading {len(pairs)} pairs and aligning derivatives by information availability...", flush=True)
    for i, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        price = load_price(config, Path(args.datadir), pair)
        if price.empty:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: NO PRICE DATA", flush=True)
            continue
        price["date"] = as_ns(price["date"])
        start_ms = int((price["date"].min() - pd.Timedelta("1d")).timestamp() * 1000)
        end_ms = int((price["date"].max() + pd.Timedelta("1d")).timestamp() * 1000)
        deriv = load_derivatives(db, pair_to_symbol(pair), start_ms, end_ms)
        feat, cov = build_features(price, deriv)
        cov["pair"] = pair
        coverage_rows.append(cov)

        core_cov = min(cov["oi_pct"], cov["taker_pct"])
        if core_cov < 50.0:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: SKIP core derivative coverage={core_cov:.1f}%", flush=True)
            continue

        for split, (start, end) in ranges.items():
            for h, bars in HORIZONS.items():
                x = slice_horizon(feat, start, end, h, bars)
                if not x.empty:
                    x["pair"] = pair
                    datasets[split][h].append(x)

        print(
            f"  [{i:02d}/{len(pairs)}] {pair}: OI {cov['oi_pct']:.1f}% | taker {cov['taker_pct']:.1f}% | "
            f"funding {cov['funding_pct']:.1f}% | top {cov['top_pct']:.1f}% [{time.monotonic()-t0:.1f}s]",
            flush=True,
        )

    pooled: dict[str, dict[str, pd.DataFrame]] = {s: {} for s in datasets}
    for split in datasets:
        for h in HORIZONS:
            pooled[split][h] = pd.concat(datasets[split][h], ignore_index=True) if datasets[split][h] else pd.DataFrame()

    rows = []
    total = len(HORIZONS) * len(FACTORS)
    ntest = 0
    print(f"\nPre-2026 factor gate: {len(FACTORS)} factors x {len(HORIZONS)} horizons = {total} tests", flush=True)
    print(
        f"PASS requires train net>0, validation net>{GATE_MIN_VAL_NET_BPS:.1f} bps after {args.cost_bps:.1f} bps cost, "
        f"validation IC>{GATE_MIN_VAL_IC:.3f}, >= {GATE_MIN_POSITIVE_MONTHS}/12 positive months, n>={GATE_MIN_VAL_POSITIONS}.",
        flush=True,
    )

    for h in HORIZONS:
        tr = pooled["train"][h]
        va = pooled["val"][h]
        target = f"y_{h}"
        if tr.empty or va.empty:
            continue
        for factor in FACTORS:
            ntest += 1
            base = tr[[factor, target]].dropna()
            if len(base) < 100:
                continue
            raw_ic = spearman(base[factor], base[target])
            orient = 1.0 if not np.isfinite(raw_ic) or raw_ic >= 0 else -1.0
            score = orient * base[factor]
            qlo, qhi = np.quantile(score, [0.25, 0.75])
            tr_s = position_stats(tr, factor, target, orient, qlo, qhi, args.cost_bps)
            va_s = position_stats(va, factor, target, orient, qlo, qhi, args.cost_bps)
            gate = (
                FACTOR_FAMILY[factor] != "price_baseline"
                and tr_s["net_bps"] > 0
                and va_s["net_bps"] > GATE_MIN_VAL_NET_BPS
                and va_s["ic"] > GATE_MIN_VAL_IC
                and va_s["positive_months"] >= GATE_MIN_POSITIVE_MONTHS
                and va_s["n"] >= GATE_MIN_VAL_POSITIONS
            )
            rows.append({
                "horizon": h,
                "factor": factor,
                "family": FACTOR_FAMILY[factor],
                "orientation": int(orient),
                "train_q25": float(qlo),
                "train_q75": float(qhi),
                **{f"train_{k}": v for k, v in tr_s.items()},
                **{f"val_{k}": v for k, v in va_s.items()},
                "gate_pass": bool(gate),
            })
            print(
                f"  [{ntest:02d}/{total}] {h:>3} {factor:<22} train={tr_s['net_bps']:+6.2f} | "
                f"val={va_s['net_bps']:+6.2f} bps | IC={va_s['ic']:+.4f} | "
                f"months={va_s['positive_months']}/{va_s['months']} | {'PASS' if gate else 'fail'}",
                flush=True,
            )

    grid = pd.DataFrame(rows)
    if grid.empty:
        raise RuntimeError("No factor tests could be constructed. Check derivatives coverage.")
    grid = grid.sort_values(["gate_pass", "val_net_bps", "val_ic"], ascending=[False, False, False])
    passed = grid[grid["gate_pass"]]
    selected = passed.iloc[0] if not passed.empty else grid[grid["family"] != "price_baseline"].iloc[0]

    print("\n=== DERIVATIVES ALPHA PRE-2026 GATE ===")
    show = ["horizon", "factor", "family", "train_net_bps", "val_net_bps", "val_ic", "val_positive_months", "val_months", "gate_pass"]
    print(grid[show].head(20).to_string(index=False))
    if passed.empty:
        print("GATE: FAIL - no derivatives factor cleared the predeclared pre-2026 robustness gate.")
    else:
        print(f"GATE: PASS - {len(passed)} factor/horizon combinations cleared the gate.")

    # 2026 is diagnostic only and never participates in selection/orientation/thresholds.
    h = str(selected["horizon"])
    factor = str(selected["factor"])
    te = pooled["test"][h]
    te_s = position_stats(
        te,
        factor,
        f"y_{h}",
        float(selected["orientation"]),
        float(selected["train_q25"]),
        float(selected["train_q75"]),
        args.cost_bps,
    )
    print("\n=== 2026 DIAGNOSTIC FOR PRE-2026 SELECTED FACTOR ===")
    print(pd.Series({"horizon": h, "factor": factor, "family": selected["family"], **te_s}).to_string())
    print("2026 is diagnostic only; it was not used to choose factor sign, thresholds, horizon, or gate.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage_rows).to_csv(outdir / "coverage.csv", index=False)
    grid.to_csv(outdir / "factor_grid_pre2026.csv", index=False)
    pd.DataFrame([{**{"horizon": h, "factor": factor, "family": selected["family"]}, **te_s}]).to_csv(outdir / "selected_2026.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps({
        "pass": bool(not passed.empty),
        "passed": int(len(passed)),
        "selected_horizon": h,
        "selected_factor": factor,
        "cost_bps": args.cost_bps,
        "criteria": {
            "train_net_bps_gt": 0.0,
            "val_net_bps_gt": GATE_MIN_VAL_NET_BPS,
            "val_ic_gt": GATE_MIN_VAL_IC,
            "val_positive_months_gte": GATE_MIN_POSITIVE_MONTHS,
            "val_positions_gte": GATE_MIN_VAL_POSITIONS,
        },
    }, indent=2))
    print(f"\nOutput: {outdir}")
    print(f"Total runtime: {time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
