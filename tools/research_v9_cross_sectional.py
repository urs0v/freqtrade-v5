#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

LOOKBACKS = {"1h": 4, "4h": 16, "16h": 64}
HOLDS = {"1h": 4, "4h": 16, "8h": 32}
MODES = ["reversal", "momentum"]


def as_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def load_pair(config: dict, datadir: Path, pair: str) -> pd.DataFrame:
    return load_pair_history(
        pair=pair,
        timeframe="15m",
        datadir=datadir,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"),
        candle_type=CandleType.FUTURES,
    )


def build_panel(config: dict, datadir: Path, pairs: list[str]) -> pd.DataFrame:
    frames = []
    total = len(pairs)
    print(f"[1/3] Loading 15m data for {total} pairs...", flush=True)
    for i, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        df = load_pair(config, datadir, pair)
        if df.empty:
            print(f"  [{i:02d}/{total}] {pair}: NO DATA", flush=True)
            continue
        df = df[["date", "open", "close"]].copy().sort_values("date")
        df["date"] = as_ns(df["date"])
        df["pair"] = pair
        frames.append(df)
        elapsed = time.monotonic() - t0
        print(
            f"  [{i:02d}/{total}] {pair}: {len(df):,} candles "
            f"({df['date'].min().date()} -> {df['date'].max().date()}) [{elapsed:.1f}s]",
            flush=True,
        )
    if not frames:
        raise RuntimeError("No futures 15m data loaded.")
    print("[1/3] Data loading complete. Building cross-sectional panel...", flush=True)
    long = pd.concat(frames, ignore_index=True)
    return long.sort_values(["date", "pair"]).reset_index(drop=True)


def make_wide(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    close = panel.pivot(index="date", columns="pair", values="close").sort_index()
    open_ = panel.pivot(index="date", columns="pair", values="open").sort_index()
    common = close.index.intersection(open_.index)
    close = close.loc[common]
    open_ = open_.loc[common]
    print(
        f"[2/3] Panel ready: {len(common):,} timestamps x {close.shape[1]} pairs. "
        f"Range {common.min()} -> {common.max()}",
        flush=True,
    )
    return close, open_


def evaluate_candidate(
    close: pd.DataFrame,
    open_: pd.DataFrame,
    start: str,
    end: str,
    lookback_bars: int,
    hold_bars: int,
    mode: str,
    q: float,
    signal_step: int,
    roundtrip_cost_bps: float,
) -> dict:
    past = close / close.shift(lookback_bars) - 1.0
    # Cross-sectional market-neutral residual of past move.
    resid = past.sub(past.median(axis=1), axis=0)

    entry = open_.shift(-1)
    exit_ = open_.shift(-(1 + hold_bars))
    fwd = exit_ / entry - 1.0

    idx = resid.index
    mask_time = (idx >= pd.Timestamp(start, tz="UTC")) & (idx < pd.Timestamp(end, tz="UTC"))
    eligible_idx = idx[mask_time]
    if signal_step > 1:
        eligible_idx = eligible_idx[::signal_step]

    portfolio_returns: list[float] = []
    n_positions = 0

    for ts in eligible_idx:
        s = resid.loc[ts].dropna()
        r = fwd.loc[ts].reindex(s.index).dropna()
        s = s.reindex(r.index).dropna()
        if len(s) < 8:
            continue
        lo = s.quantile(q)
        hi = s.quantile(1.0 - q)
        low_names = s.index[s <= lo]
        high_names = s.index[s >= hi]
        if len(low_names) == 0 or len(high_names) == 0:
            continue

        if mode == "reversal":
            x = pd.concat([r.reindex(low_names), -r.reindex(high_names)]).dropna()
        else:
            x = pd.concat([-r.reindex(low_names), r.reindex(high_names)]).dropna()
        if x.empty:
            continue
        portfolio_returns.append(float(x.mean()))
        n_positions += int(len(x))

    if not portfolio_returns:
        return {
            "n_signals": 0,
            "n_positions": 0,
            "gross_mean_bps": np.nan,
            "gross_median_bps": np.nan,
            "win_pct": np.nan,
            "net_mean_bps": np.nan,
        }

    arr = np.asarray(portfolio_returns, dtype=float)
    gross_mean_bps = float(arr.mean() * 10000.0)
    return {
        "n_signals": int(len(arr)),
        "n_positions": int(n_positions),
        "gross_mean_bps": gross_mean_bps,
        "gross_median_bps": float(np.median(arr) * 10000.0),
        "win_pct": float(np.mean(arr > 0) * 100.0),
        "net_mean_bps": gross_mean_bps - float(roundtrip_cost_bps),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="V9 cross-sectional momentum/reversal research")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--outdir", default="/freqtrade/user_data/v9/research")
    ap.add_argument("--train-start", default="2022-01-01")
    ap.add_argument("--train-end", default="2025-01-01")
    ap.add_argument("--val-start", default="2025-01-01")
    ap.add_argument("--val-end", default="2026-01-01")
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--test-end", default="2026-08-19")
    ap.add_argument("--quantile", type=float, default=0.25)
    ap.add_argument("--signal-step", type=int, default=4, help="Evaluate every Nth 15m candle")
    ap.add_argument("--roundtrip-cost-bps", type=float, default=8.0)
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist in config")

    panel = build_panel(config, Path(args.datadir), pairs)
    close, open_ = make_wide(panel)

    rows = []
    total_candidates = len(LOOKBACKS) * len(HOLDS) * len(MODES)
    candidate_no = 0
    print(f"[3/3] Evaluating {total_candidates} candidates (train + 2025 validation)...", flush=True)

    for lb_name, lb in LOOKBACKS.items():
        for hold_name, hold in HOLDS.items():
            for mode in MODES:
                candidate_no += 1
                base = {"lookback": lb_name, "hold": hold_name, "mode": mode}
                t0 = time.monotonic()
                print(
                    f"  [{candidate_no:02d}/{total_candidates}] lookback={lb_name}, hold={hold_name}, mode={mode} ...",
                    end=" ",
                    flush=True,
                )
                tr = evaluate_candidate(
                    close, open_, args.train_start, args.train_end, lb, hold, mode,
                    args.quantile, args.signal_step, args.roundtrip_cost_bps,
                )
                va = evaluate_candidate(
                    close, open_, args.val_start, args.val_end, lb, hold, mode,
                    args.quantile, args.signal_step, args.roundtrip_cost_bps,
                )
                elapsed = time.monotonic() - t0
                print(
                    f"train={tr['net_mean_bps']:+.2f} bps | val={va['net_mean_bps']:+.2f} bps "
                    f"| {elapsed:.1f}s",
                    flush=True,
                )
                rows.append({
                    **base,
                    **{f"train_{k}": v for k, v in tr.items()},
                    **{f"val_{k}": v for k, v in va.items()},
                })

    grid = pd.DataFrame(rows)
    grid["robust_pre2026"] = (grid["train_net_mean_bps"] > 0) & (grid["val_net_mean_bps"] > 0)
    grid = grid.sort_values(
        ["robust_pre2026", "val_net_mean_bps", "train_net_mean_bps"],
        ascending=[False, False, False],
    )

    print("\n=== V9 CROSS-SECTIONAL PRE-2026 GRID ===")
    show_cols = [
        "lookback", "hold", "mode", "train_net_mean_bps",
        "val_net_mean_bps", "val_win_pct", "robust_pre2026",
    ]
    print(grid[show_cols].to_string(index=False))

    robust = grid[grid["robust_pre2026"]]
    selected = robust.iloc[0] if not robust.empty else grid.iloc[0]
    print("\n=== PRE-2026 SELECTION ===")
    print(selected[show_cols].to_string())
    if robust.empty:
        print("GATE: FAIL - no candidate is net-positive in both train and 2025 validation.")
    else:
        print("GATE: PASS - candidate is net-positive in both train and 2025 validation.")

    lb = LOOKBACKS[str(selected["lookback"])]
    hold = HOLDS[str(selected["hold"])]
    mode = str(selected["mode"])
    print("\nEvaluating selected candidate on 2026 diagnostic...", flush=True)
    te = evaluate_candidate(
        close, open_, args.test_start, args.test_end, lb, hold, mode,
        args.quantile, args.signal_step, args.roundtrip_cost_bps,
    )
    print("\n=== 2026 DIAGNOSTIC FOR PRE-2026 SELECTED CANDIDATE ===")
    print(pd.Series({"lookback": selected["lookback"], "hold": selected["hold"], "mode": mode, **te}).to_string())
    print("NOTE: 2026 has already been inspected in prior experiments, so treat this as diagnostic, not pristine OOS.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(outdir / "grid_pre2026.csv", index=False)
    pd.DataFrame([
        {**{"lookback": selected["lookback"], "hold": selected["hold"], "mode": mode}, **te}
    ]).to_csv(outdir / "selected_2026.csv", index=False)
    print(f"\nOutput: {outdir}")
    print(f"Total runtime: {time.monotonic() - started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
