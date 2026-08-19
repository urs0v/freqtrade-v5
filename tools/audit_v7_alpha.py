#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType


HORIZONS = {"1h": 4, "4h": 16, "8h": 32, "12h": 48}
SCORE_BINS = [-np.inf, 0.64, 0.68, 0.72, 0.76, 0.80, 0.85, np.inf]
SCORE_LABELS = ["<.64", ".64-.68", ".68-.72", ".72-.76", ".76-.80", ".80-.85", ">=.85"]


def walk_frames(obj: Any, path: tuple[Any, ...] = ()):
    if isinstance(obj, pd.DataFrame):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():
            yield from walk_frames(value, path + (key,))
    elif isinstance(obj, (list, tuple)):
        for i, value in enumerate(obj):
            yield from walk_frames(value, path + (i,))


def infer_pair(path: tuple[Any, ...], df: pd.DataFrame) -> str | None:
    if "pair" in df.columns and not df["pair"].dropna().empty:
        return str(df["pair"].dropna().iloc[0])
    for item in reversed(path):
        if isinstance(item, str) and "/" in item and (":USDT" in item or "USDT" in item):
            return item
    return None


def extract_signals(signals_file: Path) -> pd.DataFrame:
    obj = pd.read_pickle(signals_file)
    chunks: list[pd.DataFrame] = []

    for path, frame in walk_frames(obj):
        if frame.empty or "date" not in frame.columns:
            continue
        if "enter_long" not in frame.columns and "enter_short" not in frame.columns:
            continue

        pair = infer_pair(path, frame)
        work = frame.copy()
        long_col = pd.to_numeric(work.get("enter_long", 0), errors="coerce").fillna(0)
        short_col = pd.to_numeric(work.get("enter_short", 0), errors="coerce").fillna(0)
        work = work[(long_col > 0) | (short_col > 0)].copy()
        if work.empty:
            continue

        work["date"] = pd.to_datetime(work["date"], utc=True)
        work["pair"] = pair if pair is not None else work.get("pair", "UNKNOWN")
        work["side"] = np.where(pd.to_numeric(work.get("enter_short", 0), errors="coerce").fillna(0) > 0, "short", "long")
        work["score"] = np.where(
            work["side"].eq("short"),
            pd.to_numeric(work.get("short_score", np.nan), errors="coerce"),
            pd.to_numeric(work.get("long_score", np.nan), errors="coerce"),
        )
        if "enter_tag" not in work.columns:
            work["enter_tag"] = "unknown"
        keep = ["date", "pair", "side", "enter_tag", "score", "close"]
        for extra in ["long_score", "short_score", "trend_strength", "adx_quality", "vol_stress", "volume_quality"]:
            if extra in work.columns:
                keep.append(extra)
        chunks.append(work[keep])

    if not chunks:
        raise RuntimeError(f"No entry-signal DataFrames found in {signals_file}")

    out = pd.concat(chunks, ignore_index=True)
    out = out.dropna(subset=["date", "pair", "score", "close"])
    out = out.drop_duplicates(subset=["pair", "date", "side", "enter_tag"])
    return out.sort_values(["pair", "date", "side"]).reset_index(drop=True)


def resolve_datadir(config: dict, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    if config.get("datadir"):
        return Path(config["datadir"])
    exchange = str(config.get("exchange", {}).get("name", "binance")).lower()
    return Path("/freqtrade/user_data/data") / exchange


def add_forward_returns(signals: pd.DataFrame, config: dict, datadir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    data_format = config.get("dataformat_ohlcv")

    for pair, sigs in signals.groupby("pair", sort=True):
        hist = load_pair_history(
            pair=str(pair),
            timeframe="15m",
            datadir=datadir,
            fill_up_missing=False,
            drop_incomplete=False,
            data_format=data_format,
            candle_type=CandleType.FUTURES,
        )
        if hist.empty:
            print(f"WARN no 15m futures history for {pair}")
            continue

        hist = hist.sort_values("date").reset_index(drop=True)
        hist["date"] = pd.to_datetime(hist["date"], utc=True)
        pos_by_date = {ts: i for i, ts in enumerate(hist["date"])}

        for _, sig in sigs.iterrows():
            pos = pos_by_date.get(sig["date"])
            if pos is None:
                continue
            entry = float(hist.at[pos, "close"])
            if not np.isfinite(entry) or entry <= 0:
                continue
            direction = -1.0 if sig["side"] == "short" else 1.0

            for horizon, bars in HORIZONS.items():
                end = pos + bars
                if end >= len(hist):
                    continue
                future_close = float(hist.at[end, "close"])
                if not np.isfinite(future_close):
                    continue

                price_ret = future_close / entry - 1.0
                side_ret = direction * price_ret
                window = hist.iloc[pos + 1 : end + 1]
                if window.empty:
                    continue
                if direction > 0:
                    mfe = float(window["high"].max() / entry - 1.0)
                    mae = float(window["low"].min() / entry - 1.0)
                else:
                    mfe = float(1.0 - window["low"].min() / entry)
                    mae = float(1.0 - window["high"].max() / entry)

                row = sig.to_dict()
                row.update(
                    {
                        "horizon": horizon,
                        "bars": bars,
                        "entry_close": entry,
                        "future_close": future_close,
                        "side_return": side_ret,
                        "mfe": mfe,
                        "mae": mae,
                    }
                )
                rows.append(row)

    if not rows:
        raise RuntimeError("No forward-return rows could be constructed. Check datadir/pair naming.")
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame):
    df = df.copy()
    df["score_bin"] = pd.cut(df["score"], SCORE_BINS, labels=SCORE_LABELS, right=False)
    df["win"] = df["side_return"] > 0

    def agg(g: pd.DataFrame) -> pd.Series:
        x = g["side_return"].dropna()
        if x.empty:
            return pd.Series(dtype=float)
        std = float(x.std(ddof=1)) if len(x) > 1 else np.nan
        tstat = float(x.mean() / (std / math.sqrt(len(x)))) if len(x) > 1 and std > 0 else np.nan
        return pd.Series(
            {
                "n": len(x),
                "mean_bps": float(x.mean() * 10000.0),
                "median_bps": float(x.median() * 10000.0),
                "win_pct": float((x > 0).mean() * 100.0),
                "mfe_bps": float(g.loc[x.index, "mfe"].mean() * 10000.0),
                "mae_bps": float(g.loc[x.index, "mae"].mean() * 10000.0),
                "t_stat": tstat,
            }
        )

    overall = df.groupby(["horizon"], observed=True).apply(agg, include_groups=False).reset_index()
    by_side = df.groupby(["horizon", "side"], observed=True).apply(agg, include_groups=False).reset_index()
    by_tag = df.groupby(["horizon", "enter_tag"], observed=True).apply(agg, include_groups=False).reset_index()
    by_score = df.groupby(["horizon", "score_bin"], observed=True).apply(agg, include_groups=False).reset_index()
    by_side_score = df.groupby(["horizon", "side", "score_bin"], observed=True).apply(agg, include_groups=False).reset_index()

    corr_rows = []
    for horizon, group in df.groupby("horizon"):
        for side in ["all", "long", "short"]:
            g = group if side == "all" else group[group["side"] == side]
            g = g[["score", "side_return"]].dropna()
            corr = float(g["score"].corr(g["side_return"], method="spearman")) if len(g) >= 3 else np.nan
            corr_rows.append({"horizon": horizon, "side": side, "n": len(g), "spearman_score_vs_return": corr})
    corr = pd.DataFrame(corr_rows)
    return overall, by_side, by_tag, by_score, by_side_score, corr


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit V7 score vs forward side-adjusted returns.")
    ap.add_argument("--signals", required=True, help="Freqtrade *_signals.pkl file")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default=None)
    ap.add_argument("--outdir", default="/freqtrade/user_data/v7/alpha_audit")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    datadir = resolve_datadir(config, args.datadir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    signals = extract_signals(Path(args.signals))
    forward = add_forward_returns(signals, config, datadir)
    overall, by_side, by_tag, by_score, by_side_score, corr = summarize(forward)

    signals.to_csv(outdir / "signals.csv", index=False)
    forward.to_csv(outdir / "forward_returns.csv", index=False)
    overall.to_csv(outdir / "overall.csv", index=False)
    by_side.to_csv(outdir / "by_side.csv", index=False)
    by_tag.to_csv(outdir / "by_tag.csv", index=False)
    by_score.to_csv(outdir / "by_score.csv", index=False)
    by_side_score.to_csv(outdir / "by_side_score.csv", index=False)
    corr.to_csv(outdir / "score_correlation.csv", index=False)

    print(f"Signals: {len(signals)} | forward rows: {len(forward)} | datadir: {datadir}")
    print("\n=== FORWARD EDGE (side-adjusted, before fees/leverage) ===")
    print(overall.to_string(index=False))
    print("\n=== SCORE BUCKETS ===")
    print(by_score.to_string(index=False))
    print("\n=== SCORE MONOTONICITY ===")
    print(corr.to_string(index=False))
    print(f"\nCSV output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
