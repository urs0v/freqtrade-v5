#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from freqtrade.data.btanalysis import load_backtest_analysis_data, load_backtest_data
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType


HORIZONS = {"1h": 4, "4h": 16, "8h": 32, "12h": 48}
SCORE_BINS = [-np.inf, 0.64, 0.68, 0.72, 0.76, 0.80, 0.85, np.inf]
SCORE_LABELS = ["<.64", ".64-.68", ".68-.72", ".72-.76", ".76-.80", ".80-.85", ">=.85"]
KEEP_EXTRA = [
    "long_score",
    "short_score",
    "trend_strength",
    "adx_quality",
    "vol_stress",
    "volume_quality",
]


def _utc_ns(values: pd.Series) -> pd.Series:
    """Normalize tz-aware timestamps to one exact pandas resolution for joins/lookups."""
    return pd.to_datetime(values, utc=True).dt.as_unit("ns")


def _pick_strategy(signal_obj: dict[str, Any]) -> str:
    if not signal_obj:
        raise RuntimeError("Signal export is empty.")
    if len(signal_obj) == 1:
        return next(iter(signal_obj))
    for name in signal_obj:
        if "AdaptivePerp15mV7Audit" in name:
            return name
    raise RuntimeError(f"Multiple strategies in signal export: {list(signal_obj)}")


def extract_signals(backtest_zip: Path) -> pd.DataFrame:
    """Merge Freqtrade's exported signal candles with trade metadata.

    Freqtrade's *_signals.pkl contains strategy -> pair -> DataFrame of the
    indicator candle preceding each trade. It does not need to contain
    enter_long/enter_short. Side and enter_tag are therefore taken from the
    trade records in the same backtest ZIP, matching Freqtrade's own
    backtesting-analysis semantics (last signal candle strictly before open).
    """
    if backtest_zip.suffix.lower() != ".zip":
        raise RuntimeError("Alpha audit now expects the complete backtest .zip file.")

    signal_obj = load_backtest_analysis_data(backtest_zip, "signals")
    strategy_name = _pick_strategy(signal_obj)
    trades = load_backtest_data(backtest_zip, strategy_name)
    if trades.empty:
        raise RuntimeError("Backtest ZIP contains no trades.")

    trades = trades.copy()
    trades["open_date"] = _utc_ns(trades["open_date"])
    chunks: list[pd.DataFrame] = []

    pair_frames = signal_obj.get(strategy_name, {})
    print(f"Signal strategy: {strategy_name} | trade rows: {len(trades)} | pairs: {len(pair_frames)}")

    for pair, frame in pair_frames.items():
        if frame is None or frame.empty or "date" not in frame.columns:
            continue

        pair_trades = trades.loc[trades["pair"] == pair].copy()
        if pair_trades.empty:
            continue

        sig = frame.copy()
        sig["date"] = _utc_ns(sig["date"])
        sig = sig.sort_values("date")
        pair_trades = pair_trades.sort_values("open_date")

        # Same rule Freqtrade uses in backtesting-analysis: attach the last
        # signal candle strictly before the trade's open_date.
        merged = pd.merge_asof(
            pair_trades,
            sig,
            left_on="open_date",
            right_on="date",
            direction="backward",
            allow_exact_matches=False,
            suffixes=("", "_signal"),
        )
        merged = merged.dropna(subset=["date"])
        if merged.empty:
            continue

        missing = [c for c in ("long_score", "short_score", "close") if c not in merged.columns]
        if missing:
            raise RuntimeError(
                f"Signal export for {pair} is missing {missing}. "
                f"Available indicator columns include: {list(sig.columns)[:80]}"
            )

        merged["side"] = np.where(merged["is_short"].fillna(False), "short", "long")
        merged["score"] = np.where(
            merged["side"].eq("short"),
            pd.to_numeric(merged["short_score"], errors="coerce"),
            pd.to_numeric(merged["long_score"], errors="coerce"),
        )
        merged["enter_tag"] = merged["enter_tag"].fillna("unknown")

        keep = ["date", "pair", "side", "enter_tag", "score", "close"]
        keep.extend([c for c in KEEP_EXTRA if c in merged.columns])
        chunks.append(merged[keep])

    if not chunks:
        raise RuntimeError("Could not merge any signal candles with trades from the ZIP.")

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
        hist["date"] = _utc_ns(hist["date"])
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

                side_ret = direction * (future_close / entry - 1.0)
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

    overall = df.groupby("horizon", observed=True).apply(agg, include_groups=False).reset_index()
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
            corr_rows.append(
                {"horizon": horizon, "side": side, "n": len(g), "spearman_score_vs_return": corr}
            )
    return overall, by_side, by_tag, by_score, by_side_score, pd.DataFrame(corr_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit V7 score vs forward side-adjusted returns.")
    ap.add_argument("--signals", required=True, help="Complete Freqtrade backtest ZIP")
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
