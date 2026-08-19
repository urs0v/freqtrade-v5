#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.btanalysis import load_backtest_analysis_data, load_backtest_data
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

HORIZONS = {"1h": 4, "4h": 16, "8h": 32, "12h": 48}

DIRECTIONAL = [
    "trend_signed",
    "continuation_signed",
    "trend_dir_1h",
    "trend_dir_4h",
    "momentum_1h",
    "momentum_4h",
    "ret1",
    "ret4",
    "ret16",
    "ema24_slope6_1h",
    "ema18_slope4_4h",
]

QUALITY = [
    "adx_quality",
    "volume_quality",
    "vol_stress",
    "volume_z",
    "vol_ratio",
    "adx",
    "adx_1h",
    "adx_4h",
    "atr_pct",
    "atr_pct_1h",
    "atr_pct_4h",
    "score_gap",
    "confidence",
]


def as_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def pick_strategy(obj: dict) -> str:
    if not obj:
        raise RuntimeError("Signal export is empty")
    if len(obj) == 1:
        return next(iter(obj))
    for name in obj:
        if "AdaptivePerp15mV7Audit" in name:
            return name
    raise RuntimeError(f"Multiple strategies found: {list(obj)}")


def merge_signal_candles(zip_path: Path) -> pd.DataFrame:
    sig_obj = load_backtest_analysis_data(zip_path, "signals")
    strategy = pick_strategy(sig_obj)
    trades = load_backtest_data(zip_path, strategy).copy()
    trades["open_date"] = as_ns(trades["open_date"])

    chunks: list[pd.DataFrame] = []
    for pair, frame in sig_obj[strategy].items():
        if frame is None or frame.empty or "date" not in frame:
            continue
        left = trades.loc[trades["pair"] == pair].copy()
        if left.empty:
            continue
        right = frame.copy()
        right["date"] = as_ns(right["date"])
        left = left.sort_values("open_date")
        right = right.sort_values("date")
        m = pd.merge_asof(
            left,
            right,
            left_on="open_date",
            right_on="date",
            direction="backward",
            allow_exact_matches=False,
            suffixes=("", "_signal"),
        )
        m = m.dropna(subset=["date"])
        if m.empty:
            continue
        m["side"] = np.where(m["is_short"].fillna(False), "short", "long")
        m["direction"] = np.where(m["side"].eq("short"), -1.0, 1.0)
        chunks.append(m)

    if not chunks:
        raise RuntimeError("Could not merge signal candles with trades")
    out = pd.concat(chunks, ignore_index=True)
    print(f"Strategy: {strategy} | merged trades: {len(out)}")
    return out


def add_oriented_factors(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    out = df.copy()
    factors: list[str] = []

    for c in DIRECTIONAL:
        if c in out:
            name = f"side_{c}"
            out[name] = pd.to_numeric(out[c], errors="coerce") * out["direction"]
            factors.append(name)

    if "donch_pos" in out:
        out["side_donch"] = (2.0 * pd.to_numeric(out["donch_pos"], errors="coerce") - 1.0) * out["direction"]
        factors.append("side_donch")

    for c in ["rsi", "rsi_1h", "rsi_4h"]:
        if c in out:
            name = f"side_{c}"
            out[name] = ((pd.to_numeric(out[c], errors="coerce") - 50.0) / 50.0) * out["direction"]
            factors.append(name)

    for base in ["score", "trend_score", "pull_score", "pull_event"]:
        if base == "score" and {"long_score", "short_score"}.issubset(out.columns):
            out[base] = np.where(out["side"].eq("short"), out["short_score"], out["long_score"])
            factors.append(base)
        elif base == "trend_score" and {"trend_long_score", "trend_short_score"}.issubset(out.columns):
            out[base] = np.where(out["side"].eq("short"), out["trend_short_score"], out["trend_long_score"])
            factors.append(base)
        elif base == "pull_score" and {"pull_long_score", "pull_short_score"}.issubset(out.columns):
            out[base] = np.where(out["side"].eq("short"), out["pull_short_score"], out["pull_long_score"])
            factors.append(base)
        elif base == "pull_event" and {"pull_event_long", "pull_event_short"}.issubset(out.columns):
            out[base] = np.where(out["side"].eq("short"), out["pull_event_short"], out["pull_event_long"])
            factors.append(base)

    for c in QUALITY:
        if c in out:
            out[c] = pd.to_numeric(out[c], errors="coerce")
            factors.append(c)

    # preserve order while removing duplicates
    factors = list(dict.fromkeys(factors))
    return out, factors


def resolve_datadir(config: dict) -> Path:
    if config.get("datadir"):
        return Path(config["datadir"])
    exchange = str(config.get("exchange", {}).get("name", "binance")).lower()
    return Path("/freqtrade/user_data/data") / exchange


def add_forward_returns(df: pd.DataFrame, config: dict, datadir: Path) -> pd.DataFrame:
    rows = []
    fmt = config.get("dataformat_ohlcv")

    for pair, g in df.groupby("pair", sort=True):
        hist = load_pair_history(
            pair=str(pair),
            timeframe="15m",
            datadir=datadir,
            fill_up_missing=False,
            drop_incomplete=False,
            data_format=fmt,
            candle_type=CandleType.FUTURES,
        )
        if hist.empty:
            print(f"WARN no history: {pair}")
            continue
        hist = hist.sort_values("date").reset_index(drop=True)
        hist["date"] = as_ns(hist["date"])
        pos = {ts: i for i, ts in enumerate(hist["date"])}

        for _, r in g.iterrows():
            i = pos.get(r["date"])
            if i is None:
                continue
            entry = float(hist.at[i, "close"])
            direction = float(r["direction"])
            for horizon, bars in HORIZONS.items():
                j = i + bars
                if j >= len(hist):
                    continue
                future = float(hist.at[j, "close"])
                rec = r.to_dict()
                rec["horizon"] = horizon
                rec["side_return"] = direction * (future / entry - 1.0)
                rows.append(rec)

    if not rows:
        raise RuntimeError("No forward rows generated")
    return pd.DataFrame(rows)


def factor_stats(forward: pd.DataFrame, factors: list[str]) -> pd.DataFrame:
    rows = []
    for horizon, hg in forward.groupby("horizon"):
        for factor in factors:
            if factor not in hg:
                continue
            g = hg[[factor, "side_return"]].replace([np.inf, -np.inf], np.nan).dropna()
            if len(g) < 30 or g[factor].nunique() < 3:
                continue
            corr = g[factor].corr(g["side_return"], method="spearman")
            try:
                q = pd.qcut(g[factor], 4, labels=False, duplicates="drop")
                lo = g.loc[q == q.min(), "side_return"].mean() * 10000.0
                hi = g.loc[q == q.max(), "side_return"].mean() * 10000.0
                spread = hi - lo
            except Exception:
                lo = hi = spread = np.nan
            rows.append(
                {
                    "horizon": horizon,
                    "factor": factor,
                    "n": len(g),
                    "spearman": float(corr),
                    "low_q_mean_bps": float(lo),
                    "high_q_mean_bps": float(hi),
                    "high_minus_low_bps": float(spread),
                }
            )
    return pd.DataFrame(rows)


def tag_stats(forward: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (h, tag, side), g in forward.groupby(["horizon", "enter_tag", "side"], dropna=False):
        x = g["side_return"].dropna()
        if x.empty:
            continue
        rows.append(
            {
                "horizon": h,
                "enter_tag": tag,
                "side": side,
                "n": len(x),
                "mean_bps": x.mean() * 10000.0,
                "median_bps": x.median() * 10000.0,
                "win_pct": (x > 0).mean() * 100.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Component-level audit of V7 signal factors")
    ap.add_argument("--zip", required=True, help="Freqtrade backtest ZIP with signal export")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--outdir", default="/freqtrade/user_data/v7/alpha_audit")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    merged = merge_signal_candles(Path(args.zip))
    merged, factors = add_oriented_factors(merged)
    forward = add_forward_returns(merged, config, resolve_datadir(config))
    stats = factor_stats(forward, factors)
    tags = tag_stats(forward)

    stats.to_csv(outdir / "component_factors.csv", index=False)
    tags.to_csv(outdir / "component_tags.csv", index=False)

    print("\n=== COMPONENT FACTORS: 4h ===")
    x = stats[stats["horizon"] == "4h"].copy()
    if not x.empty:
        x["abs_spearman"] = x["spearman"].abs()
        print(x.sort_values("abs_spearman", ascending=False).head(15).drop(columns="abs_spearman").to_string(index=False))

    print("\n=== COMPONENT FACTORS: 8h ===")
    x = stats[stats["horizon"] == "8h"].copy()
    if not x.empty:
        x["abs_spearman"] = x["spearman"].abs()
        print(x.sort_values("abs_spearman", ascending=False).head(15).drop(columns="abs_spearman").to_string(index=False))

    print("\n=== ENTRY TAG EDGE: 4h / 8h ===")
    print(tags[tags["horizon"].isin(["4h", "8h"])].sort_values(["horizon", "mean_bps"], ascending=[True, False]).to_string(index=False))
    print(f"\nCSV output: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
