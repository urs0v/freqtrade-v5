#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

VOL_FLOOR_24H = 70_000_000.0
TOP_N = 3
SPIKE_MULT = 5.0
SPIKE_MOVE_PCT = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="Build causal Digash-style activity panel from cached OHLCV")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    return p.parse_args()


def as_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def load_tf(config: dict, datadir: Path, pair: str, timeframe: str) -> pd.DataFrame:
    return load_pair_history(
        pair=pair,
        timeframe=timeframe,
        datadir=datadir,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"),
        candle_type=CandleType.FUTURES,
    )


def load_5m(config: dict, datadir: Path, pair: str) -> tuple[pd.DataFrame, str]:
    d5 = load_tf(config, datadir, pair, "5m")
    if not d5.empty:
        x = d5[["date", "open", "high", "low", "close", "volume"]].copy()
        x["date"] = as_ns(x["date"])
        return x.sort_values("date").drop_duplicates("date").reset_index(drop=True), "5m"
    d1 = load_tf(config, datadir, pair, "1m")
    if d1.empty:
        return pd.DataFrame(), "none"
    x = d1[["date", "open", "high", "low", "close", "volume"]].copy()
    x["date"] = as_ns(x["date"])
    y = (
        x.set_index("date").sort_index()
        .resample("5min", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna().reset_index()
    )
    return y, "1m->5m"


def pair_activity(config: dict, datadir: Path, pair: str) -> tuple[pd.DataFrame, str]:
    d15 = load_tf(config, datadir, pair, "15m")
    d5, source = load_5m(config, datadir, pair)
    if d15.empty or d5.empty:
        return pd.DataFrame(), source

    a = d15[["date", "open", "high", "low", "close", "volume"]].copy()
    a["date"] = as_ns(a["date"])
    a = a.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    quote15 = a["volume"] * a["close"]
    a["ret_24h"] = a["close"].pct_change(96)
    a["quote_vol_24h"] = quote15.rolling(96, min_periods=48).sum()
    a["ctx_time"] = a["date"] + pd.Timedelta(minutes=15)

    x = d5.copy().sort_values("date").reset_index(drop=True)
    x["signal_time"] = as_ns(x["date"] + pd.Timedelta(minutes=5))
    q5 = x["volume"] * x["close"]
    prev = x["close"].shift()
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    x["natr_local"] = tr.rolling(12, min_periods=6).mean() / x["close"] * 100.0
    x["ret_1h"] = x["close"].pct_change(12)
    x["ret_6h"] = x["close"].pct_change(72)
    x["move_5m_pct"] = (x["close"] / x["open"] - 1.0) * 100.0
    # Current 5m quote volume vs PREVIOUS 2h average, so the current spike is not in its own baseline.
    base2h = q5.shift(1).rolling(24, min_periods=12).mean()
    x["volume_spike"] = q5 / base2h.replace(0, np.nan)

    ctx = a[["ctx_time", "ret_24h", "quote_vol_24h"]].sort_values("ctx_time")
    x = pd.merge_asof(
        x.sort_values("signal_time"), ctx,
        left_on="signal_time", right_on="ctx_time",
        direction="backward", tolerance=pd.Timedelta("30min"),
    )
    x["pair"] = pair
    x["detail_source"] = source
    return x[[
        "signal_time", "pair", "detail_source", "ret_24h", "quote_vol_24h",
        "natr_local", "ret_1h", "ret_6h", "move_5m_pct", "volume_spike"
    ]], source


def main() -> int:
    a = parse_args()
    config = json.loads(Path(a.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(a.start, tz="UTC") - pd.Timedelta(days=3)
    end = pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=2)

    frames = []
    sources = {}
    for i, pair in enumerate(pairs, 1):
        x, source = pair_activity(config, Path(a.datadir), pair)
        sources[pair] = source
        if x.empty:
            print(f"ACTIVITY|{i}|{len(pairs)}|{pair}|NO_DATA", flush=True)
            continue
        x = x[(x.signal_time >= start) & (x.signal_time < end)].copy()
        frames.append(x)
        print(f"ACTIVITY|{i}|{len(pairs)}|{pair}|rows={len(x)}", flush=True)
    if not frames:
        raise RuntimeError("No activity data")

    panel = pd.concat(frames, ignore_index=True)
    panel["liquid70"] = panel["quote_vol_24h"] >= VOL_FLOOR_24H

    # Digash uses separate lists. Preserve them separately rather than blending into one arbitrary score.
    grp = panel.groupby("signal_time", sort=False)
    panel["growth_rank"] = grp["ret_24h"].rank(ascending=False, method="first")
    panel["decline_rank"] = grp["ret_24h"].rank(ascending=True, method="first")
    panel["volatility_rank"] = grp["natr_local"].rank(ascending=False, method="first")
    panel["spike_rank"] = grp["volume_spike"].rank(ascending=False, method="first")

    panel["top_growth"] = panel["liquid70"] & (panel["growth_rank"] <= TOP_N)
    panel["top_decline"] = panel["liquid70"] & (panel["decline_rank"] <= TOP_N)
    panel["top_volatility"] = panel["liquid70"] & (panel["volatility_rank"] <= TOP_N)
    panel["spike_alert"] = (
        (panel["volume_spike"] >= SPIKE_MULT)
        & (panel["move_5m_pct"].abs() >= SPIKE_MOVE_PCT)
    )
    votes = panel[["top_growth", "top_decline", "top_volatility"]].astype(int).sum(axis=1)
    panel["active_votes"] = votes
    panel["active_any"] = (votes >= 1) | panel["spike_alert"]
    panel["active_strict"] = (votes >= 2) | panel["spike_alert"]

    keep = [
        "signal_time", "ret_24h", "quote_vol_24h", "natr_local", "ret_1h", "ret_6h",
        "move_5m_pct", "volume_spike", "liquid70", "growth_rank", "decline_rank",
        "volatility_rank", "spike_rank", "top_growth", "top_decline", "top_volatility",
        "spike_alert", "active_votes", "active_any", "active_strict"
    ]
    manifest = []
    for pair, g in panel.groupby("pair", sort=False):
        safe = pair.replace("/", "_").replace(":", "_")
        path = outdir / f"{safe}.pkl"
        gg = g[keep].sort_values("signal_time").copy()
        gg["signal_time"] = as_ns(gg["signal_time"])
        gg.to_pickle(path)
        manifest.append({"pair": pair, "file": str(path), "rows": len(gg), "detail_source": sources.get(pair)})
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    print(f"ACTIVITY_DONE|pairs={len(manifest)}|rows={len(panel)}|out={outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
