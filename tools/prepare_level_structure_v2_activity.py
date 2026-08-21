#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType

def parse_args():
    p = argparse.ArgumentParser(description="Build causal cross-sectional activity ranks for Level/Structure V2")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    return p.parse_args()

def load_pair(config: dict, datadir: Path, pair: str) -> pd.DataFrame:
    d = load_pair_history(
        pair=pair, timeframe="15m", datadir=datadir,
        fill_up_missing=False, drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"),
        candle_type=CandleType.FUTURES,
    )
    if d.empty:
        return d
    x = d[["date","open","high","low","close","volume"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True).astype("datetime64[ns, UTC]")
    x = x.sort_values("date").drop_duplicates("date")
    prev = x["close"].shift()
    tr = pd.concat([
        x["high"] - x["low"],
        (x["high"] - prev).abs(),
        (x["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=14).mean()
    quote = x["volume"] * x["close"]
    q24 = quote.rolling(96, min_periods=48).sum()
    q1h = quote.rolling(4, min_periods=2).sum()
    q1h_med = q1h.rolling(96*30, min_periods=96*7).median()
    x["ret_4h"] = x["close"].pct_change(16)
    x["ret_24h"] = x["close"].pct_change(96)
    x["quote_vol_24h"] = q24
    x["volume_anom"] = q1h / q1h_med.replace(0, np.nan)
    x["atr_pct"] = atr / x["close"]
    x["signal_time"] = (x["date"] + pd.Timedelta(minutes=15)).astype("datetime64[ns, UTC]")
    return x[["signal_time","ret_4h","ret_24h","quote_vol_24h","volume_anom","atr_pct"]]

def main() -> int:
    a = parse_args()
    config = json.loads(Path(a.config).read_text())
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    start = pd.Timestamp(a.start, tz="UTC") - pd.Timedelta(days=35)
    end = pd.Timestamp(a.end, tz="UTC") + pd.Timedelta(days=2)

    frames = []
    for i, pair in enumerate(pairs, 1):
        x = load_pair(config, Path(a.datadir), pair)
        if x.empty:
            print(f"ACTIVITY|{i}|{len(pairs)}|{pair}|NO_DATA", flush=True)
            continue
        x = x[(x["signal_time"] >= start) & (x["signal_time"] < end)].copy()
        x["pair"] = pair
        frames.append(x)
        print(f"ACTIVITY|{i}|{len(pairs)}|{pair}|rows={len(x)}", flush=True)

    if not frames:
        raise RuntimeError("No activity data")
    panel = pd.concat(frames, ignore_index=True)
    panel["signal_time"] = pd.to_datetime(panel["signal_time"], utc=True).astype("datetime64[ns, UTC]")

    # Larger value = more active. Every feature is known at the just-closed 15m candle.
    panel["abs_ret_4h"] = panel["ret_4h"].abs()
    panel["abs_ret_24h"] = panel["ret_24h"].abs()
    components = ["abs_ret_4h", "abs_ret_24h", "quote_vol_24h", "volume_anom", "atr_pct"]
    for c in components:
        panel[f"{c}_pct"] = panel.groupby("signal_time")[c].rank(pct=True, method="average")
    panel["activity_score"] = panel[[f"{c}_pct" for c in components]].mean(axis=1, skipna=True)
    panel["activity_rank"] = panel.groupby("signal_time")["activity_score"].rank(
        ascending=False, method="first"
    )
    panel["volume_rank"] = panel.groupby("signal_time")["quote_vol_24h"].rank(
        ascending=False, method="first"
    )
    panel["active_top5"] = panel["activity_rank"] <= 5
    panel["active_top10"] = panel["activity_rank"] <= 10

    keep = [
        "signal_time","activity_score","activity_rank","volume_rank",
        "active_top5","active_top10","ret_4h","ret_24h","quote_vol_24h","volume_anom","atr_pct"
    ]
    manifest = []
    for pair, g in panel.groupby("pair", sort=False):
        safe = pair.replace("/", "_").replace(":", "_")
        path = outdir / f"{safe}.pkl"
        out = g[keep].sort_values("signal_time").copy()
        out["signal_time"] = pd.to_datetime(out["signal_time"], utc=True).astype("datetime64[ns, UTC]")
        out.to_pickle(path)
        manifest.append({"pair": pair, "file": str(path), "rows": len(g)})
    pd.DataFrame(manifest).to_csv(outdir / "manifest.csv", index=False)
    print(f"ACTIVITY_DONE|pairs={len(manifest)}|rows={len(panel)}|out={outdir}", flush=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
