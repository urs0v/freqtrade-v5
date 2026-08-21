#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from prepare_digash_activity_v3 import pair_activity, as_ns

VOL_FLOOR_24H = 70_000_000.0
TOP_N = 5  # user-supplied/public workflow: watch roughly top 5-10 active names; use the tighter bound
SPIKE_MULT = 5.0
SPIKE_MOVE_PCT = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="Build causal Digash-style activity lists for V3.1")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", required=True)
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    return p.parse_args()


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
    grp = panel.groupby("signal_time", sort=False)

    # Keep the public screener lists separate. Quote-volume is a real list of its own;
    # it is NOT treated as a substitute for unavailable historical trade-count ranking.
    panel["growth_rank"] = grp["ret_24h"].rank(ascending=False, method="first")
    panel["decline_rank"] = grp["ret_24h"].rank(ascending=True, method="first")
    panel["volatility_rank"] = grp["natr_local"].rank(ascending=False, method="first")
    panel["volume_rank"] = grp["quote_vol_24h"].rank(ascending=False, method="first")
    panel["spike_rank"] = grp["volume_spike"].rank(ascending=False, method="first")

    panel["top_growth"] = panel["liquid70"] & (panel["growth_rank"] <= TOP_N)
    panel["top_decline"] = panel["liquid70"] & (panel["decline_rank"] <= TOP_N)
    panel["top_volatility"] = panel["liquid70"] & (panel["volatility_rank"] <= TOP_N)
    panel["top_volume"] = panel["liquid70"] & (panel["volume_rank"] <= TOP_N)
    panel["spike_alert"] = (
        (panel["volume_spike"] >= SPIKE_MULT)
        & (panel["move_5m_pct"].abs() >= SPIKE_MOVE_PCT)
    )

    list_cols = ["top_growth", "top_decline", "top_volatility", "top_volume"]
    votes = panel[list_cols].astype(int).sum(axis=1)
    panel["active_votes"] = votes
    panel["active_any"] = (votes >= 1) | panel["spike_alert"]
    panel["active_strict"] = (votes >= 2) | panel["spike_alert"]

    keep = [
        "signal_time", "ret_24h", "quote_vol_24h", "natr_local", "ret_1h", "ret_6h",
        "move_5m_pct", "volume_spike", "liquid70", "growth_rank", "decline_rank",
        "volatility_rank", "volume_rank", "spike_rank", "top_growth", "top_decline",
        "top_volatility", "top_volume", "spike_alert", "active_votes", "active_any", "active_strict"
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
