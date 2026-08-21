#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

import level_edge_highroi_v1 as m


def parse_args():
    p = argparse.ArgumentParser(description="Prepare cached causal events and stage2 shortlist for Level Edge High-ROI V1")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/level_edge_highroi_v1")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rescan", action="store_true")
    return p.parse_args()


def log(s: str) -> None:
    print(s, flush=True)


def load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    for c in [x for x in df.columns if x.startswith("exit_")]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def main() -> int:
    a = parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(Path(a.config).read_text())
    pairs = list(cfg.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        pairs = [
            "AAVE/USDT:USDT", "ADA/USDT:USDT", "ATOM/USDT:USDT", "AVAX/USDT:USDT",
            "BCH/USDT:USDT", "BNB/USDT:USDT", "BTC/USDT:USDT", "DOGE/USDT:USDT",
            "DOT/USDT:USDT", "ETC/USDT:USDT", "ETH/USDT:USDT", "FIL/USDT:USDT",
            "LINK/USDT:USDT", "LTC/USDT:USDT", "NEAR/USDT:USDT", "SOL/USDT:USDT",
            "TRX/USDT:USDT", "UNI/USDT:USDT", "XLM/USDT:USDT", "XRP/USDT:USDT",
        ]

    log("=== LEVEL EDGE HIGH-ROI V1 — PREPARE ===")
    events_path = out / "causal_events.csv"
    if events_path.exists() and not a.rescan:
        log(f"reusing cached causal events: {events_path}")
        df = load_events(events_path)
    else:
        df = m.scan_events(a, out, pairs)

    train = m._split(df, "TRAIN")
    valid = m._split(df, "VALID")
    test = m._split(df, "HIST_TEST")
    log(f"events train={len(train):,} valid={len(valid):,} hist_test={len(test):,}")

    s1 = m.stage1(train)
    s1.to_csv(out / "stage1_train.csv", index=False)
    log(f"stage1 train-positive configs={len(s1):,}")
    if s1.empty:
        raise RuntimeError("No positive TRAIN configurations")

    s2 = m.stage2(train, s1, top_n=40)
    s2.to_csv(out / "stage2_train.csv", index=False)
    log(f"stage2 structural configs={len(s2):,}")
    if s2.empty:
        raise RuntimeError("No positive stage2 TRAIN configurations")

    log("PREPARE COMPLETE — cached scan/stage2 ready for parallel resume")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
