"""Compatibility wrapper for the optimized free Binance backfill.

Keeps the optimized downloader/chunking from backfill_free.py while replacing
its metrics parser with a column-safe implementation (pandas itertuples can
rename underscore-prefixed fields such as _bucket/_symbol).
"""
from __future__ import annotations

import pandas as pd
import backfill_free as b


def parse_metrics_fixed(job: b.Job, data: bytes) -> list[tuple]:
    df = b.csv_from_zip(data)
    if df.empty:
        return []

    t = b.col(df, "create_time", "timestamp", "time")
    s = b.col(df, "symbol")
    oi = b.col(df, "sum_open_interest", "open_interest", "openInterest")
    taker = b.col(df, "sum_taker_long_short_vol_ratio", "taker_long_short_ratio", "buySellRatio")
    top = b.col(df, "sum_toptrader_long_short_ratio", "top_long_short_ratio", "longShortRatio")
    if t is None:
        return []

    df["_date"] = b.to_utc_series(df[t])
    df = df.dropna(subset=["_date"])
    if df.empty:
        return []

    df["_bucket"] = df["_date"].dt.floor("15min")
    df["_symbol"] = df[s].astype(str).str.upper() if s else job.symbol

    agg: dict[str, str] = {}
    if oi:
        agg[oi] = "last"
    if taker:
        agg[taker] = "last"
    if top:
        agg[top] = "last"
    if not agg:
        return []

    grouped = df.groupby(["_symbol", "_bucket"], as_index=False).agg(agg)
    rows: list[tuple] = []
    for _, r in grouped.iterrows():
        bucket = int(pd.Timestamp(r["_bucket"]).timestamp() * 1000)
        rows.append((
            bucket,
            str(r["_symbol"]),
            float(r[oi]) if oi and pd.notna(r[oi]) else None,
            float(r[taker]) if taker and pd.notna(r[taker]) else None,
            float(r[top]) if top and pd.notna(r[top]) else None,
            bucket,
        ))
    return rows


b.parse_metrics = parse_metrics_fixed

if __name__ == "__main__":
    b.main()
