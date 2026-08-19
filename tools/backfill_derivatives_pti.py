"""Point-in-time-safe wrapper around backfill_free.py.

This writes to a separate research DB. Binance metrics observed inside a 15m
bucket are timestamped at the bucket CLOSE, not the bucket open. Funding events
are only propagated to rows strictly after the event timestamp.
"""
from __future__ import annotations

import sqlite3
import pandas as pd
import backfill_free as b

BAR = pd.Timedelta("15min")


def parse_metrics_pti(job: b.Job, data: bytes) -> list[tuple]:
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

    # A value sampled anywhere inside [10:00, 10:15) is not considered usable
    # until 10:15. This removes the historical within-candle lookahead.
    df["_available"] = df["_date"].dt.floor("15min") + BAR
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

    grouped = df.groupby(["_symbol", "_available"], as_index=False).agg(agg)
    rows: list[tuple] = []
    for _, r in grouped.iterrows():
        available_ms = int(pd.Timestamp(r["_available"]).timestamp() * 1000)
        rows.append((
            available_ms,
            str(r["_symbol"]),
            float(r[oi]) if oi and pd.notna(r[oi]) else None,
            float(r[taker]) if taker and pd.notna(r[taker]) else None,
            float(r[top]) if top and pd.notna(r[top]) else None,
            available_ms,
        ))
    return rows


def apply_funding_pti(
    con: sqlite3.Connection,
    symbol: str,
    events: list[tuple[int, str, float]],
    start_ms: int,
    end_ms: int,
) -> int:
    if not events:
        return 0
    events = sorted(events, key=lambda x: x[0])
    feature_rows = con.execute(
        "SELECT bucket_ms FROM features WHERE symbol=? AND bucket_ms BETWEEN ? AND ? ORDER BY bucket_ms",
        (symbol, start_ms, end_ms),
    ).fetchall()
    if not feature_rows:
        return 0

    i = 0
    last_rate = None
    updates: list[tuple[float, str, int]] = []
    for (available_ms,) in feature_rows:
        # Strict inequality is intentional: a funding print occurring exactly at
        # the next trade open is not treated as information known before entry.
        while i < len(events) and events[i][0] < available_ms:
            last_rate = events[i][2]
            i += 1
        if last_rate is not None:
            updates.append((last_rate, symbol, available_ms))

    con.executemany(
        "UPDATE features SET funding_rate=? WHERE symbol=? AND bucket_ms=?",
        updates,
    )
    return len(updates)


b.parse_metrics = parse_metrics_pti
b.apply_funding = apply_funding_pti

if __name__ == "__main__":
    b.main()
