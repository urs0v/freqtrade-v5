#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


START_BALANCE = 100.0


def parse_args():
    p = argparse.ArgumentParser(description="Compact report for FrozenFakeoutV1 Freqtrade dry-run")
    p.add_argument("--db", default="/freqtrade/user_data/trades-frozen-fakeout.sqlite")
    p.add_argument("--feed", default="/freqtrade/user_data/frozen_fakeout_feed")
    return p.parse_args()


def _feed_info(path: Path):
    state = {}
    snap = {}
    try:
        state = json.loads((path / "state.json").read_text())
    except Exception:
        pass
    try:
        snap = json.loads((path / "snapshot.json").read_text())
    except Exception:
        pass
    return state, snap


def _colset(conn) -> set[str]:
    return {str(r[1]) for r in conn.execute("PRAGMA table_info(trades)").fetchall()}


def main():
    a = parse_args()
    feed_state, feed_snap = _feed_info(Path(a.feed))
    print("=== FROZEN FAKEOUT V1 — FREQTRADE DRY-RUN ===")
    if feed_state:
        print(f"cutoff={feed_state.get('cutoff')} | source={feed_state.get('cutoff_origin')}")
    if feed_snap:
        print(
            f"feed published={feed_snap.get('published_at')} bucket={feed_snap.get('entry_bucket')} "
            f"signals_since_cutoff={feed_snap.get('signals_since_cutoff', 0)} "
            f"active_now={feed_snap.get('active_for_freqtrade', 0)}"
        )

    db = Path(a.db)
    if not db.exists():
        print("database: not created yet")
        return 0

    conn = sqlite3.connect(str(db))
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "trades" not in tables:
            print("database: trades table not created yet")
            return 0
        cols = _colset(conn)
        required = {"id", "pair", "is_open"}
        if not required.issubset(cols):
            print(f"database: unexpected trades schema, columns={sorted(cols)}")
            return 1

        use = [c for c in [
            "id", "pair", "is_open", "is_short", "open_date", "close_date",
            "open_rate", "close_rate", "stake_amount", "leverage", "close_profit",
            "close_profit_abs", "exit_reason", "enter_tag"
        ] if c in cols]
        q = "SELECT " + ",".join(use) + " FROM trades ORDER BY id"
        df = pd.read_sql_query(q, conn)
    finally:
        conn.close()

    if df.empty:
        print("trades=0 | waiting for first executable signal")
        return 0

    closed = df[df["is_open"].astype(int).eq(0)].copy()
    opened = df[df["is_open"].astype(int).ne(0)].copy()
    print(f"trades={len(df)} closed={len(closed)} open={len(opened)}")

    if len(closed) and "close_profit_abs" in closed:
        p = pd.to_numeric(closed["close_profit_abs"], errors="coerce").dropna().astype(float)
        if len(p):
            pos = float(p[p > 0].sum())
            neg = float(-p[p < 0].sum())
            pf = pos / neg if neg > 0 else np.inf
            wr = float((p > 0).mean() * 100.0)
            total = float(p.sum())
            balance = START_BALANCE + total
            curve = START_BALANCE + p.cumsum()
            peak = curve.cummax()
            dd = ((peak - curve) / peak.replace(0, np.nan)).max() * 100.0
            print(
                f"closed PF={pf:.2f} WR={wr:.1f}% net=${total:+.2f} "
                f"balance≈${balance:.2f} ROI≈{(balance/START_BALANCE-1)*100:+.1f}% DD≈{float(dd):.1f}%"
            )

    if len(closed) and "close_profit" in closed:
        cp = pd.to_numeric(closed["close_profit"], errors="coerce").dropna().astype(float)
        if len(cp):
            print(f"avg Freqtrade trade return={cp.mean()*100:+.2f}% median={cp.median()*100:+.2f}%")

    show_cols = [c for c in ["id", "pair", "is_short", "open_date", "close_date", "close_profit_abs", "exit_reason"] if c in df]
    print("\n=== LATEST TRADES ===")
    print(df[show_cols].tail(10).to_string(index=False))
    print(f"\ncheckpoint: closed={len(closed)}/50 preliminary | {len(closed)}/100 primary")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
