#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Report for live BTCUSDT shadow market maker")
    p.add_argument("--db", default="/freqtrade/user_data/mm_shadow_btc/mm_shadow.sqlite")
    return p.parse_args()


def meta(con: sqlite3.Connection) -> dict:
    out = {}
    for k, v in con.execute("SELECT key,value FROM meta"):
        try:
            out[k] = json.loads(v)
        except Exception:
            out[k] = v
    return out


def fmt(v: float, nd: int = 3) -> str:
    return "nan" if not np.isfinite(v) else f"{v:.{nd}f}"


def main() -> int:
    cfg = parse_args()
    if not Path(cfg.db).exists():
        raise RuntimeError(f"Missing DB: {cfg.db}")
    con = sqlite3.connect(cfg.db)
    m = meta(con)
    snaps = pd.read_sql_query("SELECT * FROM snapshots ORDER BY ts", con)
    fills = pd.read_sql_query("SELECT * FROM fills ORDER BY ts", con)
    marks = pd.read_sql_query("SELECT * FROM markouts ORDER BY fill_id,horizon_ms", con)
    if snaps.empty:
        print("No snapshots yet.")
        return 0

    hours = (snaps.ts.iloc[-1] - snaps.ts.iloc[0]) / 3_600_000.0
    last = snaps.iloc[-1]
    capital = float(m.get("virtual_capital", 100.0))
    net = float(last.net_equity)
    gross = float(last.gross_equity)
    fees = float(last.fees)
    funding = float(last.funding_pnl)
    max_inv = float(snaps.inventory_notional.abs().max())
    gate_share = float(snaps.vol_gate.mean()) if "vol_gate" in snaps else math.nan

    buys = int((fills.side == "BUY").sum()) if not fills.empty else 0
    sells = int((fills.side == "SELL").sum()) if not fills.empty else 0
    fill_notional = float(fills.notional.sum()) if not fills.empty else 0.0
    capture = float(fills.capture_bps.mean()) if not fills.empty else math.nan
    net_capture = float(fills.net_capture_bps.mean()) if not fills.empty else math.nan
    qmed = float(fills.queue_initial.median()) if not fills.empty else math.nan
    age = float(fills.quote_age_ms.median()) if not fills.empty else math.nan

    print("=== BTCUSDT LIVE SHADOW MARKET-MAKING REPORT ===")
    print(f"Runtime: {hours:.2f}h | snapshots={len(snaps):,}")
    print(f"Fills: {len(fills):,} | buys={buys} sells={sells} | filled notional=${fill_notional:.2f}")
    print(f"Virtual capital=${capital:.2f} | gross PnL=${gross:+.4f} | fees=${fees:.4f} | funding=${funding:+.4f} | net=${net:+.4f} ({100*net/capital:+.3f}%)")
    print(f"Max |inventory|=${max_inv:.2f} | volatility-gate active share={100*gate_share:.1f}%")
    print(f"Immediate capture: gross={fmt(capture)} bps/fill | after maker fee={fmt(net_capture)} bps/fill")
    print(f"Median queue ahead={fmt(qmed,4)} BTC | median quote age at fill={fmt(age,0)} ms")

    print("\nMARKOUTS")
    rows = []
    if not marks.empty:
        for h, g in marks.groupby("horizon_ms"):
            rows.append({
                "horizon": f"{h/1000:g}s",
                "n": len(g),
                "gross_mean_bps": g.markout_bps.mean(),
                "gross_median_bps": g.markout_bps.median(),
                "net_mean_bps": g.net_markout_bps.mean(),
                "net_positive": (g.net_markout_bps > 0).mean(),
                "p10_net_bps": g.net_markout_bps.quantile(0.10),
            })
        r = pd.DataFrame(rows)
        for c in ["gross_mean_bps", "gross_median_bps", "net_mean_bps", "p10_net_bps"]:
            r[c] = r[c].map(lambda x: f"{x:+.3f}")
        r["net_positive"] = r["net_positive"].map(lambda x: f"{100*x:.1f}%")
        print(r.to_string(index=False))
    else:
        print("No matured markouts yet.")

    m30 = marks[marks.horizon_ms == 30000] if not marks.empty else pd.DataFrame()
    mean30 = float(m30.net_markout_bps.mean()) if len(m30) else math.nan
    gates = [
        ("Runtime >= 2h", hours >= 2.0),
        ("At least 20 conservative simulated fills", len(fills) >= 20),
        ("At least 5 fills on each side", buys >= 5 and sells >= 5),
        ("30s post-fill edge after maker fee > 0", np.isfinite(mean30) and mean30 > 0),
        ("Net marked-to-market PnL > 0", net > 0),
    ]
    print("\nSHADOW GATES")
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    if all(ok for _, ok in gates):
        print("VERDICT: [PROMISING] Queue-aware live shadow economics clear the first gate. Next step is tiny-live GTX execution with real fills.")
    else:
        print("VERDICT: [KEEP SHADOW / DIAGNOSE] Do not send real orders yet. The live execution economics have not cleared the fixed first gate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
