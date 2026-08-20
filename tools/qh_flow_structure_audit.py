#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import qh_orderflow_v0 as qh

SYMBOLS = qh.SYMBOLS
HORIZON_MS = qh.HORIZON_MS
RT_COST_BPS = qh.ROUND_TRIP_COST_BPS


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Use existing QH aggTrade data to test synchronized, persistent, and residual order flow")
    p.add_argument("--db", default="/freqtrade/user_data/qh_orderflow_v0/qh.sqlite")
    p.add_argument("--start", default="2025-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--output-dir", default="/freqtrade/user_data/qh_flow_structure")
    return p.parse_args()


def hac_slope(df: pd.DataFrame, ycol: str, xcol: str, lag: int = 32) -> tuple[float, float, int]:
    q = df[[ycol, xcol]].dropna()
    n = len(q)
    if n < 100:
        return math.nan, math.nan, n
    x = q[xcol].to_numpy(float)
    y = q[ycol].to_numpy(float)
    X = np.column_stack([np.ones(n), x])
    inv = np.linalg.pinv(X.T @ X)
    b = inv @ (X.T @ y)
    e = y - X @ b
    z = X * e[:, None]
    meat = z.T @ z
    L = min(lag, n - 1)
    for ell in range(1, L + 1):
        w = 1.0 - ell / (L + 1.0)
        g = z[ell:].T @ z[:-ell]
        meat += w * (g + g.T)
    cov = inv @ meat @ inv
    se = math.sqrt(max(float(cov[1, 1]), 0.0))
    slope = float(b[1])
    return slope, slope / se if se > 0 else math.nan, n


def edge_stats(df: pd.DataFrame, signal: str, target: str) -> tuple[float, float, float]:
    q = df[[signal, target]].dropna().copy()
    if q.empty:
        return math.nan, math.nan, math.nan
    sig = q[signal].to_numpy(float)
    ret = q[target].to_numpy(float)
    denom = float(np.mean(np.abs(sig)))
    linear = float(np.mean(sig * ret) / denom) if denom > 0 else math.nan
    signed = np.sign(sig) * ret
    tmp = pd.DataFrame({"symbol": "SYNTH", "boundary_ms": np.arange(len(signed)), "v": signed})
    mean_signed, t_signed, _ = qh.hac_mean_t(tmp, "v", lag=32)
    return linear, mean_signed, t_signed


def load_panel(con: sqlite3.Connection, start: date, end: date) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lo = int(datetime.combine(start, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
    hi_core = int(datetime.combine(end + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000) - 1
    hi = hi_core + HORIZON_MS
    ph = ",".join("?" for _ in SYMBOLS)
    rows = con.execute(
        f"""
        SELECT symbol,boundary_ms,oi,next10_last_price
        FROM qh_events
        WHERE symbol IN ({ph}) AND boundary_ms BETWEEN ? AND ?
        ORDER BY boundary_ms,symbol
        """,
        (*SYMBOLS, lo, hi),
    ).fetchall()
    if not rows:
        raise RuntimeError("No qh_events in DB. Run qh_orderflow_v0 first.")
    df = pd.DataFrame(rows, columns=["symbol", "boundary_ms", "oi", "p_start"])
    exits = df[["symbol", "boundary_ms", "p_start"]].rename(columns={"boundary_ms": "exit_boundary_ms", "p_start": "p_exit"})
    core = df[(df.boundary_ms >= lo) & (df.boundary_ms <= hi_core)].copy()
    core["exit_boundary_ms"] = core.boundary_ms + HORIZON_MS
    core = core.merge(exits, on=["symbol", "exit_boundary_ms"], how="left")
    good = core.p_start.notna() & core.p_exit.notna() & (core.p_start > 0) & (core.p_exit > 0)
    core = core[good].copy()
    core["fwd_bps"] = 10_000.0 * np.log(core.p_exit / core.p_start)
    core["timestamp"] = pd.to_datetime(core.boundary_ms, unit="ms", utc=True)
    core["year"] = core.timestamp.dt.year

    oi_w = core.pivot(index="boundary_ms", columns="symbol", values="oi").reindex(columns=SYMBOLS)
    ret_w = core.pivot(index="boundary_ms", columns="symbol", values="fwd_bps").reindex(columns=SYMBOLS)
    sync = oi_w.notna().all(axis=1) & ret_w.notna().all(axis=1)
    oi_w = oi_w[sync].sort_index()
    ret_w = ret_w[sync].sort_index()
    return core, oi_w, ret_w


def build_common(oi_w: pd.DataFrame, ret_w: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=oi_w.index)
    x["common_flow"] = oi_w.mean(axis=1)
    x["breadth"] = np.sign(oi_w).mean(axis=1)
    x["persistent_common_1h"] = x.common_flow.rolling(4, min_periods=4).mean()
    x["basket_fwd_bps"] = ret_w.mean(axis=1)
    x["btc_fwd_bps"] = ret_w["BTCUSDT"]
    x["eth_fwd_bps"] = ret_w["ETHUSDT"]
    x["timestamp"] = pd.to_datetime(x.index, unit="ms", utc=True)
    x["year"] = x.timestamp.dt.year
    return x.reset_index()


def build_relative(oi_w: pd.DataFrame, ret_w: pd.DataFrame) -> pd.DataFrame:
    n = len(SYMBOLS)
    oi_sum = oi_w.sum(axis=1)
    ret_sum = ret_w.sum(axis=1)
    rows = []
    for s in SYMBOLS:
        other_oi = (oi_sum - oi_w[s]) / (n - 1)
        other_ret = (ret_sum - ret_w[s]) / (n - 1)
        z = pd.DataFrame({
            "symbol": s,
            "boundary_ms": oi_w.index,
            "rel_flow": oi_w[s].to_numpy(float) - other_oi.to_numpy(float),
            "rel_fwd_bps": ret_w[s].to_numpy(float) - other_ret.to_numpy(float),
        })
        z["timestamp"] = pd.to_datetime(z.boundary_ms, unit="ms", utc=True)
        z["year"] = z.timestamp.dt.year
        rows.append(z)
    return pd.concat(rows, ignore_index=True)


def common_row(name: str, df: pd.DataFrame, signal: str, target: str) -> dict:
    slope, t, n = hac_slope(df, target, signal, lag=32)
    linear, sign_edge, sign_t = edge_stats(df, signal, target)
    return {
        "signal": name,
        "target": target,
        "n": n,
        "slope_bps_per_signal": slope,
        "hac_t": t,
        "linear_edge_bps": linear,
        "sign_edge_bps": sign_edge,
        "sign_t": sign_t,
    }


def main() -> int:
    args = parse_args()
    start = datetime.strptime(args.start, "%Y-%m-%d").date()
    end = datetime.strptime(args.end, "%Y-%m-%d").date()
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    if not Path(args.db).exists():
        raise RuntimeError(f"Missing existing QH DB: {args.db}")

    print("=== QH FLOW STRUCTURE AUDIT ===")
    print(f"Evaluation: {start} -> {end}")
    print("Reuses existing qh.sqlite; NO downloads and NO parameter search.")
    print("Fixed hypotheses:")
    print("  1) common_flow = mean simultaneous OI across BTC/ETH/XRP/SOL/DOGE/ADA")
    print("  2) breadth = mean sign(OI) across the six markets")
    print("  3) persistent_common_1h = causal mean of current + previous 3 quarter-hours")
    print("  4) residual flow = asset OI minus mean OI of the other five; target is relative 8h return")
    print(f"Economic hurdle: {RT_COST_BPS:.1f} bps taker round-trip.\n")

    con = sqlite3.connect(args.db, timeout=120)
    core, oi_w, ret_w = load_panel(con, start, end)
    common = build_common(oi_w, ret_w)
    relative = build_relative(oi_w, ret_w)
    print(f"Raw valid asset-events: {len(core):,}")
    print(f"Fully synchronized six-asset boundaries: {len(common):,}")

    summary_rows = [
        common_row("COMMON_FLOW_TO_BASKET", common, "common_flow", "basket_fwd_bps"),
        common_row("BREADTH_TO_BASKET", common, "breadth", "basket_fwd_bps"),
        common_row("PERSISTENT_COMMON_1H_TO_BASKET", common, "persistent_common_1h", "basket_fwd_bps"),
        common_row("COMMON_FLOW_TO_BTC", common, "common_flow", "btc_fwd_bps"),
        common_row("COMMON_FLOW_TO_ETH", common, "common_flow", "eth_fwd_bps"),
    ]

    rel_slope, rel_t, rel_n = qh.panel_hac_beta_t(relative, "rel_fwd_bps", "rel_flow", lag=32)
    rel_linear = float((relative.rel_flow * relative.rel_fwd_bps).mean() / relative.rel_flow.abs().mean())
    relative["signed_rel_bps"] = np.sign(relative.rel_flow) * relative.rel_fwd_bps
    rel_sign, rel_sign_t, _ = qh.hac_mean_t(relative.rename(columns={"signed_rel_bps": "v"}), "v", lag=32)
    summary_rows.append({
        "signal": "RESIDUAL_FLOW_TO_RELATIVE_RETURN",
        "target": "rel_fwd_bps",
        "n": rel_n,
        "slope_bps_per_signal": rel_slope,
        "hac_t": rel_t,
        "linear_edge_bps": rel_linear,
        "sign_edge_bps": rel_sign,
        "sign_t": rel_sign_t,
    })
    summary = pd.DataFrame(summary_rows)

    year_rows = []
    for year, g in common.groupby("year"):
        for name, sig in [
            ("COMMON_FLOW_TO_BASKET", "common_flow"),
            ("BREADTH_TO_BASKET", "breadth"),
            ("PERSISTENT_COMMON_1H_TO_BASKET", "persistent_common_1h"),
        ]:
            s, t, n = hac_slope(g, "basket_fwd_bps", sig, lag=32)
            lin, se, st = edge_stats(g, sig, "basket_fwd_bps")
            year_rows.append({"year": int(year), "signal": name, "n": n, "slope": s, "hac_t": t, "linear_edge_bps": lin, "sign_edge_bps": se, "sign_t": st})
    for year, g in relative.groupby("year"):
        s, t, n = qh.panel_hac_beta_t(g, "rel_fwd_bps", "rel_flow", lag=32)
        lin = float((g.rel_flow * g.rel_fwd_bps).mean() / g.rel_flow.abs().mean())
        gg = g.copy(); gg["v"] = np.sign(gg.rel_flow) * gg.rel_fwd_bps
        se, st, _ = qh.hac_mean_t(gg, "v", lag=32)
        year_rows.append({"year": int(year), "signal": "RESIDUAL_FLOW_TO_RELATIVE_RETURN", "n": n, "slope": s, "hac_t": t, "linear_edge_bps": lin, "sign_edge_bps": se, "sign_t": st})
    year_df = pd.DataFrame(year_rows)

    asset_rows = []
    for s, g in relative.groupby("symbol"):
        sl, tt = qh.asset_slope(g.rename(columns={"rel_fwd_bps": "fwd_bps", "rel_flow": "oi"}), lag=32)
        lin = float((g.rel_flow * g.rel_fwd_bps).mean() / g.rel_flow.abs().mean())
        gg = g.copy(); gg["v"] = np.sign(gg.rel_flow) * gg.rel_fwd_bps
        se, st, _ = qh.hac_mean_t(gg, "v", lag=32)
        asset_rows.append({"symbol": s, "n": len(g), "slope": sl, "hac_t": tt, "linear_edge_bps": lin, "sign_edge_bps": se, "sign_t": st})
    asset_df = pd.DataFrame(asset_rows)

    # Cross-sectional information coefficient at each synchronized boundary.
    ic_rows = []
    for b in oi_w.index:
        x = oi_w.loc[b]
        y = ret_w.loc[b]
        ic = x.corr(y, method="spearman")
        ic_rows.append((int(b), float(ic) if np.isfinite(ic) else np.nan))
    ic_df = pd.DataFrame(ic_rows, columns=["boundary_ms", "cross_asset_rank_ic"])
    ic_df["timestamp"] = pd.to_datetime(ic_df.boundary_ms, unit="ms", utc=True)
    ic_mean = float(ic_df.cross_asset_rank_ic.mean())
    ic_tmp = ic_df.dropna().copy(); ic_tmp["symbol"] = "X"; ic_tmp["v"] = ic_tmp.cross_asset_rank_ic
    _, ic_t, _ = qh.hac_mean_t(ic_tmp, "v", lag=32)

    print("\n=== QH FLOW STRUCTURE RESULT ===")
    p = summary.copy()
    for c in ["slope_bps_per_signal", "linear_edge_bps", "sign_edge_bps"]:
        p[c] = p[c].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "nan")
    for c in ["hac_t", "sign_t"]:
        p[c] = p[c].map(lambda v: f"{v:+.2f}" if np.isfinite(v) else "nan")
    print(p.to_string(index=False))

    print("\nYEAR BREAKDOWN")
    yp = year_df.copy()
    for c in ["slope", "linear_edge_bps", "sign_edge_bps"]:
        yp[c] = yp[c].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "nan")
    for c in ["hac_t", "sign_t"]:
        yp[c] = yp[c].map(lambda v: f"{v:+.2f}" if np.isfinite(v) else "nan")
    print(yp.to_string(index=False))

    print("\nRESIDUAL ASSET BREAKDOWN")
    ap = asset_df.copy()
    for c in ["slope", "linear_edge_bps", "sign_edge_bps"]:
        ap[c] = ap[c].map(lambda v: f"{v:+.3f}" if np.isfinite(v) else "nan")
    for c in ["hac_t", "sign_t"]:
        ap[c] = ap[c].map(lambda v: f"{v:+.2f}" if np.isfinite(v) else "nan")
    print(ap.to_string(index=False))

    print("\nCROSS-ASSET RANK IC")
    print(f"Mean simultaneous OI -> 8h relative-return rank IC: {ic_mean:+.4f} | HAC t={ic_t:+.2f}")

    primary = summary.set_index("signal")
    p1 = primary.loc["PERSISTENT_COMMON_1H_TO_BASKET"]
    p2 = primary.loc["RESIDUAL_FLOW_TO_RELATIVE_RETURN"]
    y_persist = year_df[year_df.signal == "PERSISTENT_COMMON_1H_TO_BASKET"].set_index("year")
    y_rel = year_df[year_df.signal == "RESIDUAL_FLOW_TO_RELATIVE_RETURN"].set_index("year")
    pos_assets = int((asset_df.slope > 0).sum())

    gates = [
        ("Persistent common slope > 0", p1.slope_bps_per_signal > 0),
        ("Persistent common HAC t > 2", p1.hac_t > 2),
        ("Persistent common 2025/2026 slopes > 0", all(y in y_persist.index and y_persist.loc[y, "slope"] > 0 for y in [2025, 2026])),
        (f"Persistent common linear edge > {RT_COST_BPS:.0f} bps", p1.linear_edge_bps > RT_COST_BPS),
        ("Residual relative-flow slope > 0", p2.slope_bps_per_signal > 0),
        ("Residual relative-flow HAC t > 2", p2.hac_t > 2),
        ("Residual 2025/2026 slopes > 0", all(y in y_rel.index and y_rel.loc[y, "slope"] > 0 for y in [2025, 2026])),
        ("Residual slope positive in at least 4/6 assets", pos_assets >= 4),
        (f"Residual linear edge > {RT_COST_BPS:.0f} bps", p2.linear_edge_bps > RT_COST_BPS),
    ]
    print("\nPRE-REGISTERED STRUCTURE GATES")
    for label, ok in gates:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")

    persist_route = all(ok for _, ok in gates[:4])
    residual_route = all(ok for _, ok in gates[4:])
    print("\nSTRUCTURE VERDICT")
    if persist_route or residual_route:
        kept = []
        if persist_route: kept.append("persistent synchronized market flow")
        if residual_route: kept.append("residual cross-asset flow")
        print("[KEEP] " + " + ".join(kept) + " clears its full pre-registered statistical/year/cost gates. Build one execution backtest next.")
    else:
        print("[CLOSE QH AGGTRADES FAMILY] Neither persistent common flow nor residual relative flow clears the full statistical/year/cost hurdle. Do not threshold-fit this dataset.")

    summary.to_csv(outdir / "summary.csv", index=False)
    year_df.to_csv(outdir / "year_breakdown.csv", index=False)
    asset_df.to_csv(outdir / "residual_asset_breakdown.csv", index=False)
    common.to_csv(outdir / "common_flow_panel.csv", index=False)
    relative.to_csv(outdir / "relative_flow_panel.csv", index=False)
    ic_df.to_csv(outdir / "cross_asset_rank_ic.csv", index=False)
    pd.DataFrame([{"gate": label, "pass": bool(ok)} for label, ok in gates]).to_csv(outdir / "gates.csv", index=False)
    print(f"\nSaved under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
