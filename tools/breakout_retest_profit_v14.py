#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc

BASE_COST_BPS = 8.0
STRESS_COST_BPS = 12.0
RR = 3.0
HOLD_BARS = 48
BAR_MINUTES = 5
START_BALANCE = 100.0
MAINT_MARGIN_FRAC = 0.005  # conservative small-notional approximation, not an exchange liquidation calculator
MIN_NOTIONAL = 5.0
REFERENCE_RISK_PCT = 1.0
REFERENCE_LEVERAGE = 5.0
REFERENCE_MAX_OPEN = 3
ADVERSE_FUNDING_BPS = 3.0


def parse_args():
    p = argparse.ArgumentParser(description="Profit V1.4: $100 portfolio realism for the frozen V1.3 signal")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v13dir", default="/freqtrade/user_data/breakout_retest_profit_v13")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v14")
    p.add_argument("--skip-funding-download", action="store_true")
    return p.parse_args()


def _symbol(pair: str) -> str:
    return pair.split(":", 1)[0].replace("/", "").replace("-", "")


def _fetch_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp, cache: Path):
    cache.mkdir(parents=True, exist_ok=True)
    a = int(start.timestamp() * 1000)
    b = int(end.timestamp() * 1000)
    path = cache / f"{symbol}_{start:%Y%m%d}_{end:%Y%m%d}.json"
    if path.exists():
        try:
            data = json.loads(path.read_text())
            return data, True, "CACHE"
        except Exception:
            pass
    rows = []
    cur = a
    try:
        while cur <= b:
            qs = urllib.parse.urlencode({"symbol": symbol, "startTime": cur, "endTime": b, "limit": 1000})
            url = "https://fapi.binance.com/fapi/v1/fundingRate?" + qs
            req = urllib.request.Request(url, headers={"User-Agent": "rmv5-profit-v14/1.0"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                chunk = json.loads(resp.read().decode("utf-8"))
            if not isinstance(chunk, list):
                raise RuntimeError(f"unexpected funding response: {type(chunk).__name__}")
            if not chunk:
                break
            rows.extend(chunk)
            last = max(int(x.get("fundingTime", 0)) for x in chunk)
            if last < cur:
                break
            cur = last + 1
            if len(chunk) < 1000:
                break
            time.sleep(0.05)
        path.write_text(json.dumps(rows))
        return rows, True, "API"
    except Exception as e:
        return [], False, f"{type(e).__name__}: {e}"


def _funding_df(rows) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["funding_time", "rate"])
    out = []
    for x in rows:
        try:
            out.append((pd.to_datetime(int(x["fundingTime"]), unit="ms", utc=True), float(x["fundingRate"])))
        except Exception:
            continue
    if not out:
        return pd.DataFrame(columns=["funding_time", "rate"])
    return pd.DataFrame(out, columns=["funding_time", "rate"]).drop_duplicates("funding_time").sort_values("funding_time")


def _scheduled_funding_count(entry: pd.Timestamp, exit_time: pd.Timestamp) -> int:
    # Binance USD-M commonly settles at 00:00/08:00/16:00 UTC. This is used only for an adverse stress,
    # not as a claim about the actual rate or interval for every historical symbol.
    if exit_time <= entry:
        return 0
    day0 = entry.floor("D") - pd.Timedelta(days=1)
    day1 = exit_time.ceil("D") + pd.Timedelta(days=1)
    n = 0
    d = day0
    while d <= day1:
        for h in (0, 8, 16):
            t = d + pd.Timedelta(hours=h)
            if entry < t <= exit_time:
                n += 1
        d += pd.Timedelta(days=1)
    return n


def _reconstruct_pair(pair: str, g: pd.DataFrame, cfg: dict, datadir: Path):
    raw5, source = dc.load_5m(cfg, datadir, pair)
    if raw5.empty:
        return [], {"pair": pair, "status": "NO_5M", "source": source}
    x = dc.prep_ohlcv(raw5, 5)
    x["date"] = pd.to_datetime(x["date"], utc=True)
    dates = x.date.astype("int64").to_numpy()
    o = x.open.to_numpy(float)
    h = x.high.to_numpy(float)
    l = x.low.to_numpy(float)
    c = x.close.to_numpy(float)
    out = []
    missing = 0
    for r in g.itertuples(index=False):
        et = pd.Timestamp(r.entry_time)
        ns = et.value
        j = int(np.searchsorted(dates, ns, side="left"))
        if j >= len(x) or dates[j] != ns:
            missing += 1
            continue
        entry = float(o[j])
        risk_bps = float(r.risk_bps)
        if not np.isfinite(entry) or entry <= 0 or not np.isfinite(risk_bps) or risk_bps <= 0:
            continue
        side = int(r.side)
        risk_abs = entry * risk_bps / 10000.0
        stop = entry - side * risk_abs
        target = entry + side * RR * risk_abs
        end = min(len(x) - 1, j + HOLD_BARS - 1)
        exit_idx = end
        exit_price = float(c[end])
        reason = "TIME"
        for i in range(j, end + 1):
            if side > 0:
                stop_hit = l[i] <= stop
                target_hit = h[i] >= target
            else:
                stop_hit = h[i] >= stop
                target_hit = l[i] <= target
            if stop_hit:
                exit_idx, exit_price, reason = i, stop, "STOP"
                break
            if target_hit:
                exit_idx, exit_price, reason = i, target, "TARGET"
                break
        # Treat the position as occupied until the end of the exit bar. This is conservative by <=5 minutes.
        xt = pd.Timestamp(x.iloc[exit_idx].date) + pd.Timedelta(minutes=BAR_MINUTES)
        raw_bps = side * (exit_price / entry - 1.0) * 10000.0
        reconstructed_net8 = (raw_bps - BASE_COST_BPS) / risk_bps
        stored_net8 = float(r.net8_r) if np.isfinite(r.net8_r) else np.nan
        d = r._asdict()
        d.update({
            "detail_source": source,
            "entry_price": entry,
            "stop_price": stop,
            "target_price": target,
            "exit_time_exact": xt,
            "exit_price_exact": exit_price,
            "exit_reason_exact": reason,
            "duration_min": (xt - et).total_seconds() / 60.0,
            "reconstructed_net8_r": reconstructed_net8,
            "r_match_abs": abs(reconstructed_net8 - stored_net8) if np.isfinite(stored_net8) else np.nan,
            "scheduled_funding_count": _scheduled_funding_count(et, xt),
        })
        out.append(d)
    return out, {"pair": pair, "status": "OK", "source": source, "rows": len(out), "missing_entry": missing}


def _attach_funding(z: pd.DataFrame, outdir: Path, skip_download: bool):
    z = z.copy()
    z["actual_funding_ok"] = False
    z["actual_funding_events"] = 0
    z["actual_funding_rate_sum"] = np.nan
    z["actual_funding_cost_r"] = np.nan
    cache = outdir / "funding_cache"
    status_rows = []
    for pair, idx in z.groupby("pair").groups.items():
        g = z.loc[idx]
        start = g.entry_time.min() - pd.Timedelta(days=1)
        end = g.exit_time_exact.max() + pd.Timedelta(days=1)
        if skip_download:
            rows, ok, src = [], False, "SKIPPED"
        else:
            rows, ok, src = _fetch_funding(_symbol(pair), start, end, cache)
        f = _funding_df(rows)
        status_rows.append({"pair": pair, "symbol": _symbol(pair), "ok": ok, "source": src, "funding_rows": len(f)})
        if not ok:
            continue
        for i in idx:
            e = z.at[i, "entry_time"]
            x = z.at[i, "exit_time_exact"]
            q = f[(f.funding_time > e) & (f.funding_time <= x)]
            rate_sum = float(q.rate.sum()) if len(q) else 0.0
            side = int(z.at[i, "side"])
            risk_bps = float(z.at[i, "risk_bps"])
            # Positive funding: longs pay, shorts receive. Positive value below means a portfolio cost.
            cost_r = side * rate_sum * 10000.0 / risk_bps
            z.at[i, "actual_funding_ok"] = True
            z.at[i, "actual_funding_events"] = int(len(q))
            z.at[i, "actual_funding_rate_sum"] = rate_sum
            z.at[i, "actual_funding_cost_r"] = cost_r
    z["net8_actual_funding_r"] = pd.to_numeric(z.net8_r, errors="coerce") - pd.to_numeric(z.actual_funding_cost_r, errors="coerce")
    z["stress12_actual_funding_r"] = pd.to_numeric(z.stress12_r, errors="coerce") - pd.to_numeric(z.actual_funding_cost_r, errors="coerce")
    z["net8_adverse3fund_r"] = pd.to_numeric(z.net8_r, errors="coerce") - ADVERSE_FUNDING_BPS * pd.to_numeric(z.scheduled_funding_count, errors="coerce") / pd.to_numeric(z.risk_bps, errors="coerce")
    z["stress12_adverse3fund_r"] = pd.to_numeric(z.stress12_r, errors="coerce") - ADVERSE_FUNDING_BPS * pd.to_numeric(z.scheduled_funding_count, errors="coerce") / pd.to_numeric(z.risk_bps, errors="coerce")
    return z, pd.DataFrame(status_rows)


def _close_due(heap, now_ns, state, eq_points):
    while heap and heap[0][0] <= now_ns:
        _, _, p = heapq.heappop(heap)
        state["reserved_margin"] -= p["margin"]
        state["open_risk"] -= p["risk_amt"]
        state["balance"] += p["pnl"]
        state["closed"] += 1
        state["peak"] = max(state["peak"], state["balance"])
        if state["peak"] > 0:
            state["maxdd"] = max(state["maxdd"], 1.0 - state["balance"] / state["peak"])
        eq_points.append((pd.Timestamp(p["exit_time"]), state["balance"]))


def simulate_portfolio(z: pd.DataFrame, r_col: str, risk_pct: float, leverage: float, max_open: int):
    x = z.copy()
    x[r_col] = pd.to_numeric(x[r_col], errors="coerce")
    x["risk_bps"] = pd.to_numeric(x.risk_bps, errors="coerce")
    x = x[np.isfinite(x[r_col]) & np.isfinite(x.risk_bps) & (x.risk_bps > 0)].copy()
    x = x.sort_values(["entry_time", "pair"]).reset_index(drop=True)
    state = {
        "balance": START_BALANCE,
        "peak": START_BALANCE,
        "maxdd": 0.0,
        "reserved_margin": 0.0,
        "open_risk": 0.0,
        "closed": 0,
        "accepted": 0,
        "skip_slots": 0,
        "skip_margin": 0,
        "skip_min_notional": 0,
        "skip_liq": 0,
        "max_open": 0,
        "max_margin_util": 0.0,
        "max_open_risk_pct": 0.0,
    }
    heap = []
    seq = 0
    eq_points = [(x.entry_time.min() if len(x) else pd.Timestamp.utcnow(), START_BALANCE)]
    accepted_rows = []
    for r in x.itertuples(index=False):
        et = pd.Timestamp(r.entry_time)
        _close_due(heap, et.value, state, eq_points)
        if len(heap) >= max_open:
            state["skip_slots"] += 1
            continue
        if state["balance"] <= 0:
            break
        stop_frac = float(r.risk_bps) / 10000.0
        liq_buffer = 1.0 / float(leverage) - MAINT_MARGIN_FRAC
        if liq_buffer <= 0 or stop_frac >= liq_buffer:
            state["skip_liq"] += 1
            continue
        risk_amt = state["balance"] * float(risk_pct) / 100.0
        notional = risk_amt / stop_frac
        if notional < MIN_NOTIONAL:
            state["skip_min_notional"] += 1
            continue
        margin = notional / float(leverage)
        available = state["balance"] - state["reserved_margin"]
        if margin > available + 1e-12:
            state["skip_margin"] += 1
            continue
        pnl = risk_amt * float(getattr(r, r_col))
        xt = pd.Timestamp(r.exit_time_exact)
        p = {"margin": margin, "risk_amt": risk_amt, "pnl": pnl, "exit_time": xt, "pair": r.pair}
        heapq.heappush(heap, (xt.value, seq, p))
        seq += 1
        state["reserved_margin"] += margin
        state["open_risk"] += risk_amt
        state["accepted"] += 1
        state["max_open"] = max(state["max_open"], len(heap))
        if state["balance"] > 0:
            state["max_margin_util"] = max(state["max_margin_util"], state["reserved_margin"] / state["balance"])
            state["max_open_risk_pct"] = max(state["max_open_risk_pct"], state["open_risk"] / state["balance"] * 100.0)
        accepted_rows.append({"entry_time": et, "exit_time": xt, "pair": r.pair, "risk_amt": risk_amt, "notional": notional, "margin": margin, "pnl": pnl})
    _close_due(heap, 2**63 - 1, state, eq_points)
    final = state["balance"]
    total_skips = state["skip_slots"] + state["skip_margin"] + state["skip_min_notional"] + state["skip_liq"]
    return {
        "risk_pct": float(risk_pct), "leverage": float(leverage), "max_open_limit": int(max_open),
        "r_col": r_col, "eligible": int(len(x)), "accepted": int(state["accepted"]), "skipped": int(total_skips),
        "skip_slots": int(state["skip_slots"]), "skip_margin": int(state["skip_margin"]),
        "skip_min_notional": int(state["skip_min_notional"]), "skip_liq": int(state["skip_liq"]),
        "final_balance": float(final), "roi_pct": float((final / START_BALANCE - 1.0) * 100.0),
        "realized_maxdd_pct": float(state["maxdd"] * 100.0), "max_open_actual": int(state["max_open"]),
        "max_margin_util_pct": float(state["max_margin_util"] * 100.0), "max_open_risk_pct": float(state["max_open_risk_pct"]),
    }, pd.DataFrame(accepted_rows), pd.DataFrame(eq_points, columns=["time", "balance"])


def _print_port(label: str, m: dict):
    print(
        f"{label:25s} final=${m['final_balance']:7.2f} ROI={m['roi_pct']:+7.1f}% DD={m['realized_maxdd_pct']:5.1f}% "
        f"accepted={m['accepted']:3d}/{m['eligible']:3d} skip(slot/margin/min/liq)={m['skip_slots']}/{m['skip_margin']}/{m['skip_min_notional']}/{m['skip_liq']} "
        f"maxOpen={m['max_open_actual']} marginUse={m['max_margin_util_pct']:5.1f}% openRisk={m['max_open_risk_pct']:4.1f}%"
    )


def main():
    a = parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    src = Path(a.v13dir) / "holdout_selected_trades.csv"
    if not src.exists():
        raise RuntimeError(f"Missing V1.3 selected trades: {src}")
    z0 = pd.read_csv(src)
    z0["entry_time"] = pd.to_datetime(z0.entry_time, utc=True)
    cfg = json.loads(Path(a.config).read_text())
    datadir = Path(a.datadir)

    print("=== BREAKOUT / RETEST PROFIT V1.4 — $100 PORTFOLIO REALISM ===", flush=True)
    print("Frozen V1.3 signal only. NO signal/filter/RR tuning. Reconstructs exact 5m exits, then tests capital, concurrency, leverage and funding impact.", flush=True)
    print(f"source trades={len(z0)} pairs={z0.pair.nunique()} start_balance=${START_BALANCE:.2f}", flush=True)

    all_rows, metas = [], []
    for n, (pair, g) in enumerate(z0.groupby("pair"), 1):
        rr, meta = _reconstruct_pair(pair, g, cfg, datadir)
        all_rows.extend(rr); metas.append(meta)
        print(f"reconstruct {n:2d}/{z0.pair.nunique()} {pair:24s} {meta.get('status')} rows={len(rr)} missing={meta.get('missing_entry',0)}", flush=True)
    z = pd.DataFrame(all_rows)
    if z.empty:
        raise RuntimeError("No trades reconstructed")
    z["entry_time"] = pd.to_datetime(z.entry_time, utc=True)
    z["exit_time_exact"] = pd.to_datetime(z.exit_time_exact, utc=True)
    pd.DataFrame(metas).to_csv(out / "reconstruction_coverage.csv", index=False)

    z, fstatus = _attach_funding(z, out, a.skip_funding_download)
    fstatus.to_csv(out / "funding_coverage.csv", index=False)
    z.to_csv(out / "portfolio_trades_exact.csv", index=False)

    print("\n=== EXECUTION RECONSTRUCTION SANITY ===")
    finite_match = pd.to_numeric(z.r_match_abs, errors="coerce").dropna()
    print(f"reconstructed={len(z)} finite stored outcomes={len(finite_match)} match_abs median={finite_match.median():.6g} max={finite_match.max():.6g}")
    print("exit reasons: " + ", ".join(f"{k}={v}" for k,v in z.exit_reason_exact.value_counts().items()))
    print(f"duration median={z.duration_min.median():.1f}m p90={z.duration_min.quantile(.9):.1f}m max={z.duration_min.max():.1f}m")
    print(f"scheduled 8h-funding crossings: {(z.scheduled_funding_count>0).mean()*100:.1f}% trades | total events={int(z.scheduled_funding_count.sum())}")

    print("\n=== HISTORICAL FUNDING COVERAGE ===")
    ok_pairs = int(fstatus.ok.sum()) if len(fstatus) else 0
    ok_trades = int(z.actual_funding_ok.sum())
    print(f"funding pairs={ok_pairs}/{len(fstatus)} | trades covered={ok_trades}/{len(z)} ({ok_trades/len(z)*100:.1f}%)")
    if len(fstatus):
        bad = fstatus[~fstatus.ok]
        if len(bad):
            print("uncovered: " + ", ".join(f"{r.pair}[{r.source}]" for r in bad.head(8).itertuples(index=False)))
    if ok_trades:
        q = z[z.actual_funding_ok].copy()
        print(f"actual funding cost R/trade median={q.actual_funding_cost_r.median():+.4f} mean={q.actual_funding_cost_r.mean():+.4f} | events={int(q.actual_funding_events.sum())}")

    print("\n=== $100 REFERENCE PORTFOLIO ===")
    # Reference capital mechanics are predeclared capacity settings, not selected for best return:
    # 1% equity risk/trade, isolated 5x, at most 3 simultaneous positions.
    ref_specs = [
        ("8bps no funding", "net8_r"),
        ("8bps + adverse3bp/fund", "net8_adverse3fund_r"),
        ("12bps + adverse3bp/fund", "stress12_adverse3fund_r"),
    ]
    reference_rows = []
    for label, col in ref_specs:
        m, trades, curve = simulate_portfolio(z, col, REFERENCE_RISK_PCT, REFERENCE_LEVERAGE, REFERENCE_MAX_OPEN)
        reference_rows.append({"label": label, **m})
        _print_port(label, m)
        if label == "8bps + adverse3bp/fund":
            trades.to_csv(out / "reference_portfolio_trades.csv", index=False)
            curve.to_csv(out / "reference_equity_curve.csv", index=False)
    if ok_trades == len(z):
        m, _, _ = simulate_portfolio(z, "net8_actual_funding_r", REFERENCE_RISK_PCT, REFERENCE_LEVERAGE, REFERENCE_MAX_OPEN)
        reference_rows.append({"label": "8bps + actual funding", **m})
        _print_port("8bps + actual funding", m)
    pd.DataFrame(reference_rows).to_csv(out / "reference_portfolio.csv", index=False)

    print("\n=== CAPITAL / LEVERAGE SENSITIVITY — ADVERSE 3bp PER FUNDING EVENT ===")
    grid = []
    for rp in (0.5, 1.0, 2.0, 3.0):
        for lev in (1.0, 2.0, 3.0, 5.0, 10.0, 20.0):
            for mo in (1, 2, 3, 5, 99):
                m, _, _ = simulate_portfolio(z, "net8_adverse3fund_r", rp, lev, mo)
                grid.append(m)
    gd = pd.DataFrame(grid)
    gd.to_csv(out / "portfolio_sensitivity.csv", index=False)

    print("-- leverage sensitivity at risk=1%, maxOpen=3 --")
    for r in gd[(gd.risk_pct==1.0)&(gd.max_open_limit==3)].sort_values("leverage").itertuples(index=False):
        print(f"lev={r.leverage:>4.0f}x final=${r.final_balance:7.2f} ROI={r.roi_pct:+7.1f}% DD={r.realized_maxdd_pct:5.1f}% accepted={r.accepted:3d} marginSkip={r.skip_margin:3d} liqSkip={r.skip_liq:3d}")
    print("-- risk sensitivity at leverage=5x, maxOpen=3 --")
    for r in gd[(gd.leverage==5.0)&(gd.max_open_limit==3)].sort_values("risk_pct").itertuples(index=False):
        print(f"risk={r.risk_pct:>3.1f}% final=${r.final_balance:7.2f} ROI={r.roi_pct:+7.1f}% DD={r.realized_maxdd_pct:5.1f}% accepted={r.accepted:3d} maxOpenRisk={r.max_open_risk_pct:4.1f}%")
    print("-- concurrency sensitivity at risk=1%, leverage=5x --")
    for r in gd[(gd.risk_pct==1.0)&(gd.leverage==5.0)].sort_values("max_open_limit").itertuples(index=False):
        lab = "ALL" if r.max_open_limit==99 else str(int(r.max_open_limit))
        print(f"maxOpen={lab:>3s} final=${r.final_balance:7.2f} ROI={r.roi_pct:+7.1f}% DD={r.realized_maxdd_pct:5.1f}% accepted={r.accepted:3d} slotSkip={r.skip_slots:3d}")

    print("\n=== INTERPRETATION RULE ===")
    print("This stage does NOT re-select the signal from HOLDOUT. Leverage is only a margin/capacity parameter because position notional is sized from stop risk.")
    print("A useful result must remain positive with $100 capital under realistic concurrency and costs, without requiring liquidation-unsafe leverage or skipping most trades.")
    print("The 3bp funding-event case is deliberately adverse. Actual historical funding is reported separately when Binance coverage is available.")
    print("Because V1.3 was flat/negative in 2026 despite passing the full HOLDOUT, no HOLDOUT-derived pair/regime filter may be added. A live/prospective dry-run is required before risking capital.")
    print(f"Reports: {out}")
    print("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
