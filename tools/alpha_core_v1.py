#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PAIRS = ("BTC", "ETH", "SOL", "BNB", "XRP", "DOGE", "LINK")
LOOKBACKS = (12, 20, 36)
BREAK_ATRS = (0.05, 0.10, 0.20)
RV_MINS = (1.2, 1.4, 1.7)
STOP_ATRS = (0.70, 0.85, 1.00)
RRS = (1.5, 2.0, 2.5, 3.0)
HOLD_BARS = (6, 12, 18)  # 30/60/90 min on 5m data
COSTS_BPS = (8.0, 12.0, 20.0)
RISK_PCTS = (1.5, 2.0, 2.5, 3.0)
LEVERAGE = 10.0
MAX_OPEN = 3
MAX_MARGIN_FRAC = 0.75
MAX_OPEN_RISK_FRAC = 0.05
MIN_STOP_BPS = 60.0
MAX_STOP_BPS = 950.0
MAX_ENTRY_DRIFT_ATR = 0.35
START_EQUITY = 100.0
SELECT_END = pd.Timestamp("2026-01-01", tz="UTC")


def parse_args():
    p = argparse.ArgumentParser(description="Alpha Core V1: causal 5m volatility breakout research")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance/futures")
    p.add_argument("--outdir", default="/freqtrade/user_data/alpha_core_v1")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=7)
    p.add_argument("--rescan", action="store_true")
    return p.parse_args()


def log(s: str) -> None:
    print(s, flush=True)


def _atr14(df: pd.DataFrame) -> pd.Series:
    prev = df["close"].shift(1)
    tr = pd.concat([
        df["high"] - df["low"],
        (df["high"] - prev).abs(),
        (df["low"] - prev).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(14, min_periods=14).mean()


def _load_pair(datadir: Path, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    path = datadir / f"{symbol}_USDT_USDT-5m-futures.feather"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_feather(path)
    need = {"date", "open", "high", "low", "close", "volume"}
    missing = need.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)}")
    x = df[["date", "open", "high", "low", "close", "volume"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    warm = start - pd.Timedelta(days=10)
    x = x[(x.date >= warm) & (x.date < end + pd.Timedelta(days=1))].reset_index(drop=True)
    for c in ("open", "high", "low", "close", "volume"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna(subset=["open", "high", "low", "close", "volume"]).reset_index(drop=True)


def _simulate_one(x: pd.DataFrame, entry_idx: int, side: int, risk_abs: float, rr: float, hold_bars: int):
    entry = float(x.iloc[entry_idx]["open"])
    stop = entry - side * risk_abs
    target = entry + side * risk_abs * rr
    last_idx = min(len(x) - 1, entry_idx + hold_bars - 1)
    for j in range(entry_idx, last_idx + 1):
        bar = x.iloc[j]
        lo = float(bar["low"])
        hi = float(bar["high"])
        stop_hit = lo <= stop if side > 0 else hi >= stop
        target_hit = hi >= target if side > 0 else lo <= target
        # Pessimistic same-bar ordering: stop wins.
        if stop_hit:
            return -1.0, j, "STOP"
        if target_hit:
            return float(rr), j, "TARGET"
    exit_px = float(x.iloc[last_idx]["close"])
    gross_r = side * (exit_px - entry) / risk_abs
    return float(gross_r), last_idx, "TIME"


def process_pair(symbol: str, datadir_s: str, start_s: str, end_s: str):
    datadir = Path(datadir_s)
    start = pd.Timestamp(start_s, tz="UTC")
    end = pd.Timestamp(end_s, tz="UTC")
    x = _load_pair(datadir, symbol, start, end)
    if x.empty:
        return [], {"pair": symbol, "status": "NO_DATA"}

    x["atr14"] = _atr14(x)
    lr = np.log(x["close"] / x["close"].shift(1))
    prior_rms_60m = np.sqrt((lr.shift(1).pow(2)).rolling(12, min_periods=12).mean())
    x["rv_ratio"] = lr.abs() / prior_rms_60m.replace(0, np.nan)
    x["mom_1h_prior"] = x["close"].shift(1) / x["close"].shift(13) - 1.0
    x["mom_4h_prior"] = x["close"].shift(1) / x["close"].shift(49) - 1.0
    x["vol_ratio20"] = x["volume"] / x["volume"].shift(1).rolling(20, min_periods=20).median()
    for lb in LOOKBACKS:
        x[f"ph_{lb}"] = x["high"].shift(1).rolling(lb, min_periods=lb).max()
        x[f"pl_{lb}"] = x["low"].shift(1).rolling(lb, min_periods=lb).min()

    rows = []
    min_break = min(BREAK_ATRS)
    min_rv = min(RV_MINS)
    min_lb = min(LOOKBACKS)
    for i in range(max(60, min_lb), len(x) - 1):
        t = pd.Timestamp(x.iloc[i]["date"])
        if t < start or t > end:
            continue
        atr = float(x.iloc[i]["atr14"])
        rv = float(x.iloc[i]["rv_ratio"])
        if not (np.isfinite(atr) and atr > 0 and np.isfinite(rv) and rv >= min_rv):
            continue
        sig_close = float(x.iloc[i]["close"])
        m1 = float(x.iloc[i]["mom_1h_prior"])
        m4 = float(x.iloc[i]["mom_4h_prior"])
        for lb in LOOKBACKS:
            ph = float(x.iloc[i][f"ph_{lb}"])
            pl = float(x.iloc[i][f"pl_{lb}"])
            if not (np.isfinite(ph) and np.isfinite(pl)):
                continue
            candidates = []
            bd_long = (sig_close - ph) / atr
            bd_short = (pl - sig_close) / atr
            if bd_long >= min_break and (m1 > 0 or m4 > 0):
                candidates.append((1, ph, bd_long))
            if bd_short >= min_break and (m1 < 0 or m4 < 0):
                candidates.append((-1, pl, bd_short))
            for side, level, break_atr in candidates:
                entry_idx = i + 1
                entry = float(x.iloc[entry_idx]["open"])
                # Signal must still be valid at the first executable 5m price.
                if side > 0 and entry <= level:
                    continue
                if side < 0 and entry >= level:
                    continue
                drift_atr = abs(entry - sig_close) / atr
                if drift_atr > MAX_ENTRY_DRIFT_ATR:
                    continue
                for stop_atr in STOP_ATRS:
                    structural_dist = side * (entry - level)
                    risk_abs = max(float(stop_atr) * atr, structural_dist, entry * MIN_STOP_BPS / 10000.0)
                    risk_bps = risk_abs / entry * 10000.0
                    if not (MIN_STOP_BPS <= risk_bps < MAX_STOP_BPS):
                        continue
                    base = {
                        "pair": symbol,
                        "signal_time": t + pd.Timedelta(minutes=5),
                        "entry_time": pd.Timestamp(x.iloc[entry_idx]["date"]),
                        "side": side,
                        "lookback": lb,
                        "break_atr": float(break_atr),
                        "rv_ratio": rv,
                        "mom_1h_prior": m1,
                        "mom_4h_prior": m4,
                        "volume_ratio20": float(x.iloc[i]["vol_ratio20"]),
                        "signal_close": sig_close,
                        "entry": entry,
                        "level": level,
                        "atr14": atr,
                        "entry_drift_atr": drift_atr,
                        "stop_atr": stop_atr,
                        "risk_abs": risk_abs,
                        "risk_bps": risk_bps,
                    }
                    for rr in RRS:
                        for hb in HOLD_BARS:
                            gross_r, exit_idx, reason = _simulate_one(x, entry_idx, side, risk_abs, rr, hb)
                            suffix = f"rr{str(rr).replace('.', 'p')}_h{hb}"
                            base[f"gross_{suffix}"] = gross_r
                            base[f"exit_{suffix}"] = pd.Timestamp(x.iloc[exit_idx]["date"]) + pd.Timedelta(minutes=5)
                            base[f"reason_{suffix}"] = reason
                    rows.append(base.copy())

    return rows, {
        "pair": symbol,
        "status": "OK",
        "bars": int(len(x)),
        "start": str(x.date.min()),
        "end": str(x.date.max()),
        "events": int(len(rows)),
    }


def _pf(r: np.ndarray) -> float:
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    return pos / neg if neg > 0 else math.inf


def _metrics(g: pd.DataFrame, gross_col: str, cost_bps: float, period_start: pd.Timestamp, period_end: pd.Timestamp):
    if g.empty:
        return {"n": 0, "pf": np.nan, "exp": np.nan, "wr": np.nan, "tpm": 0.0, "positive_quarters": 0.0}
    cost_r = cost_bps / pd.to_numeric(g["risk_bps"], errors="coerce").to_numpy(float)
    gross = pd.to_numeric(g[gross_col], errors="coerce").to_numpy(float)
    r = gross - cost_r
    good = np.isfinite(r)
    r = r[good]
    gg = g.iloc[np.flatnonzero(good)].copy()
    if len(r) == 0:
        return {"n": 0, "pf": np.nan, "exp": np.nan, "wr": np.nan, "tpm": 0.0, "positive_quarters": 0.0}
    months = max(1, (period_end.year - period_start.year) * 12 + period_end.month - period_start.month)
    qidx = pd.period_range(period_start.tz_localize(None).to_period("Q"), (period_end - pd.Timedelta(seconds=1)).tz_localize(None).to_period("Q"), freq="Q")
    qsum = pd.Series(0.0, index=qidx)
    qp = gg["entry_time"].dt.tz_localize(None).dt.to_period("Q")
    for q, val in pd.Series(r).groupby(qp.reset_index(drop=True)).sum().items():
        if q in qsum.index:
            qsum.loc[q] = float(val)
    return {
        "n": int(len(r)),
        "pf": float(_pf(r)),
        "exp": float(np.mean(r)),
        "wr": float(np.mean(r > 0) * 100.0),
        "tpm": float(len(r) / months),
        "positive_quarters": float((qsum > 0).mean() * 100.0) if len(qsum) else 0.0,
    }


def _filtered(events: pd.DataFrame, lb: int, break_min: float, rv_min: float, stop_atr: float):
    return events[
        (events["lookback"] == lb)
        & (events["break_atr"] >= break_min)
        & (events["rv_ratio"] >= rv_min)
        & np.isclose(events["stop_atr"], stop_atr)
    ].copy()


def search(events: pd.DataFrame, start: pd.Timestamp):
    sel = events[(events.entry_time >= start) & (events.entry_time < SELECT_END)].copy()
    rows = []
    for lb in LOOKBACKS:
        for br in BREAK_ATRS:
            for rv in RV_MINS:
                for sa in STOP_ATRS:
                    g = _filtered(sel, lb, br, rv, sa)
                    if len(g) < 80:
                        continue
                    for rr in RRS:
                        for hb in HOLD_BARS:
                            gross_col = f"gross_rr{str(rr).replace('.', 'p')}_h{hb}"
                            m8 = _metrics(g, gross_col, 8.0, start, SELECT_END)
                            m12 = _metrics(g, gross_col, 12.0, start, SELECT_END)
                            m20 = _metrics(g, gross_col, 20.0, start, SELECT_END)
                            score = (
                                m12["exp"] * min(m12["tpm"], 100.0)
                                + 0.20 * min(m12["pf"], 3.0)
                                + 0.004 * m12["positive_quarters"]
                            )
                            rows.append({
                                "lookback": lb,
                                "break_min_atr": br,
                                "rv_min": rv,
                                "stop_atr": sa,
                                "rr": rr,
                                "hold_bars": hb,
                                "hold_min": hb * 5,
                                "n": m12["n"],
                                "pf8": m8["pf"],
                                "pf12": m12["pf"],
                                "pf20": m20["pf"],
                                "exp8": m8["exp"],
                                "exp12": m12["exp"],
                                "exp20": m20["exp"],
                                "wr12": m12["wr"],
                                "trades_month": m12["tpm"],
                                "positive_quarters": m12["positive_quarters"],
                                "score": score,
                            })
    z = pd.DataFrame(rows)
    if z.empty:
        return z
    return z.sort_values(["score", "exp12", "pf12"], ascending=False).reset_index(drop=True)


def _select_winner(search_df: pd.DataFrame):
    if search_df.empty:
        return None, False
    gates = search_df[
        (search_df.pf12 >= 1.35)
        & (search_df.exp12 >= 0.20)
        & (search_df.positive_quarters >= 70.0)
        & (search_df.trades_month >= 20.0)
        & (search_df.exp20 > 0)
    ]
    if not gates.empty:
        return gates.iloc[0].to_dict(), True
    return search_df.iloc[0].to_dict(), False


def _winner_events(events: pd.DataFrame, w: dict):
    return _filtered(events, int(w["lookback"]), float(w["break_min_atr"]), float(w["rv_min"]), float(w["stop_atr"]))


def _portfolio(g: pd.DataFrame, w: dict, cost_bps: float, risk_pct: float, start: pd.Timestamp, end: pd.Timestamp):
    rr = float(w["rr"])
    hb = int(w["hold_bars"])
    suffix = f"rr{str(rr).replace('.', 'p')}_h{hb}"
    x = g[(g.entry_time >= start) & (g.entry_time < end)].copy()
    if x.empty:
        return {"accepted": 0, "final": START_EQUITY, "roi": 0.0, "maxdd": 0.0, "median_monthly": 0.0, "mean_monthly": 0.0, "positive_months": 0.0}
    x["net_r"] = pd.to_numeric(x[f"gross_{suffix}"], errors="coerce") - cost_bps / pd.to_numeric(x["risk_bps"], errors="coerce")
    x["exit_time"] = pd.to_datetime(x[f"exit_{suffix}"], utc=True)
    x["strength"] = pd.to_numeric(x["break_atr"], errors="coerce") + 0.25 * pd.to_numeric(x["rv_ratio"], errors="coerce")
    x = x.sort_values(["entry_time", "strength", "pair"], ascending=[True, False, True]).reset_index(drop=True)

    equity = START_EQUITY
    peak = equity
    maxdd = 0.0
    active = []
    closed = []
    accepted = 0
    for _, row in x.iterrows():
        t = row.entry_time
        still = []
        for p in active:
            if p["exit_time"] <= t:
                equity += p["risk_amount"] * p["net_r"]
                closed.append((p["exit_time"], equity))
                peak = max(peak, equity)
                maxdd = max(maxdd, (peak - equity) / peak if peak > 0 else 1.0)
            else:
                still.append(p)
        active = still
        if equity <= 0:
            break
        if any(p["pair"] == row.pair for p in active):
            continue
        if len(active) >= MAX_OPEN:
            continue
        rp = risk_pct / 100.0
        open_risk = sum(p["risk_amount"] for p in active)
        risk_amount = equity * rp
        if open_risk + risk_amount > equity * MAX_OPEN_RISK_FRAC + 1e-12:
            continue
        stop_frac = float(row.risk_bps) / 10000.0
        notional = risk_amount / stop_frac
        margin = notional / LEVERAGE
        if sum(p["margin"] for p in active) + margin > equity * MAX_MARGIN_FRAC + 1e-12:
            continue
        active.append({
            "pair": row.pair,
            "exit_time": row.exit_time,
            "risk_amount": risk_amount,
            "margin": margin,
            "net_r": float(row.net_r),
        })
        accepted += 1
    for p in sorted(active, key=lambda q: q["exit_time"]):
        equity += p["risk_amount"] * p["net_r"]
        closed.append((p["exit_time"], equity))
        peak = max(peak, equity)
        maxdd = max(maxdd, (peak - equity) / peak if peak > 0 else 1.0)

    month_index = pd.period_range(start.tz_localize(None).to_period("M"), (end - pd.Timedelta(seconds=1)).tz_localize(None).to_period("M"), freq="M")
    month_end_equity = []
    prev = START_EQUITY
    c = pd.DataFrame(closed, columns=["time", "equity"]) if closed else pd.DataFrame(columns=["time", "equity"])
    if not c.empty:
        c["month"] = pd.to_datetime(c.time, utc=True).dt.tz_localize(None).dt.to_period("M")
    rets = []
    for m in month_index:
        cur = prev
        if not c.empty:
            cm = c[c.month == m]
            if not cm.empty:
                cur = float(cm.iloc[-1].equity)
        rets.append((cur / prev - 1.0) * 100.0 if prev > 0 else -100.0)
        prev = cur
        month_end_equity.append(cur)
    return {
        "accepted": int(accepted),
        "final": float(equity),
        "roi": float((equity / START_EQUITY - 1.0) * 100.0),
        "maxdd": float(maxdd * 100.0),
        "median_monthly": float(np.median(rets)) if rets else 0.0,
        "mean_monthly": float(np.mean(rets)) if rets else 0.0,
        "positive_months": float(np.mean(np.array(rets) > 0) * 100.0) if rets else 0.0,
    }


def main():
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache = outdir / "events.csv"
    start = pd.Timestamp(a.start, tz="UTC")
    end = pd.Timestamp(a.end, tz="UTC")

    log("=== ALPHA CORE V1 — CAUSAL VOLATILITY BREAKOUT ===")
    log("Universe: BTC ETH SOL BNB XRP DOGE LINK | 5m 2022-2026 | research only")
    if cache.exists() and not a.rescan:
        log(f"reusing cached events: {cache}")
        events = pd.read_csv(cache, parse_dates=["signal_time", "entry_time"] + [f"exit_rr{str(rr).replace('.', 'p')}_h{hb}" for rr in RRS for hb in HOLD_BARS])
        coverage = pd.DataFrame()
    else:
        all_rows = []
        metas = []
        workers = min(max(1, int(a.workers)), len(PAIRS))
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_pair, p, a.datadir, a.start, a.end): p for p in PAIRS}
            for fut in as_completed(futs):
                p = futs[fut]
                rows, meta = fut.result()
                all_rows.extend(rows)
                metas.append(meta)
                log(f"pair {p}: {meta['status']} bars={meta.get('bars', 0):,} events={meta.get('events', 0):,}")
        events = pd.DataFrame(all_rows)
        coverage = pd.DataFrame(metas).sort_values("pair")
        coverage.to_csv(outdir / "coverage.csv", index=False)
        if events.empty:
            raise SystemExit("No events generated")
        events.to_csv(cache, index=False)
        log(f"cached {len(events):,} event/stop rows -> {cache}")

    events["signal_time"] = pd.to_datetime(events["signal_time"], utc=True)
    events["entry_time"] = pd.to_datetime(events["entry_time"], utc=True)
    log(f"searching {len(events):,} event/stop rows across {len(LOOKBACKS)*len(BREAK_ATRS)*len(RV_MINS)*len(STOP_ATRS)*len(RRS)*len(HOLD_BARS):,} configs")
    z = search(events, start)
    z.to_csv(outdir / "search.csv", index=False)
    if z.empty:
        raise SystemExit("Search produced no candidates")
    z.head(100).to_csv(outdir / "top100.csv", index=False)
    w, gate_pass = _select_winner(z)
    assert w is not None
    g = _winner_events(events, w)

    hist = {}
    for cost in COSTS_BPS:
        rr = float(w["rr"])
        hb = int(w["hold_bars"])
        col = f"gross_rr{str(rr).replace('.', 'p')}_h{hb}"
        hist[str(int(cost))] = _metrics(g[(g.entry_time >= SELECT_END) & (g.entry_time <= end)], col, cost, SELECT_END, end + pd.Timedelta(days=1))

    portfolio_rows = []
    for cost in COSTS_BPS:
        for rp in RISK_PCTS:
            selp = _portfolio(g, w, cost, rp, start, SELECT_END)
            testp = _portfolio(g, w, cost, rp, SELECT_END, end + pd.Timedelta(days=1))
            portfolio_rows.append({"cost_bps": cost, "risk_pct": rp, **{f"select_{k}": v for k, v in selp.items()}, **{f"hist2026_{k}": v for k, v in testp.items()}})
    portfolios = pd.DataFrame(portfolio_rows)
    portfolios.to_csv(outdir / "portfolio.csv", index=False)

    summary = {
        "research_only": True,
        "strategy": "Alpha Core V1",
        "universe": list(PAIRS),
        "selection_period": f"{a.start}..2025-12-31",
        "historical_2026_benchmark": f"2026-01-01..{a.end}",
        "2026_is_not_pristine_holdout": True,
        "signal": "5m close breakout of prior N bars by ATR threshold; 5m volatility shock; prior 1h OR 4h momentum regime; next-5m-open execution; drift veto",
        "cost_models_bps": list(COSTS_BPS),
        "promotion_gate": "PF12>=1.35, exp12>=0.20R, >=70% positive quarters, >=20 trades/month, exp20>0",
        "gate_pass": bool(gate_pass),
        "winner": w,
        "historical_2026_event_metrics": hist,
        "best_portfolio_rows": portfolios.to_dict(orient="records"),
    }
    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    log("\n=== WINNER ===")
    log(
        f"lookback={int(w['lookback'])} break>={w['break_min_atr']:.2f}ATR rv>={w['rv_min']:.1f} "
        f"stop={w['stop_atr']:.2f}ATR RR={w['rr']:.1f} hold={int(w['hold_min'])}m"
    )
    log(
        f"SELECT 2022-2025: N={int(w['n'])} TPM={w['trades_month']:.1f} PF12={w['pf12']:.2f} "
        f"EXP12={w['exp12']:+.3f}R PF20={w['pf20']:.2f} EXP20={w['exp20']:+.3f}R "
        f"positiveQuarters={w['positive_quarters']:.1f}%"
    )
    h12 = hist["12"]
    log(
        f"HIST 2026 @12bps: N={h12['n']} TPM={h12['tpm']:.1f} PF={h12['pf']:.2f} "
        f"EXP={h12['exp']:+.3f}R positiveQuarters={h12['positive_quarters']:.1f}%"
    )
    log(f"PROMOTION GATE: {'PASS' if gate_pass else 'MISS'}")
    log("\n=== PORTFOLIO @12bps ===")
    p12 = portfolios[portfolios.cost_bps == 12.0]
    for _, r in p12.iterrows():
        log(
            f"risk={r.risk_pct:.1f}% | select ROI={r.select_roi:+.1f}% med/mo={r.select_median_monthly:+.1f}% DD={r.select_maxdd:.1f}% "
            f"| 2026 ROI={r.hist2026_roi:+.1f}% med/mo={r.hist2026_median_monthly:+.1f}% DD={r.hist2026_maxdd:.1f}%"
        )
    log(f"reports: {outdir}")


if __name__ == "__main__":
    main()
