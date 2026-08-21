#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc
import breakout_retest_profit_v1 as v1
from breakout_retest_profit_v16 import causal_dedup_events

FROZEN_PAIRS = (
    "AAVE/USDT:USDT", "ADA/USDT:USDT", "ATOM/USDT:USDT", "AVAX/USDT:USDT",
    "BCH/USDT:USDT", "BNB/USDT:USDT", "BTC/USDT:USDT", "DOGE/USDT:USDT",
    "DOT/USDT:USDT", "ETC/USDT:USDT", "ETH/USDT:USDT", "FIL/USDT:USDT",
    "LINK/USDT:USDT", "LTC/USDT:USDT", "NEAR/USDT:USDT", "SOL/USDT:USDT",
    "TRX/USDT:USDT", "UNI/USDT:USDT", "XLM/USDT:USDT", "XRP/USDT:USDT",
)
THRESH = 1.5
RISK_MIN_BPS = 160.0
RR = 3.0
HOLD_BARS = 48
BASE_COST_BPS = 8.0
STRESS_COST_BPS = 12.0
WARMUP_DAYS = 60
LEVEL_TFS = ("15m", "1h", "4h")
LEVEL_PERIODS = (20, 30)

START_BALANCE = 100.0
RISK_PCT = 1.0
LEVERAGE = 5.0
MAX_OPEN = 3
MIN_NOTIONAL = 5.0
MAINT_MARGIN_FRAC = 0.005

BINANCE_KLINES = "https://fapi.binance.com/fapi/v1/klines"
BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
UA = "rmv5-prospective-v2/1.0"


def parse_args():
    p = argparse.ArgumentParser(description="Frozen fully-causal FAKEOUT prospective paper tracker")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/prospective_fakeout_v2")
    p.add_argument("--loop", action="store_true")
    p.add_argument("--poll-seconds", type=int, default=30)
    p.add_argument("--reset-cutoff", action="store_true",
                   help="Destructive experiment reset: start a new prospective cutoff at the next 5m boundary.")
    return p.parse_args()


def utc_now() -> pd.Timestamp:
    return pd.Timestamp.now(tz="UTC")


def next_5m_boundary(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.Timestamp(ts).tz_convert("UTC")
    return ts.floor("5min") + pd.Timedelta(minutes=5)


def symbol(pair: str) -> str:
    return pair.split(":", 1)[0].replace("/", "")


def _http_json(url: str, params: dict, timeout: int = 20):
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(url + "?" + qs, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_klines(pair: str, timeframe: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    rows = []
    cursor = int(start_ms)
    while cursor <= end_ms:
        chunk = _http_json(BINANCE_KLINES, {
            "symbol": symbol(pair),
            "interval": timeframe,
            "startTime": cursor,
            "endTime": int(end_ms),
            "limit": 1500,
        })
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        last_open = int(chunk[-1][0])
        nxt = last_open + 1
        if nxt <= cursor:
            break
        cursor = nxt
        if len(chunk) < 1500:
            break
        time.sleep(0.03)
    if not rows:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    x = pd.DataFrame(rows, columns=[
        "open_ms", "open", "high", "low", "close", "volume",
        "close_ms", "quote_volume", "trades", "taker_base", "taker_quote", "ignore",
    ])
    x["date"] = pd.to_datetime(x["open_ms"].astype("int64"), unit="ms", utc=True).astype("datetime64[ns, UTC]")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x[["date", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fetch_funding(pair: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    rows = []
    cursor = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int(pd.Timestamp(end).timestamp() * 1000)
    while cursor <= end_ms:
        chunk = _http_json(BINANCE_FUNDING, {
            "symbol": symbol(pair), "startTime": cursor, "endTime": end_ms, "limit": 1000
        })
        if not isinstance(chunk, list) or not chunk:
            break
        rows.extend(chunk)
        last = max(int(z["fundingTime"]) for z in chunk)
        if last < cursor:
            break
        cursor = last + 1
        if len(chunk) < 1000:
            break
        time.sleep(0.03)
    out = []
    for z in rows:
        try:
            out.append((pd.to_datetime(int(z["fundingTime"]), unit="ms", utc=True), float(z["fundingRate"])))
        except Exception:
            pass
    return pd.DataFrame(out, columns=["funding_time", "rate"]).drop_duplicates("funding_time").sort_values("funding_time") if out else pd.DataFrame(columns=["funding_time", "rate"])


def read_csv_candles(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    x = pd.read_csv(path)
    if x.empty:
        return x
    x["date"] = pd.to_datetime(x["date"], utc=True).astype("datetime64[ns, UTC]")
    for c in ["open", "high", "low", "close", "volume"]:
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x[["date", "open", "high", "low", "close", "volume"]].dropna().drop_duplicates("date").sort_values("date").reset_index(drop=True)


def write_csv_atomic(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


def sync_tf(cfg: dict, datadir: Path, outdir: Path, pair: str, tf: str, now: pd.Timestamp) -> pd.DataFrame:
    minutes = 5 if tf == "5m" else 15
    live_path = outdir / "market_cache" / tf / f"{symbol(pair)}.csv"
    live = read_csv_candles(live_path)
    base = dc.load_tf(cfg, datadir, pair, tf)
    if not base.empty:
        base = base[["date", "open", "high", "low", "close", "volume"]].copy()
        base["date"] = pd.to_datetime(base["date"], utc=True).astype("datetime64[ns, UTC]")
        base = base.sort_values("date").drop_duplicates("date").reset_index(drop=True)

    latest_candidates = []
    if not base.empty:
        latest_candidates.append(pd.Timestamp(base.date.max()))
    if not live.empty:
        latest_candidates.append(pd.Timestamp(live.date.max()))
    if latest_candidates:
        fetch_start = max(latest_candidates) - pd.Timedelta(minutes=minutes)
    else:
        fetch_start = now - pd.Timedelta(days=WARMUP_DAYS + 7)

    fetched = fetch_klines(
        pair, tf,
        int(fetch_start.timestamp() * 1000),
        int(now.timestamp() * 1000),
    )
    if not fetched.empty:
        fetched = fetched[fetched.date + pd.Timedelta(minutes=minutes) <= now].copy()

    if live.empty:
        live_new = fetched
    else:
        live_new = pd.concat([live, fetched], ignore_index=True)
        live_new = live_new.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    live_new = live_new[live_new.date >= now - pd.Timedelta(days=WARMUP_DAYS + 14)].reset_index(drop=True)
    write_csv_atomic(live_new, live_path)

    parts = [z for z in (base, live_new) if z is not None and not z.empty]
    if not parts:
        return pd.DataFrame()
    x = pd.concat(parts, ignore_index=True).drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    return x


def load_combined_tf(cfg: dict, datadir: Path, outdir: Path, pair: str, tf: str) -> pd.DataFrame:
    live_path = outdir / "market_cache" / tf / f"{symbol(pair)}.csv"
    live = read_csv_candles(live_path)
    base = dc.load_tf(cfg, datadir, pair, tf)
    if not base.empty:
        base = base[["date", "open", "high", "low", "close", "volume"]].copy()
        base["date"] = pd.to_datetime(base["date"], utc=True).astype("datetime64[ns, UTC]")
        base = base.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    parts = [z for z in (base, live) if z is not None and not z.empty]
    if not parts:
        return pd.DataFrame()
    return (
        pd.concat(parts, ignore_index=True)
        .drop_duplicates("date", keep="last")
        .sort_values("date")
        .reset_index(drop=True)
    )


def compute_pair_worker(pair: str, cfg_path: str, datadir_s: str, outdir_s: str, cutoff_s: str, now_s: str):
    cfg = json.loads(Path(cfg_path).read_text())
    datadir = Path(datadir_s)
    outdir = Path(outdir_s)
    cutoff = pd.Timestamp(cutoff_s)
    now = pd.Timestamp(now_s)
    raw5 = load_combined_tf(cfg, datadir, outdir, pair, "5m")
    raw15 = load_combined_tf(cfg, datadir, outdir, pair, "15m")
    rows = compute_pair(pair, raw5, raw15, cutoff, now)
    return rows, {"pair": pair, "status": "OK", "rows5": len(raw5), "rows15": len(raw15), "signals": len(rows)}


def init_state(outdir: Path, reset: bool) -> dict:
    path = outdir / "state.json"
    if reset and path.exists():
        stamp = utc_now().strftime("%Y%m%dT%H%M%SZ")
        path.rename(outdir / f"state.reset.{stamp}.json")
    if path.exists():
        return json.loads(path.read_text())
    now = utc_now()
    cutoff = next_5m_boundary(now)
    state = {
        "experiment": "prospective_fakeout_v2",
        "created_at": now.isoformat(),
        "cutoff": cutoff.isoformat(),
        "parameters": {
            "setup": "FAKEOUT",
            "activity_min": THRESH,
            "risk_min_bps": RISK_MIN_BPS,
            "rr": RR,
            "hold_bars_5m": HOLD_BARS,
            "base_cost_bps": BASE_COST_BPS,
            "stress_cost_bps": STRESS_COST_BPS,
            "warmup_days": WARMUP_DAYS,
            "level_tfs": list(LEVEL_TFS),
            "level_periods": list(LEVEL_PERIODS),
            "portfolio": {
                "start_balance": START_BALANCE,
                "risk_pct": RISK_PCT,
                "leverage": LEVERAGE,
                "max_open": MAX_OPEN,
                "min_notional": MIN_NOTIONAL,
            },
            "pairs": list(FROZEN_PAIRS),
        },
    }
    outdir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2))
    return state


def _metric(g: pd.DataFrame, col: str) -> dict:
    if g.empty or col not in g:
        return {"N": 0, "PF": np.nan, "WR": np.nan, "EXP": np.nan, "DD": np.nan}
    r = pd.to_numeric(g[col], errors="coerce").dropna().astype(float)
    if r.empty:
        return {"N": 0, "PF": np.nan, "WR": np.nan, "EXP": np.nan, "DD": np.nan}
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    curve = r.cumsum()
    dd = float((curve.cummax() - curve).max()) if len(curve) else 0.0
    return {
        "N": int(len(r)), "PF": float(pf), "WR": float((r > 0).mean() * 100.0),
        "EXP": float(r.mean()), "DD": dd,
    }


def _event_outcome(x5: pd.DataFrame, e) -> dict | None:
    si = int(e.signal_idx)
    ei = int(e.entry_idx)
    if si < 0 or ei < 0 or si >= len(x5) or ei >= len(x5):
        return None
    entry = float(x5.iloc[ei].open)
    stop = float(e.stop)
    side = int(e.side)
    if not (np.isfinite(entry) and entry > 0 and np.isfinite(stop)):
        return None
    risk_abs = side * (entry - stop)
    if not np.isfinite(risk_abs) or risk_abs <= 0:
        return None
    risk_bps = risk_abs / entry * 10000.0
    if risk_bps < 2 or risk_bps > 3000:
        return None
    target = entry + side * RR * risk_abs
    end_needed = ei + HOLD_BARS - 1
    last = len(x5) - 1
    end_scan = min(last, end_needed)
    reason = "OPEN"
    exit_idx = None
    exit_price = np.nan
    for i in range(ei, end_scan + 1):
        if side > 0:
            stop_hit = float(x5.iloc[i].low) <= stop
            target_hit = float(x5.iloc[i].high) >= target
        else:
            stop_hit = float(x5.iloc[i].high) >= stop
            target_hit = float(x5.iloc[i].low) <= target
        if stop_hit:
            reason, exit_idx, exit_price = "STOP", i, stop
            break
        if target_hit:
            reason, exit_idx, exit_price = "TARGET", i, target
            break
    if exit_idx is None and last >= end_needed:
        reason, exit_idx, exit_price = "TIME", end_needed, float(x5.iloc[end_needed].close)

    entry_time = pd.Timestamp(x5.iloc[ei].date)
    signal_time = pd.Timestamp(x5.iloc[si].signal_time)
    if exit_idx is None:
        exit_time = pd.NaT
        net8_r = np.nan
        stress12_r = np.nan
    else:
        exit_time = pd.Timestamp(x5.iloc[exit_idx].date) + pd.Timedelta(minutes=5)
        raw_bps = side * (float(exit_price) / entry - 1.0) * 10000.0
        net8_r = (raw_bps - BASE_COST_BPS) / risk_bps
        stress12_r = (raw_bps - STRESS_COST_BPS) / risk_bps

    return {
        "signal_time": signal_time,
        "entry_time": entry_time,
        "entry_price": entry,
        "side": side,
        "stop_price": stop,
        "target_price": target,
        "risk_bps": risk_bps,
        "status": "CLOSED" if exit_idx is not None else "OPEN",
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": reason,
        "net8_r": net8_r,
        "stress12_r": stress12_r,
    }


def compute_pair(pair: str, raw5: pd.DataFrame, raw15: pd.DataFrame, cutoff: pd.Timestamp, now: pd.Timestamp) -> list[dict]:
    warm_start = cutoff - pd.Timedelta(days=WARMUP_DAYS)
    raw5 = raw5[(raw5.date >= warm_start) & (raw5.date < now)].reset_index(drop=True)
    raw15 = raw15[(raw15.date >= warm_start) & (raw15.date < now)].reset_index(drop=True)
    if raw5.empty or raw15.empty:
        return []
    x15 = dc.prep_ohlcv(raw15, 15)
    x5 = v1._prep_exec(raw5)
    x5 = v1._add_activity(x5, v1._activity15(x15))

    tfs = {
        "15m": x15,
        "1h": dc.resample_from_15(x15, "1h", 60),
        "4h": dc.resample_from_15(x15, "4h", 240),
    }
    levels = []
    lid = 0
    for tf in LEVEL_TFS:
        for period in LEVEL_PERIODS:
            zz = dc.build_levels(tfs[tf], tf, period, lid)
            levels.extend(zz)
            lid += len(zz)
    if not levels:
        return []

    raw_events = __import__("digash_v31_events").detect_events(x5, levels)
    events = causal_dedup_events(raw_events)
    out = []
    for e in events:
        if e.setup != "H_FAKEOUT":
            continue
        outcome = _event_outcome(x5, e)
        if outcome is None or outcome["entry_time"] < cutoff:
            continue
        si = int(e.signal_idx)
        activity = float(x5.iloc[si].get("activity_score", np.nan))
        if not np.isfinite(activity) or activity < THRESH:
            continue
        if outcome["risk_bps"] < RISK_MIN_BPS:
            continue
        d = {
            "pair": pair,
            "tf": str(e.tf),
            "period": int(e.period),
            "level_price": float(e.level_price),
            "level_kind": str(e.level_kind),
            "approach_no": int(e.approach_no),
            "confluence_tfs": int(e.confluence_tfs),
            "touch_error_pct": float(e.touch_error_pct),
            "activity_score": activity,
            "natr_ratio30d": float(x5.iloc[si].get("natr_ratio30d", np.nan)),
            "qvol24_ratio30d": float(x5.iloc[si].get("qvol24_ratio30d", np.nan)),
            "stop_source": str(e.stop_source),
            **outcome,
        }
        d["signal_id"] = (
            f"{symbol(pair)}|{pd.Timestamp(d['signal_time']).isoformat()}|{d['side']}|"
            f"{d['tf']}|{d['period']}|{d['level_price']:.10g}"
        )
        out.append(d)
    return out


def attach_actual_funding(signals: pd.DataFrame, funding_by_pair: dict[str, pd.DataFrame], now: pd.Timestamp) -> pd.DataFrame:
    z = signals.copy()
    if z.empty:
        for c in ["funding_events", "funding_rate_sum", "actual_funding_cost_r", "net8_actual_funding_r", "stress12_actual_funding_r"]:
            z[c] = np.nan
        return z
    z["funding_events"] = 0
    z["funding_rate_sum"] = 0.0
    z["actual_funding_cost_r"] = 0.0
    for i, r in z.iterrows():
        f = funding_by_pair.get(r["pair"])
        if f is None or f.empty:
            continue
        end = pd.Timestamp(r["exit_time"]) if pd.notna(r["exit_time"]) else now
        q = f[(f.funding_time > pd.Timestamp(r["entry_time"])) & (f.funding_time <= end)]
        rate_sum = float(q.rate.sum()) if len(q) else 0.0
        cost_r = int(r["side"]) * rate_sum * 10000.0 / float(r["risk_bps"])
        z.at[i, "funding_events"] = int(len(q))
        z.at[i, "funding_rate_sum"] = rate_sum
        z.at[i, "actual_funding_cost_r"] = cost_r
    z["net8_actual_funding_r"] = pd.to_numeric(z["net8_r"], errors="coerce") - pd.to_numeric(z["actual_funding_cost_r"], errors="coerce")
    z["stress12_actual_funding_r"] = pd.to_numeric(z["stress12_r"], errors="coerce") - pd.to_numeric(z["actual_funding_cost_r"], errors="coerce")
    return z


def simulate_reference_portfolio(signals: pd.DataFrame, now: pd.Timestamp):
    if signals.empty:
        return {
            "start_balance": START_BALANCE, "realized_balance": START_BALANCE, "roi_pct": 0.0,
            "accepted": 0, "closed": 0, "open": 0, "skip_slots": 0, "skip_margin": 0,
            "skip_min_notional": 0, "skip_liq": 0, "maxdd_pct": 0.0, "reserved_margin": 0.0,
        }, pd.DataFrame()

    x = signals.sort_values(["entry_time", "pair", "signal_id"]).reset_index(drop=True)
    balance = START_BALANCE
    peak = START_BALANCE
    maxdd = 0.0
    reserved_margin = 0.0
    heap = []
    seq = 0
    decisions = []
    stats = {"accepted": 0, "closed": 0, "skip_slots": 0, "skip_margin": 0, "skip_min_notional": 0, "skip_liq": 0}

    def close_due(t: pd.Timestamp):
        nonlocal balance, peak, maxdd, reserved_margin
        while heap and heap[0][0] <= t.value:
            _, _, p = heapq.heappop(heap)
            reserved_margin -= p["margin"]
            if p["closed"]:
                balance += p["pnl"]
                stats["closed"] += 1
                peak = max(peak, balance)
                if peak > 0:
                    maxdd = max(maxdd, 1.0 - balance / peak)

    for r in x.itertuples(index=False):
        et = pd.Timestamp(r.entry_time)
        close_due(et)
        decision = {
            "signal_id": r.signal_id, "pair": r.pair, "entry_time": et,
            "decision": "ACCEPT", "reason": "", "balance_before": balance,
        }
        if len(heap) >= MAX_OPEN:
            stats["skip_slots"] += 1
            decision.update(decision="SKIP", reason="SLOT")
            decisions.append(decision)
            continue
        stop_frac = float(r.risk_bps) / 10000.0
        liq_buffer = 1.0 / LEVERAGE - MAINT_MARGIN_FRAC
        if liq_buffer <= 0 or stop_frac >= liq_buffer:
            stats["skip_liq"] += 1
            decision.update(decision="SKIP", reason="LIQ")
            decisions.append(decision)
            continue
        risk_amt = balance * RISK_PCT / 100.0
        notional = risk_amt / stop_frac
        if notional < MIN_NOTIONAL:
            stats["skip_min_notional"] += 1
            decision.update(decision="SKIP", reason="MIN_NOTIONAL")
            decisions.append(decision)
            continue
        margin = notional / LEVERAGE
        if margin > balance - reserved_margin + 1e-12:
            stats["skip_margin"] += 1
            decision.update(decision="SKIP", reason="MARGIN")
            decisions.append(decision)
            continue

        closed = str(r.status) == "CLOSED" and pd.notna(r.exit_time) and np.isfinite(float(r.net8_actual_funding_r))
        if closed:
            xt = pd.Timestamp(r.exit_time)
            pnl = risk_amt * float(r.net8_actual_funding_r)
        else:
            xt = now + pd.Timedelta(days=3650)
            pnl = 0.0
        heapq.heappush(heap, (xt.value, seq, {"margin": margin, "pnl": pnl, "closed": closed}))
        seq += 1
        reserved_margin += margin
        stats["accepted"] += 1
        decision.update({
            "risk_amt": risk_amt, "notional": notional, "margin": margin,
            "model_exit_time": pd.Timestamp(r.exit_time) if closed else pd.NaT,
            "model_pnl": pnl if closed else np.nan,
            "position_status": "CLOSED" if closed else "OPEN",
        })
        decisions.append(decision)

    close_due(now)
    result = {
        "start_balance": START_BALANCE,
        "realized_balance": float(balance),
        "roi_pct": float((balance / START_BALANCE - 1.0) * 100.0),
        "accepted": int(stats["accepted"]),
        "closed": int(stats["closed"]),
        "open": int(len(heap)),
        "skip_slots": int(stats["skip_slots"]),
        "skip_margin": int(stats["skip_margin"]),
        "skip_min_notional": int(stats["skip_min_notional"]),
        "skip_liq": int(stats["skip_liq"]),
        "maxdd_pct": float(maxdd * 100.0),
        "reserved_margin": float(reserved_margin),
    }
    return result, pd.DataFrame(decisions)


def render_summary(state: dict, now: pd.Timestamp, signals: pd.DataFrame, portfolio: dict) -> str:
    cutoff = pd.Timestamp(state["cutoff"])
    closed = signals[signals["status"].eq("CLOSED")].sort_values("entry_time") if not signals.empty else signals
    m8 = _metric(closed, "net8_actual_funding_r")
    m12 = _metric(closed, "stress12_actual_funding_r")
    open_n = int((signals["status"] == "OPEN").sum()) if not signals.empty else 0
    days = max((now - cutoff).total_seconds() / 86400.0, 1e-9)
    weeks = days / 7.0
    lines = [
        "=== PROSPECTIVE FAKEOUT V2 — FROZEN PAPER TRACKER ===",
        f"cutoff={cutoff.isoformat()} | data_through={now.isoformat()}",
        f"frozen: FAKEOUT activity>={THRESH} risk>={RISK_MIN_BPS:.0f}bps RR={RR:g} hold={HOLD_BARS}x5m causal_activity+causal_dedup",
        f"universe={len(FROZEN_PAIRS)} pairs | signals={len(signals)} closed={len(closed)} open={open_n} | signals/week={len(signals)/weeks:.2f}",
        "",
        "=== CLOSED SIGNAL PERFORMANCE (ACTUAL FUNDING) ===",
        f"8bps  N={m8['N']:4d} PF={m8['PF']:5.2f} WR={m8['WR']:5.1f}% EXP={m8['EXP']:+.3f}R DD={m8['DD']:6.1f}R",
        f"12bps N={m12['N']:4d} PF={m12['PF']:5.2f} WR={m12['WR']:5.1f}% EXP={m12['EXP']:+.3f}R DD={m12['DD']:6.1f}R",
        "",
        "=== $100 REFERENCE PAPER PORTFOLIO ===",
        f"balance=${portfolio['realized_balance']:.2f} ROI={portfolio['roi_pct']:+.1f}% DD={portfolio['maxdd_pct']:.1f}% "
        f"accepted={portfolio['accepted']} closed={portfolio['closed']} open={portfolio['open']}",
        f"skip(slot/margin/min/liq)={portfolio['skip_slots']}/{portfolio['skip_margin']}/"
        f"{portfolio['skip_min_notional']}/{portfolio['skip_liq']} | reserved_margin=${portfolio.get('reserved_margin',0.0):.2f}",
        "",
        "=== CHECKPOINT ===",
        f"closed={len(closed)}/50 first checkpoint | {len(closed)}/100 primary prospective checkpoint",
        "No pair/regime/time filter may be changed from these prospective results before the checkpoint.",
    ]
    return "\n".join(lines) + "\n"


def run_once(args, state: dict):
    outdir = Path(args.outdir)
    cfg = json.loads(Path(args.config).read_text())
    datadir = Path(args.datadir)
    cutoff = pd.Timestamp(state["cutoff"])
    now = utc_now().floor("s")
    all_rows = []
    coverage = []

    for n, pair in enumerate(FROZEN_PAIRS, 1):
        try:
            x5 = sync_tf(cfg, datadir, outdir, pair, "5m", now)
            x15 = sync_tf(cfg, datadir, outdir, pair, "15m", now)
            print(f"sync {n:2d}/{len(FROZEN_PAIRS)} {pair:24s} 5m={len(x5)} 15m={len(x15)}", flush=True)
        except Exception as e:
            coverage.append({"pair": pair, "status": f"SYNC_ERROR:{type(e).__name__}:{e}"})
            print(f"sync {n:2d}/{len(FROZEN_PAIRS)} {pair:24s} ERROR {type(e).__name__}: {e}", flush=True)

    bad = {r["pair"] for r in coverage if str(r.get("status", "")).startswith("SYNC_ERROR")}
    worker_pairs = [p for p in FROZEN_PAIRS if p not in bad]
    with ProcessPoolExecutor(max_workers=min(16, max(1, len(worker_pairs)))) as ex:
        futs = {
            ex.submit(compute_pair_worker, p, args.config, args.datadir, args.outdir, state["cutoff"], now.isoformat()): p
            for p in worker_pairs
        }
        done = 0
        for fut in as_completed(futs):
            pair = futs[fut]
            done += 1
            try:
                rows, meta = fut.result()
                all_rows.extend(rows)
                coverage.append(meta)
                print(f"scan {done:2d}/{len(worker_pairs)} {pair:24s} OK signals={len(rows)}", flush=True)
            except Exception as e:
                coverage.append({"pair": pair, "status": f"SCAN_ERROR:{type(e).__name__}:{e}"})
                print(f"scan {done:2d}/{len(worker_pairs)} {pair:24s} ERROR {type(e).__name__}: {e}", flush=True)

    funding = {}
    for pair in FROZEN_PAIRS:
        try:
            funding[pair] = fetch_funding(pair, cutoff - pd.Timedelta(hours=8), now)
        except Exception as e:
            funding[pair] = pd.DataFrame(columns=["funding_time", "rate"])
            print(f"funding {pair} ERROR {type(e).__name__}: {e}", flush=True)

    signals = pd.DataFrame(all_rows)
    if signals.empty:
        signals = pd.DataFrame(columns=[
            "signal_id", "pair", "signal_time", "entry_time", "status", "exit_time",
            "risk_bps", "net8_r", "stress12_r",
        ])
    else:
        signals["signal_time"] = pd.to_datetime(signals["signal_time"], utc=True)
        signals["entry_time"] = pd.to_datetime(signals["entry_time"], utc=True)
        signals["exit_time"] = pd.to_datetime(signals["exit_time"], utc=True)
        signals = signals.drop_duplicates("signal_id").sort_values(["entry_time", "pair"]).reset_index(drop=True)
    signals = attach_actual_funding(signals, funding, now)
    portfolio, decisions = simulate_reference_portfolio(signals, now)

    write_csv_atomic(signals, outdir / "signals.csv")
    write_csv_atomic(signals[signals["status"].eq("CLOSED")].copy(), outdir / "closed_trades.csv")
    write_csv_atomic(decisions, outdir / "portfolio_decisions.csv")
    write_csv_atomic(pd.DataFrame(coverage), outdir / "coverage.csv")
    summary = render_summary(state, now, signals, portfolio)
    (outdir / "summary.txt").write_text(summary)
    snapshot = {
        "cutoff": state["cutoff"], "data_through": now.isoformat(),
        "signals": int(len(signals)), "closed": int((signals["status"] == "CLOSED").sum()) if len(signals) else 0,
        "open": int((signals["status"] == "OPEN").sum()) if len(signals) else 0,
        "portfolio": portfolio,
    }
    (outdir / "snapshot.json").write_text(json.dumps(snapshot, indent=2, default=str))
    print("\n" + summary, flush=True)


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    state = init_state(outdir, args.reset_cutoff)
    print("=== PROSPECTIVE FAKEOUT V2 ===", flush=True)
    print(f"cutoff={state['cutoff']}", flush=True)
    print("This is paper-only. It sends no orders and does not alter the frozen signal.", flush=True)

    if not args.loop:
        run_once(args, state)
        return 0

    last_bucket = None
    while True:
        now = utc_now()
        bucket = now.floor("5min")
        if now.second >= 15 and bucket != last_bucket:
            try:
                run_once(args, state)
                last_bucket = bucket
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print(f"cycle ERROR {type(e).__name__}: {e}", flush=True)
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
