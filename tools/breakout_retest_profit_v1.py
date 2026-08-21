#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc
from digash_v31_events import detect_events, dedup_events

RRS = (1.0, 1.5, 2.0, 3.0)
HOLDS = (48, 144)
BASE_COST_BPS = 8.0
STRESS_COST_BPS = 12.0
WARMUP_DAYS = 60
LEVEL_TFS = ("15m", "1h", "4h")
LEVEL_PERIODS = (20, 30)
SETUPS = ("BREAK", "HOLD2", "RETEST", "FAKEOUT")
ACTIVITY_GATES = ("ALL", "ACTIVE_1P2", "ACTIVE_1P5")


def parse_args():
    p = argparse.ArgumentParser(description="Profit-first breakout/retest research V1")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/breakout_retest_profit_v1")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--risk-pct", type=float, default=1.0)
    return p.parse_args()


def _prep_exec(raw5: pd.DataFrame) -> pd.DataFrame:
    x = dc.prep_ohlcv(raw5, 5)
    q = x["volume"].astype(float) * x["close"].astype(float)
    x["volume_spike_local"] = q / q.shift(1).rolling(24, min_periods=12).mean().replace(0, np.nan)
    rng = x["high"].astype(float) - x["low"].astype(float)
    med = rng.shift(1).rolling(288, min_periods=72).median()
    mad = (rng.shift(1) - med).abs().rolling(288, min_periods=72).median()
    x["range_z"] = ((rng - med) / (1.4826 * mad.replace(0, np.nan))).clip(-8, 8)
    return x


def _activity15(x15: pd.DataFrame) -> pd.DataFrame:
    z = x15[["signal_time", "close", "volume", "atr"]].copy()
    close = z["close"].astype(float)
    natr = z["atr"].astype(float) / close.replace(0, np.nan)
    base_natr = natr.shift(1).rolling(96 * 30, min_periods=96 * 7).median()
    z["natr_ratio30d"] = natr / base_natr.replace(0, np.nan)
    q = z["volume"].astype(float) * close
    q24 = q.rolling(96, min_periods=48).sum()
    base_q = q24.shift(1).rolling(96 * 30, min_periods=96 * 7).median()
    z["qvol24_ratio30d"] = q24 / base_q.replace(0, np.nan)
    z["activity_score"] = z[["natr_ratio30d", "qvol24_ratio30d"]].max(axis=1, skipna=True)
    return z[["signal_time", "natr_ratio30d", "qvol24_ratio30d", "activity_score"]]


def _add_activity(x5: pd.DataFrame, a15: pd.DataFrame) -> pd.DataFrame:
    x = x5.copy()
    x["signal_time"] = pd.to_datetime(x["signal_time"], utc=True)
    a = a15.copy()
    a["signal_time"] = pd.to_datetime(a["signal_time"], utc=True)
    return pd.merge_asof(
        x.sort_values("signal_time"), a.sort_values("signal_time"),
        on="signal_time", direction="backward", tolerance=pd.Timedelta("30min")
    ).reset_index(drop=True)


def _hold2_from_break(e, x5: pd.DataFrame):
    if e.setup != "H_BREAK":
        return None
    j = int(e.signal_idx) + 1
    entry_idx = j + 1
    if entry_idx >= len(x5):
        return None
    c = x5["close"].to_numpy(float)
    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    atr = x5["atr"].to_numpy(float)
    level = float(e.level_price)
    side = int(e.side)
    if side > 0 and c[j] <= level:
        return None
    if side < 0 and c[j] >= level:
        return None
    a = float(atr[j]) if np.isfinite(atr[j]) and atr[j] > 0 else abs(level) * 0.001
    if side > 0:
        structural = float(np.min(l[max(0, j - 12):j + 1]))
        stop = structural if structural < level else level - 0.10 * a
    else:
        structural = float(np.max(h[max(0, j - 12):j + 1]))
        stop = structural if structural > level else level + 0.10 * a
    return {
        "setup": "HOLD2", "signal_idx": j, "entry_idx": entry_idx, "side": side,
        "stop": stop, "level_id": int(e.level_id), "level_price": level,
        "tf": str(e.tf), "period": int(e.period), "approach_no": int(e.approach_no),
        "confluence_tfs": int(e.confluence_tfs), "protor_proxy": bool(e.protor_proxy),
        "impulse_proxy": bool(e.impulse_proxy), "stop_source": "hold2_recent_structure",
    }


def _event_dict(e):
    setup_map = {"H_BREAK": "BREAK", "H_RETEST": "RETEST", "H_FAKEOUT": "FAKEOUT"}
    if e.setup not in setup_map:
        return None
    d = asdict(e)
    d["setup"] = setup_map[e.setup]
    return d


def _simulate_one(x5: pd.DataFrame, ev: dict, rr: float, hold_bars: int, cost_bps: float):
    ei = int(ev["entry_idx"])
    if ei < 0 or ei >= len(x5):
        return np.nan, "NO_ENTRY", np.nan, np.nan
    side = int(ev["side"])
    entry = float(x5.iloc[ei]["open"])
    stop = float(ev["stop"])
    if not (np.isfinite(entry) and np.isfinite(stop) and entry > 0):
        return np.nan, "BAD_PRICE", np.nan, np.nan
    risk_abs = side * (entry - stop)
    if not np.isfinite(risk_abs) or risk_abs <= 0:
        return np.nan, "BAD_STOP", np.nan, np.nan
    risk_bps = risk_abs / entry * 10000.0
    if risk_bps < 2.0 or risk_bps > 3000.0:
        return np.nan, "RISK_RANGE", risk_bps, np.nan
    target = entry + side * rr * risk_abs
    h = x5["high"].to_numpy(float)
    l = x5["low"].to_numpy(float)
    c = x5["close"].to_numpy(float)
    end = min(len(x5) - 1, ei + int(hold_bars) - 1)
    exit_price = float(c[end])
    reason = "TIME"
    exit_idx = end
    for i in range(ei, end + 1):
        if side > 0:
            stop_hit = l[i] <= stop
            target_hit = h[i] >= target
        else:
            stop_hit = h[i] >= stop
            target_hit = l[i] <= target
        if stop_hit:
            exit_price, reason, exit_idx = stop, "STOP", i
            break
        if target_hit:
            exit_price, reason, exit_idx = target, "TARGET", i
            break
    raw_bps = side * (exit_price / entry - 1.0) * 10000.0
    net_bps = raw_bps - float(cost_bps)
    net_r = net_bps / risk_bps
    return float(net_r), reason, float(risk_bps), int(exit_idx)


def process_pair(pair: str, cfg_path: str, datadir_s: str, start_s: str, end_s: str):
    t0 = time.monotonic()
    cfg = json.loads(Path(cfg_path).read_text())
    datadir = Path(datadir_s)
    start = pd.Timestamp(start_s, tz="UTC")
    end = pd.Timestamp(end_s, tz="UTC") + pd.Timedelta(days=1)
    warm = pd.Timedelta(days=WARMUP_DAYS)
    raw15 = dc.load_tf(cfg, datadir, pair, "15m")
    raw5, detail_source = dc.load_5m(cfg, datadir, pair)
    if raw15.empty or raw5.empty:
        return [], {"pair": pair, "status": "NO_DATA", "elapsed_s": time.monotonic() - t0}
    x15 = dc.prep_ohlcv(raw15, 15)
    x5 = _prep_exec(raw5)
    x15 = x15[(x15.date >= start - warm) & (x15.date < end + pd.Timedelta(hours=16))].reset_index(drop=True)
    x5 = x5[(x5.date >= start - warm) & (x5.date < end + pd.Timedelta(hours=16))].reset_index(drop=True)
    if x15.empty or x5.empty:
        return [], {"pair": pair, "status": "NO_RANGE", "elapsed_s": time.monotonic() - t0}
    x5 = _add_activity(x5, _activity15(x15))

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
        return [], {"pair": pair, "status": "NO_LEVELS", "elapsed_s": time.monotonic() - t0}

    raw_events = detect_events(x5, levels)
    events = dedup_events(raw_events)
    candidates = []
    for e in events:
        d = _event_dict(e)
        if d is not None:
            candidates.append(d)
        h2 = _hold2_from_break(e, x5)
        if h2 is not None:
            candidates.append(h2)

    rows = []
    for ev in candidates:
        ei = int(ev["entry_idx"])
        if ei >= len(x5):
            continue
        et = pd.Timestamp(x5.iloc[ei]["date"])
        if not (start <= et < end):
            continue
        row = {
            "pair": pair, "setup": ev["setup"], "entry_time": et,
            "side": int(ev["side"]), "tf": str(ev["tf"]), "period": int(ev["period"]),
            "level_price": float(ev["level_price"]), "approach_no": int(ev.get("approach_no", 0)),
            "confluence_tfs": int(ev.get("confluence_tfs", 1)),
            "protor_proxy": bool(ev.get("protor_proxy", False)),
            "impulse_proxy": bool(ev.get("impulse_proxy", False)),
            "stop_source": str(ev.get("stop_source", "")),
            "activity_score": float(x5.iloc[ei].get("activity_score", np.nan)),
            "natr_ratio30d": float(x5.iloc[ei].get("natr_ratio30d", np.nan)),
            "qvol24_ratio30d": float(x5.iloc[ei].get("qvol24_ratio30d", np.nan)),
        }
        first_risk = np.nan
        for rr in RRS:
            rtag = str(rr).replace(".", "p")
            for hb in HOLDS:
                rb, reason, risk_bps, _ = _simulate_one(x5, ev, rr, hb, BASE_COST_BPS)
                rs, _, _, _ = _simulate_one(x5, ev, rr, hb, STRESS_COST_BPS)
                row[f"r_{rtag}_h{hb}_b"] = rb
                row[f"r_{rtag}_h{hb}_s"] = rs
                row[f"x_{rtag}_h{hb}"] = reason
                if not np.isfinite(first_risk) and np.isfinite(risk_bps):
                    first_risk = risk_bps
        row["risk_bps"] = first_risk
        rows.append(row)
    meta = {
        "pair": pair, "status": "OK", "detail_source": detail_source,
        "bars5": len(x5), "bars15": len(x15), "levels": len(levels),
        "raw_events": len(raw_events), "dedup_events": len(events), "rows": len(rows),
        "elapsed_s": time.monotonic() - t0,
    }
    return rows, meta


def _gate_mask(df: pd.DataFrame, gate: str) -> pd.Series:
    if gate == "ALL":
        return pd.Series(True, index=df.index)
    x = pd.to_numeric(df["activity_score"], errors="coerce")
    thr = 1.2 if gate == "ACTIVE_1P2" else 1.5
    return x >= thr


def _metrics(df: pd.DataFrame, col: str, risk_pct: float):
    r = pd.to_numeric(df[col], errors="coerce").dropna()
    if r.empty:
        return {"N": 0, "PF": np.nan, "WR": np.nan, "EXP_R": np.nan, "DD_R": np.nan, "ROI_RISK": np.nan, "DD_EQ": np.nan}
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    wr = float((r > 0).mean() * 100.0)
    exp = float(r.mean())
    eq = 100.0
    peak = eq
    maxdd = 0.0
    curve = r.cumsum()
    dd_r = float((curve.cummax() - curve).max()) if len(curve) else 0.0
    f = float(risk_pct) / 100.0
    for v in r:
        eq *= max(0.001, 1.0 + f * float(v))
        peak = max(peak, eq)
        if peak > 0:
            maxdd = max(maxdd, 1.0 - eq / peak)
    return {
        "N": int(len(r)), "PF": float(pf), "WR": wr, "EXP_R": exp,
        "DD_R": dd_r, "ROI_RISK": float(eq - 100.0), "DD_EQ": float(maxdd * 100.0),
    }


def _split_bounds(start_s: str, end_s: str):
    a = pd.Timestamp(start_s, tz="UTC")
    b = pd.Timestamp(end_s, tz="UTC") + pd.Timedelta(days=1)
    span = b - a
    t1 = a + span * 0.60
    t2 = a + span * 0.80
    return a, t1, t2, b


def _split_name(t: pd.Series, bounds):
    _, t1, t2, _ = bounds
    return np.where(t < t1, "TRAIN", np.where(t < t2, "VALID", "HOLDOUT"))


def _configs():
    for setup in SETUPS:
        for gate in ACTIVITY_GATES:
            for rr in RRS:
                for hb in HOLDS:
                    yield setup, gate, rr, hb


def _eval_configs(df: pd.DataFrame, risk_pct: float):
    out = []
    for setup, gate, rr, hb in _configs():
        rtag = str(rr).replace(".", "p")
        cb = f"r_{rtag}_h{hb}_b"
        cs = f"r_{rtag}_h{hb}_s"
        base = df[df.setup.eq(setup) & _gate_mask(df, gate)]
        for split in ("TRAIN", "VALID", "HOLDOUT"):
            g = base[base.split.eq(split)].sort_values("entry_time")
            mb = _metrics(g, cb, risk_pct)
            ms = _metrics(g, cs, risk_pct)
            weeks = max((g.entry_time.max() - g.entry_time.min()).total_seconds() / (7 * 86400), 1.0) if len(g) > 1 else np.nan
            out.append({
                "setup": setup, "gate": gate, "rr": rr, "hold_bars": hb, "split": split,
                **{f"BASE_{k}": v for k, v in mb.items()},
                **{f"STRESS_{k}": v for k, v in ms.items()},
                "TRADES_WEEK": float(mb["N"] / weeks) if np.isfinite(weeks) else np.nan,
            })
    return pd.DataFrame(out)


def _select(eval_df: pd.DataFrame):
    tr = eval_df[eval_df.split.eq("TRAIN")].copy()
    tr = tr[(tr.BASE_N >= 100) & np.isfinite(tr.BASE_PF) & (tr.BASE_EXP_R > 0)].copy()
    if tr.empty:
        return None, pd.DataFrame()
    tr["train_score"] = tr.BASE_EXP_R * np.sqrt(np.minimum(tr.BASE_N, 1500)) * np.minimum(tr.BASE_PF, 3.0)
    top = tr.sort_values(["train_score", "BASE_PF"], ascending=False).head(12)
    keys = ["setup", "gate", "rr", "hold_bars"]
    val = eval_df[eval_df.split.eq("VALID")].merge(top[keys + ["train_score"]], on=keys, how="inner")
    val = val[(val.BASE_N >= 30) & np.isfinite(val.BASE_PF)].copy()
    if val.empty:
        chosen = top.iloc[0][keys].to_dict()
        return chosen, top
    val["val_score"] = val.BASE_EXP_R * np.sqrt(np.minimum(val.BASE_N, 750)) * np.minimum(val.BASE_PF, 3.0)
    val = val.sort_values(["val_score", "BASE_PF"], ascending=False)
    chosen = val.iloc[0][keys].to_dict()
    return chosen, top


def _fmt_metric(r):
    return f"N={int(r.BASE_N):5d} PF={r.BASE_PF:5.2f} WR={r.BASE_WR:5.1f}% EXP={r.BASE_EXP_R:+.3f}R DD={r.BASE_DD_R:6.1f}R ROI1%={r.BASE_ROI_RISK:+7.1f}% | stress PF={r.STRESS_PF:5.2f} EXP={r.STRESS_EXP_R:+.3f}R"


def main() -> int:
    a = parse_args()
    cfg = json.loads(Path(a.config).read_text())
    pairs = list(cfg.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist in config")
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    print("=== BREAKOUT / RETEST PROFIT RESEARCH V1 ===", flush=True)
    print("PROFIT-FIRST. Digash is only an idea source; no Telegram/source-fidelity comparison.", flush=True)
    print("CACHE ONLY. 15m/1h/4h horizontal levels; 5m execution; structural stops; next-open entries.", flush=True)
    print("Setups: BREAK, HOLD2, RETEST, FAKEOUT. RR=1/1.5/2/3. Max hold=4h/12h. Costs=8bps base / 12bps stress.", flush=True)
    print("Chronological 60/20/20 TRAIN/VALID/HOLDOUT. Config chosen on TRAIN then VALID; HOLDOUT untouched.", flush=True)
    print(f"pairs={len(pairs)} workers={min(a.workers, len(pairs))} start={a.start} end={a.end}", flush=True)

    rows, metas = [], []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=max(1, min(a.workers, len(pairs)))) as ex:
        futs = {ex.submit(process_pair, p, a.config, a.datadir, a.start, a.end): p for p in pairs}
        done = 0
        for f in as_completed(futs):
            pair = futs[f]
            try:
                rr, meta = f.result()
            except Exception as e:
                rr, meta = [], {"pair": pair, "status": "ERROR", "error": f"{type(e).__name__}: {e}"}
            rows.extend(rr)
            metas.append(meta)
            done += 1
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(futs) - done) / done if done else np.nan
            print(f"[{done:2d}/{len(futs)}] {pair:24s} {meta.get('status')} trades={len(rr):6d} levels={meta.get('levels','-')} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(outdir / "coverage.csv", index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        print("No trades generated.", flush=True)
        return 2
    df["entry_time"] = pd.to_datetime(df.entry_time, utc=True)
    bounds = _split_bounds(a.start, a.end)
    df["split"] = _split_name(df.entry_time, bounds)
    df = df.sort_values(["entry_time", "pair", "setup"]).reset_index(drop=True)
    df.to_csv(outdir / "base_trades.csv", index=False)

    ev = _eval_configs(df, a.risk_pct)
    ev.to_csv(outdir / "config_metrics.csv", index=False)
    chosen, top = _select(ev)

    print("\n=== DATA / SPLIT SANITY ===", flush=True)
    print(f"base rows={len(df):,} | pairs={df.pair.nunique()} | TRAIN={sum(df.split.eq('TRAIN')):,} VALID={sum(df.split.eq('VALID')):,} HOLDOUT={sum(df.split.eq('HOLDOUT')):,}", flush=True)
    print(f"split1={bounds[1]} | split2={bounds[2]}", flush=True)
    print("setups: " + ", ".join(f"{k}={v}" for k, v in df.setup.value_counts().items()), flush=True)
    print("risk bps median/p90: " + f"{pd.to_numeric(df.risk_bps, errors='coerce').median():.1f}/{pd.to_numeric(df.risk_bps, errors='coerce').quantile(.9):.1f}", flush=True)

    print("\n=== TRAIN TOP CANDIDATES ===", flush=True)
    if top.empty:
        print("No positive TRAIN candidate with N>=100. Strategy family fails before holdout selection.", flush=True)
    else:
        for r in top.head(8).itertuples(index=False):
            print(f"{r.setup:7s} {r.gate:11s} RR={r.rr:<3} hold={int(r.hold_bars):3d} | {_fmt_metric(r)}", flush=True)

    print("\n=== SELECTED CONFIG: TRAIN -> VALID -> UNTOUCHED HOLDOUT ===", flush=True)
    if chosen is None:
        print("NO_SELECTION", flush=True)
    else:
        print("selected=" + " ".join(f"{k}={v}" for k, v in chosen.items()), flush=True)
        mask = pd.Series(True, index=ev.index)
        for k, v in chosen.items():
            mask &= ev[k].eq(v)
        zz = ev[mask].sort_values("split", key=lambda s: s.map({"TRAIN": 0, "VALID": 1, "HOLDOUT": 2}))
        for r in zz.itertuples(index=False):
            print(f"{r.split:7s} | {_fmt_metric(r)} | trades/week={r.TRADES_WEEK:.2f}", flush=True)

        sel_df = df[df.setup.eq(chosen["setup"]) & _gate_mask(df, chosen["gate"])].copy()
        rr = float(chosen["rr"])
        hb = int(chosen["hold_bars"])
        rtag = str(rr).replace(".", "p")
        col = f"r_{rtag}_h{hb}_b"
        print("\n=== HOLDOUT ROBUSTNESS / CONCENTRATION ===", flush=True)
        h = sel_df[sel_df.split.eq("HOLDOUT")].copy()
        if len(h):
            per_pair = []
            for pair, g in h.groupby("pair"):
                m = _metrics(g.sort_values("entry_time"), col, a.risk_pct)
                per_pair.append((pair, m["N"], m["PF"], m["EXP_R"]))
            per_pair.sort(key=lambda x: x[1], reverse=True)
            print("top pair counts: " + ", ".join(f"{p}:{n}" for p, n, _, _ in per_pair[:8]), flush=True)
            n_total = sum(x[1] for x in per_pair)
            top_share = sum(x[1] for x in per_pair[:3]) / n_total * 100 if n_total else np.nan
            print(f"top3 pair trade-share={top_share:.1f}%", flush=True)
            for year, g in h.groupby(h.entry_time.dt.year):
                m = _metrics(g.sort_values("entry_time"), col, a.risk_pct)
                print(f"year {year}: N={m['N']} PF={m['PF']:.2f} WR={m['WR']:.1f}% EXP={m['EXP_R']:+.3f}R", flush=True)

    print("\n=== DECISION RULE ===", flush=True)
    print("PROMOTE only if untouched HOLDOUT is positive after 8bps, stress 12bps remains near/above breakeven, sample size is adequate, and results are not dominated by a few pairs.", flush=True)
    print("If RETEST/HOLD2 beats immediate BREAK out-of-sample, keep developing this same strategy family. If all fail, reject this level/breakout implementation instead of changing to an unrelated strategy.", flush=True)
    print(f"Reports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
