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

import breakout_retest_profit_v1 as v1
import breakout_retest_profit_v16 as v16
import digash_v3_common as dc
import digash_v31_events as de


RRS = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
HOLDS = (12, 24, 48, 96)
ACTIVITY_MINS = (1.0, 1.25, 1.5, 1.75, 2.0, 2.5)
RISK_MINS_BPS = (40.0, 80.0, 120.0, 160.0, 240.0, 320.0)
RISK_PCTS = (1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0)
BASE_COST_BPS = 8.0
STRESS_COST_BPS = 12.0
WARMUP_DAYS = 60
LEVEL_TFS = ("15m", "1h", "4h")
LEVEL_PERIODS = (20, 30)
LEVERAGE = 10.0
MAINT_MARGIN_FRAC = 0.005
MAX_OPEN = 3
MIN_NOTIONAL = 5.0
START_EQUITY = 100.0

SETUP_GROUPS = {
    "BREAK": ("BREAK",),
    "HOLD2": ("HOLD2",),
    "RETEST": ("RETEST",),
    "FAKEOUT": ("FAKEOUT",),
    "BOUNCE": ("BOUNCE",),
    "REVERSALS": ("FAKEOUT", "BOUNCE"),
    "CONTINUATION": ("BREAK", "HOLD2", "RETEST"),
    "RETEST_FAKEOUT": ("RETEST", "FAKEOUT"),
}

TF_MODES = ("ALL", "15m", "1h", "4h", "HTF")
CONFLUENCE_MINS = (1, 2)
IMPULSE_MODES = ("ANY", "TRUE")
APPROACH_MINS = (0, 2)

TRAIN_START = pd.Timestamp("2022-01-01", tz="UTC")
TRAIN_END = pd.Timestamp("2025-01-01", tz="UTC")
VALID_END = pd.Timestamp("2026-01-01", tz="UTC")


def parse_args():
    p = argparse.ArgumentParser(description="High-ROI causal level-event research lab")
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


def _event_dict(e) -> dict | None:
    setup_map = {
        "H_BREAK": "BREAK",
        "H_RETEST": "RETEST",
        "H_FAKEOUT": "FAKEOUT",
        "H_BOUNCE": "BOUNCE",
    }
    if e.setup not in setup_map:
        return None
    d = asdict(e)
    d["setup"] = setup_map[e.setup]
    return d


def _risk_bps(x5: pd.DataFrame, ev: dict) -> float:
    ei = int(ev["entry_idx"])
    if ei < 0 or ei >= len(x5):
        return np.nan
    entry = float(x5.iloc[ei]["open"])
    stop = float(ev["stop"])
    side = int(ev["side"])
    risk_abs = side * (entry - stop)
    if not (np.isfinite(entry) and entry > 0 and np.isfinite(risk_abs) and risk_abs > 0):
        return np.nan
    return float(risk_abs / entry * 10000.0)


def _simulate_grid(x5: pd.DataFrame, ev: dict) -> dict:
    out = {}
    for rr in RRS:
        rt = str(rr).replace(".", "p")
        for hb in HOLDS:
            rb, _, _, exit_idx = v1._simulate_one(x5, ev, rr, hb, BASE_COST_BPS)
            rs, _, _, _ = v1._simulate_one(x5, ev, rr, hb, STRESS_COST_BPS)
            out[f"r_b_{rt}_h{hb}"] = rb
            out[f"r_s_{rt}_h{hb}"] = rs
            out[f"exit_{rt}_h{hb}"] = (
                pd.Timestamp(x5.iloc[int(exit_idx)]["signal_time"])
                if exit_idx is not None and np.isfinite(exit_idx) and 0 <= int(exit_idx) < len(x5)
                else pd.NaT
            )
    return out


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
    x5 = v1._prep_exec(raw5)
    x15 = x15[(x15.date >= start - warm) & (x15.date < end + pd.Timedelta(hours=16))].reset_index(drop=True)
    x5 = x5[(x5.date >= start - warm) & (x5.date < end + pd.Timedelta(hours=16))].reset_index(drop=True)
    if x15.empty or x5.empty:
        return [], {"pair": pair, "status": "NO_RANGE", "elapsed_s": time.monotonic() - t0}

    # Same entry-time causal activity timing as V1.6.
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
        return [], {"pair": pair, "status": "NO_LEVELS", "elapsed_s": time.monotonic() - t0}

    raw_events = de.detect_events(x5, levels)
    causal = v16.causal_dedup_events(raw_events)
    candidates: list[dict] = []
    for e in causal:
        d = _event_dict(e)
        if d is not None:
            candidates.append(d)
        if e.setup == "H_BREAK":
            h2 = v1._hold2_from_break(e, x5)
            if h2 is not None:
                candidates.append(h2)

    rows = []
    for ev in candidates:
        si = int(ev["signal_idx"])
        ei = int(ev["entry_idx"])
        if si < 0 or ei < 0 or si >= len(x5) or ei >= len(x5):
            continue
        entry_time = pd.Timestamp(x5.iloc[ei]["date"])
        if not (start <= entry_time < end):
            continue
        risk_bps = _risk_bps(x5, ev)
        if not np.isfinite(risk_bps) or risk_bps < 2.0 or risk_bps > 3000.0:
            continue

        # Critical: use the signal bar. Its signal_time equals intended entry time.
        # Using entry_idx here would leak one extra 5m bar of activity information.
        feat = x5.iloc[si]
        row = {
            "pair": pair,
            "setup": str(ev["setup"]),
            "signal_time": pd.Timestamp(x5.iloc[si]["signal_time"]),
            "entry_time": entry_time,
            "side": int(ev["side"]),
            "tf": str(ev["tf"]),
            "tf_minutes": int(dc.TF_MINUTES[str(ev["tf"])]),
            "period": int(ev["period"]),
            "level_price": float(ev["level_price"]),
            "level_id": int(ev.get("level_id", -1)),
            "approach_no": int(ev.get("approach_no", 0)),
            "confluence_tfs": int(ev.get("confluence_tfs", 1)),
            "touch_error_pct": float(ev.get("touch_error_pct", np.nan)),
            "protor_proxy": bool(ev.get("protor_proxy", False)),
            "impulse_proxy": bool(ev.get("impulse_proxy", False)),
            "near_bars_6": int(ev.get("near_bars_6", 0)),
            "reclaim_bars": int(ev.get("reclaim_bars", 0)),
            "stop_source": str(ev.get("stop_source", "")),
            "activity_score": float(feat.get("activity_score", np.nan)),
            "natr_ratio30d": float(feat.get("natr_ratio30d", np.nan)),
            "qvol24_ratio30d": float(feat.get("qvol24_ratio30d", np.nan)),
            "risk_bps": float(risk_bps),
        }
        row.update(_simulate_grid(x5, ev))
        rows.append(row)

    meta = {
        "pair": pair,
        "status": "OK",
        "detail_source": detail_source,
        "bars5": len(x5),
        "bars15": len(x15),
        "levels": len(levels),
        "raw_events": len(raw_events),
        "causal_events": len(causal),
        "candidates": len(candidates),
        "rows": len(rows),
        "elapsed_s": round(time.monotonic() - t0, 3),
    }
    return rows, meta


def _metric(g: pd.DataFrame, col: str) -> dict:
    r = pd.to_numeric(g[col], errors="coerce").dropna().astype(float)
    if r.empty:
        return {"n": 0, "pf": np.nan, "wr": np.nan, "exp": np.nan, "dd_r": np.nan}
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    curve = r.cumsum()
    dd = float((curve.cummax() - curve).max()) if len(curve) else 0.0
    return {
        "n": int(len(r)),
        "pf": float(pf),
        "wr": float((r > 0).mean() * 100.0),
        "exp": float(r.mean()),
        "dd_r": dd,
    }


def _split(df: pd.DataFrame, name: str) -> pd.DataFrame:
    t = df["entry_time"]
    if name == "TRAIN":
        return df[(t >= TRAIN_START) & (t < TRAIN_END)]
    if name == "VALID":
        return df[(t >= TRAIN_END) & (t < VALID_END)]
    return df[t >= VALID_END]


def _months_in(g: pd.DataFrame) -> int:
    if g.empty:
        return 1
    return max(1, int(g["entry_time"].dt.tz_localize(None).dt.to_period("M").nunique()))


def _base_filter(df: pd.DataFrame, setup_group: str, activity_min: float, risk_min: float) -> pd.DataFrame:
    setups = SETUP_GROUPS[setup_group]
    liq_max_bps = (1.0 / LEVERAGE - MAINT_MARGIN_FRAC) * 10000.0
    x = df[
        df["setup"].isin(setups)
        & (pd.to_numeric(df["activity_score"], errors="coerce") >= float(activity_min))
        & (pd.to_numeric(df["risk_bps"], errors="coerce") >= float(risk_min))
        & (pd.to_numeric(df["risk_bps"], errors="coerce") < liq_max_bps)
    ].copy()
    return x


def _struct_filter(g: pd.DataFrame, tf_mode: str, conf_min: int, impulse_mode: str, approach_min: int) -> pd.DataFrame:
    x = g
    if tf_mode == "15m":
        x = x[x["tf"].eq("15m")]
    elif tf_mode == "1h":
        x = x[x["tf"].eq("1h")]
    elif tf_mode == "4h":
        x = x[x["tf"].eq("4h")]
    elif tf_mode == "HTF":
        x = x[x["tf"].isin(("1h", "4h"))]
    x = x[pd.to_numeric(x["confluence_tfs"], errors="coerce") >= int(conf_min)]
    if impulse_mode == "TRUE":
        x = x[x["impulse_proxy"].astype(bool)]
    if approach_min > 0:
        x = x[pd.to_numeric(x["approach_no"], errors="coerce") >= int(approach_min)]
    return x.copy()


def _outcome_cols(rr: float, hb: int) -> tuple[str, str, str]:
    rt = str(float(rr)).replace(".", "p")
    return f"r_b_{rt}_h{hb}", f"r_s_{rt}_h{hb}", f"exit_{rt}_h{hb}"


def stage1(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    months = _months_in(train)
    for setup_group in SETUP_GROUPS:
        for activity_min in ACTIVITY_MINS:
            for risk_min in RISK_MINS_BPS:
                g0 = _base_filter(train, setup_group, activity_min, risk_min)
                if len(g0) < 48:
                    continue
                for rr in RRS:
                    for hb in HOLDS:
                        cb, cs, _ = _outcome_cols(rr, hb)
                        mb = _metric(g0, cb)
                        ms = _metric(g0, cs)
                        if ms["n"] < 48 or not np.isfinite(ms["exp"]):
                            continue
                        rpm = float(ms["exp"] * ms["n"] / months)
                        rows.append({
                            "setup_group": setup_group,
                            "activity_min": activity_min,
                            "risk_min_bps": risk_min,
                            "rr": rr,
                            "hold_bars": hb,
                            "n": ms["n"],
                            "pf8": mb["pf"],
                            "pf12": ms["pf"],
                            "wr12": ms["wr"],
                            "exp8": mb["exp"],
                            "exp12": ms["exp"],
                            "dd_r12": ms["dd_r"],
                            "stress_r_per_month": rpm,
                        })
    z = pd.DataFrame(rows)
    if z.empty:
        return z
    z = z[(z["exp12"] > 0) & (z["pf12"] > 1.0)].copy()
    return z.sort_values(["stress_r_per_month", "pf12", "n"], ascending=[False, False, False]).reset_index(drop=True)


def stage2(train: pd.DataFrame, first: pd.DataFrame, top_n: int = 40) -> pd.DataFrame:
    rows = []
    months = _months_in(train)
    for r in first.head(top_n).itertuples(index=False):
        g0 = _base_filter(train, r.setup_group, r.activity_min, r.risk_min_bps)
        cb, cs, _ = _outcome_cols(r.rr, r.hold_bars)
        for tf_mode in TF_MODES:
            for conf_min in CONFLUENCE_MINS:
                for impulse_mode in IMPULSE_MODES:
                    for approach_min in APPROACH_MINS:
                        g = _struct_filter(g0, tf_mode, conf_min, impulse_mode, approach_min)
                        ms = _metric(g, cs)
                        if ms["n"] < 36 or not np.isfinite(ms["exp"]) or ms["exp"] <= 0:
                            continue
                        mb = _metric(g, cb)
                        rows.append({
                            "setup_group": r.setup_group,
                            "activity_min": float(r.activity_min),
                            "risk_min_bps": float(r.risk_min_bps),
                            "rr": float(r.rr),
                            "hold_bars": int(r.hold_bars),
                            "tf_mode": tf_mode,
                            "confluence_min": int(conf_min),
                            "impulse_mode": impulse_mode,
                            "approach_min": int(approach_min),
                            "n": ms["n"],
                            "pf8": mb["pf"],
                            "pf12": ms["pf"],
                            "wr12": ms["wr"],
                            "exp8": mb["exp"],
                            "exp12": ms["exp"],
                            "dd_r12": ms["dd_r"],
                            "stress_r_per_month": float(ms["exp"] * ms["n"] / months),
                        })
    z = pd.DataFrame(rows)
    if z.empty:
        return z
    z = z.drop_duplicates([
        "setup_group", "activity_min", "risk_min_bps", "rr", "hold_bars",
        "tf_mode", "confluence_min", "impulse_mode", "approach_min",
    ])
    return z.sort_values(["stress_r_per_month", "pf12", "n"], ascending=[False, False, False]).reset_index(drop=True)


def _candidate_events(df: pd.DataFrame, c: dict) -> pd.DataFrame:
    g = _base_filter(df, c["setup_group"], c["activity_min"], c["risk_min_bps"])
    g = _struct_filter(g, c["tf_mode"], int(c["confluence_min"]), c["impulse_mode"], int(c["approach_min"]))
    return g.sort_values(["entry_time", "pair", "setup"]).reset_index(drop=True)


def _quality_tuple(r) -> tuple:
    return (
        -float(getattr(r, "activity_score", 0.0) if np.isfinite(getattr(r, "activity_score", np.nan)) else 0.0),
        -int(getattr(r, "confluence_tfs", 1)),
        -int(getattr(r, "tf_minutes", 0)),
        float(getattr(r, "touch_error_pct", 999.0) if np.isfinite(getattr(r, "touch_error_pct", np.nan)) else 999.0),
        str(getattr(r, "pair", "")),
        str(getattr(r, "setup", "")),
    )


def _monthly_stats(points: list[tuple[pd.Timestamp, float]], start_equity: float) -> dict:
    if not points:
        return {
            "mean_monthly_roi": 0.0, "median_monthly_roi": 0.0, "positive_months_pct": 0.0,
            "months_ge_50_pct": 0.0, "min_monthly_roi": 0.0, "max_monthly_roi": 0.0, "months": 0,
        }
    points = sorted(points, key=lambda z: z[0])
    start_p = points[0][0].tz_localize(None).to_period("M")
    end_p = points[-1][0].tz_localize(None).to_period("M")
    periods = pd.period_range(start_p, end_p, freq="M")
    rois = []
    prev = float(start_equity)
    j = 0
    last = float(start_equity)
    for p in periods:
        end_ts = pd.Timestamp(p.end_time, tz="UTC")
        while j < len(points) and points[j][0] <= end_ts:
            last = float(points[j][1])
            j += 1
        roi = (last / prev - 1.0) * 100.0 if prev > 0 else -100.0
        rois.append(float(roi))
        prev = last
    a = np.asarray(rois, dtype=float)
    return {
        "mean_monthly_roi": float(np.mean(a)) if len(a) else 0.0,
        "median_monthly_roi": float(np.median(a)) if len(a) else 0.0,
        "positive_months_pct": float(np.mean(a > 0) * 100.0) if len(a) else 0.0,
        "months_ge_50_pct": float(np.mean(a >= 50.0) * 100.0) if len(a) else 0.0,
        "min_monthly_roi": float(np.min(a)) if len(a) else 0.0,
        "max_monthly_roi": float(np.max(a)) if len(a) else 0.0,
        "months": int(len(a)),
    }


def simulate_portfolio(g: pd.DataFrame, rr: float, hb: int, risk_pct: float, stress: bool = True) -> dict:
    cb, cs, ce = _outcome_cols(rr, hb)
    rcol = cs if stress else cb
    if g.empty:
        return {"final_equity": START_EQUITY, "roi_pct": 0.0, "accepted": 0, "max_dd_pct": 0.0, **_monthly_stats([], START_EQUITY)}

    z = g.dropna(subset=[rcol, ce, "risk_bps", "entry_time"]).copy()
    z[ce] = pd.to_datetime(z[ce], utc=True, errors="coerce")
    z = z.dropna(subset=[ce]).sort_values(["entry_time", "pair", "setup"])
    if z.empty:
        return {"final_equity": START_EQUITY, "roi_pct": 0.0, "accepted": 0, "max_dd_pct": 0.0, **_monthly_stats([], START_EQUITY)}

    equity = float(START_EQUITY)
    peak = equity
    max_dd = 0.0
    max_open_risk_pct = 0.0
    max_margin_use_pct = 0.0
    open_pos: list[dict] = []
    points: list[tuple[pd.Timestamp, float]] = []
    accepted = slot_skips = margin_skips = pair_skips = liq_skips = min_notional_skips = 0
    liq_room = 1.0 / LEVERAGE - MAINT_MARGIN_FRAC

    def close_due(t: pd.Timestamp) -> None:
        nonlocal equity, peak, max_dd, open_pos
        due = sorted((p for p in open_pos if p["exit_time"] <= t), key=lambda p: p["exit_time"])
        if not due:
            return
        keep = [p for p in open_pos if p["exit_time"] > t]
        for p in due:
            equity += float(p["risk_cash"] * p["r"])
            equity = max(0.0, equity)
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, 1.0 - equity / peak)
            points.append((p["exit_time"], equity))
        open_pos = keep

    for t, batch in z.groupby("entry_time", sort=True):
        t = pd.Timestamp(t)
        close_due(t)
        if equity <= 0:
            break
        batch_rows = sorted(batch.itertuples(index=False), key=_quality_tuple)
        for r in batch_rows:
            pair = str(r.pair)
            if any(p["pair"] == pair for p in open_pos):
                pair_skips += 1
                continue
            if len(open_pos) >= MAX_OPEN:
                slot_skips += 1
                continue
            rbps = float(r.risk_bps)
            risk_frac = rbps / 10000.0
            if not np.isfinite(risk_frac) or risk_frac <= 0 or risk_frac >= liq_room:
                liq_skips += 1
                continue
            risk_cash = equity * float(risk_pct) / 100.0
            notional = risk_cash / risk_frac
            if notional < MIN_NOTIONAL:
                min_notional_skips += 1
                continue
            collateral = notional / LEVERAGE
            used = sum(float(p["collateral"]) for p in open_pos)
            if used + collateral > equity:
                margin_skips += 1
                continue
            rv = float(getattr(r, rcol))
            exit_time = pd.Timestamp(getattr(r, ce))
            open_pos.append({
                "pair": pair, "exit_time": exit_time, "risk_cash": risk_cash,
                "collateral": collateral, "r": rv,
            })
            accepted += 1
            used_now = used + collateral
            open_risk = sum(float(p["risk_cash"]) for p in open_pos)
            if equity > 0:
                max_margin_use_pct = max(max_margin_use_pct, used_now / equity * 100.0)
                max_open_risk_pct = max(max_open_risk_pct, open_risk / equity * 100.0)
                worst_eq = max(0.0, equity - open_risk)
                max_dd = max(max_dd, 1.0 - worst_eq / max(peak, 1e-12))

    if open_pos:
        for p in sorted(open_pos, key=lambda p: p["exit_time"]):
            equity += float(p["risk_cash"] * p["r"])
            equity = max(0.0, equity)
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, 1.0 - equity / peak)
            points.append((p["exit_time"], equity))

    monthly = _monthly_stats(points, START_EQUITY)
    return {
        "final_equity": float(equity),
        "roi_pct": float((equity / START_EQUITY - 1.0) * 100.0),
        "accepted": int(accepted),
        "slot_skips": int(slot_skips),
        "margin_skips": int(margin_skips),
        "pair_skips": int(pair_skips),
        "liq_skips": int(liq_skips),
        "min_notional_skips": int(min_notional_skips),
        "max_dd_pct": float(max_dd * 100.0),
        "max_open_risk_pct": float(max_open_risk_pct),
        "max_margin_use_pct": float(max_margin_use_pct),
        **monthly,
    }


def _portfolio_score(m: dict) -> float:
    if m.get("accepted", 0) <= 0 or m.get("final_equity", 0) <= 0:
        return -1e9
    return (
        float(m["median_monthly_roi"])
        + 0.25 * float(m["mean_monthly_roi"])
        + 0.10 * float(m["positive_months_pct"])
        - 0.35 * float(m["max_dd_pct"])
    )


def _candidate_dict(r) -> dict:
    return {
        "setup_group": str(r.setup_group),
        "activity_min": float(r.activity_min),
        "risk_min_bps": float(r.risk_min_bps),
        "rr": float(r.rr),
        "hold_bars": int(r.hold_bars),
        "tf_mode": str(r.tf_mode),
        "confluence_min": int(r.confluence_min),
        "impulse_mode": str(r.impulse_mode),
        "approach_min": int(r.approach_min),
    }


def portfolio_train(train: pd.DataFrame, second: pd.DataFrame, top_n: int = 120) -> pd.DataFrame:
    rows = []
    for rank, r in enumerate(second.head(top_n).itertuples(index=False), 1):
        c = _candidate_dict(r)
        g = _candidate_events(train, c)
        for risk_pct in RISK_PCTS:
            ms = simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=True)
            mb = simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=False)
            rows.append({
                **c,
                "risk_pct": float(risk_pct),
                "train_event_rank": rank,
                "train_pf12": float(r.pf12),
                "train_exp12": float(r.exp12),
                "train_r_per_month": float(r.stress_r_per_month),
                "train_final_stress": ms["final_equity"],
                "train_roi_stress": ms["roi_pct"],
                "train_median_monthly_stress": ms["median_monthly_roi"],
                "train_mean_monthly_stress": ms["mean_monthly_roi"],
                "train_positive_months_stress": ms["positive_months_pct"],
                "train_months_ge50_stress": ms["months_ge_50_pct"],
                "train_maxdd_stress": ms["max_dd_pct"],
                "train_accepted": ms["accepted"],
                "train_max_margin_use": ms["max_margin_use_pct"],
                "train_final_base": mb["final_equity"],
                "train_median_monthly_base": mb["median_monthly_roi"],
                "train_score": _portfolio_score(ms),
            })
    z = pd.DataFrame(rows)
    if z.empty:
        return z
    viable = z[
        (z["train_accepted"] >= 36)
        & (z["train_positive_months_stress"] >= 50.0)
        & (z["train_maxdd_stress"] <= 65.0)
        & (z["train_final_stress"] > START_EQUITY)
    ].copy()
    if viable.empty:
        viable = z.copy()
    return viable.sort_values(["train_score", "train_median_monthly_stress"], ascending=False).reset_index(drop=True)


def evaluate_validation(valid: pd.DataFrame, train_port: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    rows = []
    for r in train_port.head(top_n).itertuples(index=False):
        c = _candidate_dict(r)
        g = _candidate_events(valid, c)
        ms = simulate_portfolio(g, c["rr"], c["hold_bars"], float(r.risk_pct), stress=True)
        mb = simulate_portfolio(g, c["rr"], c["hold_bars"], float(r.risk_pct), stress=False)
        _, cs, _ = _outcome_cols(c["rr"], c["hold_bars"])
        em = _metric(g, cs)
        rows.append({
            **c,
            "risk_pct": float(r.risk_pct),
            "train_score": float(r.train_score),
            "train_median_monthly_stress": float(r.train_median_monthly_stress),
            "valid_n": em["n"],
            "valid_pf12": em["pf"],
            "valid_exp12": em["exp"],
            "valid_final_stress": ms["final_equity"],
            "valid_roi_stress": ms["roi_pct"],
            "valid_median_monthly_stress": ms["median_monthly_roi"],
            "valid_mean_monthly_stress": ms["mean_monthly_roi"],
            "valid_positive_months_stress": ms["positive_months_pct"],
            "valid_months_ge50_stress": ms["months_ge_50_pct"],
            "valid_maxdd_stress": ms["max_dd_pct"],
            "valid_accepted": ms["accepted"],
            "valid_final_base": mb["final_equity"],
            "valid_median_monthly_base": mb["median_monthly_roi"],
            "valid_score": _portfolio_score(ms),
        })
    z = pd.DataFrame(rows)
    if z.empty:
        return z
    viable = z[
        (z["valid_accepted"] >= 12)
        & (z["valid_final_stress"] > 0)
        & (z["valid_exp12"] > 0)
        & (z["valid_pf12"] > 1.0)
        & (z["valid_positive_months_stress"] >= 50.0)
        & (z["valid_maxdd_stress"] <= 65.0)
    ].copy()
    if viable.empty:
        viable = z.copy()
    return viable.sort_values(["valid_score", "valid_median_monthly_stress"], ascending=False).reset_index(drop=True)


def evaluate_test(test: pd.DataFrame, winner: dict) -> dict:
    c = {k: winner[k] for k in (
        "setup_group", "activity_min", "risk_min_bps", "rr", "hold_bars",
        "tf_mode", "confluence_min", "impulse_mode", "approach_min",
    )}
    g = _candidate_events(test, c)
    cb, cs, _ = _outcome_cols(c["rr"], c["hold_bars"])
    em8 = _metric(g, cb)
    em12 = _metric(g, cs)
    ms = simulate_portfolio(g, c["rr"], c["hold_bars"], float(winner["risk_pct"]), stress=True)
    mb = simulate_portfolio(g, c["rr"], c["hold_bars"], float(winner["risk_pct"]), stress=False)
    return {
        **c,
        "risk_pct": float(winner["risk_pct"]),
        "hist_test_n": em12["n"],
        "hist_test_pf8": em8["pf"],
        "hist_test_pf12": em12["pf"],
        "hist_test_exp8": em8["exp"],
        "hist_test_exp12": em12["exp"],
        "hist_test_final_stress": ms["final_equity"],
        "hist_test_roi_stress": ms["roi_pct"],
        "hist_test_median_monthly_stress": ms["median_monthly_roi"],
        "hist_test_mean_monthly_stress": ms["mean_monthly_roi"],
        "hist_test_positive_months_stress": ms["positive_months_pct"],
        "hist_test_months_ge50_stress": ms["months_ge_50_pct"],
        "hist_test_maxdd_stress": ms["max_dd_pct"],
        "hist_test_accepted": ms["accepted"],
        "hist_test_final_base": mb["final_equity"],
        "hist_test_median_monthly_base": mb["median_monthly_roi"],
    }


def scan_events(a, out: Path, pairs: list[str]) -> pd.DataFrame:
    rows = []
    metas = []
    log(f"scan pairs={len(pairs)} workers={a.workers} range={a.start}..{a.end}")
    with ProcessPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
        futs = {ex.submit(process_pair, p, a.config, a.datadir, a.start, a.end): p for p in pairs}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            try:
                rr, meta = fut.result()
            except Exception as e:
                rr, meta = [], {"pair": p, "status": f"ERROR:{type(e).__name__}:{e}"}
            rows.extend(rr)
            metas.append(meta)
            done += 1
            log(
                f"pair {done:2d}/{len(pairs)} {p:24s} {meta.get('status')} "
                f"rows={meta.get('rows', 0)} causal={meta.get('causal_events', 0)} "
                f"t={meta.get('elapsed_s', 0):.1f}s"
            )
    pd.DataFrame(metas).to_csv(out / "pair_coverage.csv", index=False)
    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No causal event rows produced")
    for c in ("entry_time", "signal_time"):
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in [x for x in df.columns if x.startswith("exit_")]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    df = df.sort_values(["entry_time", "pair", "setup"]).reset_index(drop=True)
    df.to_csv(out / "causal_events.csv", index=False)
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

    log("=== LEVEL EDGE HIGH-ROI V1 ===")
    log("Research only. Current frozen WS dry-run strategy is not modified.")
    log(
        f"fixed execution model: ${START_EQUITY:.0f}, {LEVERAGE:g}x, maxOpen={MAX_OPEN}, "
        f"minNotional=${MIN_NOTIONAL:g}, maint={MAINT_MARGIN_FRAC*100:.2f}%, costs={BASE_COST_BPS:g}/{STRESS_COST_BPS:g}bps"
    )
    log(f"10x structural-stop liquidation ceiling: <{(1/LEVERAGE-MAINT_MARGIN_FRAC)*10000:.0f} bps")
    log("search discipline: TRAIN=2022-2024, VALID=2025, HIST_TEST=2026; test is evaluated only for the validation-selected winner.")

    events_path = out / "causal_events.csv"
    if events_path.exists() and not a.rescan:
        log(f"reusing {events_path}")
        df = pd.read_csv(events_path)
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
        for c in [x for x in df.columns if x.startswith("exit_")]:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    else:
        df = scan_events(a, out, pairs)

    train = _split(df, "TRAIN")
    valid = _split(df, "VALID")
    test = _split(df, "HIST_TEST")
    log(f"events train={len(train):,} valid={len(valid):,} hist_test={len(test):,}")

    s1 = stage1(train)
    s1.to_csv(out / "stage1_train.csv", index=False)
    log(f"stage1 train-positive configs={len(s1):,}")
    if s1.empty:
        raise RuntimeError("No positive TRAIN configurations")

    s2 = stage2(train, s1, top_n=40)
    s2.to_csv(out / "stage2_train.csv", index=False)
    log(f"stage2 structural configs={len(s2):,}")
    if s2.empty:
        raise RuntimeError("No positive stage2 TRAIN configurations")

    tp = portfolio_train(train, s2, top_n=120)
    tp.to_csv(out / "train_portfolio.csv", index=False)
    log(f"train portfolio candidates={len(tp):,}")
    if tp.empty:
        raise RuntimeError("No train portfolio candidates")

    va = evaluate_validation(valid, tp, top_n=50)
    va.to_csv(out / "validation_shortlist.csv", index=False)
    if va.empty:
        raise RuntimeError("No validation candidates")

    winner = va.iloc[0].to_dict()
    hist = evaluate_test(test, winner)
    pd.DataFrame([hist]).to_csv(out / "historical_test_winner.csv", index=False)

    target_hit_valid = bool(
        float(winner["valid_median_monthly_stress"]) >= 50.0
        and float(winner["valid_positive_months_stress"]) >= 60.0
        and float(winner["valid_maxdd_stress"]) <= 65.0
    )
    target_hit_test = bool(
        float(hist["hist_test_median_monthly_stress"]) >= 50.0
        and float(hist["hist_test_positive_months_stress"]) >= 60.0
        and float(hist["hist_test_maxdd_stress"]) <= 65.0
    )

    summary = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "research_only": True,
        "production_frozen_untouched": True,
        "target": "median monthly ROI >= 50% under 12bps stress, >=60% positive months, maxDD <=65%",
        "execution": {
            "start_equity": START_EQUITY,
            "leverage": LEVERAGE,
            "max_open": MAX_OPEN,
            "min_notional": MIN_NOTIONAL,
            "maintenance_margin_frac": MAINT_MARGIN_FRAC,
            "max_structural_risk_bps_exclusive": (1.0 / LEVERAGE - MAINT_MARGIN_FRAC) * 10000.0,
            "risk_pct_scenarios": list(RISK_PCTS),
            "base_cost_bps": BASE_COST_BPS,
            "stress_cost_bps": STRESS_COST_BPS,
        },
        "splits": {
            "train": "2022-01-01..2024-12-31",
            "valid": "2025-01-01..2025-12-31",
            "historical_test": "2026-01-01..2026-08-19",
            "note": "2026 is historical test, not a pristine new holdout; truly new evidence remains post prospective cutoff.",
        },
        "winner": winner,
        "historical_test": hist,
        "target_hit_valid": target_hit_valid,
        "target_hit_historical_test": target_hit_test,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    log("\n=== VALIDATION-SELECTED WINNER ===")
    log(
        f"{winner['setup_group']} act>={winner['activity_min']} risk>={winner['risk_min_bps']}bps "
        f"RR={winner['rr']} hold={int(winner['hold_bars'])}x5m tf={winner['tf_mode']} "
        f"conf>={int(winner['confluence_min'])} impulse={winner['impulse_mode']} "
        f"approach>={int(winner['approach_min'])} risk/trade={winner['risk_pct']}% @10x"
    )
    log(
        f"TRAIN stress median/mo={winner['train_median_monthly_stress']:+.1f}% | "
        f"VALID stress median/mo={winner['valid_median_monthly_stress']:+.1f}% "
        f"mean/mo={winner['valid_mean_monthly_stress']:+.1f}% positiveMonths={winner['valid_positive_months_stress']:.1f}% "
        f"DD={winner['valid_maxdd_stress']:.1f}% PF={winner['valid_pf12']:.2f}"
    )
    log("\n=== HISTORICAL TEST 2026 — WINNER ONLY ===")
    log(
        f"stress median/mo={hist['hist_test_median_monthly_stress']:+.1f}% "
        f"mean/mo={hist['hist_test_mean_monthly_stress']:+.1f}% positiveMonths={hist['hist_test_positive_months_stress']:.1f}% "
        f"DD={hist['hist_test_maxdd_stress']:.1f}% PF={hist['hist_test_pf12']:.2f} "
        f"accepted={hist['hist_test_accepted']} final=${hist['hist_test_final_stress']:.2f}"
    )
    log(f"TARGET valid={'PASS' if target_hit_valid else 'MISS'} | hist_test={'PASS' if target_hit_test else 'MISS'}")
    log(f"reports: {out}")
    log("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
