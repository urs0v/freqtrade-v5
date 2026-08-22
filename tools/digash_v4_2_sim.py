#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v4_stage0 as s0


def parse_args():
    p = argparse.ArgumentParser(
        description="Digash V4.2 post-signal PnL replay. Detector stays causal; replay uses future 1m only after signal."
    )
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance/futures")
    p.add_argument("--events", default="/freqtrade/user_data/digash_v4_2_fidelity/events.csv")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_v4_2_fidelity/sim")
    p.add_argument("--starting-equity", type=float, default=100.0)
    p.add_argument("--risk-pcts", default="0.01,0.02,0.03")
    p.add_argument("--max-leverage", type=float, default=10.0)
    p.add_argument("--max-concurrent", type=int, default=3)
    p.add_argument("--fee-bps-side", type=float, default=5.0)
    p.add_argument("--slippage-bps-side", type=float, default=1.0)
    p.add_argument("--max-hold-hours", type=float, default=24.0)
    return p.parse_args()


def log(x):
    print(x, flush=True)


def load_symbol_1m(datadir: Path, sym: str) -> pd.DataFrame:
    path = s0._data_path(datadir, sym, "1m")
    if not path.exists():
        raise FileNotFoundError(path)
    x = pd.read_feather(path)[["date", "open", "high", "low", "close"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x = x.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)
    for c in ("open", "high", "low", "close"):
        x[c] = pd.to_numeric(x[c], errors="coerce")
    return x.dropna().reset_index(drop=True)


def _touched_stop(side, lo, hi, stop):
    return lo <= stop if side > 0 else hi >= stop


def _touched_target(side, lo, hi, target):
    return hi >= target if side > 0 else lo <= target


def _net_return(entry: float, exit_price: float, side: int, fee_bps: float, slip_bps: float) -> float:
    fee = fee_bps / 10000.0
    slip = slip_bps / 10000.0
    entry_eff = entry * (1.0 + side * slip)
    exit_eff = exit_price * (1.0 - side * slip)
    gross = side * (exit_eff - entry_eff) / entry_eff
    fees = fee * (1.0 + exit_eff / entry_eff)
    return gross - fees


def replay_one(ev: pd.Series, x: pd.DataFrame, max_hold_hours: float, management: str,
               fee_bps: float, slip_bps: float) -> dict:
    t = pd.Timestamp(ev["entry_time"])
    side = 1 if str(ev["side"]).upper() == "LONG" else -1
    entry = float(ev["entry_price"])
    stop0 = float(ev["initial_stop"])
    t1 = float(ev["target_1"])
    final = float(ev["final_target"])
    risk = side * (entry - stop0)

    base = {
        "pair": ev["pair"], "entry_time": t, "side": ev["side"],
        "entry_family": ev["entry_family"], "management": management,
        "entry_price": entry, "initial_stop": stop0, "target_1": t1,
        "final_target": final, "risk_abs": risk,
        "available_R_T1": float(ev.get("available_R_T1", np.nan)),
        "available_R_final": float(ev.get("available_R_final", ev.get("available_R", np.nan))),
        "setup_episode_id": ev.get("setup_episode_id", ""),
    }
    if not (np.isfinite(risk) and risk > 0):
        return {**base, "status": "BAD_RISK"}

    dates = x["date"].to_numpy(dtype="datetime64[ns]")
    start_i = int(np.searchsorted(dates, np.datetime64(t.to_datetime64()), side="left"))
    if start_i >= len(x):
        return {**base, "status": "NO_ENTRY_BAR"}
    end_t = t + pd.Timedelta(hours=max_hold_hours)
    end_i = int(np.searchsorted(dates, np.datetime64(end_t.to_datetime64()), side="right"))
    end_i = min(end_i, len(x))
    if end_i <= start_i:
        return {**base, "status": "NO_REPLAY_WINDOW"}

    lo = x["low"].to_numpy(float)
    hi = x["high"].to_numpy(float)
    cl = x["close"].to_numpy(float)

    t1_hit = False
    t1_bar = None
    exit_price = None
    exit_i = None
    reason = None
    active_stop = stop0
    mfe_r = -math.inf
    mae_r = math.inf
    same_target = abs(final - t1) <= max(abs(entry) * 1e-12, 1e-12)

    for i in range(start_i, end_i):
        if side > 0:
            mfe_r = max(mfe_r, (hi[i] - entry) / risk)
            mae_r = min(mae_r, (lo[i] - entry) / risk)
        else:
            mfe_r = max(mfe_r, (entry - lo[i]) / risk)
            mae_r = min(mae_r, (entry - hi[i]) / risk)

        stop_hit = _touched_stop(side, lo[i], hi[i], active_stop)
        final_hit = _touched_target(side, lo[i], hi[i], final)
        one_hit = _touched_target(side, lo[i], hi[i], t1)

        # Same-bar ambiguity is always adverse-first.
        if stop_hit:
            exit_price = active_stop
            exit_i = i
            reason = "STOP" if not t1_hit or management == "FINAL_ONLY" else "BE"
            break

        if final_hit:
            exit_price = final
            exit_i = i
            reason = "FINAL"
            if one_hit:
                t1_hit = True
            break

        if management == "T1_BE_FINAL" and (not t1_hit) and one_hit:
            t1_hit = True
            t1_bar = i
            # BE becomes active only from the next 1m bar.
            continue

        if management == "T1_BE_FINAL" and t1_hit and t1_bar is not None and i > t1_bar:
            active_stop = entry

        if management == "FINAL_ONLY" and same_target and one_hit:
            exit_price = final
            exit_i = i
            reason = "FINAL"
            t1_hit = True
            break

    if exit_i is None:
        exit_i = end_i - 1
        exit_price = float(cl[exit_i])
        reason = "TIMEOUT"

    exit_time = pd.Timestamp(x.iloc[exit_i]["date"]) + pd.Timedelta(minutes=1)
    gross_r = side * (exit_price - entry) / risk
    net_ret = _net_return(entry, exit_price, side, fee_bps, slip_bps)
    stop_pct = risk / entry
    net_r = net_ret / stop_pct

    return {
        **base,
        "status": "OK",
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "exit_reason": reason,
        "t1_hit": bool(t1_hit),
        "gross_R": float(gross_r),
        "net_R": float(net_r),
        "net_return_per_notional": float(net_ret),
        "stop_pct": float(stop_pct),
        "mfe_R": float(mfe_r) if np.isfinite(mfe_r) else None,
        "mae_R": float(mae_r) if np.isfinite(mae_r) else None,
        "hold_minutes": float((exit_time - t) / pd.Timedelta(minutes=1)),
    }


def replay_all(events: pd.DataFrame, datadir: Path, max_hold_hours: float,
               fee_bps: float, slip_bps: float) -> pd.DataFrame:
    rows = []
    for sym, grp in events.groupby("pair", sort=True):
        x = load_symbol_1m(datadir, sym)
        log(f"replay {sym}: events={len(grp)}")
        for _, ev in grp.iterrows():
            for management in ("FINAL_ONLY", "T1_BE_FINAL"):
                rows.append(replay_one(ev, x, max_hold_hours, management, fee_bps, slip_bps))
    z = pd.DataFrame(rows)
    if not z.empty:
        for c in ("entry_time", "exit_time"):
            if c in z.columns:
                z[c] = pd.to_datetime(z[c], utc=True, errors="coerce")
    return z


def _pf(vals: pd.Series):
    pos = vals[vals > 0].sum()
    neg = -vals[vals < 0].sum()
    return float(pos / neg) if neg > 0 else (math.inf if pos > 0 else None)


def trade_stats(z: pd.DataFrame) -> dict:
    if z.empty:
        return {}
    out = {}
    for mgmt, g in z[z["status"] == "OK"].groupby("management"):
        net = g["net_R"].astype(float)
        months = max(1.0, (g["entry_time"].max() - g["entry_time"].min()) / pd.Timedelta(days=30.4375)) if len(g) > 1 else 1.0
        out[mgmt] = {
            "trades": int(len(g)),
            "trades_per_month": float(len(g) / months),
            "winrate_net": float((net > 0).mean()),
            "expectancy_net_R": float(net.mean()),
            "median_net_R": float(net.median()),
            "profit_factor_R": _pf(net),
            "avg_mfe_R": float(g["mfe_R"].mean()),
            "avg_mae_R": float(g["mae_R"].mean()),
            "avg_hold_minutes": float(g["hold_minutes"].mean()),
            "exit_reasons": dict(Counter(g["exit_reason"])),
            "by_family": {},
        }
        for fam, f in g.groupby("entry_family"):
            n = f["net_R"].astype(float)
            out[mgmt]["by_family"][fam] = {
                "trades": int(len(f)),
                "winrate_net": float((n > 0).mean()),
                "expectancy_net_R": float(n.mean()),
                "median_net_R": float(n.median()),
                "profit_factor_R": _pf(n),
                "exit_reasons": dict(Counter(f["exit_reason"])),
            }
    return out


def portfolio_sim(g: pd.DataFrame, starting_equity: float, risk_pct: float,
                  max_leverage: float, max_concurrent: int):
    g = g[(g["status"] == "OK") & g["exit_time"].notna()].sort_values(
        ["entry_time", "pair", "entry_family"]
    ).reset_index(drop=True)
    equity = float(starting_equity)
    peak = equity
    max_dd = 0.0
    active = []
    ledger = []
    skipped_slots = 0
    skipped_cap = 0

    def settle(until):
        nonlocal equity, peak, max_dd, active
        keep = []
        for p in active:
            if p["exit_time"] <= until:
                equity += p["pnl"]
                peak = max(peak, equity)
                dd = (peak - equity) / peak if peak > 0 else 1.0
                max_dd = max(max_dd, dd)
                p["equity_after"] = equity
                ledger.append(p)
            else:
                keep.append(p)
        active = keep

    for _, r in g.iterrows():
        et = pd.Timestamp(r["entry_time"])
        settle(et)
        if equity <= 0:
            break
        if len(active) >= max_concurrent:
            skipped_slots += 1
            continue

        active_notional = sum(p["notional"] for p in active)
        notional_cap = max(0.0, equity * max_leverage - active_notional)
        if notional_cap <= 0:
            skipped_cap += 1
            continue

        stop_pct = float(r["stop_pct"])
        if not (np.isfinite(stop_pct) and stop_pct > 0):
            continue
        desired = equity * risk_pct / stop_pct
        notional = min(desired, notional_cap)
        if notional <= 0:
            skipped_cap += 1
            continue

        pnl = notional * float(r["net_return_per_notional"])
        active.append({
            "entry_time": et,
            "exit_time": pd.Timestamp(r["exit_time"]),
            "pair": r["pair"],
            "side": r["side"],
            "family": r["entry_family"],
            "exit_reason": r["exit_reason"],
            "notional": float(notional),
            "risk_dollars_nominal": float(notional * stop_pct),
            "pnl": float(pnl),
            "equity_at_entry": float(equity),
            "net_R": float(r["net_R"]),
        })

    settle(pd.Timestamp.max.tz_localize("UTC"))
    led = pd.DataFrame(ledger)
    if led.empty:
        return {
            "starting_equity": starting_equity, "final_equity": equity,
            "pnl": equity - starting_equity, "roi_pct": (equity / starting_equity - 1) * 100,
            "max_drawdown_pct": max_dd * 100, "trades_taken": 0,
            "skipped_slots": skipped_slots, "skipped_leverage_cap": skipped_cap,
        }, led

    vals = led["pnl"]
    summary = {
        "starting_equity": float(starting_equity),
        "final_equity": float(equity),
        "pnl": float(equity - starting_equity),
        "roi_pct": float((equity / starting_equity - 1.0) * 100.0),
        "max_drawdown_pct": float(max_dd * 100.0),
        "trades_taken": int(len(led)),
        "wins": int((vals > 0).sum()),
        "losses": int((vals < 0).sum()),
        "winrate": float((vals > 0).mean()),
        "profit_factor_dollars": _pf(vals),
        "skipped_slots": int(skipped_slots),
        "skipped_leverage_cap": int(skipped_cap),
        "avg_notional": float(led["notional"].mean()),
        "max_notional": float(led["notional"].max()),
    }
    return summary, led


def monthly_from_ledger(led: pd.DataFrame, starting_equity: float) -> pd.DataFrame:
    if led.empty:
        return pd.DataFrame(columns=["month", "trades", "pnl", "equity_end", "roi_from_start_pct"])
    z = led.copy().sort_values("exit_time")
    z["month"] = pd.to_datetime(z["exit_time"], utc=True).dt.strftime("%Y-%m")
    z["equity_end"] = starting_equity + z["pnl"].cumsum()
    rows = []
    for month, g in z.groupby("month", sort=True):
        rows.append({
            "month": month,
            "trades": len(g),
            "pnl": float(g["pnl"].sum()),
            "equity_end": float(g["equity_end"].iloc[-1]),
            "roi_from_start_pct": float((g["equity_end"].iloc[-1] / starting_equity - 1) * 100),
        })
    return pd.DataFrame(rows)


def main():
    A = parse_args()
    datadir = Path(A.datadir)
    events_path = Path(A.events)
    outdir = Path(A.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not events_path.exists():
        raise SystemExit(f"events.csv not found: {events_path}")

    ev = pd.read_csv(events_path)
    if ev.empty:
        raise SystemExit("events.csv is empty")
    ev["entry_time"] = pd.to_datetime(ev["entry_time"], utc=True)
    for c in ("entry_price", "initial_stop", "target_1", "final_target"):
        ev[c] = pd.to_numeric(ev[c], errors="coerce")
    ev = ev.dropna(subset=["entry_time", "entry_price", "initial_stop", "target_1", "final_target"])

    log("=== DIGASH V4.2 POST-SIGNAL PNL REPLAY ===")
    log("Detector output is frozen first; future 1m is used only here, after signal creation.")
    log(f"events={len(ev)} fee={A.fee_bps_side:.2f}bps/side slippage={A.slippage_bps_side:.2f}bps/side hold={A.max_hold_hours:g}h")

    rep = replay_all(ev, datadir, A.max_hold_hours, A.fee_bps_side, A.slippage_bps_side)
    rep.to_csv(outdir / "trade_replay.csv", index=False)

    tstats = trade_stats(rep)
    risk_pcts = [float(x) for x in A.risk_pcts.split(",") if x.strip()]
    portfolios = {}

    for mgmt in ("FINAL_ONLY", "T1_BE_FINAL"):
        g = rep[rep["management"] == mgmt].copy()
        portfolios[mgmt] = {}
        for rp in risk_pcts:
            key = f"risk_{rp*100:.1f}pct"
            s, led = portfolio_sim(g, A.starting_equity, rp, A.max_leverage, A.max_concurrent)
            portfolios[mgmt][key] = s
            led.to_csv(outdir / f"portfolio_{mgmt.lower()}_{rp*100:.1f}pct.csv", index=False)
            monthly_from_ledger(led, A.starting_equity).to_csv(
                outdir / f"monthly_{mgmt.lower()}_{rp*100:.1f}pct.csv", index=False
            )

    summary = {
        "stage": "Digash V4.2 post-signal PnL replay",
        "detector_future_used": False,
        "replay_future_used": True,
        "assumptions": {
            "starting_equity_usd": A.starting_equity,
            "risk_pcts": risk_pcts,
            "max_leverage": A.max_leverage,
            "max_concurrent": A.max_concurrent,
            "fee_bps_per_side": A.fee_bps_side,
            "slippage_bps_per_side": A.slippage_bps_side,
            "all_in_nominal_roundtrip_bps": 2 * (A.fee_bps_side + A.slippage_bps_side),
            "max_hold_hours": A.max_hold_hours,
            "intrabar_ambiguity": "adverse-first",
            "management_models": {
                "FINAL_ONLY": "initial structural stop vs final structural target; timeout at horizon",
                "T1_BE_FINAL": "after T1 touch, breakeven becomes active from next 1m bar; final target remains; timeout at horizon",
            },
            "note": "Diagnostic simulation, not a profitability claim. Funding, partial fills, latency, liquidation mechanics, and discretionary dynamic management are not yet modeled.",
        },
        "trade_stats": tstats,
        "portfolio_scenarios": portfolios,
    }

    (outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    log("")
    log("=== REPLAY RESULT ===")
    for mgmt, st in tstats.items():
        log(f"{mgmt}: n={st['trades']} win={st['winrate_net']:.1%} E={st['expectancy_net_R']:.3f}R PF={st['profit_factor_R']}")
    if "T1_BE_FINAL" in portfolios:
        for k, s in portfolios["T1_BE_FINAL"].items():
            log(f"T1_BE_FINAL {k}: ROI={s['roi_pct']:.2f}% final=${s['final_equity']:.2f} DD={s['max_drawdown_pct']:.2f}% trades={s['trades_taken']}")
    log(f"summary={outdir/'summary.json'}")
    log(f"trade_replay={outdir/'trade_replay.csv'}")


if __name__ == "__main__":
    main()
