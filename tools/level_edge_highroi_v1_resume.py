#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import level_edge_highroi_v1 as m


TRAIN = None
VALID = None


def parse_args():
    p = argparse.ArgumentParser(description="Resume Level Edge High-ROI V1 from cached causal events/stage2 and parallelize portfolio search")
    p.add_argument("--outdir", default="/freqtrade/user_data/level_edge_highroi_v1")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def log(s: str) -> None:
    print(s, flush=True)


def _load_events(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    for c in [x for x in df.columns if x.startswith("exit_")]:
        df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df


def _init_worker(train: pd.DataFrame, valid: pd.DataFrame) -> None:
    global TRAIN, VALID
    TRAIN = train
    VALID = valid


def _candidate_from_dict(r: dict) -> dict:
    return {
        "setup_group": str(r["setup_group"]),
        "activity_min": float(r["activity_min"]),
        "risk_min_bps": float(r["risk_min_bps"]),
        "rr": float(r["rr"]),
        "hold_bars": int(r["hold_bars"]),
        "tf_mode": str(r["tf_mode"]),
        "confluence_min": int(r["confluence_min"]),
        "impulse_mode": str(r["impulse_mode"]),
        "approach_min": int(r["approach_min"]),
    }


def _train_one(rank: int, r: dict) -> list[dict]:
    c = _candidate_from_dict(r)
    g = m._candidate_events(TRAIN, c)
    rows = []
    for risk_pct in m.RISK_PCTS:
        ms = m.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=True)
        mb = m.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=False)
        rows.append({
            **c,
            "risk_pct": float(risk_pct),
            "train_event_rank": int(rank),
            "train_pf12": float(r["pf12"]),
            "train_exp12": float(r["exp12"]),
            "train_r_per_month": float(r["stress_r_per_month"]),
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
            "train_score": m._portfolio_score(ms),
        })
    return rows


def _valid_one(r: dict) -> dict:
    c = _candidate_from_dict(r)
    g = m._candidate_events(VALID, c)
    risk_pct = float(r["risk_pct"])
    ms = m.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=True)
    mb = m.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=False)
    _, cs, _ = m._outcome_cols(c["rr"], c["hold_bars"])
    em = m._metric(g, cs)
    return {
        **c,
        "risk_pct": risk_pct,
        "train_score": float(r["train_score"]),
        "train_median_monthly_stress": float(r["train_median_monthly_stress"]),
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
        "valid_score": m._portfolio_score(ms),
    }


def _finalize_train(rows: list[dict]) -> pd.DataFrame:
    z = pd.DataFrame(rows)
    viable = z[
        (z["train_accepted"] >= 36)
        & (z["train_positive_months_stress"] >= 50.0)
        & (z["train_maxdd_stress"] <= 65.0)
        & (z["train_final_stress"] > m.START_EQUITY)
    ].copy()
    if viable.empty:
        viable = z.copy()
    return viable.sort_values(["train_score", "train_median_monthly_stress"], ascending=False).reset_index(drop=True)


def _finalize_valid(rows: list[dict]) -> pd.DataFrame:
    z = pd.DataFrame(rows)
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


def main() -> int:
    a = parse_args()
    out = Path(a.outdir)
    events_path = out / "causal_events.csv"
    stage2_path = out / "stage2_train.csv"
    if not events_path.exists() or not stage2_path.exists():
        raise FileNotFoundError("Need causal_events.csv and stage2_train.csv from the completed scan/stage2 before resume")

    log("=== LEVEL EDGE HIGH-ROI V1 — PARALLEL RESUME ===")
    log(f"reusing cached scan: {events_path}")
    log(f"reusing cached stage2: {stage2_path}")

    df = _load_events(events_path)
    train = m._split(df, "TRAIN")
    valid = m._split(df, "VALID")
    test = m._split(df, "HIST_TEST")
    s2 = pd.read_csv(stage2_path)
    top_train = s2.head(120).reset_index(drop=True)
    workers = max(1, int(a.workers))
    log(f"events train={len(train):,} valid={len(valid):,} hist_test={len(test):,}; stage2={len(s2):,}")
    log(f"portfolio TRAIN: {len(top_train)} candidates x {len(m.RISK_PCTS)} risk levels on {workers} workers")

    ctx = mp.get_context("fork")
    train_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(train, valid)) as ex:
        futs = {
            ex.submit(_train_one, i + 1, row.to_dict()): i + 1
            for i, row in top_train.iterrows()
        }
        done = 0
        for fut in as_completed(futs):
            train_rows.extend(fut.result())
            done += 1
            if done == 1 or done % 5 == 0 or done == len(futs):
                log(f"portfolio TRAIN {done}/{len(futs)} candidates complete")

    tp = _finalize_train(train_rows)
    tp.to_csv(out / "train_portfolio.csv", index=False)
    if tp.empty:
        raise RuntimeError("No train portfolio candidates")
    log(f"train portfolio candidates={len(tp):,}")

    top_valid = tp.head(50).reset_index(drop=True)
    log(f"VALID: {len(top_valid)} candidates on {workers} workers")
    valid_rows: list[dict] = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx, initializer=_init_worker, initargs=(train, valid)) as ex:
        futs = [ex.submit(_valid_one, row.to_dict()) for _, row in top_valid.iterrows()]
        done = 0
        for fut in as_completed(futs):
            valid_rows.append(fut.result())
            done += 1
            if done == 1 or done % 5 == 0 or done == len(futs):
                log(f"VALID {done}/{len(futs)} candidates complete")

    va = _finalize_valid(valid_rows)
    va.to_csv(out / "validation_shortlist.csv", index=False)
    if va.empty:
        raise RuntimeError("No validation candidates")

    winner = va.iloc[0].to_dict()
    hist = m.evaluate_test(test, winner)
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
        "parallel_resume": True,
        "workers": workers,
        "target": "median monthly ROI >= 50% under 12bps stress, >=60% positive months, maxDD <=65%",
        "execution": {
            "start_equity": m.START_EQUITY,
            "leverage": m.LEVERAGE,
            "max_open": m.MAX_OPEN,
            "min_notional": m.MIN_NOTIONAL,
            "maintenance_margin_frac": m.MAINT_MARGIN_FRAC,
            "max_structural_risk_bps_exclusive": (1.0 / m.LEVERAGE - m.MAINT_MARGIN_FRAC) * 10000.0,
            "risk_pct_scenarios": list(m.RISK_PCTS),
            "base_cost_bps": m.BASE_COST_BPS,
            "stress_cost_bps": m.STRESS_COST_BPS,
        },
        "splits": {
            "train": "2022-01-01..2024-12-31",
            "valid": "2025-01-01..2025-12-31",
            "historical_test": "2026-01-01..2026-08-19",
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
