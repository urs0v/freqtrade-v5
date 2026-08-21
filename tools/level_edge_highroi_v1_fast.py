#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import level_edge_highroi_v1 as lab


_WORK_DF: pd.DataFrame | None = None


def parse_args():
    p = argparse.ArgumentParser(description="Parallel post-scan optimizer for LEVEL EDGE HIGH-ROI V1")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/level_edge_highroi_v1")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-08-19")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--rescan", action="store_true")
    return p.parse_args()


def _load_events(a, out: Path) -> pd.DataFrame:
    events_path = out / "causal_events.csv"
    if events_path.exists() and not a.rescan:
        lab.log(f"reusing {events_path}")
        df = pd.read_csv(events_path)
        df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
        df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
        for c in [x for x in df.columns if x.startswith("exit_")]:
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
        return df

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
    return lab.scan_events(a, out, pairs)


def _candidate_dict_from_mapping(r: dict) -> dict:
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


def _portfolio_worker(payload: tuple[int, dict]) -> tuple[int, list[dict]]:
    rank, src = payload
    if _WORK_DF is None:
        raise RuntimeError("worker dataframe not initialized")
    c = _candidate_dict_from_mapping(src)
    g = lab._candidate_events(_WORK_DF, c)
    rows = []
    for risk_pct in lab.RISK_PCTS:
        ms = lab.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=True)
        mb = lab.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=False)
        rows.append({
            **c,
            "risk_pct": float(risk_pct),
            "train_event_rank": int(rank),
            "train_pf12": float(src["pf12"]),
            "train_exp12": float(src["exp12"]),
            "train_r_per_month": float(src["stress_r_per_month"]),
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
            "train_score": lab._portfolio_score(ms),
        })
    return rank, rows


def portfolio_train_parallel(train: pd.DataFrame, second: pd.DataFrame, workers: int, top_n: int = 120) -> pd.DataFrame:
    global _WORK_DF
    _WORK_DF = train
    items = [(i, row) for i, row in enumerate(second.head(top_n).to_dict("records"), 1)]
    rows: list[dict] = []
    if not items:
        return pd.DataFrame()

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=ctx) as ex:
        futs = [ex.submit(_portfolio_worker, x) for x in items]
        done = 0
        for fut in as_completed(futs):
            rank, rr = fut.result()
            rows.extend(rr)
            done += 1
            if done == 1 or done % 5 == 0 or done == len(futs):
                lab.log(f"portfolio TRAIN {done:3d}/{len(futs)} candidate groups complete (last rank={rank})")

    z = pd.DataFrame(rows)
    if z.empty:
        return z
    viable = z[
        (z["train_accepted"] >= 36)
        & (z["train_positive_months_stress"] >= 50.0)
        & (z["train_maxdd_stress"] <= 65.0)
        & (z["train_final_stress"] > lab.START_EQUITY)
    ].copy()
    if viable.empty:
        viable = z.copy()
    return viable.sort_values(
        ["train_score", "train_median_monthly_stress", "train_event_rank", "risk_pct"],
        ascending=[False, False, True, True], kind="mergesort",
    ).reset_index(drop=True)


def _validation_worker(payload: tuple[int, dict]) -> tuple[int, dict]:
    rank, src = payload
    if _WORK_DF is None:
        raise RuntimeError("worker dataframe not initialized")
    c = _candidate_dict_from_mapping(src)
    g = lab._candidate_events(_WORK_DF, c)
    risk_pct = float(src["risk_pct"])
    ms = lab.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=True)
    mb = lab.simulate_portfolio(g, c["rr"], c["hold_bars"], risk_pct, stress=False)
    _, cs, _ = lab._outcome_cols(c["rr"], c["hold_bars"])
    em = lab._metric(g, cs)
    row = {
        **c,
        "risk_pct": risk_pct,
        "train_rank": int(rank),
        "train_score": float(src["train_score"]),
        "train_median_monthly_stress": float(src["train_median_monthly_stress"]),
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
        "valid_score": lab._portfolio_score(ms),
    }
    return rank, row


def validation_parallel(valid: pd.DataFrame, train_port: pd.DataFrame, workers: int, top_n: int = 50) -> pd.DataFrame:
    global _WORK_DF
    _WORK_DF = valid
    items = [(i, row) for i, row in enumerate(train_port.head(top_n).to_dict("records"), 1)]
    rows: list[dict] = []
    if not items:
        return pd.DataFrame()

    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=max(1, int(workers)), mp_context=ctx) as ex:
        futs = [ex.submit(_validation_worker, x) for x in items]
        done = 0
        for fut in as_completed(futs):
            rank, row = fut.result()
            rows.append(row)
            done += 1
            if done == 1 or done % 5 == 0 or done == len(futs):
                lab.log(f"portfolio VALID {done:3d}/{len(futs)} candidates complete (last rank={rank})")

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
    return viable.sort_values(
        ["valid_score", "valid_median_monthly_stress", "train_rank"],
        ascending=[False, False, True], kind="mergesort",
    ).reset_index(drop=True)


def _load_or_build_stage2(train: pd.DataFrame, out: Path) -> pd.DataFrame:
    p2 = out / "stage2_train.csv"
    if p2.exists():
        s2 = pd.read_csv(p2)
        if not s2.empty:
            lab.log(f"reusing {p2} ({len(s2):,} structural configs)")
            return s2

    p1 = out / "stage1_train.csv"
    if p1.exists():
        s1 = pd.read_csv(p1)
        lab.log(f"reusing {p1} ({len(s1):,} configs)")
    else:
        s1 = lab.stage1(train)
        s1.to_csv(p1, index=False)
        lab.log(f"stage1 train-positive configs={len(s1):,}")
    if s1.empty:
        raise RuntimeError("No positive TRAIN configurations")

    s2 = lab.stage2(train, s1, top_n=40)
    s2.to_csv(p2, index=False)
    lab.log(f"stage2 structural configs={len(s2):,}")
    if s2.empty:
        raise RuntimeError("No positive stage2 TRAIN configurations")
    return s2


def main() -> int:
    a = parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)

    lab.log("=== LEVEL EDGE HIGH-ROI V1 — PARALLEL OPTIMIZER ===")
    lab.log("Research only. Signal generation/search semantics are unchanged; only post-scan portfolio evaluation is parallelized.")
    lab.log(
        f"fixed execution model: ${lab.START_EQUITY:.0f}, {lab.LEVERAGE:g}x, maxOpen={lab.MAX_OPEN}, "
        f"minNotional=${lab.MIN_NOTIONAL:g}, costs={lab.BASE_COST_BPS:g}/{lab.STRESS_COST_BPS:g}bps, workers={a.workers}"
    )

    df = _load_events(a, out)
    train = lab._split(df, "TRAIN")
    valid = lab._split(df, "VALID")
    test = lab._split(df, "HIST_TEST")
    lab.log(f"events train={len(train):,} valid={len(valid):,} hist_test={len(test):,}")

    s2 = _load_or_build_stage2(train, out)

    tp = portfolio_train_parallel(train, s2, workers=a.workers, top_n=120)
    tp.to_csv(out / "train_portfolio.csv", index=False)
    lab.log(f"train portfolio candidates={len(tp):,}")
    if tp.empty:
        raise RuntimeError("No train portfolio candidates")

    va = validation_parallel(valid, tp, workers=a.workers, top_n=50)
    va.to_csv(out / "validation_shortlist.csv", index=False)
    if va.empty:
        raise RuntimeError("No validation candidates")

    winner = va.iloc[0].to_dict()
    hist = lab.evaluate_test(test, winner)
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
        "optimizer": "parallel_postscan_v1",
        "target": "median monthly ROI >= 50% under 12bps stress, >=60% positive months, maxDD <=65%",
        "execution": {
            "start_equity": lab.START_EQUITY,
            "leverage": lab.LEVERAGE,
            "max_open": lab.MAX_OPEN,
            "min_notional": lab.MIN_NOTIONAL,
            "maintenance_margin_frac": lab.MAINT_MARGIN_FRAC,
            "max_structural_risk_bps_exclusive": (1.0 / lab.LEVERAGE - lab.MAINT_MARGIN_FRAC) * 10000.0,
            "risk_pct_scenarios": list(lab.RISK_PCTS),
            "base_cost_bps": lab.BASE_COST_BPS,
            "stress_cost_bps": lab.STRESS_COST_BPS,
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

    lab.log("\n=== VALIDATION-SELECTED WINNER ===")
    lab.log(
        f"{winner['setup_group']} act>={winner['activity_min']} risk>={winner['risk_min_bps']}bps "
        f"RR={winner['rr']} hold={int(winner['hold_bars'])}x5m tf={winner['tf_mode']} "
        f"conf>={int(winner['confluence_min'])} impulse={winner['impulse_mode']} "
        f"approach>={int(winner['approach_min'])} risk/trade={winner['risk_pct']}% @10x"
    )
    lab.log(
        f"TRAIN stress median/mo={winner['train_median_monthly_stress']:+.1f}% | "
        f"VALID stress median/mo={winner['valid_median_monthly_stress']:+.1f}% "
        f"mean/mo={winner['valid_mean_monthly_stress']:+.1f}% positiveMonths={winner['valid_positive_months_stress']:.1f}% "
        f"DD={winner['valid_maxdd_stress']:.1f}% PF={winner['valid_pf12']:.2f}"
    )
    lab.log("\n=== HISTORICAL TEST 2026 — WINNER ONLY ===")
    lab.log(
        f"stress median/mo={hist['hist_test_median_monthly_stress']:+.1f}% "
        f"mean/mo={hist['hist_test_mean_monthly_stress']:+.1f}% positiveMonths={hist['hist_test_positive_months_stress']:.1f}% "
        f"DD={hist['hist_test_maxdd_stress']:.1f}% PF={hist['hist_test_pf12']:.2f} "
        f"accepted={hist['hist_test_accepted']} final=${hist['hist_test_final_stress']:.2f}"
    )
    lab.log(f"TARGET valid={'PASS' if target_hit_valid else 'MISS'} | hist_test={'PASS' if target_hit_test else 'MISS'}")
    lab.log(f"reports: {out}")
    lab.log("=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
