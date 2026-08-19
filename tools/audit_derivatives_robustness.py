#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research_derivatives_alpha as r

CANDIDATES = ["funding_z", "taker_minus_funding"]
HORIZON = "12h"
BARS = r.HORIZONS[HORIZON]
THRESHOLD_QS = [0.20, 0.25, 0.30]
COSTS_BPS = [4.0, 8.0, 12.0]
CANONICAL_Q = 0.25
CANONICAL_COST = 8.0

# Predeclared before this audit is run. 2026 is never part of this gate.
ROBUST_MIN_POSITIVE_YEARS = 3          # of 2022, 2023, 2024, 2025
ROBUST_MIN_POSITIVE_MONTHS_2025 = 6    # of 12
ROBUST_MIN_LOO_POSITIVE_FRAC = 0.90    # leave-one-pair-out validation

_original_load_derivatives = r.load_derivatives


def load_derivatives_ns(db: Path, symbol: str, start_ms: int, end_ms: int) -> pd.DataFrame:
    df = _original_load_derivatives(db, symbol, start_ms, end_ms)
    if not df.empty:
        df = df.copy()
        df["available_time"] = pd.to_datetime(df["available_time"], utc=True).astype("datetime64[ns, UTC]")
    return df


r.load_derivatives = load_derivatives_ns


def oriented_thresholds(train: pd.DataFrame, factor: str, q: float) -> tuple[float, float, float]:
    target = f"y_{HORIZON}"
    base = train[[factor, target]].dropna()
    if len(base) < 100:
        raise RuntimeError(f"Not enough train rows for {factor}: {len(base)}")
    ic = r.spearman(base[factor], base[target])
    orient = 1.0 if not np.isfinite(ic) or ic >= 0 else -1.0
    score = orient * base[factor]
    qlo, qhi = np.quantile(score, [q, 1.0 - q])
    return orient, float(qlo), float(qhi)


def selected_rows(df: pd.DataFrame, factor: str, orient: float, qlo: float, qhi: float) -> pd.DataFrame:
    target = f"y_{HORIZON}"
    x = df[["signal_time", "pair", factor, target]].dropna().copy()
    if x.empty:
        return pd.DataFrame(columns=["signal_time", "pair", "gross_ret"])
    score = orient * x[factor].to_numpy(dtype=float)
    side = np.where(score >= qhi, 1.0, np.where(score <= qlo, -1.0, 0.0))
    chosen = side != 0
    out = x.loc[chosen, ["signal_time", "pair"]].reset_index(drop=True)
    out["gross_ret"] = side[chosen] * x.loc[chosen, target].to_numpy(dtype=float)
    return out


def stats(rows: pd.DataFrame, cost_bps: float) -> dict:
    if rows.empty:
        return {"n": 0, "gross_bps": np.nan, "net_bps": np.nan, "win_pct": np.nan}
    gross = float(rows["gross_ret"].mean() * 10000.0)
    return {
        "n": int(len(rows)),
        "gross_bps": gross,
        "net_bps": gross - cost_bps,
        "win_pct": float((rows["gross_ret"] > 0).mean() * 100.0),
    }


def bootstrap_month_ci(rows: pd.DataFrame, cost_bps: float, seed: int = 20260819, n_boot: int = 10000) -> dict:
    if rows.empty:
        return {"boot_mean_net_bps": np.nan, "boot_p05_net_bps": np.nan, "boot_p95_net_bps": np.nan, "boot_prob_positive": np.nan}
    x = rows.copy()
    x["month"] = x["signal_time"].dt.to_period("M").astype(str)
    month_means = x.groupby("month")["gross_ret"].mean().to_numpy(dtype=float) * 10000.0 - cost_bps
    if len(month_means) == 0:
        return {"boot_mean_net_bps": np.nan, "boot_p05_net_bps": np.nan, "boot_p95_net_bps": np.nan, "boot_prob_positive": np.nan}
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(month_means), size=(n_boot, len(month_means)))
    sims = month_means[idx].mean(axis=1)
    return {
        "boot_mean_net_bps": float(sims.mean()),
        "boot_p05_net_bps": float(np.quantile(sims, 0.05)),
        "boot_p95_net_bps": float(np.quantile(sims, 0.95)),
        "boot_prob_positive": float((sims > 0).mean()),
    }


def load_all(config: dict, datadir: Path, db: Path) -> dict[str, pd.DataFrame]:
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No pair_whitelist in config")

    ranges = {
        "train": ("2022-01-01", "2025-01-01"),
        "val": ("2025-01-01", "2026-01-01"),
        "test": ("2026-01-01", "2026-08-19"),
    }
    buckets: dict[str, list[pd.DataFrame]] = {k: [] for k in ranges}

    print(f"Loading {len(pairs)} pairs for 12h robustness audit...", flush=True)
    for i, pair in enumerate(pairs, 1):
        t0 = time.monotonic()
        price = r.load_price(config, datadir, pair)
        if price.empty:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: NO PRICE", flush=True)
            continue
        price["date"] = r.as_ns(price["date"])
        start_ms = int((price["date"].min() - pd.Timedelta("1D")).timestamp() * 1000)
        end_ms = int((price["date"].max() + pd.Timedelta("1D")).timestamp() * 1000)
        deriv = r.load_derivatives(db, r.pair_to_symbol(pair), start_ms, end_ms)
        feat, cov = r.build_features(price, deriv)
        core_cov = min(cov["oi_pct"], cov["taker_pct"])
        if core_cov < 50.0:
            print(f"  [{i:02d}/{len(pairs)}] {pair}: SKIP coverage={core_cov:.1f}%", flush=True)
            continue
        for split, (start, end) in ranges.items():
            x = r.slice_horizon(feat, start, end, HORIZON, BARS)
            if not x.empty:
                x["pair"] = pair
                buckets[split].append(x)
        print(f"  [{i:02d}/{len(pairs)}] {pair}: ok [{time.monotonic()-t0:.1f}s]", flush=True)

    return {
        split: pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        for split, parts in buckets.items()
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Robustness audit for pre-2026 derivatives alpha candidates")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--db", default="/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/robustness")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    db = Path(args.db)
    if not db.exists():
        raise RuntimeError(f"Missing DB: {db}")
    data = load_all(config, Path(args.datadir), db)
    train, val, test = data["train"], data["val"], data["test"]
    if train.empty or val.empty:
        raise RuntimeError("Missing train/validation data")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detail_rows = []

    print("\n=== PREDECLARED ROBUSTNESS RULES ===")
    print("2026 is diagnostic only and cannot rescue/fail a pre-2026 candidate.")
    print(f"PASS requires: all q={THRESHOLD_QS} positive net at 8bps on 2025; >= {ROBUST_MIN_POSITIVE_YEARS}/4 positive years; >= {ROBUST_MIN_POSITIVE_MONTHS_2025}/12 positive 2025 months; >= {ROBUST_MIN_LOO_POSITIVE_FRAC:.0%} leave-one-pair-out 2025 runs positive.")

    for factor in CANDIDATES:
        print(f"\n=== {factor} / {HORIZON} ===")
        canonical_orient, canonical_lo, canonical_hi = oriented_thresholds(train, factor, CANONICAL_Q)

        # Threshold stability on untouched 2025, thresholds always learned on 2022-24.
        threshold_results = []
        for q in THRESHOLD_QS:
            orient, qlo, qhi = oriented_thresholds(train, factor, q)
            va_rows = selected_rows(val, factor, orient, qlo, qhi)
            s = stats(va_rows, CANONICAL_COST)
            threshold_results.append((q, s["net_bps"]))
            detail_rows.append({"factor": factor, "check": "threshold", "key": f"q{q:.2f}", **s})
            print(f"threshold q={q:.2f}: val net={s['net_bps']:+.2f} bps n={s['n']}")

        # Cost sensitivity at canonical threshold.
        val_rows = selected_rows(val, factor, canonical_orient, canonical_lo, canonical_hi)
        for c in COSTS_BPS:
            s = stats(val_rows, c)
            detail_rows.append({"factor": factor, "check": "cost", "key": f"{c:.0f}bps", **s})
            print(f"cost {c:>4.0f}bps: val net={s['net_bps']:+.2f} bps")

        # Year stability, canonical thresholds learned once on all 2022-24.
        pre = pd.concat([train, val], ignore_index=True)
        pre_rows = selected_rows(pre, factor, canonical_orient, canonical_lo, canonical_hi)
        yearly = []
        for year in [2022, 2023, 2024, 2025]:
            y = pre_rows[pre_rows["signal_time"].dt.year == year]
            s = stats(y, CANONICAL_COST)
            yearly.append(s["net_bps"])
            detail_rows.append({"factor": factor, "check": "year", "key": str(year), **s})
            print(f"year {year}: net={s['net_bps']:+.2f} bps n={s['n']}")

        # Month stability in 2025.
        month_nets = []
        for month in range(1, 13):
            m = val_rows[val_rows["signal_time"].dt.month == month]
            s = stats(m, CANONICAL_COST)
            month_nets.append(s["net_bps"])
            detail_rows.append({"factor": factor, "check": "month_2025", "key": f"2025-{month:02d}", **s})

        # Pair dependence: remove one pair at a time from 2025.
        loo_nets = []
        pairs = sorted(val_rows["pair"].unique())
        for pair in pairs:
            x = val_rows[val_rows["pair"] != pair]
            s = stats(x, CANONICAL_COST)
            loo_nets.append(s["net_bps"])
            detail_rows.append({"factor": factor, "check": "loo_pair", "key": pair, **s})

        # Pair contribution diagnostics (not a gate by itself).
        pair_stats = []
        for pair in pairs:
            s = stats(val_rows[val_rows["pair"] == pair], CANONICAL_COST)
            pair_stats.append((pair, s["net_bps"], s["n"]))
        pair_stats.sort(key=lambda z: z[1], reverse=True)
        print("top pair net bps:", ", ".join(f"{p} {v:+.1f}" for p, v, _ in pair_stats[:5]))
        print("bottom pair net bps:", ", ".join(f"{p} {v:+.1f}" for p, v, _ in pair_stats[-5:]))

        boot = bootstrap_month_ci(val_rows, CANONICAL_COST)
        print(f"month-block bootstrap: mean={boot['boot_mean_net_bps']:+.2f}, p05={boot['boot_p05_net_bps']:+.2f}, p95={boot['boot_p95_net_bps']:+.2f}, P(>0)={boot['boot_prob_positive']:.1%}")

        test_rows = selected_rows(test, factor, canonical_orient, canonical_lo, canonical_hi)
        test_s = stats(test_rows, CANONICAL_COST)
        print(f"2026 diagnostic: net={test_s['net_bps']:+.2f} bps n={test_s['n']} (NOT GATE)")

        positive_thresholds = int(sum(np.isfinite(v) and v > 0 for _, v in threshold_results))
        positive_years = int(sum(np.isfinite(v) and v > 0 for v in yearly))
        positive_months = int(sum(np.isfinite(v) and v > 0 for v in month_nets))
        loo_positive = int(sum(np.isfinite(v) and v > 0 for v in loo_nets))
        loo_frac = loo_positive / len(loo_nets) if loo_nets else 0.0
        robust = (
            positive_thresholds == len(THRESHOLD_QS)
            and positive_years >= ROBUST_MIN_POSITIVE_YEARS
            and positive_months >= ROBUST_MIN_POSITIVE_MONTHS_2025
            and loo_frac >= ROBUST_MIN_LOO_POSITIVE_FRAC
        )

        summary_rows.append({
            "factor": factor,
            "horizon": HORIZON,
            "orientation": int(canonical_orient),
            "train_q25": canonical_lo,
            "train_q75": canonical_hi,
            "positive_thresholds": positive_thresholds,
            "thresholds_total": len(THRESHOLD_QS),
            "positive_years": positive_years,
            "years_total": 4,
            "positive_months_2025": positive_months,
            "months_total": 12,
            "loo_positive": loo_positive,
            "loo_total": len(loo_nets),
            "loo_positive_frac": loo_frac,
            **boot,
            "test_2026_net_bps": test_s["net_bps"],
            "robustness_pass": bool(robust),
        })
        print("ROBUSTNESS:", "PASS" if robust else "FAIL")

    summary = pd.DataFrame(summary_rows)
    detail = pd.DataFrame(detail_rows)
    summary.to_csv(outdir / "summary.csv", index=False)
    detail.to_csv(outdir / "details.csv", index=False)
    (outdir / "rules.json").write_text(json.dumps({
        "candidates": CANDIDATES,
        "horizon": HORIZON,
        "threshold_qs": THRESHOLD_QS,
        "canonical_cost_bps": CANONICAL_COST,
        "cost_sensitivity_bps": COSTS_BPS,
        "gate": {
            "all_threshold_variants_positive_2025": True,
            "positive_years_gte": ROBUST_MIN_POSITIVE_YEARS,
            "positive_months_2025_gte": ROBUST_MIN_POSITIVE_MONTHS_2025,
            "loo_pair_positive_fraction_gte": ROBUST_MIN_LOO_POSITIVE_FRAC,
            "uses_2026": False,
        },
    }, indent=2))

    print("\n=== ROBUSTNESS SUMMARY ===")
    cols = ["factor", "positive_thresholds", "positive_years", "positive_months_2025", "loo_positive_frac", "boot_p05_net_bps", "test_2026_net_bps", "robustness_pass"]
    print(summary[cols].to_string(index=False))
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
