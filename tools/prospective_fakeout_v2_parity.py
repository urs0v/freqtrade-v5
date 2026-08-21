#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc
from prospective_fakeout_v2 import FROZEN_PAIRS, HISTORY_START, compute_pair

HIST_END_EXCLUSIVE = pd.Timestamp("2026-08-20", tz="UTC")
MIN_RISK_BPS = 2.0
MAX_RISK_BPS = 3000.0


def parse_args():
    p = argparse.ArgumentParser(description="Parity gate: prospective engine vs frozen V1.6 causal historical output")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v16dir", default="/freqtrade/user_data/breakout_retest_profit_v16")
    p.add_argument("--outdir", default="/freqtrade/user_data/prospective_fakeout_v2")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def worker(pair, config, datadir):
    cfg = json.loads(Path(config).read_text())
    d = Path(datadir)
    raw5, src = dc.load_5m(cfg, d, pair)
    raw15 = dc.load_tf(cfg, d, pair, "15m")
    rows = compute_pair(pair, raw5, raw15, HISTORY_START, HIST_END_EXCLUSIVE)
    return rows, src


def key_frame(df: pd.DataFrame) -> pd.Series:
    et = pd.to_datetime(df["entry_time"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    lvl = pd.to_numeric(df["level_price"], errors="coerce").map(lambda x: f"{x:.10g}")
    return (
        df["pair"].astype(str) + "|" + et + "|" +
        pd.to_numeric(df["side"], errors="coerce").astype("Int64").astype(str) + "|" +
        df["tf"].astype(str) + "|" +
        pd.to_numeric(df["period"], errors="coerce").astype("Int64").astype(str) + "|" + lvl
    )


def metric(df, col):
    r = pd.to_numeric(df[col], errors="coerce").dropna().astype(float)
    if r.empty:
        return 0, np.nan, np.nan
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else np.inf
    return len(r), pf, float(r.mean())


def executable_frame(df: pd.DataFrame):
    """Mirror the frozen V1/V2 executable-outcome domain.

    V1.6 selected rows by activity/risk lower bound, but `_simulate_one()` still
    returns NaN for stops outside its frozen 2..3000bps admissible risk range.
    The prospective engine intentionally does not emit those non-executable rows.
    Parity therefore compares the finite executable domain, while separately
    requiring every excluded historical row to be explained by that same risk rule.
    """
    risk = pd.to_numeric(df.get("risk_bps"), errors="coerce")
    r8 = pd.to_numeric(df.get("net8_r"), errors="coerce")
    r12 = pd.to_numeric(df.get("stress12_r"), errors="coerce")
    in_risk = risk.between(MIN_RISK_BPS, MAX_RISK_BPS, inclusive="both")
    finite_r = np.isfinite(r8) & np.isfinite(r12)
    keep = in_risk & finite_r
    excluded = df.loc[~keep].copy()
    unexplained = int((~keep & in_risk).sum())
    return df.loc[keep].copy(), excluded, unexplained


def main():
    a = parse_args()
    v16 = Path(a.v16dir) / "causal_selected.csv"
    if not v16.exists():
        raise FileNotFoundError(f"Missing V1.6 causal reference: {v16}")
    ref = pd.read_csv(v16)
    ref["entry_time"] = pd.to_datetime(ref["entry_time"], utc=True)
    ref = ref[(ref.entry_time >= HISTORY_START) & (ref.entry_time < HIST_END_EXCLUSIVE)].copy()

    rows = []
    print("=== PROSPECTIVE V2 IMPLEMENTATION PARITY GATE ===", flush=True)
    print("Rebuilds the frozen fully-causal signal on the already-observed historical range and compares it to V1.6.", flush=True)
    with ProcessPoolExecutor(max_workers=max(1, int(a.workers))) as ex:
        futs = {ex.submit(worker, p, a.config, a.datadir): p for p in FROZEN_PAIRS}
        done = 0
        for fut in as_completed(futs):
            p = futs[fut]
            done += 1
            rr, src = fut.result()
            rows.extend(rr)
            print(f"pair {done:2d}/{len(FROZEN_PAIRS)} {p:24s} rows={len(rr)} source={src}", flush=True)

    got = pd.DataFrame(rows)
    if got.empty:
        raise RuntimeError("Prospective implementation produced no historical parity rows")
    got["entry_time"] = pd.to_datetime(got["entry_time"], utc=True)
    got = got[(got.entry_time >= HISTORY_START) & (got.entry_time < HIST_END_EXCLUSIVE)].copy()

    ref_raw_n = len(ref)
    got_raw_n = len(got)
    ref, ref_excluded, ref_unexplained = executable_frame(ref)
    got, got_excluded, got_unexplained = executable_frame(got)

    ref["key"] = key_frame(ref)
    got["key"] = key_frame(got)
    ref_keys = set(ref.key)
    got_keys = set(got.key)
    inter = ref_keys & got_keys
    union = ref_keys | got_keys
    jac = len(inter) / len(union) if union else 1.0

    joined = ref[["key", "net8_r", "stress12_r"]].merge(
        got[["key", "net8_r", "stress12_r"]], on="key", suffixes=("_ref", "_got")
    )
    d8 = (pd.to_numeric(joined.net8_r_ref, errors="coerce") - pd.to_numeric(joined.net8_r_got, errors="coerce")).abs()
    d12 = (pd.to_numeric(joined.stress12_r_ref, errors="coerce") - pd.to_numeric(joined.stress12_r_got, errors="coerce")).abs()
    max8 = float(d8.dropna().max()) if d8.notna().any() else np.nan
    max12 = float(d12.dropna().max()) if d12.notna().any() else np.nan

    rn, rpf, rexp = metric(ref, "net8_r")
    gn, gpf, gexp = metric(got, "net8_r")
    print("\n=== PARITY DOMAIN SANITY ===")
    print(
        f"raw reference rows={ref_raw_n} | executable={len(ref)} | excluded_by_frozen_risk_domain={len(ref_excluded)} | "
        f"unexplained_exclusions={ref_unexplained}"
    )
    print(
        f"raw prospect rows={got_raw_n} | executable={len(got)} | excluded_by_frozen_risk_domain={len(got_excluded)} | "
        f"unexplained_exclusions={got_unexplained}"
    )
    if len(ref_excluded):
        rr = pd.to_numeric(ref_excluded["risk_bps"], errors="coerce")
        print(f"reference excluded risk range: min={rr.min():.2f}bps max={rr.max():.2f}bps")

    print("\n=== PARITY RESULT ===")
    print(f"reference N={rn} PF={rpf:.6f} EXP={rexp:+.9f}R")
    print(f"prospect N={gn} PF={gpf:.6f} EXP={gexp:+.9f}R")
    print(f"keys overlap={len(inter)} ref-only={len(ref_keys-inter)} prospect-only={len(got_keys-inter)} Jaccard={jac:.6f}")
    print(f"matched max_abs_R_diff: 8bps={max8:.3g} 12bps={max12:.3g}")

    passed = (
        ref_unexplained == 0
        and got_unexplained == 0
        and len(ref) == len(got)
        and jac >= 0.999999
        and np.isfinite(max8) and max8 <= 1e-10
        and np.isfinite(max12) and max12 <= 1e-10
    )
    verdict = "PARITY_PASS" if passed else "PARITY_FAIL"
    print(f"\n=== VERDICT ===\n{verdict}")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "verdict": verdict,
        "reference_raw_n": ref_raw_n,
        "prospective_raw_n": got_raw_n,
        "reference_executable_n": len(ref),
        "prospective_executable_n": len(got),
        "reference_excluded": len(ref_excluded),
        "prospective_excluded": len(got_excluded),
        "reference_unexplained_exclusions": ref_unexplained,
        "prospective_unexplained_exclusions": got_unexplained,
        "overlap": len(inter),
        "ref_only": len(ref_keys-inter),
        "prospective_only": len(got_keys-inter),
        "jaccard": jac,
        "max_abs_r_diff_8bps": max8,
        "max_abs_r_diff_12bps": max12,
    }]).to_csv(out / "parity.csv", index=False)
    (out / "parity_pass.json").write_text(json.dumps({
        "verdict": verdict,
        "history_start": HISTORY_START.isoformat(),
        "history_end_exclusive": HIST_END_EXCLUSIVE.isoformat(),
        "reference_raw_n": ref_raw_n,
        "prospective_raw_n": got_raw_n,
        "reference_executable_n": len(ref),
        "prospective_executable_n": len(got),
        "reference_excluded": len(ref_excluded),
        "prospective_excluded": len(got_excluded),
        "reference_unexplained_exclusions": ref_unexplained,
        "prospective_unexplained_exclusions": got_unexplained,
        "jaccard": jac,
        "max_abs_r_diff_8bps": max8,
        "max_abs_r_diff_12bps": max12,
    }, indent=2))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
