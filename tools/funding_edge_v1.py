#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v4_stage0 as s0

PAIRS = (
    "AAVE","ADA","ATOM","AVAX","BCH","BNB","BTC","DOGE","DOT","ETC",
    "ETH","FIL","LINK","LTC","NEAR","SOL","TRX","UNI","XLM","XRP",
)
HORIZONS_MIN = (15,30,60,120,240,480)
DISCOVERY_START = pd.Timestamp("2025-11-01", tz="UTC")
VALIDATION_START = pd.Timestamp("2026-04-01", tz="UTC")
HOLDOUT_START = pd.Timestamp("2026-06-01", tz="UTC")
END_EXCL = pd.Timestamp("2026-08-20", tz="UTC")


def parse_args():
    p = argparse.ArgumentParser(description="Funding Edge V1: causal post-funding reversal research.")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance/futures")
    p.add_argument("--outdir", default="/freqtrade/user_data/funding_edge_v1")
    p.add_argument("--pairs", default=",".join(PAIRS))
    p.add_argument("--fee-bps-side", type=float, default=5.0)
    p.add_argument("--slippage-bps-side", type=float, default=1.0)
    p.add_argument("--rolling-events", type=int, default=90)
    p.add_argument("--min-history", type=int, default=45)
    return p.parse_args()


def funding_path(datadir: Path, sym: str) -> Path:
    return datadir / f"{sym}_USDT_USDT-1h-funding_rate.feather"


def load_funding(datadir: Path, sym: str, rolling_events: int, min_history: int) -> pd.DataFrame:
    p = funding_path(datadir, sym)
    if not p.exists():
        raise FileNotFoundError(p)
    x = pd.read_feather(p)[["date","open"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x["funding"] = pd.to_numeric(x["open"], errors="coerce")
    x = x.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    prior = x["funding"].shift(1)
    abs_prior = prior.abs()
    roll = prior.rolling(rolling_events, min_periods=min_history)
    aroll = abs_prior.rolling(rolling_events, min_periods=min_history)
    x["prior_mean"] = roll.mean()
    x["prior_std"] = roll.std(ddof=0)
    x["z"] = (x["funding"] - x["prior_mean"]) / x["prior_std"].replace(0, np.nan)
    x["q90"] = roll.quantile(.90)
    x["q95"] = roll.quantile(.95)
    x["q10"] = roll.quantile(.10)
    x["q05"] = roll.quantile(.05)
    x["abs_q90"] = aroll.quantile(.90)
    x["abs_q95"] = aroll.quantile(.95)
    x["symbol"] = sym
    return x


def attach_cross_section(z: pd.DataFrame) -> pd.DataFrame:
    z = z.copy()
    counts = z.groupby("date")["funding"].transform("size")
    z["cs_pct"] = z.groupby("date")["funding"].rank(method="average", pct=True)
    z["abs_funding"] = z["funding"].abs()
    z["cs_abs_pct"] = z.groupby("date")["abs_funding"].rank(method="average", pct=True)
    z.loc[counts < 10, ["cs_pct", "cs_abs_pct"]] = np.nan
    return z.drop(columns=["abs_funding"])


def load_1m(datadir: Path, sym: str) -> pd.DataFrame:
    p = s0._data_path(datadir, sym, "1m")
    x = pd.read_feather(p)[["date","open"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    x["open"] = pd.to_numeric(x["open"], errors="coerce")
    return x.dropna().sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)


def add_forward_returns(g: pd.DataFrame, px: pd.DataFrame, roundtrip_cost: float) -> pd.DataFrame:
    dates = px["date"].to_numpy(dtype="datetime64[ns]")
    op = px["open"].to_numpy(float)
    rows = []
    for r in g.itertuples(index=False):
        t = pd.Timestamp(r.date)
        # Funding at t must already be known. Enter only on first 1m bar strictly after t.
        ei = int(np.searchsorted(dates, np.datetime64(t.to_datetime64()), side="right"))
        if ei >= len(px):
            continue
        entry_time = pd.Timestamp(px.iloc[ei]["date"])
        entry = float(op[ei])
        q = r._asdict()
        q["entry_time"] = entry_time
        q["entry_price"] = entry
        for h in HORIZONS_MIN:
            target = entry_time + pd.Timedelta(minutes=h)
            xi = int(np.searchsorted(dates, np.datetime64(target.to_datetime64()), side="left"))
            if xi >= len(px):
                q[f"ret_{h}m"] = np.nan
            else:
                q[f"ret_{h}m"] = float(op[xi] / entry - 1.0)
        rows.append(q)
    return pd.DataFrame(rows)


def variant_side(df: pd.DataFrame, variant: str) -> pd.Series:
    f = df["funding"]
    base = variant.replace("_CONT", "_REV")
    if base == "ABS_Q90_REV":
        sig = f.abs() >= df["abs_q90"]
    elif base == "ABS_Q95_REV":
        sig = f.abs() >= df["abs_q95"]
    elif base == "SIGNED_Q90_REV":
        sig = (f >= df["q90"]) | (f <= df["q10"])
    elif base == "SIGNED_Q95_REV":
        sig = (f >= df["q95"]) | (f <= df["q05"])
    elif base == "Z2_REV":
        sig = df["z"].abs() >= 2.0
    elif base == "Z2_5_REV":
        sig = df["z"].abs() >= 2.5
    elif base == "CS_TOPBOTTOM10_REV":
        sig = (df["cs_pct"] >= .90) | (df["cs_pct"] <= .10)
    elif base == "CS_TOPBOTTOM20_REV":
        sig = (df["cs_pct"] >= .80) | (df["cs_pct"] <= .20)
    else:
        raise ValueError(variant)
    # Positive funding => short on reversal, long on continuation; inverse for negative funding.
    reversal = pd.Series(np.where(f > 0, -1.0, np.where(f < 0, 1.0, 0.0)), index=df.index)
    side = -reversal if variant.endswith("_CONT") else reversal
    return side.where(sig, 0.0)


BASE_VARIANTS = (
    "ABS_Q90_REV","ABS_Q95_REV","SIGNED_Q90_REV","SIGNED_Q95_REV",
    "Z2_REV","Z2_5_REV","CS_TOPBOTTOM10_REV","CS_TOPBOTTOM20_REV",
)
VARIANTS = BASE_VARIANTS + tuple(v.replace("_REV", "_CONT") for v in BASE_VARIANTS)


def split_name(t: pd.Timestamp) -> str | None:
    if DISCOVERY_START <= t < VALIDATION_START:
        return "DISCOVERY"
    if VALIDATION_START <= t < HOLDOUT_START:
        return "VALIDATION"
    if HOLDOUT_START <= t < END_EXCL:
        return "HOLDOUT"
    return None


def pf(v: pd.Series):
    pos = v[v > 0].sum()
    neg = -v[v < 0].sum()
    if neg > 0:
        return float(pos / neg)
    return math.inf if pos > 0 else None


def stats_for(x: pd.DataFrame, col: str) -> dict:
    v = x[col].dropna().astype(float)
    if len(v) == 0:
        return {"n": 0}
    sd = float(v.std(ddof=1)) if len(v) > 1 else 0.0
    se = sd / math.sqrt(len(v)) if len(v) else np.nan
    return {
        "n": int(len(v)),
        "mean": float(v.mean()),
        "median": float(v.median()),
        "winrate": float((v > 0).mean()),
        "pf": pf(v),
        "stdev": sd,
        "se": float(se),
        "ci95_lo": float(v.mean() - 1.96 * se) if np.isfinite(se) else None,
        "ci95_hi": float(v.mean() + 1.96 * se) if np.isfinite(se) else None,
    }


def main():
    a = parse_args()
    datadir = Path(a.datadir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pairs = [s.strip().upper() for s in a.pairs.split(",") if s.strip()]
    roundtrip_cost = 2.0 * (a.fee_bps_side + a.slippage_bps_side) / 10000.0

    fund = []
    meta = []
    for sym in pairs:
        x = load_funding(datadir, sym, a.rolling_events, a.min_history)
        meta.append({
            "symbol": sym, "funding_rows": int(len(x)),
            "funding_start": str(x["date"].min()), "funding_end": str(x["date"].max()),
        })
        fund.append(x)
    f = pd.concat(fund, ignore_index=True)
    f = attach_cross_section(f)

    enriched = []
    for sym, g in f.groupby("symbol", sort=True):
        px = load_1m(datadir, sym)
        z = add_forward_returns(g, px, roundtrip_cost)
        enriched.append(z)
    e = pd.concat(enriched, ignore_index=True)
    e["split"] = e["date"].map(split_name)
    e = e[e["split"].notna()].copy()

    long_rows = []
    detail = []
    for variant in VARIANTS:
        side = variant_side(e, variant)
        mask = side != 0
        base = e.loc[mask].copy()
        base["side"] = side.loc[mask].astype(int)
        base["variant"] = variant
        for h in HORIZONS_MIN:
            raw = base[f"ret_{h}m"].astype(float)
            base[f"gross_{h}m"] = base["side"] * raw
            base[f"net_{h}m"] = base[f"gross_{h}m"] - roundtrip_cost

            for split in ("DISCOVERY","VALIDATION","HOLDOUT"):
                q = base[base["split"] == split]
                gross = stats_for(q, f"gross_{h}m")
                net = stats_for(q, f"net_{h}m")
                long_rows.append({
                    "variant": variant, "horizon_min": h, "split": split,
                    "gross_n": gross.get("n", 0),
                    "gross_mean": gross.get("mean"), "gross_median": gross.get("median"),
                    "gross_winrate": gross.get("winrate"), "gross_pf": gross.get("pf"),
                    "net_mean": net.get("mean"), "net_median": net.get("median"),
                    "net_winrate": net.get("winrate"), "net_pf": net.get("pf"),
                    "net_ci95_lo": net.get("ci95_lo"), "net_ci95_hi": net.get("ci95_hi"),
                })
        cols = ["symbol","date","entry_time","funding","z","cs_pct","cs_abs_pct","split","side","variant"]
        cols += [f"net_{h}m" for h in HORIZONS_MIN] + [f"gross_{h}m" for h in HORIZONS_MIN]
        detail.append(base[cols])

    stats = pd.DataFrame(long_rows)
    events = pd.concat(detail, ignore_index=True) if detail else pd.DataFrame()

    # Pre-defined automatic selection: no holdout information is used.
    candidates = []
    for (variant, h), g in stats.groupby(["variant","horizon_min"]):
        d = g[g["split"] == "DISCOVERY"]
        v = g[g["split"] == "VALIDATION"]
        ho = g[g["split"] == "HOLDOUT"]
        if d.empty or v.empty:
            continue
        dr, vr = d.iloc[0], v.iloc[0]
        passes = (
            dr["gross_n"] >= 30 and vr["gross_n"] >= 15
            and pd.notna(dr["net_mean"]) and dr["net_mean"] > 0
            and pd.notna(vr["net_mean"]) and vr["net_mean"] > 0
            and pd.notna(dr["net_pf"]) and dr["net_pf"] > 1.0
            and pd.notna(vr["net_pf"]) and vr["net_pf"] > 1.0
        )
        if passes:
            hr = ho.iloc[0] if not ho.empty else None
            candidates.append({
                "variant": variant, "horizon_min": int(h),
                "discovery_n": int(dr["gross_n"]), "discovery_net_mean": float(dr["net_mean"]),
                "validation_n": int(vr["gross_n"]), "validation_net_mean": float(vr["net_mean"]),
                "holdout_n": int(hr["gross_n"]) if hr is not None else 0,
                "holdout_net_mean": float(hr["net_mean"]) if hr is not None and pd.notna(hr["net_mean"]) else None,
                "holdout_net_pf": float(hr["net_pf"]) if hr is not None and pd.notna(hr["net_pf"]) else None,
                "holdout_winrate": float(hr["net_winrate"]) if hr is not None and pd.notna(hr["net_winrate"]) else None,
            })

    stats.to_csv(outdir / "horizon_stats.csv", index=False)
    events.to_csv(outdir / "signal_events.csv", index=False)

    monthly = []
    if not events.empty:
        events["month"] = pd.to_datetime(events["date"], utc=True).dt.to_period("M").astype(str)
        for variant, g in events.groupby("variant"):
            for h in HORIZONS_MIN:
                for month, q in g.groupby("month"):
                    vv = q[f"net_{h}m"].dropna()
                    if len(vv):
                        monthly.append({
                            "variant": variant, "horizon_min": h, "month": month,
                            "n": int(len(vv)), "net_mean": float(vv.mean()),
                            "winrate": float((vv > 0).mean()),
                        })
    pd.DataFrame(monthly).to_csv(outdir / "monthly_stats.csv", index=False)

    summary = {
        "stage": "Funding Edge V1 causal post-funding directional test",
        "pairs": pairs,
        "period": {"discovery": ["2025-11-01","2026-04-01"],
                   "validation": ["2026-04-01","2026-06-01"],
                   "locked_holdout": ["2026-06-01","2026-08-20"]},
        "causality": {
            "funding_feature": "current funding event plus rolling features computed only from prior funding events",
            "entry": "first 1m bar strictly after funding timestamp",
            "forward_returns_minutes": list(HORIZONS_MIN),
            "holdout_selection_used": False,
        },
        "costs": {
            "fee_bps_side": a.fee_bps_side,
            "slippage_bps_side": a.slippage_bps_side,
            "roundtrip_bps": roundtrip_cost * 10000.0,
            "funding_payment_at_signal": "not included; entry occurs after the funding timestamp",
        },
        "predefined_variants": list(VARIANTS),
        "selection_rule": ">=30 discovery and >=15 validation events; positive net mean and PF>1 in both; holdout not used for selection",
        "surviving_candidates": candidates,
        "data_meta": meta,
        "files": {
            "horizon_stats": str(outdir / "horizon_stats.csv"),
            "signal_events": str(outdir / "signal_events.csv"),
            "monthly_stats": str(outdir / "monthly_stats.csv"),
        },
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("=== FUNDING EDGE V1 ===")
    print(f"pairs={len(pairs)} event_rows={len(e)} roundtrip_cost={roundtrip_cost*10000:.1f} bps")
    print(f"survivors={len(candidates)}")
    for q in candidates:
        hmean = q["holdout_net_mean"]
        print(
            f"{q['variant']} {q['horizon_min']}m: "
            f"D n={q['discovery_n']} E={q['discovery_net_mean']*100:.4f}% | "
            f"V n={q['validation_n']} E={q['validation_net_mean']*100:.4f}% | "
            f"H n={q['holdout_n']} E={(hmean if hmean is not None else 0)*100:.4f}% "
            f"PF={q['holdout_net_pf']}"
        )
    print(f"summary={outdir / 'summary.json'}")


if __name__ == "__main__":
    main()
