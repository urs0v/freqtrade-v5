#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import research_derivatives_alpha as r
import research_quarter_hour_orderflow as qh

SYMBOLS = qh.SYMBOLS
HORIZON = "12h"
BARS = 48
COST_BPS = 8.0
TRAIN_START, TRAIN_END = "2024-01-01", "2025-01-01"
VAL_START, VAL_END = "2025-01-01", "2026-01-01"
TEST_START, TEST_END = "2026-01-01", "2026-08-19"

# This is a paper-inspired OOS adaptation of Kim & Hansen (2026), Section 6.2:
# OI_t is decomposed into 12 own quarter-hour lags + 28 public OHLCV indicators + residual.
# Unlike the paper's full-sample inference regression, every coefficient here is fit on 2024 only,
# then frozen for the untouched 2025 validation and 2026 diagnostic.
# Exact last-trade aggressor-side reversal controls cannot be reconstructed from the compact DB,
# so this is deliberately labelled an OOS economic test rather than an exact paper replication.

TI_WINDOWS_PRICE = [4, 6, 12, 20, 32, 48, 96]       # 1h,1.5h,3h,5h,8h,12h,1d
TI_WINDOWS_VOLUME = [4, 6, 12, 16, 24, 32, 48]      # 1h,1.5h,3h,4h,6h,8h,12h
LAG_COLS = [f"oi_lag_{i}" for i in range(1, 13)]

# Predeclared before this test is run. 2026 cannot rescue/fail the gate.
GATE_MIN_TRAIN_NET_BPS = 0.0
GATE_MIN_VAL_NET_BPS = 2.0
GATE_MIN_VAL_IC = 0.005
GATE_MIN_POSITIVE_MONTHS = 7
GATE_MIN_VAL_POSITIONS = 500
GATE_REQUIRE_Q20_Q30_POSITIVE = True


def safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return a / b.where(b.abs() > eps)


def ema(s: pd.Series, bars: int) -> pd.Series:
    return s.ewm(span=bars, adjust=False, min_periods=bars).mean()


def wilder_rsi(close: pd.Series, bars: int = 24) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = (-d).clip(lower=0.0)
    au = up.ewm(alpha=1.0 / bars, adjust=False, min_periods=bars).mean()
    ad = dn.ewm(alpha=1.0 / bars, adjust=False, min_periods=bars).mean()
    rs = safe_div(au, ad)
    return 100.0 - 100.0 / (1.0 + rs)


def ti28(price: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    df = price[["date", "open", "high", "low", "close", "volume"]].copy().sort_values("date")
    df["date"] = r.as_ns(df["date"])
    o = pd.to_numeric(df["open"], errors="coerce")
    h = pd.to_numeric(df["high"], errors="coerce")
    l = pd.to_numeric(df["low"], errors="coerce")
    c = pd.to_numeric(df["close"], errors="coerce")
    v = pd.to_numeric(df["volume"], errors="coerce")

    out = pd.DataFrame({"feature_time": df["date"] + pd.Timedelta(minutes=15)})
    names: list[str] = []

    # Momentum: 6h RSI, 6h stochastic K/D, 6h stochastic RSI with double 1.5h smoothing, 6h CCI.
    rsi = wilder_rsi(c, 24)
    out["rsi_6h"] = rsi / 100.0
    names.append("rsi_6h")

    hh = h.rolling(24, min_periods=24).max()
    ll = l.rolling(24, min_periods=24).min()
    stoch_k = safe_div(c - ll, hh - ll) * 100.0
    stoch_d = stoch_k.rolling(6, min_periods=6).mean()
    out["stoch_k_6h"] = stoch_k / 100.0
    out["stoch_d_1p5h"] = stoch_d / 100.0
    names += ["stoch_k_6h", "stoch_d_1p5h"]

    rsi_lo = rsi.rolling(24, min_periods=24).min()
    rsi_hi = rsi.rolling(24, min_periods=24).max()
    srsi = safe_div(rsi - rsi_lo, rsi_hi - rsi_lo)
    srsi = srsi.rolling(6, min_periods=6).mean().rolling(6, min_periods=6).mean()
    out["stoch_rsi_6h"] = srsi
    names.append("stoch_rsi_6h")

    tp = (h + l + c) / 3.0
    tp_ma = tp.rolling(24, min_periods=24).mean()
    tp_mad = tp.rolling(24, min_periods=24).apply(lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True)
    out["cci_6h"] = safe_div(tp - tp_ma, 0.015 * tp_mad).clip(-10, 10)
    names.append("cci_6h")

    # Trend: seven SMA-relative close features + price MACD + MACD signal difference.
    for n in TI_WINDOWS_PRICE:
        col = f"px_sma_rel_{n}"
        out[col] = safe_div(c, c.rolling(n, min_periods=n).mean()) - 1.0
        names.append(col)
    p_macd = safe_div(ema(c, 8) - ema(c, 32), c)
    out["price_macd"] = p_macd
    out["price_macd_diff"] = p_macd - ema(p_macd, 6)
    names += ["price_macd", "price_macd_diff"]

    # Volume: seven SMA-relative features + relative volume MACD + signal difference + Chaikin-like flow.
    for n in TI_WINDOWS_VOLUME:
        col = f"vol_sma_rel_{n}"
        out[col] = safe_div(v, v.rolling(n, min_periods=n).mean()) - 1.0
        names.append(col)
    vslow = ema(v, 32)
    v_macd = safe_div(ema(v, 8) - vslow, vslow)
    out["volume_macd"] = v_macd
    out["volume_macd_diff"] = v_macd - ema(v_macd, 6)
    names += ["volume_macd", "volume_macd_diff"]

    clv = safe_div((c - l) - (h - c), h - l).fillna(0.0)
    adl = (clv * v.fillna(0.0)).cumsum()
    out["chaikin_flow"] = safe_div(ema(adl, 4) - ema(adl, 32), v.rolling(32, min_periods=32).mean()).clip(-20, 20)
    names.append("chaikin_flow")

    # Volatility: position vs. 6h Bollinger lower/mid/upper plus relative bandwidth.
    mid = c.rolling(24, min_periods=24).mean()
    sd = c.rolling(24, min_periods=24).std(ddof=0)
    lower = mid - 2.0 * sd
    upper = mid + 2.0 * sd
    out["bb_dist_lower"] = safe_div(c - lower, c)
    out["bb_dist_mid"] = safe_div(c - mid, c)
    out["bb_dist_upper"] = safe_div(c - upper, c)
    out["bb_width"] = safe_div(upper - lower, mid)
    names += ["bb_dist_lower", "bb_dist_mid", "bb_dist_upper", "bb_width"]

    assert len(names) == 28, len(names)
    return out, names


def build_symbol(config: dict, datadir: Path, qh_db: Path, symbol: str) -> tuple[pd.DataFrame, list[str]]:
    base = qh.build_symbol_frame(config, datadir, qh_db, symbol)
    if base.empty:
        return base, []
    price = r.load_price(config, datadir, qh.pair_for_symbol(symbol))
    if price.empty:
        return pd.DataFrame(), []

    ti, ti_cols = ti28(price)
    x = base.copy().sort_values("bucket_time")
    x["oi"] = pd.to_numeric(x["imbalance_qty"], errors="coerce").clip(-1.0, 1.0)
    for k in range(1, 13):
        x[f"oi_lag_{k}"] = x["oi"].shift(k)

    # Paper baseline uses indicators known at least 15m before boundary T.
    # Therefore we merge features available no later than T-15m.
    x["feature_cutoff"] = x["bucket_time"] - pd.Timedelta(minutes=15)
    x = pd.merge_asof(
        x.sort_values("feature_cutoff"),
        ti.sort_values("feature_time"),
        left_on="feature_cutoff",
        right_on="feature_time",
        direction="backward",
        tolerance=pd.Timedelta(minutes=20),
    ).sort_values("bucket_time")
    return x, ti_cols


def fit_standardizer(train: pd.DataFrame, cols: list[str]) -> tuple[pd.Series, pd.Series]:
    mu = train[cols].mean()
    sd = train[cols].std(ddof=0).replace(0.0, 1.0).fillna(1.0)
    return mu, sd


def transform(df: pd.DataFrame, cols: list[str], mu: pd.Series, sd: pd.Series) -> np.ndarray:
    return ((df[cols] - mu) / sd).to_numpy(dtype=float)


def fit_asset_model(df: pd.DataFrame, ti_cols: list[str]) -> dict:
    predictors = LAG_COLS + ti_cols
    train = df[(df["available_time"] >= pd.Timestamp(TRAIN_START, tz="UTC")) &
               (df["available_time"] < pd.Timestamp(TRAIN_END, tz="UTC"))].copy()
    train = train.dropna(subset=["oi", f"y_{HORIZON}", *predictors])
    if len(train) < 1000:
        raise RuntimeError(f"Insufficient 2024 rows: {len(train)}")

    mu, sd = fit_standardizer(train, predictors)
    X = transform(train, predictors, mu, sd)
    y_oi = train["oi"].to_numpy(dtype=float)
    A = np.column_stack([np.ones(len(X)), X])
    beta, *_ = np.linalg.lstsq(A, y_oi, rcond=None)
    intercept = float(beta[0])
    phi = beta[1:1 + len(LAG_COLS)]
    psi = beta[1 + len(LAG_COLS):]

    lag_comp = X[:, :len(LAG_COLS)] @ phi
    pub_comp = X[:, len(LAG_COLS):] @ psi
    residual = y_oi - lag_comp - pub_comp  # includes the intercept, matching the paper's residual-component convention.
    fitted = intercept + lag_comp + pub_comp
    denom = float(np.sum((y_oi - np.mean(y_oi)) ** 2))
    stage1_r2 = float(1.0 - np.sum((y_oi - fitted) ** 2) / denom) if denom > 0 else np.nan

    yret = train[f"y_{HORIZON}"].to_numpy(dtype=float)
    B = np.column_stack([np.ones(len(train)), lag_comp, pub_comp, residual])
    e, *_ = np.linalg.lstsq(B, yret, rcond=None)

    return {
        "predictors": predictors,
        "mu": mu,
        "sd": sd,
        "intercept_oi": intercept,
        "phi": phi,
        "psi": psi,
        "e0": float(e[0]),
        "e_lag": float(e[1]),
        "e_pub": float(e[2]),
        "e_res": float(e[3]),
        "stage1_r2": stage1_r2,
        "n_train_fit": int(len(train)),
    }


def apply_asset_model(df: pd.DataFrame, model: dict, ti_cols: list[str]) -> pd.DataFrame:
    predictors = model["predictors"]
    x = df.dropna(subset=["oi", *predictors]).copy()
    if x.empty:
        return x
    X = transform(x, predictors, model["mu"], model["sd"])
    lag = X[:, :len(LAG_COLS)] @ model["phi"]
    pub = X[:, len(LAG_COLS):] @ model["psi"]
    res = x["oi"].to_numpy(dtype=float) - lag - pub
    x["oi_lag_component"] = lag
    x["oi_public_component"] = pub
    x["oi_residual_component"] = res
    x["score_public"] = model["e_pub"] * pub
    x["score_lag"] = model["e_lag"] * lag
    x["score_full"] = model["e0"] + model["e_lag"] * lag + model["e_pub"] * pub + model["e_res"] * res
    return x


def split_nonoverlap(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    a = pd.Timestamp(start, tz="UTC")
    b = pd.Timestamp(end, tz="UTC")
    x = df[(df["available_time"] >= a) & (df["available_time"] < b) & (df[f"exit_time_{HORIZON}"] < b)].copy()
    x = x.dropna(subset=[f"y_{HORIZON}", "score_public"])
    return x.iloc[::BARS].copy() if not x.empty else x


def spearman(x: pd.Series, y: pd.Series) -> float:
    z = pd.concat([x, y], axis=1).dropna()
    if len(z) < 3:
        return float("nan")
    return float(z.iloc[:, 0].corr(z.iloc[:, 1], method="spearman"))


def thresholds(train: pd.DataFrame, q: float) -> tuple[float, float]:
    s = train["score_public"].dropna().to_numpy(dtype=float)
    return tuple(float(v) for v in np.quantile(s, [q, 1.0 - q]))


def trade_stats(df: pd.DataFrame, qlo: float, qhi: float, cost_bps: float = COST_BPS) -> dict:
    x = df[["bucket_time", "pair", "score_public", f"y_{HORIZON}"]].dropna().copy()
    score = x["score_public"].to_numpy(dtype=float)
    side = np.where(score >= qhi, 1.0, np.where(score <= qlo, -1.0, 0.0))
    chosen = side != 0
    x = x.loc[chosen].copy()
    side = side[chosen]
    if x.empty:
        return {"n": 0, "ic": np.nan, "gross_bps": np.nan, "net_bps": np.nan, "win_pct": np.nan, "positive_months": 0, "months": 0}
    ret = side * x[f"y_{HORIZON}"].to_numpy(dtype=float)
    net = ret - cost_bps / 10000.0
    monthly = pd.DataFrame({"month": x["bucket_time"].dt.strftime("%Y-%m"), "net": net}).groupby("month")["net"].mean() * 10000.0
    return {
        "n": int(len(ret)),
        "ic": spearman(x["score_public"], x[f"y_{HORIZON}"]),
        "gross_bps": float(np.mean(ret) * 10000.0),
        "net_bps": float(np.mean(net) * 10000.0),
        "win_pct": float(np.mean(net > 0) * 100.0),
        "positive_months": int((monthly > 0).sum()),
        "months": int(len(monthly)),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="OOS public-signal decomposition of quarter-hour opening imbalance")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    ap.add_argument("--qh-db", default="/freqtrade/user_data/alpha_lab/qh_orderflow.sqlite")
    ap.add_argument("--outdir", default="/freqtrade/user_data/alpha_lab/qh_public_oos")
    args = ap.parse_args()

    started = time.monotonic()
    config = json.loads(Path(args.config).read_text())
    qh_db = Path(args.qh_db)
    if not qh_db.exists():
        raise RuntimeError(f"Missing qh DB: {qh_db}")

    fitted_frames: list[pd.DataFrame] = []
    model_rows: list[dict] = []
    print("=== QH PUBLIC-SIGNAL OOS DECOMPOSITION ===")
    print("2024 fit only | 2025 untouched validation | 2026 diagnostic only")
    print("Paper-inspired: OI = 12 quarter-hour OI lags + TI28 public state + residual; gate uses public component at 12h.")
    print("No new raw downloads are required.")

    for i, sym in enumerate(SYMBOLS, 1):
        t0 = time.monotonic()
        frame, ti_cols = build_symbol(config, Path(args.datadir), qh_db, sym)
        if frame.empty:
            print(f"  [{i}/{len(SYMBOLS)}] {sym}: NO DATA", flush=True)
            continue
        model = fit_asset_model(frame, ti_cols)
        scored = apply_asset_model(frame, model, ti_cols)
        fitted_frames.append(scored)
        model_rows.append({
            "symbol": sym,
            "stage1_r2_pct": model["stage1_r2"] * 100.0,
            "e_pub_bps_per_unit": model["e_pub"] * 10000.0,
            "e_lag_bps_per_unit": model["e_lag"] * 10000.0,
            "e_res_bps_per_unit": model["e_res"] * 10000.0,
            "n_train_fit": model["n_train_fit"],
        })
        print(
            f"  [{i}/{len(SYMBOLS)}] {sym}: stage1 R2={model['stage1_r2']*100:.2f}% "
            f"e_pub={model['e_pub']*10000:+.1f}bps/unit n={model['n_train_fit']:,} [{time.monotonic()-t0:.1f}s]",
            flush=True,
        )

    if not fitted_frames:
        raise RuntimeError("No scored frames")

    all_df = pd.concat(fitted_frames, ignore_index=True)
    train_parts, val_parts, test_parts = [], [], []
    for _, g in all_df.groupby("pair", sort=True):
        g = g.sort_values("bucket_time")
        train_parts.append(split_nonoverlap(g, TRAIN_START, TRAIN_END))
        val_parts.append(split_nonoverlap(g, VAL_START, VAL_END))
        test_parts.append(split_nonoverlap(g, TEST_START, TEST_END))
    train = pd.concat(train_parts, ignore_index=True)
    val = pd.concat(val_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    print("\n=== PUBLIC-COMPONENT 12H GATE ===")
    print("Thresholds are learned on 2024 only; 8bps roundtrip cost; 2026 cannot affect PASS/FAIL.")
    scenarios = []
    for q in (0.20, 0.25, 0.30):
        qlo, qhi = thresholds(train, q)
        tr = trade_stats(train, qlo, qhi)
        va = trade_stats(val, qlo, qhi)
        te = trade_stats(test, qlo, qhi)
        scenarios.append({"q": q, "qlo": qlo, "qhi": qhi, "train": tr, "val": va, "test": te})
        print(
            f"q{int(q*100):02d}: train={tr['net_bps']:+7.2f}bps | "
            f"val={va['net_bps']:+7.2f}bps IC={va['ic']:+.4f} months={va['positive_months']}/{va['months']} n={va['n']:4d} | "
            f"2026={te['net_bps']:+7.2f}bps IC={te['ic']:+.4f}",
            flush=True,
        )

    can = next(s for s in scenarios if abs(s["q"] - 0.25) < 1e-12)
    q20 = next(s for s in scenarios if abs(s["q"] - 0.20) < 1e-12)
    q30 = next(s for s in scenarios if abs(s["q"] - 0.30) < 1e-12)
    tr, va = can["train"], can["val"]
    passed = bool(
        tr["net_bps"] > GATE_MIN_TRAIN_NET_BPS
        and va["net_bps"] > GATE_MIN_VAL_NET_BPS
        and va["ic"] > GATE_MIN_VAL_IC
        and va["positive_months"] >= GATE_MIN_POSITIVE_MONTHS
        and va["n"] >= GATE_MIN_VAL_POSITIONS
        and (
            not GATE_REQUIRE_Q20_Q30_POSITIVE
            or (q20["val"]["net_bps"] > 0 and q30["val"]["net_bps"] > 0)
        )
    )

    print("\n=== GATE SUMMARY ===")
    print(
        "PASS requires canonical q25: train net>0; 2025 net>2bps; val IC>0.005; "
        ">=7/12 positive months; n>=500; q20/q30 2025 net>0."
    )
    print("GATE:", "PASS" if passed else "FAIL")
    print("2026 is diagnostic only.")
    print("This test does NOT yet apply leverage. 10x/20x sizing is only evaluated after an unlevered edge clears robustness.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(model_rows).to_csv(outdir / "asset_models.csv", index=False)
    flat_rows = []
    for s in scenarios:
        for split in ("train", "val", "test"):
            flat_rows.append({"q": s["q"], "qlo": s["qlo"], "qhi": s["qhi"], "split": split, **s[split]})
    pd.DataFrame(flat_rows).to_csv(outdir / "public_signal_scenarios.csv", index=False)
    (outdir / "gate.json").write_text(json.dumps({
        "pass": passed,
        "factor": "quarter_hour_public_component",
        "horizon": HORIZON,
        "cost_bps": COST_BPS,
        "train": [TRAIN_START, TRAIN_END],
        "validation": [VAL_START, VAL_END],
        "test_diagnostic": [TEST_START, TEST_END],
        "criteria": {
            "train_net_bps_gt": GATE_MIN_TRAIN_NET_BPS,
            "val_net_bps_gt": GATE_MIN_VAL_NET_BPS,
            "val_ic_gt": GATE_MIN_VAL_IC,
            "val_positive_months_gte": GATE_MIN_POSITIVE_MONTHS,
            "val_positions_gte": GATE_MIN_VAL_POSITIONS,
            "q20_q30_val_net_positive": GATE_REQUIRE_Q20_Q30_POSITIVE,
        },
        "note": "Paper-inspired OOS adaptation; exact last-trade reversal controls unavailable in compact qh DB.",
    }, indent=2))
    print(f"Output: {outdir}")
    print(f"Runtime: {time.monotonic()-started:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
