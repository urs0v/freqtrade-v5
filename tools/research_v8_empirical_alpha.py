#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import talib.abstract as ta

from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType


FEATURES = [
    "mom1",
    "mom4",
    "mom16",
    "ema20_dist",
    "ema20_50",
    "donch_signed",
    "rsi15",
    "adx15",
    "volume_z_scaled",
    "vol_state",
    "trend1h",
    "mom1h",
    "rsi1h",
    "adx1h",
    "trend4h",
    "mom4h",
    "rsi4h",
    "adx4h",
]

RIDGE_GRID = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
HORIZONS = {"1h": 4, "4h": 16, "8h": 32, "12h": 48}


def as_utc_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def safe_tanh(x: pd.Series) -> pd.Series:
    return pd.Series(np.tanh(np.asarray(x, dtype=float)), index=x.index)


def htf_frame(base: pd.DataFrame, rule: str, minutes: int) -> pd.DataFrame:
    """Build completed HTF candles from 15m data and timestamp when they become usable.

    Freqtrade informative semantics on a 15m base allow a 1h candle opened at 10:00
    to be used on the 10:45 base candle (both close at 11:00). Therefore the HTF
    availability timestamp is block_start + (HTF - 15m).
    """
    x = base[["date", "open", "high", "low", "close", "volume"]].copy().set_index("date")
    h = x.resample(rule, label="left", closed="left").agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
    ).dropna(subset=["open", "high", "low", "close"])
    h = h.reset_index()
    h["atr"] = ta.ATR(h, timeperiod=20)
    h["atr_pct"] = h["atr"] / h["close"].replace(0, np.nan)
    h["rsi"] = ta.RSI(h, timeperiod=14)
    h["adx"] = ta.ADX(h, timeperiod=14)
    h["available_date"] = as_utc_ns(h["date"] + pd.Timedelta(minutes=minutes - 15))
    return h


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy().sort_values("date").reset_index(drop=True)
    df["date"] = as_utc_ns(df["date"])

    df["ema20"] = ta.EMA(df, timeperiod=20)
    df["ema50"] = ta.EMA(df, timeperiod=50)
    df["atr"] = ta.ATR(df, timeperiod=20)
    df["atr_pct"] = df["atr"] / df["close"].replace(0, np.nan)
    df["rsi"] = ta.RSI(df, timeperiod=14)
    df["adx"] = ta.ADX(df, timeperiod=14)
    df["ret1"] = df["close"].pct_change()
    df["ret4"] = df["close"].pct_change(4)
    df["ret16"] = df["close"].pct_change(16)

    dh = df["high"].rolling(32).max().shift(1)
    dl = df["low"].rolling(32).min().shift(1)
    dr = (dh - dl).replace(0, np.nan)
    donch_pos = ((df["close"] - dl) / dr).clip(0, 1)

    logv = np.log1p(df["volume"].clip(lower=0))
    med = logv.rolling(96, min_periods=48).median()
    mad = (logv - med).abs().rolling(96, min_periods=48).median()
    volume_z = ((logv - med) / (1.4826 * mad).replace(0, np.nan)).clip(-5, 5)
    atr_med = df["atr_pct"].rolling(192, min_periods=96).median()
    vol_ratio = df["atr_pct"] / atr_med.replace(0, np.nan)

    eps = 1e-12
    df["mom1"] = safe_tanh(df["ret1"] / (1.25 * df["atr_pct"] + eps))
    df["mom4"] = safe_tanh(df["ret4"] / (2.0 * df["atr_pct"] + eps))
    df["mom16"] = safe_tanh(df["ret16"] / (4.0 * df["atr_pct"] + eps))
    df["ema20_dist"] = safe_tanh((df["close"] - df["ema20"]) / (2.0 * df["atr"].replace(0, np.nan)))
    df["ema20_50"] = safe_tanh((df["ema20"] - df["ema50"]) / (3.0 * df["atr"].replace(0, np.nan)))
    df["donch_signed"] = (2.0 * donch_pos - 1.0).clip(-1, 1)
    df["rsi15"] = ((df["rsi"] - 50.0) / 25.0).clip(-1.5, 1.5)
    df["adx15"] = ((df["adx"] - 15.0) / 30.0).clip(0, 1.5)
    df["volume_z_scaled"] = (volume_z / 3.0).clip(-1.5, 1.5)
    df["vol_state"] = safe_tanh(np.log(vol_ratio.clip(lower=0.05, upper=20.0)))

    h1 = htf_frame(df, "1h", 60)
    h1["ema24"] = ta.EMA(h1, timeperiod=24)
    h1["ema72"] = ta.EMA(h1, timeperiod=72)
    h1["ret4"] = h1["close"].pct_change(4)
    h1["trend1h"] = safe_tanh((h1["ema24"] - h1["ema72"]) / (2.0 * h1["atr"].replace(0, np.nan)))
    h1["mom1h"] = safe_tanh(h1["ret4"] / (2.5 * h1["atr_pct"].replace(0, np.nan) + eps))
    h1["rsi1h"] = ((h1["rsi"] - 50.0) / 25.0).clip(-1.5, 1.5)
    h1["adx1h"] = ((h1["adx"] - 15.0) / 30.0).clip(0, 1.5)

    h4 = htf_frame(df, "4h", 240)
    h4["ema18"] = ta.EMA(h4, timeperiod=18)
    h4["ema54"] = ta.EMA(h4, timeperiod=54)
    h4["ret6"] = h4["close"].pct_change(6)
    h4["trend4h"] = safe_tanh((h4["ema18"] - h4["ema54"]) / (2.5 * h4["atr"].replace(0, np.nan)))
    h4["mom4h"] = safe_tanh(h4["ret6"] / (3.0 * h4["atr_pct"].replace(0, np.nan) + eps))
    h4["rsi4h"] = ((h4["rsi"] - 50.0) / 25.0).clip(-1.5, 1.5)
    h4["adx4h"] = ((h4["adx"] - 15.0) / 30.0).clip(0, 1.5)

    base = df.sort_values("date")
    base = pd.merge_asof(
        base,
        h1[["available_date", "trend1h", "mom1h", "rsi1h", "adx1h"]].sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    ).drop(columns=["available_date"])
    base = pd.merge_asof(
        base,
        h4[["available_date", "trend4h", "mom4h", "rsi4h", "adx4h"]].sort_values("available_date"),
        left_on="date",
        right_on="available_date",
        direction="backward",
    ).drop(columns=["available_date"])

    # Signal is known at this candle close; simulated market entry is next candle open.
    entry = base["open"].shift(-1)
    for name, bars in HORIZONS.items():
        base[f"y_{name}"] = base["close"].shift(-bars) / entry - 1.0
    # Predeclared multi-hour target. No 2026 information is used to choose the mix.
    base["target"] = (0.5 * base["y_4h"] + 0.5 * base["y_8h"]).clip(-0.10, 0.10)
    return base


def load_pair(config: dict, datadir: Path, pair: str) -> pd.DataFrame:
    return load_pair_history(
        pair=pair,
        timeframe="15m",
        datadir=datadir,
        fill_up_missing=False,
        drop_incomplete=False,
        data_format=config.get("dataformat_ohlcv"),
        candle_type=CandleType.FUTURES,
    )


def slice_rows(df: pd.DataFrame, start: str, end: str, step: int) -> pd.DataFrame:
    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    cols = [*FEATURES, "target", *[f"y_{h}" for h in HORIZONS]]
    x = df.loc[(df["date"] >= start_ts) & (df["date"] < end_ts), ["date", *cols]].copy()
    x = x.dropna(subset=[*FEATURES, "target"])
    if step > 1:
        x = x.iloc[::step].copy()
    return x


def fit_ridge(frames: list[pd.DataFrame], lam: float) -> np.ndarray:
    p = len(FEATURES) + 1
    xtx = np.zeros((p, p), dtype=np.float64)
    xty = np.zeros(p, dtype=np.float64)
    n = 0
    for f in frames:
        if f.empty:
            continue
        x = f[FEATURES].to_numpy(dtype=np.float64)
        y = f["target"].to_numpy(dtype=np.float64)
        X = np.column_stack([np.ones(len(x)), x])
        xtx += X.T @ X
        xty += X.T @ y
        n += len(x)
    if n == 0:
        raise RuntimeError("No training rows. Download older 15m futures data first.")
    penalty = np.eye(p)
    penalty[0, 0] = 0.0
    return np.linalg.solve(xtx + (lam * n) * penalty, xty)


def predict(frame: pd.DataFrame, beta: np.ndarray) -> np.ndarray:
    x = frame[FEATURES].to_numpy(dtype=np.float64)
    return beta[0] + x @ beta[1:]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def validation_score(frames: list[pd.DataFrame], beta: np.ndarray) -> tuple[float, float, int]:
    ps, ys = [], []
    for f in frames:
        if f.empty:
            continue
        ps.append(predict(f, beta))
        ys.append(f["target"].to_numpy(dtype=float))
    if not ps:
        return float("nan"), float("nan"), 0
    p = np.concatenate(ps)
    y = np.concatenate(ys)
    ic = spearman(p, y)
    q25, q75 = np.quantile(p, [0.25, 0.75])
    low = y[p <= q25]
    high = y[p >= q75]
    spread_bps = float((high.mean() - low.mean()) * 10000.0)
    return ic, spread_bps, len(y)


def evaluate(frames: list[tuple[str, pd.DataFrame]], beta: np.ndarray, val_abs_thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    for pair, f in frames:
        if f.empty:
            continue
        pred = predict(f, beta)
        side = np.where(pred >= 0, 1.0, -1.0)
        conf = np.abs(pred)
        bucket = np.searchsorted(val_abs_thresholds, conf, side="right")
        for h in HORIZONS:
            y = f[f"y_{h}"].to_numpy(dtype=float)
            ok = np.isfinite(y)
            if not ok.any():
                continue
            sret = side[ok] * y[ok]
            for b in sorted(set(bucket[ok])):
                m = ok & (bucket == b)
                x = side[m] * f.loc[m, f"y_{h}"].to_numpy(dtype=float)
                rows.append(
                    {
                        "pair": pair,
                        "horizon": h,
                        "conf_bucket": int(b),
                        "n": int(len(x)),
                        "mean_bps": float(np.mean(x) * 10000.0),
                        "median_bps": float(np.median(x) * 10000.0),
                        "win_pct": float(np.mean(x > 0) * 100.0),
                    }
                )
    return pd.DataFrame(rows)


def aggregate_eval(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return detail
    # Weighted aggregation of means/win rates by observations. Median is omitted here because
    # pair-level medians cannot be exactly re-aggregated without raw rows.
    out = []
    for (h, b), g in detail.groupby(["horizon", "conf_bucket"]):
        n = g["n"].sum()
        out.append(
            {
                "horizon": h,
                "conf_bucket": b,
                "n": int(n),
                "mean_bps": float(np.average(g["mean_bps"], weights=g["n"])),
                "win_pct": float(np.average(g["win_pct"], weights=g["n"])),
            }
        )
    return pd.DataFrame(out).sort_values(["horizon", "conf_bucket"])


def main() -> int:
    ap = argparse.ArgumentParser(description="V8 empirical price-alpha research. Train/validate pre-2026, freeze, then score 2026 OOS.")
    ap.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    ap.add_argument("--datadir", default=None)
    ap.add_argument("--train-start", default="2022-01-01")
    ap.add_argument("--train-end", default="2025-01-01")
    ap.add_argument("--val-start", default="2025-01-01")
    ap.add_argument("--val-end", default="2026-01-01")
    ap.add_argument("--test-start", default="2026-01-01")
    ap.add_argument("--test-end", default="2026-08-19")
    ap.add_argument("--sample-step", type=int, default=2, help="Use every Nth 15m candle for fitting; evaluation remains full-resolution.")
    ap.add_argument("--outdir", default="/freqtrade/user_data/v8/research")
    args = ap.parse_args()

    config = json.loads(Path(args.config).read_text())
    datadir = Path(args.datadir) if args.datadir else Path(config.get("datadir", "/freqtrade/user_data/data/binance"))
    pairs = list(config.get("exchange", {}).get("pair_whitelist", []))
    if not pairs:
        raise RuntimeError("No exchange.pair_whitelist in config")

    train_frames: list[pd.DataFrame] = []
    val_frames: list[pd.DataFrame] = []
    test_frames: list[tuple[str, pd.DataFrame]] = []
    coverage = []

    for pair in pairs:
        raw = load_pair(config, datadir, pair)
        if raw.empty:
            print(f"WARN no 15m futures data: {pair}")
            continue
        raw["date"] = as_utc_ns(raw["date"])
        coverage.append({"pair": pair, "from": raw["date"].min(), "to": raw["date"].max(), "candles": len(raw)})
        print(f"Features {pair}: {raw['date'].min()} -> {raw['date'].max()} ({len(raw)} candles)")
        feat = build_features(raw)
        tr = slice_rows(feat, args.train_start, args.train_end, max(1, args.sample_step))
        va = slice_rows(feat, args.val_start, args.val_end, max(1, args.sample_step))
        te = slice_rows(feat, args.test_start, args.test_end, 1)
        if not tr.empty:
            train_frames.append(tr)
        if not va.empty:
            val_frames.append(va)
        if not te.empty:
            test_frames.append((pair, te))

    if not train_frames or not val_frames:
        raise RuntimeError(
            "Insufficient pre-2026 training/validation rows. Ensure 15m futures data exists back to 2022. "
            "Use freqtrade download-data --prepend for older candles."
        )

    grid_rows = []
    best = None
    for lam in RIDGE_GRID:
        beta = fit_ridge(train_frames, lam)
        ic, spread, n = validation_score(val_frames, beta)
        grid_rows.append({"lambda": lam, "val_n": n, "val_spearman": ic, "val_top_minus_bottom_bps": spread})
        # Primary criterion: validation rank IC. Spread breaks near-ties.
        key = (ic if np.isfinite(ic) else -999.0, spread if np.isfinite(spread) else -999.0)
        if best is None or key > best[0]:
            best = (key, lam)

    assert best is not None
    best_lam = float(best[1])
    print("\n=== RIDGE VALIDATION (2025 only) ===")
    grid = pd.DataFrame(grid_rows)
    print(grid.to_string(index=False))
    print(f"Selected lambda: {best_lam:g}")

    # Freeze hyperparameter, then refit on all pre-2026 data.
    pre2026 = [*train_frames, *val_frames]
    beta = fit_ridge(pre2026, best_lam)

    # Confidence thresholds are also frozen from pre-2026 predictions.
    abs_pred = []
    for f in pre2026:
        abs_pred.append(np.abs(predict(f, beta)))
    abs_pred_all = np.concatenate(abs_pred)
    # Buckets: 0=bottom 50%, 1=50-75%, 2=75-90%, 3=90-97.5%, 4=top 2.5% confidence.
    thresholds = np.quantile(abs_pred_all, [0.50, 0.75, 0.90, 0.975])

    detail = evaluate(test_frames, beta, thresholds)
    agg = aggregate_eval(detail)

    coef = pd.DataFrame({"feature": ["intercept", *FEATURES], "coefficient": beta})
    coef["abs_coefficient"] = coef["coefficient"].abs()
    coef = coef.sort_values("abs_coefficient", ascending=False)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(coverage).to_csv(outdir / "coverage.csv", index=False)
    grid.to_csv(outdir / "validation_grid.csv", index=False)
    coef.to_csv(outdir / "coefficients.csv", index=False)
    detail.to_csv(outdir / "oos_by_pair.csv", index=False)
    agg.to_csv(outdir / "oos_confidence_buckets.csv", index=False)
    (outdir / "model.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "coefficients": beta.tolist(),
                "lambda": best_lam,
                "confidence_thresholds": thresholds.tolist(),
                "train": [args.train_start, args.train_end],
                "validation": [args.val_start, args.val_end],
                "test": [args.test_start, args.test_end],
                "sample_step": args.sample_step,
            },
            indent=2,
        )
    )

    print("\n=== FROZEN MODEL COEFFICIENTS ===")
    print(coef.head(12).to_string(index=False))
    print("\n=== 2026 OOS: SIDE-ADJUSTED RETURN BY PRE-2026 CONFIDENCE BUCKET ===")
    print(agg.to_string(index=False))
    print(f"\nOutput: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
