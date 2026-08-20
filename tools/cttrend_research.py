#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import warnings

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet


FEATURES = [
    "rsi", "stochRSI", "stochK", "stochD", "cci",
    "sma_3d", "sma_5d", "sma_10d", "sma_20d", "sma_50d", "sma_100d", "sma_200d",
    "macd", "macd_diff_signal",
    "volsma_3d", "volsma_5d", "volsma_10d", "volsma_20d",
    "volsma_50d", "volsma_100d", "volsma_200d",
    "volmacd", "volmacd_diff_signal", "chaikin",
    "boll_low", "boll_mid", "boll_high", "boll_width",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Point-in-time Binance USD-M CTREND + 28d TSMOM research backtest"
    )
    ap.add_argument(
        "--db",
        default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite",
        help="SQLite DB created by backfill_adaptivetrend_core_data_fast.py",
    )
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2026-07-31")
    ap.add_argument("--universe", type=int, default=50, help="Top N by trailing 30d quote volume")
    ap.add_argument("--top-frac", type=float, default=0.20, help="Top CTREND fraction held long")
    ap.add_argument("--min-history-days", type=int, default=210)
    ap.add_argument("--train-weeks", type=int, default=52)
    ap.add_argument("--side-cost-bps", type=float, default=7.0, help="Fee+slippage per traded side")
    ap.add_argument("--min-cross-section", type=int, default=15)
    ap.add_argument("--output-dir", default="/freqtrade/user_data/cttrend_research")
    return ap.parse_args()


def ema_paper(s: pd.Series, length: int) -> pd.Series:
    # Supplement: alpha = 1 / (1 + L), not the common 2 / (L + 1).
    return s.ewm(alpha=1.0 / (1.0 + length), adjust=False, min_periods=length).mean()


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num / den.replace(0.0, np.nan)


def add_technical_features(g: pd.DataFrame) -> pd.DataFrame:
    """All 28 daily technical indicators from the CTREND online supplement."""
    g = g.sort_values("date").copy()
    close = g["close"].astype(float)
    high = g["high"].astype(float)
    low = g["low"].astype(float)
    dollar_vol = g["quote_volume"].astype(float).clip(lower=0.0)

    diff = close.diff()
    gains = diff.clip(lower=0.0)
    losses = (-diff).clip(lower=0.0)
    avg_gain = gains.rolling(14, min_periods=14).mean()
    avg_loss = losses.rolling(14, min_periods=14).mean()
    rs = safe_div(avg_gain, avg_loss)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(avg_loss > 0, 100.0)
    rsi = rsi.where(avg_gain > 0, 0.0)
    both_zero = (avg_gain == 0) & (avg_loss == 0)
    rsi = rsi.where(~both_zero, 50.0)
    g["rsi"] = rsi

    rsi_lo = rsi.rolling(14, min_periods=14).min()
    rsi_hi = rsi.rolling(14, min_periods=14).max()
    g["stochRSI"] = safe_div(rsi - rsi_lo, rsi_hi - rsi_lo)

    ll = low.rolling(14, min_periods=14).min()
    hh = high.rolling(14, min_periods=14).max()
    g["stochK"] = safe_div(close - ll, hh - ll)
    g["stochD"] = g["stochK"].rolling(3, min_periods=3).mean()

    typical = (close + high + low) / 3.0
    typical_sma = typical.rolling(20, min_periods=20).mean()
    avg_dev = typical.rolling(20, min_periods=20).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
    )
    g["cci"] = safe_div(typical - typical_sma, 0.015 * avg_dev)

    for length in (3, 5, 10, 20, 50, 100, 200):
        g[f"sma_{length}d"] = safe_div(
            close.rolling(length, min_periods=length).mean(), close
        )

    ema12 = ema_paper(close, 12)
    ema26 = ema_paper(close, 26)
    macd = safe_div(ema12 - ema26, ema12)
    g["macd"] = macd
    g["macd_diff_signal"] = macd - ema_paper(macd, 9)

    for length in (3, 5, 10, 20, 50, 100, 200):
        g[f"volsma_{length}d"] = safe_div(
            dollar_vol.rolling(length, min_periods=length).mean(), dollar_vol
        )

    vema12 = ema_paper(dollar_vol, 12)
    vema26 = ema_paper(dollar_vol, 26)
    volmacd = safe_div(vema12 - vema26, vema12)
    g["volmacd"] = volmacd
    g["volmacd_diff_signal"] = volmacd - ema_paper(volmacd, 9)

    hl = (high - low).replace(0.0, np.nan)
    ad = safe_div((close - low) - (high - close), hl) * dollar_vol
    g["chaikin"] = safe_div(
        ad.rolling(21, min_periods=21).sum(),
        dollar_vol.rolling(21, min_periods=21).sum(),
    )

    mid_raw = close.rolling(20, min_periods=20).mean()
    sd = close.rolling(20, min_periods=20).std(ddof=0)
    low_raw = mid_raw - 2.0 * sd
    high_raw = mid_raw + 2.0 * sd
    g["boll_low"] = safe_div(low_raw, close)
    g["boll_mid"] = safe_div(mid_raw, close)
    g["boll_high"] = safe_div(high_raw, close)
    g["boll_width"] = safe_div(high_raw - low_raw, mid_raw)

    g["ret_28d"] = close.pct_change(28, fill_method=None)
    g["liq_30d"] = dollar_vol.rolling(30, min_periods=30).mean()
    g["history_days"] = np.arange(1, len(g) + 1)
    return g


def load_daily(con: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    warm_start = start - pd.Timedelta(days=900)
    start_ms = int(warm_start.timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1

    symbols = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT symbol FROM candles WHERE open_time BETWEEN ? AND ? ORDER BY symbol",
            (start_ms, end_ms),
        )
    ]
    print(f"DB symbols with candles in range: {len(symbols)}", flush=True)

    frames: list[pd.DataFrame] = []
    for n, symbol in enumerate(symbols, 1):
        rows = con.execute(
            """
            SELECT open_time, open, high, low, close, quote_volume
            FROM candles
            WHERE symbol = ? AND open_time BETWEEN ? AND ?
            ORDER BY open_time
            """,
            (symbol, start_ms, end_ms),
        ).fetchall()
        if not rows:
            continue
        x = pd.DataFrame(
            rows, columns=["open_time", "open", "high", "low", "close", "quote_volume"]
        )
        x["date"] = pd.to_datetime(x["open_time"], unit="ms", utc=True).dt.floor("D")

        daily = (
            x.groupby("date", sort=True)
            .agg(
                open=("open", "first"),
                high=("high", "max"),
                low=("low", "min"),
                close=("close", "last"),
                quote_volume=("quote_volume", "sum"),
                bars=("open_time", "count"),
            )
            .reset_index()
        )
        # 6h archive should have exactly four candles per full UTC day.
        daily = daily[daily["bars"] == 4].drop(columns="bars")
        if daily.empty:
            continue
        daily["symbol"] = symbol
        daily = add_technical_features(daily)
        frames.append(daily)
        if n % 50 == 0 or n == len(symbols):
            print(f"Daily aggregation/features: {n}/{len(symbols)}", flush=True)

    if not frames:
        raise RuntimeError("No usable candle data found")
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    print(
        f"Daily rows={len(out):,}, symbols={out['symbol'].nunique()}, "
        f"date={out['date'].min().date()}..{out['date'].max().date()}",
        flush=True,
    )
    return out


def cross_section_rank(s: pd.Series) -> pd.Series:
    valid = s.notna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    n = int(valid.sum())
    if n <= 1:
        return out
    r = s[valid].rank(method="average")
    out.loc[valid] = (r - 1.0) / (n - 1.0) - 0.5
    return out


def make_weekly_panel(
    daily: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    universe_n: int,
    min_history_days: int,
    min_cross_section: int,
) -> pd.DataFrame:
    # Sunday UTC close is the information set / rebalance timestamp.
    sun = daily[daily["date"].dt.dayofweek == 6].copy()
    if sun.empty:
        raise RuntimeError("No Sunday snapshots found")

    # Compute future return before liquidity filtering. A symbol can drop from next
    # week's top-N and its next-week return must still remain observable.
    sun = sun.sort_values(["symbol", "date"])
    sun["next_close"] = sun.groupby("symbol")["close"].shift(-1)
    sun["next_date"] = sun.groupby("symbol")["date"].shift(-1)
    sun["fwd_ret"] = sun["next_close"] / sun["close"] - 1.0
    sun["target_ok"] = sun["next_date"].eq(sun["date"] + pd.Timedelta(days=7))
    sun.loc[~sun["target_ok"], "fwd_ret"] = np.nan

    sun = sun.dropna(subset=FEATURES + ["liq_30d", "close"])
    sun = sun[sun["history_days"] >= min_history_days]

    selected_weeks: list[pd.DataFrame] = []
    for dt, x in sun.groupby("date", sort=True):
        x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["liq_30d"])
        x = x[x["liq_30d"] > 0]
        if len(x) < min_cross_section:
            continue
        x = x.nlargest(min(universe_n, len(x)), "liq_30d").copy()
        if len(x) < min_cross_section:
            continue
        for f in FEATURES:
            x[f"z_{f}"] = cross_section_rank(x[f])
        # Practical futures analogue of value-weighted regressions: observable
        # trailing dollar liquidity, known at rebalance time.
        x["reg_weight"] = x["liq_30d"] / x["liq_30d"].sum()
        selected_weeks.append(x)

    if not selected_weeks:
        raise RuntimeError("No eligible weekly cross-sections")

    panel = pd.concat(selected_weeks, ignore_index=True)
    panel = panel.sort_values(["date", "symbol"]).reset_index(drop=True)
    panel = panel[panel["date"] <= end].copy()
    print(
        f"Weekly panel rows={len(panel):,}, weeks={panel['date'].nunique()}, "
        f"median universe={panel.groupby('date').size().median():.0f}",
        flush=True,
    )
    return panel


def weighted_univariate(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float] | None:
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if mask.sum() < 5:
        return None
    x = x[mask]
    y = y[mask]
    w = w[mask]
    w = w / w.sum()
    xm = float(np.sum(w * x))
    ym = float(np.sum(w * y))
    var = float(np.sum(w * (x - xm) ** 2))
    if var <= 1e-12:
        return None
    cov = float(np.sum(w * (x - xm) * (y - ym)))
    beta = cov / var
    alpha = ym - beta * xm
    return alpha, beta


@dataclass
class ModelFit:
    ctrend: pd.Series
    selected_features: list[str]
    alpha: float


def fit_predict_week(
    panel: pd.DataFrame,
    current_date: pd.Timestamp,
    train_weeks: int,
    min_cross_section: int,
) -> ModelFit | None:
    dates = sorted(d for d in panel["date"].unique() if d < current_date)
    if len(dates) < train_weeks:
        return None
    train_dates = dates[-train_weeks:]

    week_coefs: dict[str, list[tuple[float, float]]] = {f: [] for f in FEATURES}
    train_chunks: list[pd.DataFrame] = []

    for dt in train_dates:
        x = panel[(panel["date"] == dt) & panel["fwd_ret"].notna()].copy()
        if len(x) < min_cross_section:
            continue
        train_chunks.append(x)
        y = x["fwd_ret"].to_numpy(float)
        w = x["reg_weight"].to_numpy(float)
        for f in FEATURES:
            z = x[f"z_{f}"].to_numpy(float)
            coef = weighted_univariate(z, y, w)
            if coef is not None:
                week_coefs[f].append(coef)

    if len(train_chunks) < max(26, train_weeks // 2):
        return None

    avg_coef: dict[str, tuple[float, float]] = {}
    for f, vals in week_coefs.items():
        if len(vals) >= max(20, len(train_chunks) // 2):
            avg_coef[f] = (
                float(np.mean([v[0] for v in vals])),
                float(np.mean([v[1] for v in vals])),
            )

    if len(avg_coef) < 5:
        return None

    train = pd.concat(train_chunks, ignore_index=True)
    cols = list(avg_coef)
    X = np.column_stack(
        [
            avg_coef[f][0] + avg_coef[f][1] * train[f"z_{f}"].to_numpy(float)
            for f in cols
        ]
    )
    y = train["fwd_ret"].to_numpy(float)
    sw = train["reg_weight"].to_numpy(float)

    mask = np.isfinite(y) & np.isfinite(sw) & (sw > 0) & np.all(np.isfinite(X), axis=1)
    X, y, sw = X[mask], y[mask], sw[mask]
    if len(y) < 100 or X.shape[1] < 5:
        return None

    sw = sw / np.mean(sw)

    # Standardization only stabilizes the Elastic Net selection path. The final
    # CTREND score still averages the original univariate return forecasts.
    x_mean = np.average(X, axis=0, weights=sw)
    x_var = np.average((X - x_mean) ** 2, axis=0, weights=sw)
    x_std = np.sqrt(np.maximum(x_var, 1e-12))
    Xs = (X - x_mean) / x_std

    # Supplement: l1/l2 trade-off = 0.5 and lambda selected with corrected AIC.
    yc = y - np.average(y, weights=sw)
    alpha_max = float(np.max(np.abs((Xs * sw[:, None]).T @ yc)) / (sw.sum() * 0.5))
    alpha_max = max(alpha_max, 1e-8)
    alpha_grid = alpha_max * np.logspace(0, -4, 16)

    best = None
    n = len(y)
    for alpha in alpha_grid:
        model = ElasticNet(
            alpha=float(alpha),
            l1_ratio=0.5,
            fit_intercept=True,
            max_iter=5000,
            tol=1e-6,
            selection="cyclic",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(Xs, y, sample_weight=sw)
        pred = model.predict(Xs)
        rss = float(np.sum(sw * (y - pred) ** 2))
        k = int(np.count_nonzero(np.abs(model.coef_) > 1e-8) + 1)
        if rss <= 0 or n <= k + 1:
            continue
        aic = n * math.log(rss / sw.sum()) + 2.0 * k
        aicc = aic + 2.0 * k * (k + 1.0) / (n - k - 1.0)
        if best is None or aicc < best[0]:
            best = (aicc, model, float(alpha))

    if best is None:
        return None
    _, model, chosen_alpha = best
    selected = [f for f, c in zip(cols, model.coef_) if c > 1e-8]
    if not selected:
        return None

    cur = panel[panel["date"] == current_date].copy()
    if len(cur) < min_cross_section:
        return None

    preds = []
    for f in selected:
        a, b = avg_coef[f]
        preds.append(a + b * cur[f"z_{f}"].to_numpy(float))
    ctrend = pd.Series(np.mean(np.column_stack(preds), axis=1), index=cur.index)
    return ModelFit(ctrend=ctrend, selected_features=selected, alpha=chosen_alpha)


def load_weekly_funding(
    con: sqlite3.Connection,
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[tuple[str, pd.Timestamp], float]:
    start_ms = int(start.timestamp() * 1000)
    end_ms = int((end + pd.Timedelta(days=8)).timestamp() * 1000)
    out: dict[tuple[str, pd.Timestamp], float] = {}
    symbols = sorted(set(symbols))
    for n, sym in enumerate(symbols, 1):
        rows = con.execute(
            """
            SELECT event_time, rate FROM funding_events
            WHERE symbol = ? AND event_time > ? AND event_time <= ?
            ORDER BY event_time
            """,
            (sym, start_ms, end_ms),
        ).fetchall()
        if rows:
            x = pd.DataFrame(rows, columns=["event_time", "rate"])
            x["ts"] = pd.to_datetime(x["event_time"], unit="ms", utc=True)
            # Monday..Sunday funding belongs to a position opened at prior Sunday close.
            x["week_end"] = (
                x["ts"].dt.to_period("W-SUN").dt.end_time.dt.tz_localize("UTC").dt.floor("D")
            )
            sums = x.groupby("week_end")["rate"].sum()
            for dt, rate in sums.items():
                out[(sym, pd.Timestamp(dt))] = float(rate)
        if n % 100 == 0 or n == len(symbols):
            print(f"Funding aggregation: {n}/{len(symbols)}", flush=True)
    return out


def max_consecutive_losses(r: pd.Series) -> int:
    best = cur = 0
    for v in r.fillna(0.0):
        if v < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def performance_metrics(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    equity = (1.0 + r).cumprod()
    dd = equity / equity.cummax() - 1.0
    pos = float(r[r > 0].sum())
    neg = float(-r[r < 0].sum())
    pf = pos / neg if neg > 0 else float("inf")
    std = float(r.std(ddof=1))
    sharpe = math.sqrt(52.0) * float(r.mean()) / std if std > 0 else np.nan
    downside = float(r[r < 0].std(ddof=1))
    sortino = math.sqrt(52.0) * float(r.mean()) / downside if downside > 0 else np.nan
    years = len(r) / 52.0
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {
        "weeks": float(len(r)),
        "total_return": float(equity.iloc[-1] - 1.0),
        "cagr": float(cagr),
        "profit_factor": float(pf),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "max_drawdown": float(dd.min()),
        "worst_week": float(r.min()),
        "best_week": float(r.max()),
        "avg_week": float(r.mean()),
        "max_consecutive_losses": float(max_consecutive_losses(r)),
    }


def fmt_pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100.0 * x:.2f}%"


def run_backtest(
    panel: pd.DataFrame,
    funding: dict[tuple[str, pd.Timestamp], float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_weeks: int,
    top_frac: float,
    min_cross_section: int,
    side_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_dates = sorted(
        pd.Timestamp(d)
        for d in panel["date"].unique()
        if start <= pd.Timestamp(d) <= end
    )
    previous_weights: dict[str, float] = {}
    rows: list[dict] = []
    asset_rows: list[dict] = []
    side_cost = side_cost_bps / 10_000.0

    for k, dt in enumerate(all_dates, 1):
        cur = panel[(panel["date"] == dt) & panel["fwd_ret"].notna()].copy()
        if len(cur) < min_cross_section:
            continue

        prior_dates = panel.loc[panel["date"] < dt, "date"].drop_duplicates()
        if len(prior_dates) < train_weeks:
            continue

        fit = fit_predict_week(panel, dt, train_weeks, min_cross_section)
        current_weights: dict[str, float] = {}
        selected_features: list[str] = []
        chosen_alpha = np.nan

        if fit is not None:
            selected_features = fit.selected_features
            chosen_alpha = fit.alpha
            cur["cttrend"] = fit.ctrend.reindex(cur.index)
            cur = cur.dropna(subset=["cttrend"])
            if len(cur) >= min_cross_section:
                top_n = max(1, int(math.ceil(len(cur) * top_frac)))
                candidates = cur.nlargest(top_n, "cttrend").copy()
                # Independent asset-level long-only time-series momentum gate.
                candidates = candidates[candidates["ret_28d"] > 0].copy()
                if len(candidates):
                    weight = 1.0 / len(candidates)
                    current_weights = {s: weight for s in candidates["symbol"]}
        else:
            # No positive Elastic-Net selected forecasts => explicit cash week.
            cur["cttrend"] = np.nan

        union = set(previous_weights) | set(current_weights)
        turnover = sum(
            abs(current_weights.get(s, 0.0) - previous_weights.get(s, 0.0))
            for s in union
        )
        cost = side_cost * turnover

        gross = 0.0
        funding_pnl = 0.0
        funding_known = 0
        next_week_end = dt + pd.Timedelta(days=7)
        by_symbol = cur.set_index("symbol")

        for sym, w in current_weights.items():
            if sym not in by_symbol.index:
                continue
            rr = float(by_symbol.loc[sym, "fwd_ret"])
            gross_piece = w * rr
            funding_key = (sym, next_week_end)
            if funding_key in funding:
                funding_known += 1
            fr = funding.get(funding_key, 0.0)
            fund_piece = -w * fr  # long pays positive funding, receives negative funding
            gross += gross_piece
            funding_pnl += fund_piece
            asset_rows.append(
                {
                    "date": dt,
                    "symbol": sym,
                    "weight": w,
                    "fwd_ret": rr,
                    "funding_rate_sum": fr,
                    "gross_contribution": gross_piece,
                    "funding_contribution": fund_piece,
                    "cttrend": float(by_symbol.loc[sym, "cttrend"]),
                    "ret_28d": float(by_symbol.loc[sym, "ret_28d"]),
                }
            )

        net = gross + funding_pnl - cost
        rows.append(
            {
                "date": dt,
                "week_end": next_week_end,
                "positions": len(current_weights),
                "gross_return": gross,
                "funding_return": funding_pnl,
                "turnover": turnover,
                "cost_return": cost,
                "net_return": net,
                "selected_features": len(selected_features),
                "enet_alpha": chosen_alpha,
                "features": ",".join(selected_features),
                "funding_coverage": (funding_known / len(current_weights)) if current_weights else 1.0,
            }
        )
        previous_weights = current_weights

        if k % 26 == 0 or k == len(all_dates):
            print(
                f"Model/backtest progress {k}/{len(all_dates)} | "
                f"last={dt.date()} positions={len(current_weights)} "
                f"net={net:+.3%} selected_features={len(selected_features)}",
                flush=True,
            )

    if not rows:
        raise RuntimeError("No out-of-sample weeks produced")

    weekly = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    assets = pd.DataFrame(asset_rows)

    # Terminal liquidation cost, so the result cannot hide the final exit fee.
    if previous_weights and len(weekly):
        terminal_notional = sum(abs(w) for w in previous_weights.values())
        terminal_cost = side_cost * terminal_notional
        weekly.loc[weekly.index[-1], "cost_return"] += terminal_cost
        weekly.loc[weekly.index[-1], "turnover"] += terminal_notional
        weekly.loc[weekly.index[-1], "net_return"] -= terminal_cost

    weekly["equity"] = 100.0 * (1.0 + weekly["net_return"]).cumprod()
    return weekly, assets


def year_breakdown(weekly: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, x in weekly.groupby(weekly["date"].dt.year):
        m = performance_metrics(x["net_return"])
        active = x[(x["positions"] > 0) | (x["turnover"] > 1e-12)]
        rows.append(
            {
                "year": int(year),
                "weeks": int(m["weeks"]),
                "return": (1.0 + x["net_return"]).prod() - 1.0,
                "active_win_rate": float((active["net_return"] > 0).mean()) if len(active) else np.nan,
                "profit_factor": m["profit_factor"],
                "sharpe": m["sharpe"],
                "max_drawdown_local": m["max_drawdown"],
            }
        )
    return pd.DataFrame(rows)


def print_report(weekly: pd.DataFrame, assets: pd.DataFrame) -> None:
    m = performance_metrics(weekly["net_return"])
    active = weekly[(weekly["positions"] > 0) | (weekly["turnover"] > 1e-12)]
    active_wr = float((active["net_return"] > 0).mean()) if len(active) else np.nan
    held = weekly[weekly["positions"] > 0]
    funding_cov = float(held["funding_coverage"].mean()) if len(held) else np.nan

    print("\n=== CTREND + 28D TSMOM: RESEARCH RESULT ===")
    print(f"Weeks: {int(m['weeks'])}")
    print(f"$100 -> ${weekly['equity'].iloc[-1]:.2f}")
    print(f"Total return: {fmt_pct(m['total_return'])}")
    print(f"CAGR: {fmt_pct(m['cagr'])}")
    print(f"Active weekly win rate: {fmt_pct(active_wr)} ({len(active)} active weeks)")
    print(f"Profit factor: {m['profit_factor']:.3f}")
    print(f"Sharpe: {m['sharpe']:.3f}")
    print(f"Sortino: {m['sortino']:.3f}")
    print(f"Max drawdown: {fmt_pct(m['max_drawdown'])}")
    print(
        f"Avg / best / worst week: {fmt_pct(m['avg_week'])} / "
        f"{fmt_pct(m['best_week'])} / {fmt_pct(m['worst_week'])}"
    )
    print(f"Max consecutive losing weeks: {int(m['max_consecutive_losses'])}")
    print(f"Avg positions: {weekly['positions'].mean():.2f}")
    print(f"Avg weekly turnover: {weekly['turnover'].mean():.3f}")
    print(f"Avg gross: {fmt_pct(weekly['gross_return'].mean())}")
    print(f"Avg funding: {fmt_pct(weekly['funding_return'].mean())}")
    print(f"Avg costs: {fmt_pct(weekly['cost_return'].mean())}")
    print(f"Avg selected CTREND features: {weekly['selected_features'].mean():.1f}")
    print(f"Funding coverage on held positions: {funding_cov:.2%}")

    years = year_breakdown(weekly)
    print("\nYEAR BREAKDOWN")
    if not years.empty:
        z = years.copy()
        z["return"] = z["return"].map(fmt_pct)
        z["active_win_rate"] = z["active_win_rate"].map(fmt_pct)
        z["max_drawdown_local"] = z["max_drawdown_local"].map(fmt_pct)
        print(z.to_string(index=False))

    top_share = np.nan
    if not assets.empty:
        contrib = (
            assets.assign(
                net_before_cost=assets["gross_contribution"] + assets["funding_contribution"]
            )
            .groupby("symbol")["net_before_cost"]
            .sum()
            .sort_values(ascending=False)
        )
        positive_total = float(contrib[contrib > 0].sum())
        top_share = (
            float(contrib.iloc[0] / positive_total)
            if positive_total > 0 and len(contrib)
            else np.nan
        )
        print("\nTOP ASSET CONTRIBUTIONS (before shared turnover costs)")
        print(contrib.head(12).to_string())
        print(f"Top asset share of positive contribution: {fmt_pct(top_share)}")

    yr = year_breakdown(weekly)
    y2025 = yr.loc[yr["year"] == 2025, "return"]
    y2026 = yr.loc[yr["year"] == 2026, "return"]
    gates = {
        "PF > 1.30": m["profit_factor"] > 1.30,
        "Sharpe > 1.00": m["sharpe"] > 1.00,
        "Active weekly WR > 55%": active_wr > 0.55,
        "2025 positive": (not y2025.empty and float(y2025.iloc[0]) > 0),
        "2026 positive": (not y2026.empty and float(y2026.iloc[0]) > 0),
        "Top asset < 35% of positive contribution": (
            np.isfinite(top_share) and top_share < 0.35
        ),
        "Funding coverage >= 95%": (np.isfinite(funding_cov) and funding_cov >= 0.95),
    }
    print("\nPRE-REGISTERED GATES")
    for name, ok in gates.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"Overall: {'PASS' if all(gates.values()) else 'FAIL'}")


def main() -> int:
    args = parse_args()
    if not (0 < args.top_frac <= 1):
        raise ValueError("--top-frac must be in (0, 1]")
    if args.universe < args.min_cross_section:
        raise ValueError("--universe must be >= --min-cross-section")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC")
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    db = Path(args.db)
    if not db.exists():
        raise FileNotFoundError(f"DB not found: {db}")

    print("=== BINANCE CTREND RECONSTRUCTION RESEARCH ===")
    print(f"DB: {db}")
    print(f"Evaluation: {start.date()} -> {end.date()}")
    print(
        f"Universe: point-in-time top {args.universe} USDT perpetuals by trailing 30d quote volume"
    )
    print(
        f"Portfolio: long top {args.top_frac:.0%} CTREND, only assets with positive 28d TSMOM"
    )
    print(
        f"Training: rolling {args.train_weeks} weekly cross-sections, CS combined Elastic Net reconstruction"
    )
    print(
        f"Costs: {args.side_cost_bps:.1f} bps per changed notional side + actual archived funding"
    )
    print(
        "Leverage: 1x. No stop/TP/hyperopt. This test measures whether the core edge exists.\n"
    )

    con = sqlite3.connect(str(db), timeout=120)
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-262144")

    daily = load_daily(con, start, end)
    panel = make_weekly_panel(
        daily,
        start,
        end,
        args.universe,
        args.min_history_days,
        args.min_cross_section,
    )
    symbols = panel["symbol"].unique().tolist()
    funding = load_weekly_funding(con, symbols, start, end)
    weekly, assets = run_backtest(
        panel,
        funding,
        start,
        end,
        args.train_weeks,
        args.top_frac,
        args.min_cross_section,
        args.side_cost_bps,
    )

    weekly.to_csv(outdir / "weekly_results.csv", index=False)
    assets.to_csv(outdir / "asset_contributions.csv", index=False)
    year_breakdown(weekly).to_csv(outdir / "year_breakdown.csv", index=False)

    print_report(weekly, assets)
    print(f"\nSaved: {outdir}/weekly_results.csv")
    print(f"Saved: {outdir}/asset_contributions.csv")
    print(f"Saved: {outdir}/year_breakdown.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
