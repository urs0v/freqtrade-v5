#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet


STABLE_BASES = {
    "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI",
    "USDE", "USDS", "USD1", "PYUSD",
}

FEATURES = [
    "rsi", "stochRSI", "stochK", "stochD", "cci",
    "sma_3d", "sma_5d", "sma_10d", "sma_20d", "sma_50d", "sma_100d", "sma_200d",
    "macd", "macd_diff_signal",
    "volsma_3d", "volsma_5d", "volsma_10d", "volsma_20d",
    "volsma_50d", "volsma_100d", "volsma_200d",
    "volmacd", "volmacd_diff_signal", "chaikin",
    "boll_low", "boll_mid", "boll_high", "boll_width",
]


def args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unbiased Binance USD-M CTREND reconstruction + 28d long-only TSMOM gate"
    )
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--universe", type=int, default=50)
    p.add_argument("--top-frac", type=float, default=0.20)
    p.add_argument("--train-weeks", type=int, default=52)
    p.add_argument("--min-history-days", type=int, default=210)
    p.add_argument("--min-cross-section", type=int, default=15)
    p.add_argument("--side-cost-bps", type=float, default=7.0)
    p.add_argument("--output-dir", default="/freqtrade/user_data/cttrend_research")
    return p.parse_args()


def div(a: pd.Series, b: pd.Series) -> pd.Series:
    return a / b.replace(0.0, np.nan)


def ema_paper(s: pd.Series, n: int) -> pd.Series:
    # CTREND supplement uses alpha = 1/(1+L).
    return s.ewm(alpha=1.0 / (1.0 + n), adjust=False, min_periods=n).mean()


def indicators(g: pd.DataFrame) -> pd.DataFrame:
    """Exact 28 indicator definitions from CTREND online appendix, on daily data."""
    g = g.sort_values("date").copy()
    c = g["close"].astype(float)
    h = g["high"].astype(float)
    l = g["low"].astype(float)
    v = g["quote_volume"].astype(float).clip(lower=0.0)

    d = c.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = gain.rolling(14, min_periods=14).mean()
    al = loss.rolling(14, min_periods=14).mean()
    rsi = 100.0 - 100.0 / (1.0 + div(ag, al))
    rsi = rsi.where(al > 0, 100.0)
    rsi = rsi.where(ag > 0, 0.0)
    rsi = rsi.where(~((ag == 0) & (al == 0)), 50.0)
    g["rsi"] = rsi

    rlo = rsi.rolling(14, min_periods=14).min()
    rhi = rsi.rolling(14, min_periods=14).max()
    g["stochRSI"] = div(rsi - rlo, rhi - rlo)

    ll = l.rolling(14, min_periods=14).min()
    hh = h.rolling(14, min_periods=14).max()
    g["stochK"] = div(c - ll, hh - ll)
    g["stochD"] = g["stochK"].rolling(3, min_periods=3).mean()

    tp = (c + h + l) / 3.0
    tp20 = tp.rolling(20, min_periods=20).mean()
    adev = tp.rolling(20, min_periods=20).apply(
        lambda x: float(np.mean(np.abs(x - np.mean(x)))), raw=True
    )
    g["cci"] = div(tp - tp20, 0.015 * adev)

    for n in (3, 5, 10, 20, 50, 100, 200):
        g[f"sma_{n}d"] = div(c.rolling(n, min_periods=n).mean(), c)

    e12, e26 = ema_paper(c, 12), ema_paper(c, 26)
    macd = div(e12 - e26, e12)
    g["macd"] = macd
    g["macd_diff_signal"] = macd - ema_paper(macd, 9)

    for n in (3, 5, 10, 20, 50, 100, 200):
        g[f"volsma_{n}d"] = div(v.rolling(n, min_periods=n).mean(), v)

    ve12, ve26 = ema_paper(v, 12), ema_paper(v, 26)
    vmacd = div(ve12 - ve26, ve12)
    g["volmacd"] = vmacd
    g["volmacd_diff_signal"] = vmacd - ema_paper(vmacd, 9)

    spread = (h - l).replace(0.0, np.nan)
    ad = div((c - l) - (h - c), spread) * v
    g["chaikin"] = div(
        ad.rolling(21, min_periods=21).sum(),
        v.rolling(21, min_periods=21).sum(),
    )

    mid = c.rolling(20, min_periods=20).mean()
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    lo, hi = mid - 2.0 * sd, mid + 2.0 * sd
    g["boll_low"] = div(lo, c)
    g["boll_mid"] = div(mid, c)
    g["boll_high"] = div(hi, c)
    g["boll_width"] = div(hi - lo, mid)

    g["ret_28d"] = c.pct_change(28, fill_method=None)
    g["liq_30d"] = v.rolling(30, min_periods=30).mean()
    g["history_days"] = np.arange(1, len(g) + 1)
    return g


def load_daily(con: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    warm = start - pd.Timedelta(days=900)
    lo = int(warm.timestamp() * 1000)
    hi = int((end + pd.Timedelta(days=1)).timestamp() * 1000) - 1

    syms = [
        r[0]
        for r in con.execute(
            "SELECT DISTINCT symbol FROM candles WHERE open_time BETWEEN ? AND ? ORDER BY symbol",
            (lo, hi),
        )
        if r[0].endswith("USDT") and r[0][:-4] not in STABLE_BASES
    ]
    print(f"Historical non-stable USDT perp symbols in DB: {len(syms)}", flush=True)

    out: list[pd.DataFrame] = []
    for i, sym in enumerate(syms, 1):
        rows = con.execute(
            """
            SELECT open_time, open, high, low, close, quote_volume
            FROM candles
            WHERE symbol=? AND open_time BETWEEN ? AND ?
            ORDER BY open_time
            """,
            (sym, lo, hi),
        ).fetchall()
        if not rows:
            continue

        x = pd.DataFrame(
            rows, columns=["open_time", "open", "high", "low", "close", "quote_volume"]
        )
        x["date"] = pd.to_datetime(x["open_time"], unit="ms", utc=True).dt.floor("D")
        d = (
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
        # Reject incomplete UTC days rather than silently changing indicators.
        d = d[d["bars"] == 4].drop(columns="bars")
        if d.empty:
            continue
        d["symbol"] = sym
        out.append(indicators(d))
        if i % 50 == 0 or i == len(syms):
            print(f"Daily feature build: {i}/{len(syms)}", flush=True)

    if not out:
        raise RuntimeError("No daily candles could be built")
    df = pd.concat(out, ignore_index=True).sort_values(["date", "symbol"]).reset_index(drop=True)
    print(
        f"Daily panel: {len(df):,} rows | {df.symbol.nunique()} symbols | "
        f"{df.date.min().date()} -> {df.date.max().date()}",
        flush=True,
    )
    return df


def rank_pm_half(s: pd.Series) -> pd.Series:
    ok = s.notna()
    ans = pd.Series(np.nan, index=s.index, dtype=float)
    n = int(ok.sum())
    if n <= 1:
        return ans
    r = s[ok].rank(method="average")
    ans.loc[ok] = (r - 1.0) / (n - 1.0) - 0.5
    return ans


def attach_forward_exits(sun: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """
    Next-Sunday close is the normal exit. If a contract truly disappears before
    next Sunday, use its last archived daily close as forced exit.

    Crucially, today's universe is never filtered on future-price availability.
    """
    sun = sun.copy()
    sun["planned_exit_date"] = sun["date"] + pd.Timedelta(days=7)

    exact = daily[["symbol", "date", "close"]].rename(
        columns={"date": "planned_exit_date", "close": "exit_close"}
    )
    sun = sun.merge(exact, on=["symbol", "planned_exit_date"], how="left")

    last = (
        daily.sort_values(["symbol", "date"])
        .groupby("symbol", as_index=False)
        .tail(1)[["symbol", "date", "close"]]
        .rename(columns={"date": "last_date", "close": "last_close"})
    )
    sun = sun.merge(last, on="symbol", how="left")
    global_last = daily["date"].max()

    forced = (
        sun["exit_close"].isna()
        & (sun["last_date"] > sun["date"])
        & (sun["last_date"] < sun["planned_exit_date"])
        & (sun["last_date"] < global_last)
    )
    sun["forced_exit"] = forced
    sun["actual_exit_date"] = sun["planned_exit_date"]
    sun.loc[forced, "exit_close"] = sun.loc[forced, "last_close"]
    sun.loc[forced, "actual_exit_date"] = sun.loc[forced, "last_date"]
    sun["fwd_ret"] = sun["exit_close"] / sun["close"] - 1.0
    return sun


def weekly_panel(
    daily: pd.DataFrame,
    end: pd.Timestamp,
    universe_n: int,
    min_history: int,
    min_cs: int,
) -> pd.DataFrame:
    sun = daily[daily["date"].dt.dayofweek == 6].copy()
    sun = attach_forward_exits(sun, daily)
    sun = sun.dropna(subset=FEATURES + ["liq_30d", "close"])
    sun = sun[sun["history_days"] >= min_history]

    weeks: list[pd.DataFrame] = []
    for dt, x in sun.groupby("date", sort=True):
        x = x.replace([np.inf, -np.inf], np.nan).dropna(subset=FEATURES + ["liq_30d"])
        x = x[x["liq_30d"] > 0]
        if len(x) < min_cs:
            continue
        x = x.nlargest(min(universe_n, len(x)), "liq_30d").copy()
        if len(x) < min_cs:
            continue
        for f in FEATURES:
            x[f"z_{f}"] = rank_pm_half(x[f])
        # Market-cap weights are unavailable point-in-time from Binance itself.
        # Trailing observable dollar liquidity is the futures implementation proxy.
        x["reg_weight"] = x["liq_30d"] / x["liq_30d"].sum()
        weeks.append(x)

    if not weeks:
        raise RuntimeError("No eligible point-in-time weekly cross-sections")
    p = pd.concat(weeks, ignore_index=True)
    p = p[p["date"] <= end].sort_values(["date", "symbol"]).reset_index(drop=True)
    print(
        f"Weekly panel: {p.date.nunique()} weeks | median universe "
        f"{p.groupby('date').size().median():.0f}",
        flush=True,
    )
    return p


def wls1(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> tuple[float, float] | None:
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0)
    if m.sum() < 5:
        return None
    x, y, w = x[m], y[m], w[m]
    w = w / w.sum()
    xm, ym = float(np.sum(w * x)), float(np.sum(w * y))
    vx = float(np.sum(w * (x - xm) ** 2))
    if vx <= 1e-12:
        return None
    b = float(np.sum(w * (x - xm) * (y - ym))) / vx
    return ym - b * xm, b


@dataclass
class Fit:
    score: pd.Series
    features: list[str]
    alpha: float


def fit_week(panel: pd.DataFrame, now: pd.Timestamp, train_weeks: int, min_cs: int) -> Fit | None:
    dates = sorted(pd.Timestamp(x) for x in panel["date"].unique() if pd.Timestamp(x) < now)
    if len(dates) < train_weeks:
        return None
    train_dates = dates[-train_weeks:]

    chunks: list[pd.DataFrame] = []
    coefs: dict[str, list[tuple[float, float]]] = {f: [] for f in FEATURES}
    for dt in train_dates:
        x = panel[(panel["date"] == dt) & panel["fwd_ret"].notna()]
        if len(x) < min_cs:
            continue
        chunks.append(x)
        y = x["fwd_ret"].to_numpy(float)
        w = x["reg_weight"].to_numpy(float)
        for f in FEATURES:
            z = x[f"z_{f}"].to_numpy(float)
            c = wls1(z, y, w)
            if c is not None:
                coefs[f].append(c)

    if len(chunks) < max(26, train_weeks // 2):
        return None

    avg: dict[str, tuple[float, float]] = {}
    threshold = max(20, len(chunks) // 2)
    for f, vals in coefs.items():
        if len(vals) >= threshold:
            avg[f] = (
                float(np.mean([v[0] for v in vals])),
                float(np.mean([v[1] for v in vals])),
            )
    if len(avg) < 5:
        return None

    tr = pd.concat(chunks, ignore_index=True)
    names = list(avg)
    X = np.column_stack([
        avg[f][0] + avg[f][1] * tr[f"z_{f}"].to_numpy(float) for f in names
    ])
    y = tr["fwd_ret"].to_numpy(float)
    sw = tr["reg_weight"].to_numpy(float)
    good = np.isfinite(y) & np.isfinite(sw) & (sw > 0) & np.all(np.isfinite(X), axis=1)
    X, y, sw = X[good], y[good], sw[good]
    if len(y) < 100:
        return None

    sw = sw / np.mean(sw)
    mu = np.average(X, axis=0, weights=sw)
    var = np.average((X - mu) ** 2, axis=0, weights=sw)
    sd = np.sqrt(np.maximum(var, 1e-12))
    Xs = (X - mu) / sd

    # Paper: Elastic Net L1/L2 tradeoff 0.5, lambda chosen by corrected AIC.
    yc = y - np.average(y, weights=sw)
    amax = float(np.max(np.abs((Xs * sw[:, None]).T @ yc)) / (sw.sum() * 0.5))
    amax = max(amax, 1e-8)
    grid = amax * np.logspace(0, -4, 16)

    best: tuple[float, ElasticNet, float] | None = None
    n = len(y)
    for a in grid:
        model = ElasticNet(
            alpha=float(a),
            l1_ratio=0.5,
            fit_intercept=True,
            max_iter=5000,
            tol=1e-6,
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
            best = (aicc, model, float(a))

    if best is None:
        return None
    _, model, chosen = best
    picked = [f for f, b in zip(names, model.coef_) if b > 1e-8]
    if not picked:
        return None

    cur = panel[panel["date"] == now]
    forecasts = np.column_stack([
        avg[f][0] + avg[f][1] * cur[f"z_{f}"].to_numpy(float) for f in picked
    ])
    return Fit(
        score=pd.Series(np.mean(forecasts, axis=1), index=cur.index),
        features=picked,
        alpha=chosen,
    )


def funding_by_week(
    con: sqlite3.Connection,
    symbols: Iterable[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[tuple[str, pd.Timestamp], float]:
    lo = int(start.timestamp() * 1000)
    hi = int((end + pd.Timedelta(days=8)).timestamp() * 1000)
    ans: dict[tuple[str, pd.Timestamp], float] = {}

    syms = sorted(set(symbols))
    for i, sym in enumerate(syms, 1):
        rows = con.execute(
            """
            SELECT event_time, rate FROM funding_events
            WHERE symbol=? AND event_time>? AND event_time<=?
            ORDER BY event_time
            """,
            (sym, lo, hi),
        ).fetchall()
        if rows:
            x = pd.DataFrame(rows, columns=["event_time", "rate"])
            x["ts"] = pd.to_datetime(x["event_time"], unit="ms", utc=True)
            # Period end is Sunday 23:59; normalize to Sunday date.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                wend = x["ts"].dt.tz_convert(None).dt.to_period("W-SUN").dt.end_time.dt.floor("D")
            x["week_end"] = wend.dt.tz_localize("UTC")
            for dt, rate in x.groupby("week_end")["rate"].sum().items():
                ans[(sym, pd.Timestamp(dt))] = float(rate)
        if i % 100 == 0 or i == len(syms):
            print(f"Funding build: {i}/{len(syms)}", flush=True)
    return ans


def metrics(r: pd.Series) -> dict[str, float]:
    r = r.dropna()
    eq = (1.0 + r).cumprod()
    dd = eq / eq.cummax() - 1.0
    pos, neg = float(r[r > 0].sum()), float(-r[r < 0].sum())
    std = float(r.std(ddof=1))
    down = float(r[r < 0].std(ddof=1))
    years = len(r) / 52.0
    return {
        "weeks": len(r),
        "total": float(eq.iloc[-1] - 1.0),
        "cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else np.nan,
        "pf": pos / neg if neg > 0 else np.inf,
        "sharpe": math.sqrt(52.0) * float(r.mean()) / std if std > 0 else np.nan,
        "sortino": math.sqrt(52.0) * float(r.mean()) / down if down > 0 else np.nan,
        "mdd": float(dd.min()),
        "avg": float(r.mean()),
        "best": float(r.max()),
        "worst": float(r.min()),
    }


def longest_loss_streak(r: pd.Series) -> int:
    best = cur = 0
    for x in r:
        if x < 0:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def backtest(
    panel: pd.DataFrame,
    funding: dict[tuple[str, pd.Timestamp], float],
    start: pd.Timestamp,
    end: pd.Timestamp,
    train_weeks: int,
    top_frac: float,
    min_cs: int,
    side_cost_bps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Last eligible rebalance must have a complete planned week inside dataset.
    dates = sorted(
        pd.Timestamp(d)
        for d in panel["date"].unique()
        if start <= pd.Timestamp(d) <= end - pd.Timedelta(days=7)
    )
    prev: dict[str, float] = {}
    weeks: list[dict] = []
    assets: list[dict] = []
    side_cost = side_cost_bps / 10_000.0

    for n, dt in enumerate(dates, 1):
        cur = panel[panel["date"] == dt].copy()
        if len(cur) < min_cs:
            continue
        if panel.loc[panel["date"] < dt, "date"].nunique() < train_weeks:
            continue

        fit = fit_week(panel, dt, train_weeks, min_cs)
        cur_weights: dict[str, float] = {}
        selected_features: list[str] = []
        alpha = np.nan

        if fit is not None:
            selected_features = fit.features
            alpha = fit.alpha
            cur["cttrend"] = fit.score.reindex(cur.index)
            cur = cur.dropna(subset=["cttrend"])
            if len(cur) >= min_cs:
                k = max(1, int(math.ceil(len(cur) * top_frac)))
                candidates = cur.nlargest(k, "cttrend")
                candidates = candidates[candidates["ret_28d"] > 0]
                if len(candidates):
                    w = 1.0 / len(candidates)
                    cur_weights = {s: w for s in candidates["symbol"]}
        else:
            cur["cttrend"] = np.nan

        turnover = sum(abs(cur_weights.get(s, 0.0) - prev.get(s, 0.0)) for s in set(prev) | set(cur_weights))
        cost = side_cost * turnover
        gross = 0.0
        fund = 0.0
        fund_known = 0
        forced_notional = 0.0
        bysym = cur.set_index("symbol")
        planned_week_end = dt + pd.Timedelta(days=7)

        for sym, w in cur_weights.items():
            if sym not in bysym.index:
                raise RuntimeError(f"Selected symbol missing from current cross-section: {sym} {dt}")
            rr = bysym.loc[sym, "fwd_ret"]
            if pd.isna(rr):
                raise RuntimeError(
                    f"Selected {sym} on {dt.date()} has no unbiased exit price. "
                    "Aborting instead of introducing future-availability bias."
                )
            rr = float(rr)
            gross_piece = w * rr

            key = (sym, planned_week_end)
            if key in funding:
                fund_known += 1
            fr = funding.get(key, 0.0)
            fund_piece = -w * fr

            forced = bool(bysym.loc[sym, "forced_exit"])
            if forced:
                forced_notional += abs(w)

            gross += gross_piece
            fund += fund_piece
            assets.append({
                "date": dt,
                "symbol": sym,
                "weight": w,
                "cttrend": float(bysym.loc[sym, "cttrend"]),
                "ret_28d": float(bysym.loc[sym, "ret_28d"]),
                "fwd_ret": rr,
                "funding_rate_sum": fr,
                "gross_contribution": gross_piece,
                "funding_contribution": fund_piece,
                "forced_exit": forced,
                "actual_exit_date": bysym.loc[sym, "actual_exit_date"],
            })

        # Forced delisting exit pays one additional side immediately and cannot
        # remain in next week's carried position state.
        if forced_notional:
            turnover += forced_notional
            cost += side_cost * forced_notional

        net = gross + fund - cost
        weeks.append({
            "date": dt,
            "week_end": planned_week_end,
            "positions": len(cur_weights),
            "gross_return": gross,
            "funding_return": fund,
            "turnover": turnover,
            "cost_return": cost,
            "net_return": net,
            "selected_features": len(selected_features),
            "features": ",".join(selected_features),
            "enet_alpha": alpha,
            "funding_coverage": fund_known / len(cur_weights) if cur_weights else 1.0,
        })

        forced_syms = {
            a["symbol"] for a in assets if a["date"] == dt and a["forced_exit"]
        }
        prev = {s: w for s, w in cur_weights.items() if s not in forced_syms}

        if n % 26 == 0 or n == len(dates):
            print(
                f"Backtest {n}/{len(dates)} | {dt.date()} | pos={len(cur_weights)} "
                f"| net={net:+.3%} | features={len(selected_features)}",
                flush=True,
            )

    if not weeks:
        raise RuntimeError("No OOS weeks were produced")

    wk = pd.DataFrame(weeks).sort_values("date").reset_index(drop=True)
    ar = pd.DataFrame(assets)

    # Explicit terminal close.
    if prev:
        final_notional = sum(abs(x) for x in prev.values())
        final_cost = side_cost * final_notional
        wk.loc[wk.index[-1], "turnover"] += final_notional
        wk.loc[wk.index[-1], "cost_return"] += final_cost
        wk.loc[wk.index[-1], "net_return"] -= final_cost

    wk["equity"] = 100.0 * (1.0 + wk["net_return"]).cumprod()
    return wk, ar


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100.0 * x:.2f}%"


def yearly(w: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, x in w.groupby(w["date"].dt.year):
        m = metrics(x["net_return"])
        active = x[(x["positions"] > 0) | (x["turnover"] > 1e-12)]
        rows.append({
            "year": int(y),
            "weeks": len(x),
            "return": float((1.0 + x["net_return"]).prod() - 1.0),
            "active_wr": float((active["net_return"] > 0).mean()) if len(active) else np.nan,
            "pf": m["pf"],
            "sharpe": m["sharpe"],
            "mdd": m["mdd"],
        })
    return pd.DataFrame(rows)


def report(w: pd.DataFrame, a: pd.DataFrame) -> None:
    m = metrics(w["net_return"])
    active = w[(w["positions"] > 0) | (w["turnover"] > 1e-12)]
    wr = float((active["net_return"] > 0).mean()) if len(active) else np.nan
    held = w[w["positions"] > 0]
    fcov = float(held["funding_coverage"].mean()) if len(held) else np.nan

    print("\n=== CTREND + 28D TSMOM: RESEARCH RESULT ===")
    print(f"Weeks: {m['weeks']} | active weeks: {len(active)}")
    print(f"$100 -> ${w.equity.iloc[-1]:.2f}")
    print(f"Total return: {pct(m['total'])} | CAGR: {pct(m['cagr'])}")
    print(f"Active weekly WR: {pct(wr)}")
    print(f"Profit factor: {m['pf']:.3f}")
    print(f"Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f}")
    print(f"Max drawdown: {pct(m['mdd'])}")
    print(f"Avg / best / worst week: {pct(m['avg'])} / {pct(m['best'])} / {pct(m['worst'])}")
    print(f"Longest losing streak: {longest_loss_streak(w.net_return)} weeks")
    print(f"Avg positions: {w.positions.mean():.2f}")
    print(f"Avg turnover: {w.turnover.mean():.3f}")
    print(f"Avg gross / funding / cost: {pct(w.gross_return.mean())} / {pct(w.funding_return.mean())} / {pct(w.cost_return.mean())}")
    print(f"Avg CTREND features selected: {w.selected_features.mean():.1f}")
    print(f"Funding coverage while invested: {fcov:.2%}")

    yy = yearly(w)
    print("\nYEAR BREAKDOWN")
    zz = yy.copy()
    if len(zz):
        zz["return"] = zz["return"].map(pct)
        zz["active_wr"] = zz["active_wr"].map(pct)
        zz["mdd"] = zz["mdd"].map(pct)
        print(zz.to_string(index=False))

    top_share = np.nan
    if len(a):
        c = (
            a.assign(pre_cost=a.gross_contribution + a.funding_contribution)
            .groupby("symbol")["pre_cost"]
            .sum()
            .sort_values(ascending=False)
        )
        ptotal = float(c[c > 0].sum())
        top_share = float(c.iloc[0] / ptotal) if ptotal > 0 else np.nan
        print("\nTOP ASSET CONTRIBUTIONS (before shared turnover costs)")
        print(c.head(12).to_string())
        print(f"Top asset share of positive contribution: {pct(top_share)}")
        print(f"Forced delisting exits: {int(a.forced_exit.sum())}")

    y25 = yy.loc[yy.year == 2025, "return"]
    y26 = yy.loc[yy.year == 2026, "return"]
    gates = {
        "PF > 1.30": m["pf"] > 1.30,
        "Sharpe > 1.00": m["sharpe"] > 1.00,
        "Active weekly WR > 55%": wr > 0.55,
        "2025 positive": len(y25) > 0 and float(y25.iloc[0]) > 0,
        "2026 positive": len(y26) > 0 and float(y26.iloc[0]) > 0,
        "Top asset < 35% positive contribution": np.isfinite(top_share) and top_share < 0.35,
        "Funding coverage >= 95%": np.isfinite(fcov) and fcov >= 0.95,
    }
    print("\nPRE-REGISTERED GATES")
    for name, ok in gates.items():
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"Overall: {'PASS' if all(gates.values()) else 'FAIL'}")


def main() -> int:
    cfg = args()
    if not 0 < cfg.top_frac <= 1:
        raise ValueError("--top-frac must be in (0,1]")
    if cfg.universe < cfg.min_cross_section:
        raise ValueError("--universe must be >= --min-cross-section")

    start = pd.Timestamp(cfg.start, tz="UTC")
    end = pd.Timestamp(cfg.end, tz="UTC")
    db = Path(cfg.db)
    out = Path(cfg.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not db.exists():
        raise FileNotFoundError(db)

    print("=== BINANCE CTREND RECONSTRUCTION ===")
    print(f"Evaluation: {start.date()} -> {end.date()}")
    print(f"Universe: historical point-in-time top {cfg.universe} USDT perps by trailing 30d quote volume")
    print(f"Selection: CTREND top {cfg.top_frac:.0%}, long-only, own 28d return > 0")
    print(f"Training: rolling {cfg.train_weeks} weeks; 28 published features; AICc Elastic Net")
    print(f"Execution: weekly, 1x, {cfg.side_cost_bps:.1f} bps per changed side + archived funding")
    print("No stop/TP/hyperopt. This is a single edge-validation test.\n")

    con = sqlite3.connect(str(db), timeout=120)
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-262144")

    daily = load_daily(con, start, end)
    panel = weekly_panel(
        daily, end, cfg.universe, cfg.min_history_days, cfg.min_cross_section
    )
    funding = funding_by_week(con, panel.symbol.unique(), start, end)
    w, a = backtest(
        panel, funding, start, end, cfg.train_weeks, cfg.top_frac,
        cfg.min_cross_section, cfg.side_cost_bps
    )

    w.to_csv(out / "weekly_results.csv", index=False)
    a.to_csv(out / "asset_contributions.csv", index=False)
    yearly(w).to_csv(out / "year_breakdown.csv", index=False)

    report(w, a)
    print(f"\nSaved under: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
