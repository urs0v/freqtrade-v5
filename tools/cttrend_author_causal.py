#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import sqlite3
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import enet_path


STABLE_BASES = {
    "USDC", "BUSD", "FDUSD", "TUSD", "USDP", "DAI",
    "USDE", "USDS", "USD1", "PYUSD",
}
FEATURES = [
    "sma_3d", "sma_5d", "sma_10d", "sma_20d", "sma_50d", "sma_100d", "sma_200d",
    "macd", "macd_diff_signal",
    "volsma_3d", "volsma_5d", "volsma_10d", "volsma_20d", "volsma_50d", "volsma_100d", "volsma_200d",
    "volmacd", "volmacd_diff_signal",
    "rsi", "stochRSI", "stochK", "stochD",
    "boll_low", "boll_mid", "boll_high", "boll_width", "cci", "chaikin",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Author-code-derived causal CTREND forensic test on Binance USD-M perpetuals")
    p.add_argument("--db", default="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite")
    p.add_argument("--start", default="2022-01-01")
    p.add_argument("--end", default="2026-07-31")
    p.add_argument("--train-weeks", type=int, default=52)
    p.add_argument("--eval-universe", type=int, default=50)
    p.add_argument("--top-frac", type=float, default=0.20)
    p.add_argument("--min-cross-section", type=int, default=25)
    p.add_argument("--side-cost-bps", type=float, default=7.0)
    p.add_argument("--workers", type=int, default=int(os.environ.get("CTREND_WORKERS", "32")))
    p.add_argument("--output-dir", default="/freqtrade/user_data/cttrend_author_causal")
    return p.parse_args()


def liu_week_key(d: pd.Series) -> pd.Series:
    doy = d.dt.dayofyear
    wk = (((doy - 1) // 7) + 1).clip(upper=52).astype(int)
    return d.dt.year.astype(int) * 100 + wk


def liu_week_end_from_key(key: int) -> pd.Timestamp:
    year = int(key // 100)
    week = int(key % 100)
    if week < 52:
        return pd.Timestamp(year=year, month=1, day=1, tz="UTC") + pd.Timedelta(days=7 * week - 1)
    return pd.Timestamp(year=year, month=12, day=31, tz="UTC")


def raw_daily(con: sqlite3.Connection, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    warm = start - pd.Timedelta(days=900)
    lo = int(warm.timestamp() * 1000)
    hi = int((end + pd.Timedelta(days=2)).timestamp() * 1000) - 1
    syms = [r[0] for r in con.execute(
        "SELECT DISTINCT symbol FROM candles WHERE open_time BETWEEN ? AND ? ORDER BY symbol", (lo, hi)
    ) if r[0].endswith("USDT") and r[0][:-4] not in STABLE_BASES]
    print(f"Historical non-stable USDT perpetual symbols: {len(syms)}", flush=True)
    parts = []
    for i, sym in enumerate(syms, 1):
        rows = con.execute("""
            SELECT open_time, open, high, low, close, quote_volume
            FROM candles WHERE symbol=? AND open_time BETWEEN ? AND ? ORDER BY open_time
        """, (sym, lo, hi)).fetchall()
        if not rows:
            continue
        x = pd.DataFrame(rows, columns=["open_time", "open", "high", "low", "close", "quote_volume"])
        x["date"] = pd.to_datetime(x.open_time, unit="ms", utc=True).dt.floor("D")
        d = (x.groupby("date", sort=True)
             .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
                  close=("close", "last"), quote_volume=("quote_volume", "sum"), bars=("open_time", "count"))
             .reset_index())
        d = d[d.bars == 4].drop(columns="bars")
        if d.empty:
            continue
        d["symbol"] = sym
        parts.append(d)
        if i % 100 == 0 or i == len(syms):
            print(f"Raw daily aggregation: {i}/{len(syms)}", flush=True)
    if not parts:
        raise RuntimeError("No daily data")
    out = pd.concat(parts, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    prev_close = out.groupby("symbol", sort=False).close.shift(1)
    prev_date = out.groupby("symbol", sort=False).date.shift(1)
    contiguous = (out.date - prev_date).eq(pd.Timedelta(days=1))
    out["raw_ret"] = np.where(contiguous, out.close / prev_close - 1.0, np.nan)
    out["week_key"] = liu_week_key(out.date)
    print(f"Raw daily rows: {len(out):,} | {out.date.min().date()} -> {out.date.max().date()}", flush=True)
    return out


def apply_author_truncation(raw: pd.DataFrame) -> pd.DataFrame:
    q = raw.groupby("date")["raw_ret"].quantile([0.005, 0.995]).unstack()
    q.columns = ["q005", "q995"]
    x = raw.merge(q, left_on="date", right_index=True, how="left")
    tail = x.raw_ret.notna() & ((x.raw_ret < x.q005) | (x.raw_ret > x.q995))
    x["tail_truncated"] = tail
    x["clean_ret"] = x.raw_ret.mask(tail)
    clean_missing = tail | x.raw_ret.isna()
    for c in ["open", "high", "low", "close", "quote_volume"]:
        x[f"clean_{c}"] = x[c].mask(clean_missing)
    print(f"Author-style daily 0.5% tail truncations: {int(tail.sum()):,}", flush=True)
    return x.drop(columns=["q005", "q995"])


def sma_omit(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=1).mean()


def ema_author(s: pd.Series, n: int) -> pd.Series:
    a = 2.0 / (1.0 + n)
    arr = s.to_numpy(float)
    sm = sma_omit(s, n).to_numpy(float)
    out = np.full(len(arr), np.nan, dtype=float)
    for t in range(1, len(arr)):
        prev = out[t - 1]
        if not np.isfinite(prev):
            prev = sm[t - 1]
        if np.isfinite(arr[t]) and np.isfinite(prev):
            out[t] = arr[t] * a + prev * (1.0 - a)
    return pd.Series(out, index=s.index)


def rsi_rsindex_proxy(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    gain = d.clip(lower=0.0)
    loss = (-d).clip(lower=0.0)
    ag = gain.rolling(n, min_periods=n).mean()
    al = loss.rolling(n, min_periods=n).mean()
    rs = ag / al.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(al != 0, 100.0)
    rsi = rsi.where(ag != 0, 0.0)
    rsi = rsi.where(~((ag == 0) & (al == 0)), 50.0)
    return rsi


def technical_author(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    idx = pd.date_range(g.date.min(), g.date.max(), freq="D", tz="UTC")
    z = g.set_index("date").reindex(idx)
    z.index.name = "date"
    c = z.clean_close.astype(float); h = z.clean_high.astype(float); l = z.clean_low.astype(float); v = z.clean_quote_volume.astype(float)

    for n in (3, 5, 10, 20, 50, 100, 200):
        z[f"sma_{n}d"] = sma_omit(c, n) / c
    ef = ema_author(c, 12); es = ema_author(c, 26)
    ppo = (ef - es) / es.replace(0.0, np.nan)
    z["macd"] = ppo; z["macd_diff_signal"] = ppo - ema_author(ppo, 9)

    for n in (3, 5, 10, 20, 50, 100, 200):
        z[f"volsma_{n}d"] = sma_omit(v, n) / v.replace(0.0, np.nan)
    vef = ema_author(v, 12); ves = ema_author(v, 26)
    pvo = (vef - ves) / ves.replace(0.0, np.nan)
    z["volmacd"] = pvo; z["volmacd_diff_signal"] = pvo - ema_author(pvo, 9)

    rsi = rsi_rsindex_proxy(c, 14)
    z["rsi"] = rsi
    rlo = rsi.rolling(14, min_periods=5).min(); rhi = rsi.rolling(14, min_periods=5).max()
    st_rsi = (rsi - rlo) / (rhi - rlo).replace(0.0, np.nan); st_rsi.iloc[:13] = np.nan
    z["stochRSI"] = st_rsi
    ll = l.rolling(14, min_periods=14).min(); hh = h.rolling(14, min_periods=14).max()
    st_k = (c - ll) / (hh - ll).replace(0.0, np.nan)
    z["stochK"] = st_k; z["stochD"] = sma_omit(st_k, 3)

    mid = c.rolling(20, min_periods=20).mean(); sd = c.rolling(20, min_periods=20).std(ddof=1)
    blo, bhi = mid - 2.0 * sd, mid + 2.0 * sd
    z["boll_low"] = blo / c; z["boll_mid"] = mid / c; z["boll_high"] = bhi / c
    z["boll_width"] = (bhi - blo) / mid.replace(0.0, np.nan)

    tp = (c + h + l) / 3.0; tp_sma = sma_omit(tp, 20)
    cci = np.full(len(z), np.nan); tpv = tp.to_numpy(float); smv = tp_sma.to_numpy(float)
    for t in range(19, len(z)):
        win = tpv[t - 19:t + 1]; valid = np.isfinite(win)
        if valid.sum() < 10 or not np.isfinite(tpv[t]) or not np.isfinite(smv[t]):
            continue
        md = float(np.mean(np.abs(win[valid] - smv[t])))
        if md > 0:
            cci[t] = (tpv[t] - smv[t]) / (0.015 * md)
    z["cci"] = cci

    rng = (h - l).replace(0.0, np.nan)
    ad = (((c - l) - (h - c)) / rng) * v
    valid = c.notna() & h.notna() & l.notna() & v.notna() & rng.notna()
    num = ad.where(valid).rolling(21, min_periods=10).sum(); den = v.where(valid).rolling(21, min_periods=10).sum()
    cmf = num / den.replace(0.0, np.nan); cmf.iloc[:20] = np.nan
    z["chaikin"] = cmf

    raw_v = z.quote_volume.astype(float)
    z["liq_30d"] = raw_v.rolling(30, min_periods=30).mean()
    z["raw_close"] = z.close; z["raw_volume"] = z.quote_volume; z["clean_ret_daily"] = z.clean_ret
    z["week_key"] = liu_week_key(pd.Series(z.index, index=z.index)).to_numpy()
    return z.reset_index()


def build_weekly(raw_clean: pd.DataFrame, workers: int) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    grouped = [(sym, g.copy()) for sym, g in raw_clean.groupby("symbol", sort=False)]
    nsyms = len(grouped)
    def one(item):
        sym, g0 = item
        rg = g0[["date", "close"]].dropna().sort_values("date").reset_index(drop=True)
        z = technical_author(g0); z["symbol"] = sym; rows = []
        for key, q in z.groupby("week_key", sort=True):
            key = int(key)
            expected_days = 7 if key % 100 < 52 else (liu_week_end_from_key(key).dayofyear - 7 * 51)
            cr = q.clean_ret_daily
            clean_week_ret = float(np.prod(1.0 + cr.to_numpy(float)) - 1.0) if cr.notna().sum() == expected_days else np.nan
            rec = {"symbol": sym, "week_key": key, "period_end": liu_week_end_from_key(key), "clean_week_ret": clean_week_ret}
            for f in FEATURES + ["liq_30d"]:
                ss = q[f].dropna(); rec[f] = float(ss.iloc[-1]) if len(ss) else np.nan
            rr = q[["date", "raw_close"]].dropna()
            rec["formation_close"] = float(rr.raw_close.iloc[-1]) if len(rr) else np.nan
            rec["formation_date"] = pd.Timestamp(rr.date.iloc[-1]) if len(rr) else pd.NaT
            rows.append(rec)
        return sym, pd.DataFrame(rows), rg

    parts = []; raw_groups = {}; workers = max(1, workers)
    if workers == 1:
        for i, item in enumerate(grouped, 1):
            sym, part, rg = one(item); parts.append(part); raw_groups[sym] = rg
            if i % 50 == 0 or i == nsyms: print(f"Author indicators + 52-week resample: {i}/{nsyms}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(one, item) for item in grouped]
            for i, fut in enumerate(as_completed(futs), 1):
                sym, part, rg = fut.result(); parts.append(part); raw_groups[sym] = rg
                if i % 50 == 0 or i == nsyms: print(f"Author indicators + 52-week resample: {i}/{nsyms}", flush=True)
    w = pd.concat(parts, ignore_index=True)
    keys = sorted(w.week_key.unique()); order = {int(k): i for i, k in enumerate(keys)}
    w["period_ord"] = w.week_key.map(order).astype(int)
    print(f"Weekly author-style panel: {len(w):,} rows | {len(keys)} periods | {w.symbol.nunique()} symbols", flush=True)
    return w, raw_groups


def rank_pm_half(s: pd.Series) -> pd.Series:
    ok = s.notna(); out = pd.Series(np.nan, index=s.index, dtype=float); n = int(ok.sum())
    if n <= 1: return out
    r = s[ok].rank(method="average"); out.loc[ok] = (r - 1.0) / (n - 1.0) - 0.5
    return out


def build_target_panel(w: pd.DataFrame) -> pd.DataFrame:
    lag_cols = FEATURES + ["liq_30d", "formation_close", "formation_date"]
    prev = w[["symbol", "period_ord"] + lag_cols].copy(); prev["period_ord"] += 1
    prev = prev.rename(columns={c: f"lag_{c}" for c in lag_cols})
    p = w.merge(prev, on=["symbol", "period_ord"], how="left")
    for f in FEATURES:
        p[f"z_{f}"] = p.groupby("period_ord", group_keys=False)[f"lag_{f}"].transform(rank_pm_half)
    p["all_x"] = p[[f"z_{f}" for f in FEATURES]].notna().all(axis=1)
    return p.sort_values(["period_ord", "symbol"]).reset_index(drop=True)


@dataclass
class Fit:
    scores: pd.Series
    selected: list[str]
    alpha: float


def fit_period(panel: pd.DataFrame, ord_now: int, train_weeks: int, min_cs: int) -> Fit | None:
    train_ords = list(range(ord_now - train_weeks, ord_now))
    if min(train_ords, default=-1) < 0: return None
    zcols = [f"z_{f}" for f in FEATURES]
    gammas_a = []; gammas_b = []; chunks = []
    for o in train_ords:
        q = panel[(panel.period_ord == o) & panel.all_x & panel.clean_week_ret.notna()]
        if len(q) < min_cs: continue
        X = q[zcols].to_numpy(float); y0 = q.clean_week_ret.to_numpy(float); y = y0 - float(np.mean(y0))
        xm = X.mean(axis=0); ym = float(y.mean()); xc = X - xm; yc = y - ym
        var = np.mean(xc * xc, axis=0); cov = np.mean(xc * yc[:, None], axis=0)
        b = np.divide(cov, var, out=np.full_like(cov, np.nan), where=var > 1e-14); a = ym - b * xm
        gammas_a.append(a); gammas_b.append(b)
        qq = q[["symbol", "period_ord", "clean_week_ret"] + zcols].copy(); qq["y_dm"] = y; chunks.append(qq)
    if len(chunks) < max(26, train_weeks // 2): return None
    A = np.nanmean(np.vstack(gammas_a), axis=0); B = np.nanmean(np.vstack(gammas_b), axis=0)
    if np.isfinite(B).sum() < 5: return None
    tr = pd.concat(chunks, ignore_index=True); Xz = tr[zcols].to_numpy(float); y = tr.y_dm.to_numpy(float)
    Xhat = A[None, :] + Xz * B[None, :]
    good = np.isfinite(y) & np.all(np.isfinite(Xhat), axis=1); Xhat, y = Xhat[good], y[good]
    if len(y) < 100: return None
    mu = Xhat.mean(axis=0); sd = Xhat.std(axis=0, ddof=0); sd[sd < 1e-12] = 1.0; Xs = (Xhat - mu) / sd
    ybar = float(y.mean()); yc = y - ybar; l1_ratio = 0.5
    amax = float(np.max(np.abs(Xs.T @ yc)) / (len(y) * l1_ratio))
    if not np.isfinite(amax) or amax <= 0: return None
    alphas = amax * np.logspace(0.0, -4.0, 200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, coefs, _ = enet_path(Xs, yc, l1_ratio=l1_ratio, alphas=alphas, max_iter=5000, tol=1e-6)
    pred = ybar + Xs @ coefs; mse = np.mean((y[:, None] - pred) ** 2, axis=0)
    df = np.count_nonzero(np.abs(coefs) > 1e-10, axis=0); valid = (mse > 0) & ((len(y) - df - 1) > 0)
    aic = np.full(len(alphas), np.inf)
    aic[valid] = len(y) * np.log(mse[valid]) + 2.0 * df[valid] * len(y) / (len(y) - df[valid] - 1.0)
    j = int(np.argmin(aic)); beta_std = coefs[:, j]; selected_mask = beta_std > 1e-10
    if not selected_mask.any(): return None
    cur = panel[(panel.period_ord == ord_now) & panel.all_x]
    if cur.empty: return None
    Xout = cur[zcols].to_numpy(float); forecasts = A[None, :] + Xout * B[None, :]
    score = np.mean(forecasts[:, selected_mask], axis=1)
    selected = [f for f, keep in zip(FEATURES, selected_mask) if keep]
    return Fit(pd.Series(score, index=cur.index), selected, float(alphas[j]))


def fit_all(panel: pd.DataFrame, eval_ords: list[int], train_weeks: int, min_cs: int, workers: int) -> dict[int, Fit | None]:
    out = {}; workers = max(1, workers); print(f"CTREND model workers: {workers}", flush=True)
    if workers == 1:
        for i, o in enumerate(eval_ords, 1):
            out[o] = fit_period(panel, o, train_weeks, min_cs)
            if i % 20 == 0 or i == len(eval_ords): print(f"Model fits: {i}/{len(eval_ords)}", flush=True)
        return out
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fit_period, panel, o, train_weeks, min_cs): o for o in eval_ords}
        for i, fut in enumerate(as_completed(futs), 1):
            o = futs[fut]; out[o] = fut.result()
            if i % 20 == 0 or i == len(eval_ords): print(f"Model fits: {i}/{len(eval_ords)}", flush=True)
    return out


def forward_exit(raw_groups, sym: str, entry_date: pd.Timestamp, end_date: pd.Timestamp):
    g = raw_groups.get(sym)
    if g is None or g.empty or pd.isna(entry_date): return None
    q = g[(g.date > entry_date) & (g.date <= end_date)]
    if q.empty: return None
    row = q.iloc[-1]; actual = pd.Timestamp(row.date); last_all = pd.Timestamp(g.date.iloc[-1])
    forced = bool(actual < end_date and last_all <= end_date)
    return actual, float(row.close), forced


def funding_prefix(con: sqlite3.Connection, symbols: list[str], start: pd.Timestamp, end: pd.Timestamp):
    lo = int((start - pd.Timedelta(days=14)).timestamp() * 1000); hi = int((end + pd.Timedelta(days=14)).timestamp() * 1000)
    out = {}; unique = sorted(set(symbols))
    for i, sym in enumerate(unique, 1):
        rows = con.execute("SELECT event_time, rate FROM funding_events WHERE symbol=? AND event_time BETWEEN ? AND ? ORDER BY event_time", (sym, lo, hi)).fetchall()
        if rows:
            t = np.array([r[0] for r in rows], dtype=np.int64); r = np.array([r[1] for r in rows], dtype=float)
            out[sym] = (t, np.concatenate([[0.0], np.cumsum(r)]))
        if i % 100 == 0 or i == len(unique): print(f"Funding prefix: {i}/{len(unique)}", flush=True)
    return out


def funding_sum(pref, sym: str, entry: pd.Timestamp, exit_: pd.Timestamp):
    item = pref.get(sym)
    if item is None: return 0.0, False
    t, cs = item; a = int(entry.timestamp() * 1000); b = int((exit_ + pd.Timedelta(days=1)).timestamp() * 1000) - 1
    i0 = int(np.searchsorted(t, a, side="right")); i1 = int(np.searchsorted(t, b, side="right"))
    return float(cs[i1] - cs[i0]), True


def perf(r: pd.Series) -> dict[str, float]:
    r = r.fillna(0.0); eq = (1.0 + r).cumprod(); dd = eq / eq.cummax() - 1.0
    pos = float(r[r > 0].sum()); neg = float(-r[r < 0].sum()); sd = float(r.std(ddof=1)); years = len(r) / 52.0
    return {"total": float(eq.iloc[-1] - 1.0), "cagr": float(eq.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and eq.iloc[-1] > 0 else np.nan,
            "pf": pos / neg if neg > 0 else np.inf, "sharpe": math.sqrt(52.0) * float(r.mean()) / sd if sd > 0 else np.nan,
            "mdd": float(dd.min()), "equity": float(100.0 * eq.iloc[-1])}


def pct(x: float) -> str:
    return "nan" if not np.isfinite(x) else f"{100*x:.2f}%"


def main() -> int:
    cfg = parse_args(); start = pd.Timestamp(cfg.start, tz="UTC"); end = pd.Timestamp(cfg.end, tz="UTC")
    outdir = Path(cfg.output_dir); outdir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(cfg.db, timeout=120); con.execute("PRAGMA temp_store=MEMORY"); con.execute("PRAGMA cache_size=-262144")

    print("=== CTREND AUTHOR-CODE CAUSAL FORENSIC ===")
    print(f"Evaluation: {start.date()} -> {end.date()}")
    print("Source mechanics: official Dataverse b02/b05/fWalkforwardCSENET/fEstRegPanelRegression")
    print("Author formulas + 52-block/year sampling + 0.5% historical training truncation")
    print("Causal corrections: no future-return availability filter; no full-year LAST fallback; raw OOS PnL retained")
    print("Estimation objective: equal-weight CS-C-ENet (an explicit author research-design option)")
    print(f"Evaluation universe: point-in-time top {cfg.eval_universe} Binance perps by lagged 30d quote volume")
    print(f"Portfolio: top {cfg.top_frac:.0%} CTREND, long-only, 1x, {cfg.side_cost_bps:.1f} bps/changed side + real funding")
    print("No TSMOM, no stop/TP, no parameter search.\n")

    raw = raw_daily(con, start, end); rc = apply_author_truncation(raw); weekly, raw_groups = build_weekly(rc, cfg.workers); panel = build_target_panel(weekly)
    period_meta = weekly[["period_ord", "week_key", "period_end"]].drop_duplicates().sort_values("period_ord")
    eval_ords = period_meta[(period_meta.period_end >= start) & (period_meta.period_end <= end)].period_ord.astype(int).tolist()
    eval_ords = [o for o in eval_ords if o >= cfg.train_weeks]
    print(f"Evaluation periods with 52-week history: {len(eval_ords)}", flush=True)
    fits = fit_all(panel, eval_ords, cfg.train_weeks, cfg.min_cross_section, cfg.workers)
    pref = funding_prefix(con, panel.symbol.unique().tolist(), start, end)

    prev = {}; rows = []; assets = []; ic_raw = []; ic_clean = []; side_cost = cfg.side_cost_bps / 10_000.0
    for i, o in enumerate(eval_ords, 1):
        fit = fits.get(o); cur = panel[(panel.period_ord == o) & panel.all_x].copy()
        if fit is None or cur.empty: continue
        cur["cttrend"] = fit.scores.reindex(cur.index); cur = cur.dropna(subset=["cttrend"])
        if len(cur) < cfg.min_cross_section: continue
        raw_ret_map = {}; actual_map = {}; forced_map = {}; end_date = pd.Timestamp(cur.period_end.iloc[0])
        for rr in cur.itertuples():
            fx = forward_exit(raw_groups, rr.symbol, rr.lag_formation_date, end_date)
            if fx is None or not np.isfinite(rr.lag_formation_close): continue
            actual, exit_close, forced = fx
            raw_ret_map[rr.symbol] = exit_close / float(rr.lag_formation_close) - 1.0; actual_map[rr.symbol] = actual; forced_map[rr.symbol] = forced
        cur["raw_oos_ret"] = cur.symbol.map(raw_ret_map)
        qraw = cur.dropna(subset=["raw_oos_ret"])
        if len(qraw) >= cfg.min_cross_section: ic_raw.append((o, float(qraw.cttrend.corr(qraw.raw_oos_ret, method="spearman"))))
        qcl = cur.dropna(subset=["clean_week_ret"])
        if len(qcl) >= cfg.min_cross_section: ic_clean.append((o, float(qcl.cttrend.corr(qcl.clean_week_ret, method="spearman"))))

        elig = cur.dropna(subset=["lag_liq_30d", "cttrend"]); elig = elig.nlargest(min(cfg.eval_universe, len(elig)), "lag_liq_30d")
        k = max(1, int(math.ceil(len(elig) * cfg.top_frac))); pick = elig.nlargest(k, "cttrend")
        missing = [s for s in pick.symbol if s not in raw_ret_map]
        if missing: raise RuntimeError(f"Selected assets lack causal exit in period {int(cur.week_key.iloc[0])}: {missing}")
        w = 1.0 / len(pick) if len(pick) else 0.0; curw = {s: w for s in pick.symbol}
        turnover = sum(abs(curw.get(s, 0.0) - prev.get(s, 0.0)) for s in set(curw) | set(prev)); cost = turnover * side_cost
        gross = 0.0; fund = 0.0; fknown = 0; forced_notional = 0.0
        for rr in pick.itertuples():
            r = float(raw_ret_map[rr.symbol]); actual = actual_map[rr.symbol]; forced = bool(forced_map[rr.symbol])
            fr, known = funding_sum(pref, rr.symbol, pd.Timestamp(rr.lag_formation_date), actual)
            gross_piece = w * r; fund_piece = -w * fr; gross += gross_piece; fund += fund_piece; fknown += int(known)
            if forced: forced_notional += w
            assets.append({"period_ord": o, "week_key": int(rr.week_key), "symbol": rr.symbol, "weight": w, "cttrend": float(rr.cttrend),
                           "raw_return": r, "clean_return": float(rr.clean_week_ret) if np.isfinite(rr.clean_week_ret) else np.nan,
                           "funding_sum": fr, "gross_contribution": gross_piece, "funding_contribution": fund_piece,
                           "forced_exit": forced, "entry_date": rr.lag_formation_date, "exit_date": actual})
        if forced_notional: turnover += forced_notional; cost += side_cost * forced_notional
        net = gross + fund - cost
        rows.append({"period_ord": o, "week_key": int(cur.week_key.iloc[0]), "period_end": end_date, "positions": len(pick),
                     "gross_return": gross, "funding_return": fund, "turnover": turnover, "cost_return": cost, "net_return": net,
                     "selected_features": len(fit.selected), "features": ",".join(fit.selected), "enet_alpha": fit.alpha,
                     "funding_coverage": fknown / len(pick) if len(pick) else 1.0})
        forced_syms = {a["symbol"] for a in assets if a["period_ord"] == o and a["forced_exit"]}
        prev = {s: ww for s, ww in curw.items() if s not in forced_syms}
        if i % 20 == 0 or i == len(eval_ords): print(f"Portfolio pass: {i}/{len(eval_ords)} | {int(cur.week_key.iloc[0])} | pos={len(pick)} | net={net:+.3%}", flush=True)

    if not rows: raise RuntimeError("No OOS portfolio periods")
    wdf = pd.DataFrame(rows).sort_values("period_ord").reset_index(drop=True); adf = pd.DataFrame(assets)
    if prev:
        terminal = sum(abs(v) for v in prev.values()); wdf.loc[wdf.index[-1], "turnover"] += terminal
        wdf.loc[wdf.index[-1], "cost_return"] += terminal * side_cost; wdf.loc[wdf.index[-1], "net_return"] -= terminal * side_cost
    wdf["equity"] = 100.0 * (1.0 + wdf.net_return).cumprod()

    icr = pd.DataFrame(ic_raw, columns=["period_ord", "rank_ic_raw"]); icc = pd.DataFrame(ic_clean, columns=["period_ord", "rank_ic_clean"])
    pm = period_meta[["period_ord", "week_key", "period_end"]]
    icdf = pm.merge(icr, on="period_ord", how="left").merge(icc, on="period_ord", how="left"); icdf = icdf[icdf.period_ord.isin(eval_ords)]
    m = perf(wdf.net_return); active = wdf[wdf.positions > 0]; wr = float((active.net_return > 0).mean()) if len(active) else np.nan
    raw_ic = icdf.rank_ic_raw.dropna(); clean_ic = icdf.rank_ic_clean.dropna()
    raw_t = float(raw_ic.mean() / (raw_ic.std(ddof=1) / math.sqrt(len(raw_ic)))) if len(raw_ic) > 1 and raw_ic.std(ddof=1) > 0 else np.nan
    clean_t = float(clean_ic.mean() / (clean_ic.std(ddof=1) / math.sqrt(len(clean_ic)))) if len(clean_ic) > 1 and clean_ic.std(ddof=1) > 0 else np.nan

    print("\n=== AUTHOR-CODE-DERIVED CAUSAL RESULT ===")
    print(f"Periods: {len(wdf)} | active WR: {pct(wr)} | avg positions: {wdf.positions.mean():.2f}")
    print(f"$100 -> ${m['equity']:.2f} | total={pct(m['total'])} | CAGR={pct(m['cagr'])}")
    print(f"PF={m['pf']:.3f} | Sharpe={m['sharpe']:.3f} | MDD={pct(m['mdd'])}")
    print(f"Avg gross/funding/cost: {pct(wdf.gross_return.mean())} / {pct(wdf.funding_return.mean())} / {pct(wdf.cost_return.mean())}")
    print(f"Avg turnover: {wdf.turnover.mean():.3f} | avg selected features: {wdf.selected_features.mean():.1f}")
    print(f"Funding coverage: {wdf[wdf.positions>0].funding_coverage.mean():.2%}")
    print("\nRANK-IC DIAGNOSTIC")
    print(f"RAW tradable next-period return: weeks={len(raw_ic)} mean={raw_ic.mean():+.4f} median={raw_ic.median():+.4f} positive={(raw_ic>0).mean():.2%} t={raw_t:+.2f}")
    print(f"AUTHOR-cleaned return target:     weeks={len(clean_ic)} mean={clean_ic.mean():+.4f} median={clean_ic.median():+.4f} positive={(clean_ic>0).mean():.2%} t={clean_t:+.2f}")

    print("\nYEAR BREAKDOWN (RAW TRADABLE NET PNL)")
    yrrows = []
    for year, q in wdf.groupby(wdf.period_end.dt.year):
        mm = perf(q.net_return); yrrows.append({"year": int(year), "return": mm["total"], "pf": mm["pf"], "sharpe": mm["sharpe"], "mdd": mm["mdd"], "wr": float((q.net_return>0).mean())})
    ydf = pd.DataFrame(yrrows); yprint = ydf.copy()
    for c in ["return", "mdd", "wr"]: yprint[c] = yprint[c].map(pct)
    print(yprint.to_string(index=False))

    print("\nFORENSIC VERDICT")
    if raw_ic.mean() <= 0:
        print("[CLOSE] Corrected author mechanics still have non-positive RAW OOS rank IC on Binance perps.")
    elif m["pf"] <= 1.0 or m["sharpe"] <= 0:
        print("[SIGNAL ONLY] RAW rank IC is positive, but the causal long high-leg portfolio is not profitable after real costs/funding.")
    else:
        print("[KEEP INVESTIGATING] RAW rank IC and causal long high-leg are both positive; this is the first author-code-derived pass worth deeper validation.")
    if clean_ic.mean() > 0 and raw_ic.mean() <= 0:
        print("[WARNING] Signal works only on author-cleaned targets, not raw tradable returns; data-cleaning dependence is likely material.")

    wdf.to_csv(outdir / "weekly_results.csv", index=False); adf.to_csv(outdir / "asset_contributions.csv", index=False)
    icdf.to_csv(outdir / "rank_ic.csv", index=False); ydf.to_csv(outdir / "year_breakdown.csv", index=False)
    print(f"\nSaved under: {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
