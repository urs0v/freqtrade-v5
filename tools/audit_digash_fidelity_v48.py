#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

LOW_TFS = ("1m", "5m", "15m")
FEATURES = (
    "dir_imb_10s",
    "dir_imb_60s",
    "trade_accel_60v300",
    "qvol_accel_60v300",
    "near25_dir_imb_300s",
    "dir_eff_300s",
    "dir_run_frac_30s",
)
API = "https://fapi.binance.com/fapi/v1/aggTrades"


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.8: historical Binance aggTrades microstructure audit")
    p.add_argument("--v47dir", default="/freqtrade/user_data/digash_fidelity_v47")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v48")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pre-minutes", type=int, default=40)
    p.add_argument("--post-minutes", type=int, default=5)
    p.add_argument("--refresh", action="store_true")
    return p.parse_args()


def _truthy(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _symbol(pair: str) -> str:
    base = pair.split("/")[0]
    return f"{base}USDT"


def _safe(qid: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", qid)[:180]


def _fetch_json(url: str, attempts: int = 6):
    last = None
    for n in range(attempts):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            last = e
            if e.code in (418, 429):
                wait = float(e.headers.get("Retry-After") or (2 + n * 2))
                time.sleep(min(wait, 30.0))
                continue
            if e.code in (400, 404, 451):
                raise
            time.sleep(1.0 + n * 1.5)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError) as e:
            last = e
            time.sleep(1.0 + n * 1.5)
    raise RuntimeError(f"aggTrades fetch failed: {last}")


def fetch_aggtrades(symbol: str, start_ms: int, end_ms: int) -> list[dict]:
    params = urllib.parse.urlencode({"symbol": symbol, "startTime": start_ms, "endTime": end_ms, "limit": 1000})
    first = _fetch_json(f"{API}?{params}")
    if not isinstance(first, list):
        raise RuntimeError(f"unexpected Binance response for {symbol}: {first}")
    rows = list(first)
    if not rows:
        return []
    last_id = int(rows[-1]["a"])
    last_t = int(rows[-1]["T"])
    loops = 0
    while len(first) >= 1000 and last_t <= end_ms:
        loops += 1
        if loops > 5000:
            raise RuntimeError(f"pagination runaway for {symbol}")
        p2 = urllib.parse.urlencode({"symbol": symbol, "fromId": last_id + 1, "limit": 1000})
        nxt = _fetch_json(f"{API}?{p2}")
        if not isinstance(nxt, list) or not nxt:
            break
        rows.extend(x for x in nxt if int(x["T"]) <= end_ms)
        new_id = int(nxt[-1]["a"])
        new_t = int(nxt[-1]["T"])
        if new_id <= last_id:
            break
        last_id, last_t = new_id, new_t
        first = nxt
        if new_t > end_ms:
            break
        time.sleep(0.02)
    rows = [x for x in rows if start_ms <= int(x["T"]) <= end_ms]
    return rows


def load_or_fetch(row: dict, cache: Path, pre_minutes: int, post_minutes: int, refresh: bool):
    qid = str(row["query_id"])
    path = cache / f"{_safe(qid)}.json.gz"
    if path.exists() and not refresh:
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f), "CACHE"
    cross = pd.Timestamp(row["cross_time"])
    start_ms = int((cross - pd.Timedelta(minutes=pre_minutes)).timestamp() * 1000)
    end_ms = int((cross + pd.Timedelta(minutes=post_minutes)).timestamp() * 1000)
    rows = fetch_aggtrades(_symbol(str(row["pair"])), start_ms, end_ms)
    cache.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump(rows, f, separators=(",", ":"))
    return rows, "API"


def trades_df(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    z = pd.DataFrame({
        "id": [int(x["a"]) for x in rows],
        "price": [float(x["p"]) for x in rows],
        "qty": [float(x["q"]) for x in rows],
        "time_ms": [int(x["T"]) for x in rows],
        "buyer_maker": [bool(x["m"]) for x in rows],
    }).sort_values(["time_ms", "id"]).drop_duplicates("id").reset_index(drop=True)
    z["quote"] = z.price * z.qty
    z["taker_sign"] = np.where(z.buyer_maker, -1.0, 1.0)  # +1 = aggressive buyer, -1 = aggressive seller
    return z


def exact_cross_index(z: pd.DataFrame, approx_cross: pd.Timestamp, level: float, direction: int, detail_source: str):
    if z.empty or len(z) < 2:
        return None
    detail_min = 1 if str(detail_source).startswith("1m") else 5
    a_ms = int((approx_cross - pd.Timedelta(minutes=detail_min)).timestamp() * 1000)
    b_ms = int((approx_cross + pd.Timedelta(seconds=30)).timestamp() * 1000)
    p = z.price.to_numpy(float)
    t = z.time_ms.to_numpy(np.int64)
    candidates = []
    for i in range(1, len(z)):
        if t[i] < a_ms or t[i] > b_ms:
            continue
        if direction > 0:
            ok = p[i - 1] < level <= p[i]
        else:
            ok = p[i - 1] > level >= p[i]
        if ok:
            candidates.append(i)
    return candidates[0] if candidates else None


def _window(z: pd.DataFrame, cross_ms: int, seconds: int) -> pd.DataFrame:
    a = cross_ms - seconds * 1000
    return z[(z.time_ms >= a) & (z.time_ms < cross_ms)]


def _imb(w: pd.DataFrame, direction: int):
    if w.empty:
        return np.nan
    q = w.quote.to_numpy(float)
    total = float(np.sum(q))
    if total <= 0:
        return np.nan
    raw = float(np.sum(w.taker_sign.to_numpy(float) * q) / total)
    return direction * raw


def _rate(w: pd.DataFrame, seconds: float):
    return float(len(w) / seconds) if seconds > 0 else np.nan


def _qrate(w: pd.DataFrame, seconds: float):
    return float(w.quote.sum() / seconds) if seconds > 0 and not w.empty else 0.0


def _eff(w: pd.DataFrame, direction: int):
    if len(w) < 2:
        return np.nan
    p = w.price.to_numpy(float)
    denom = float(np.abs(np.diff(p)).sum())
    if denom <= 0:
        return 0.0
    return float(direction * (p[-1] - p[0]) / denom)


def _dir_run_frac(w: pd.DataFrame, direction: int):
    if w.empty:
        return np.nan
    sign = w.taker_sign.to_numpy(float) * direction
    q = w.quote.to_numpy(float)
    total = float(q.sum())
    if total <= 0:
        return np.nan
    best = 0.0
    cur = 0.0
    for s, qq in zip(sign, q):
        if s > 0:
            cur += float(qq)
            best = max(best, cur)
        else:
            cur = 0.0
    return float(best / total)


def extract_features(row: dict, rows: list[dict], source: str):
    z = trades_df(rows)
    out = dict(row)
    out["trade_source"] = source
    out["aggtrade_n"] = len(z)
    if z.empty:
        out["micro_status"] = "NO_TRADES"
        return out
    level = float(row["level_price"])
    post_close = float(row["post_close"])
    direction = 1 if level >= post_close else -1
    approx = pd.Timestamp(row["cross_time"])
    i = exact_cross_index(z, approx, level, direction, str(row.get("detail_source", "5m")))
    if i is None:
        out["micro_status"] = "NO_EXACT_CROSS"
        return out
    cross_ms = int(z.time_ms.iloc[i])
    out["micro_status"] = "OK"
    out["exact_cross_time"] = pd.to_datetime(cross_ms, unit="ms", utc=True)
    out["exact_cross_price"] = float(z.price.iloc[i])
    out["cross_time_offset_s"] = (pd.Timestamp(out["exact_cross_time"]) - approx).total_seconds()

    w10 = _window(z, cross_ms, 10)
    w30 = _window(z, cross_ms, 30)
    w60 = _window(z, cross_ms, 60)
    w300 = _window(z, cross_ms, 300)
    out["dir_imb_10s"] = _imb(w10, direction)
    out["dir_imb_60s"] = _imb(w60, direction)
    out["dir_imb_300s"] = _imb(w300, direction)
    r60, r300 = _rate(w60, 60.0), _rate(w300, 300.0)
    q60, q300 = _qrate(w60, 60.0), _qrate(w300, 300.0)
    out["trade_rate_60s"] = r60
    out["trade_rate_300s"] = r300
    out["trade_accel_60v300"] = r60 / r300 if np.isfinite(r300) and r300 > 0 else np.nan
    out["qvol_rate_60s"] = q60
    out["qvol_rate_300s"] = q300
    out["qvol_accel_60v300"] = q60 / q300 if q300 > 0 else np.nan
    near = w300[(w300.price / level - 1.0).abs() * 10000.0 <= 25.0]
    out["near25_qshare_300s"] = float(near.quote.sum() / w300.quote.sum()) if len(w300) and w300.quote.sum() > 0 else np.nan
    out["near25_dir_imb_300s"] = _imb(near, direction)
    out["dir_eff_300s"] = _eff(w300, direction)
    out["dir_run_frac_30s"] = _dir_run_frac(w30, direction)
    return out


def process_one(row: dict, cache: Path, pre_minutes: int, post_minutes: int, refresh: bool):
    t0 = time.monotonic()
    try:
        rows, src = load_or_fetch(row, cache, pre_minutes, post_minutes, refresh)
        z = extract_features(row, rows, src)
        return z, {"query_id": row["query_id"], "pair": row["pair"], "tf": row["tf"], "status": z.get("micro_status"), "trades": len(rows), "elapsed_s": time.monotonic() - t0}
    except Exception as e:
        z = dict(row)
        z["micro_status"] = "ERROR"
        z["micro_error"] = f"{type(e).__name__}: {e}"
        return z, {"query_id": row["query_id"], "pair": row["pair"], "tf": row["tf"], "status": "ERROR", "error": z["micro_error"], "elapsed_s": time.monotonic() - t0}


def auc_binary(y: pd.Series, x: pd.Series):
    d = pd.DataFrame({"y": pd.to_numeric(y, errors="coerce"), "x": pd.to_numeric(x, errors="coerce")}).dropna()
    d = d[d.y.isin([0, 1])]
    np_, nn = int((d.y == 1).sum()), int((d.y == 0).sum())
    if np_ == 0 or nn == 0:
        return np.nan, len(d)
    ranks = d.x.rank(method="average")
    s = float(ranks[d.y == 1].sum())
    auc = (s - np_ * (np_ + 1) / 2.0) / (np_ * nn)
    return float(auc), len(d)


def composite(train: pd.DataFrame, test: pd.DataFrame, label: str):
    parts_train, parts_test, dirs = [], [], []
    for f in FEATURES:
        a, _ = auc_binary(train[label], train[f])
        if not np.isfinite(a):
            continue
        sign = 1.0 if a >= 0.5 else -1.0
        mu = float(pd.to_numeric(train[f], errors="coerce").mean())
        sd = float(pd.to_numeric(train[f], errors="coerce").std(ddof=0))
        if not np.isfinite(sd) or sd <= 1e-12:
            continue
        parts_train.append(sign * (pd.to_numeric(train[f], errors="coerce") - mu) / sd)
        parts_test.append(sign * (pd.to_numeric(test[f], errors="coerce") - mu) / sd)
        dirs.append((f, a, sign))
    if not parts_train:
        return pd.Series(np.nan, index=train.index), pd.Series(np.nan, index=test.index), dirs
    tr = pd.concat(parts_train, axis=1).mean(axis=1, skipna=True)
    te = pd.concat(parts_test, axis=1).mean(axis=1, skipna=True)
    return tr, te, dirs


def report_auc(train: pd.DataFrame, hold: pd.DataFrame, label: str):
    print(f"\nOutcome: {label}", flush=True)
    print("feature                         EARLY_AUC  HOLD_AUC  HOLD_N", flush=True)
    for f in FEATURES:
        a, _ = auc_binary(train[label], train[f])
        sign = 1.0 if (np.isfinite(a) and a >= 0.5) else -1.0
        h, hn = auc_binary(hold[label], sign * pd.to_numeric(hold[f], errors="coerce"))
        ao = max(a, 1.0 - a) if np.isfinite(a) else np.nan
        print(f"{f:31s} {ao:9.3f} {h:9.3f} {hn:7d}", flush=True)
    trc, hoc, dirs = composite(train, hold, label)
    at, nt = auc_binary(train[label], trc)
    ah, nh = auc_binary(hold[label], hoc)
    print(f"COMPOSITE(all, early-oriented)    {at:9.3f} {ah:9.3f} {nh:7d}", flush=True)
    print("early directions: " + ", ".join(f"{f}={'+' if s > 0 else '-'}(rawAUC={a:.3f})" for f, a, s in dirs), flush=True)


def main() -> int:
    a = parse_args()
    v47 = Path(a.v47dir)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "aggtrades_cache"
    path = v47 / "breakout_quality_rows.csv"
    if not path.exists():
        raise FileNotFoundError("Run Fidelity V4.7 first; V4.8 consumes breakout_quality_rows.csv")
    z = pd.read_csv(path)
    z["post_time"] = pd.to_datetime(z.post_time, utc=True, errors="coerce")
    z["cross_time"] = pd.to_datetime(z.cross_time, utc=True, errors="coerce")
    z = z[z.tf.isin(LOW_TFS) & _truthy(z.cross_found) & z.cross_time.notna()].copy()
    z = z.drop_duplicates("query_id")
    if z.empty:
        print("No crossed LOW_TF source rows.", flush=True)
        return 2

    print("=== DIGASH FIDELITY V4.8 — AGGTRADES MICROSTRUCTURE ===", flush=True)
    print("This is the last historical fidelity layer before live orderbook capture.", flush=True)
    print("Downloads only public Binance USD-M aggTrades in narrow windows around already-known public Digash level crosses; cache is reused on rerun.", flush=True)
    print("No PnL fitting, no stop/target fitting, no candle-threshold search.", flush=True)
    print("Features are pre-cross only: directional taker imbalance, activity acceleration, near-level imbalance, directional price efficiency, directional aggressive-trade run.", flush=True)
    print(f"queries={len(z)} | workers={a.workers} | window=-{a.pre_minutes}m/+{a.post_minutes}m", flush=True)

    rows, metas = [], []
    t0 = time.monotonic(); done = 0
    with ThreadPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(process_one, r, cache, a.pre_minutes, a.post_minutes, a.refresh) for r in z.to_dict("records")]
        for f in as_completed(futs):
            row, meta = f.result(); rows.append(row); metas.append(meta); done += 1
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(futs) - done) / done if done else np.nan
            print(f"V4.8 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} trades={meta.get('trades',0)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(out / "coverage.csv", index=False)
    r = pd.DataFrame(rows)
    r.to_csv(out / "microstructure_rows.csv", index=False)
    ok = r[r.micro_status.eq("OK")].copy()
    print("\n=== MICROSTRUCTURE COVERAGE ===", flush=True)
    print(f"rows={len(r)} | exact trade-cross OK={len(ok)} ({len(ok)/len(r)*100:.1f}%) | errors={(r.micro_status == 'ERROR').sum()}", flush=True)
    if ok.empty:
        print("No exact aggTrade crosses reconstructed. Check coverage.csv / API accessibility.", flush=True)
        return 2

    ok["ret1h_positive"] = (pd.to_numeric(ok.ret_1h_bps, errors="coerce") > 0).astype(int)
    ok["mfe_gt_mae_1h"] = (pd.to_numeric(ok.mfe_1h_bps, errors="coerce") > pd.to_numeric(ok.mae_1h_bps, errors="coerce")).astype(int)
    train = ok[ok.cohort.eq("EARLY")].copy()
    hold = ok[ok.cohort.eq("HOLDOUT")].copy()

    print("\n=== PRE-CROSS MICROSTRUCTURE -> FOLLOW-THROUGH (EARLY-ORIENTED, HOLDOUT TESTED) ===", flush=True)
    print(f"EARLY N={len(train)} | HOLDOUT N={len(hold)}", flush=True)
    report_auc(train, hold, "ret1h_positive")
    report_auc(train, hold, "mfe_gt_mae_1h")

    print("\n=== HOLDOUT RAW MEDIANS BY 1H RETURN SIGN ===", flush=True)
    for f in FEATURES:
        gp = pd.to_numeric(hold.loc[hold.ret1h_positive.eq(1), f], errors="coerce").median()
        gn = pd.to_numeric(hold.loc[hold.ret1h_positive.eq(0), f], errors="coerce").median()
        print(f"{f:31s} positive={gp:9.4f} negative={gn:9.4f}", flush=True)

    print("\n=== DECISION RULE ===", flush=True)
    print("If the early-oriented microstructure composite and multiple individual features retain useful discrimination on HOLDOUT (roughly AUC materially above 0.5), freeze the signal and move to one Digash Replication V5 PnL backtest.", flush=True)
    print("If HOLDOUT AUC is near 0.5 or directions reverse, stop historical proxy invention: exact Digash replication then requires prospective orderbook/density capture, because OHLCV + executed trades still omit resting liquidity.", flush=True)
    print(f"Reports: {out}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
