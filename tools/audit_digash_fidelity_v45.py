#!/usr/bin/env python3
from __future__ import annotations

import argparse
import itertools
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc
from audit_digash_fidelity_v4 import load_published_tf
from audit_digash_fidelity_v43 import _touch_schedule, _active_rearmed

MATCH_BPS = 25.0
LOW_TFS = ("1m", "5m", "15m")
TOPK = (1, 3, 5, 10)


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4.5: chronological selector holdout")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--v4dir", default="/freqtrade/user_data/digash_fidelity_v4")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v45")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--warmup-days", type=int, default=120)
    p.add_argument("--train-frac", type=float, default=0.70)
    return p.parse_args()


def _prior_schedule(schedule, post_ns):
    return [q for q in schedule if q[0] <= post_ns]


def _candidate_meta(lv, schedules, post_ns, close0):
    prior = _prior_schedule(schedules[lv.level_id], post_ns)
    if prior:
        last_touch_ns = prior[-1][0]
        touch_n = len(prior)
        last_touch_age_h = (post_ns - last_touch_ns) / 3.6e12
    else:
        touch_n = 0
        last_touch_age_h = np.nan
    return {
        "level_id": int(lv.level_id),
        "level_price": float(lv.price),
        "period": int(lv.period),
        "distance_bps": abs(float(lv.price) / close0 - 1.0) * 10000.0,
        "last_touch_age_h": float(last_touch_age_h) if np.isfinite(last_touch_age_h) else np.nan,
        "touch_n": int(touch_n),
        "touch_error_pct": float(lv.touch_error_pct),
    }


def audit_group(records, config_path, datadir_s, warmup_days):
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    pair, tf = records[0]["pair"], records[0]["tf"]
    dc.TF_MINUTES.setdefault("1m", 1)
    try:
        times = pd.to_datetime([r["post_time"] for r in records], utc=True)
        x, source = load_published_tf(config, datadir, pair, tf)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_DATA", "elapsed_s": time.monotonic() - t0}
        x = x[(x.date >= times.min() - pd.Timedelta(days=warmup_days)) & (x.date < times.max() + pd.Timedelta(days=2))].reset_index(drop=True)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_RANGE", "elapsed_s": time.monotonic() - t0}

        p20 = dc.build_levels(x, tf, 20, 0)
        p30 = dc.build_levels(x, tf, 30, len(p20))
        levels = p20 + p30
        pivots = dc.local_pivots(x)
        sig_ns = pd.to_datetime(x.signal_time, utc=True).astype("int64").to_numpy()
        close = x.close.to_numpy(float)
        schedules = {lv.level_id: _touch_schedule(lv, pivots, sig_ns, close) for lv in levels}

        rows = []
        for r in records:
            t = pd.Timestamp(r["post_time"])
            post_ns = int(t.value)
            pub = float(r["published_level"])
            idx = int(np.searchsorted(sig_ns, post_ns, side="right") - 1)
            if idx < 1 or idx >= len(close) or not np.isfinite(close[idx]) or close[idx] <= 0:
                continue
            c0 = float(close[idx])
            expected_kind = "R" if pub >= c0 else "S"
            formed = [lv for lv in levels if pd.Timestamp(lv.formed_time).value <= post_ns]
            rearm = [lv for lv in formed if _active_rearmed(schedules[lv.level_id], post_ns)]
            candidates = [
                lv for lv in rearm
                if lv.kind == expected_kind
                and ((lv.kind == "R" and lv.price >= c0) or (lv.kind == "S" and lv.price <= c0))
            ]
            qid = f"{int(r['post_id'])}:{int(r.get('published_level_no', 0))}:{pair}:{tf}"
            for lv in candidates:
                z = _candidate_meta(lv, schedules, post_ns, c0)
                z.update({
                    "query_id": qid,
                    "post_id": int(r["post_id"]),
                    "post_time": t,
                    "pair": pair,
                    "tf": tf,
                    "published_level_no": int(r.get("published_level_no", 0)),
                    "published_level": pub,
                    "post_close": c0,
                    "data_source": source,
                    "match_error_bps": abs(float(lv.price) / pub - 1.0) * 10000.0,
                })
                z["is_source_match25"] = bool(z["match_error_bps"] <= MATCH_BPS)
                rows.append(z)
        return rows, {
            "pair": pair,
            "tf": tf,
            "status": "OK",
            "rows": len(rows),
            "levels": len(levels),
            "elapsed_s": time.monotonic() - t0,
        }
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic() - t0}


def _add_query_ranks(c: pd.DataFrame) -> pd.DataFrame:
    if c.empty:
        return c
    out = c.copy()
    g = out.groupby("query_id", sort=False)
    out["f_distance"] = g["distance_bps"].rank(method="average", ascending=True, pct=True)
    rec = pd.to_numeric(out["last_touch_age_h"], errors="coerce")
    rec_fill = rec.fillna(np.inf)
    out["_rec_fill"] = rec_fill
    out["f_recency"] = out.groupby("query_id", sort=False)["_rec_fill"].rank(method="average", ascending=True, pct=True)
    out["f_touch"] = g["touch_n"].rank(method="average", ascending=False, pct=True)
    out["f_period30"] = (pd.to_numeric(out["period"], errors="coerce") == 30).astype(float)
    return out.drop(columns=["_rec_fill"])


def _score(c: pd.DataFrame, weights):
    wd, wr, wt, wp = weights
    denom = wd + wr + wt + wp
    if denom <= 0:
        return np.full(len(c), np.nan)
    return (
        wd * c["f_distance"].to_numpy(float)
        + wr * c["f_recency"].to_numpy(float)
        + wt * c["f_touch"].to_numpy(float)
        + wp * c["f_period30"].to_numpy(float)
    ) / float(denom)


def _query_eval(c: pd.DataFrame, score_col: str) -> pd.DataFrame:
    rows = []
    for qid, g in c.groupby("query_id", sort=False):
        gg = g.sort_values([score_col, "distance_bps", "last_touch_age_h", "level_id"], ascending=[True, True, True, True], na_position="last")
        pos = np.flatnonzero(gg["is_source_match25"].to_numpy(bool))
        best_rank = int(pos[0] + 1) if len(pos) else np.nan
        r0 = gg.iloc[0]
        rows.append({
            "query_id": qid,
            "post_id": int(r0.post_id),
            "post_time": r0.post_time,
            "pair": r0.pair,
            "tf": r0.tf,
            "published_level": float(r0.published_level),
            "candidate_n": len(gg),
            "covered25": bool(len(pos)),
            "best_rank": best_rank,
            "rr": (1.0 / best_rank) if np.isfinite(best_rank) else 0.0,
        })
    return pd.DataFrame(rows)


def _metrics(q: pd.DataFrame):
    if q.empty:
        return {"N": 0, "covered": np.nan, "mrr": np.nan, **{f"hit{k}": np.nan for k in TOPK}}
    rank = pd.to_numeric(q["best_rank"], errors="coerce")
    return {
        "N": len(q),
        "covered": float(q["covered25"].mean()),
        "mrr": float(pd.to_numeric(q["rr"], errors="coerce").mean()),
        **{f"hit{k}": float((rank <= k).fillna(False).mean()) for k in TOPK},
    }


def _fmt_metrics(label, q):
    m = _metrics(q)
    if m["N"] == 0:
        print(f"{label:24s} N=0", flush=True)
        return
    print(
        f"{label:24s} N={m['N']:3d} cover25={m['covered']*100:5.1f}% MRR={m['mrr']:.3f} | "
        + " ".join(f"hit@{k}={m[f'hit{k}']*100:5.1f}%" for k in TOPK),
        flush=True,
    )


def _evaluate_weights(c: pd.DataFrame, weights):
    z = c.copy()
    z["score"] = _score(z, weights)
    q = _query_eval(z, "score")
    return _metrics(q)["mrr"], q


def _fit_grid(train: pd.DataFrame):
    grid = []
    best = None
    best_mrr = -1.0
    # Coarse, interpretable, predeclared grid. No holdout feedback is used.
    for wd, wr, wt in itertools.product(range(5), repeat=3):
        for wp in range(3):
            w = (wd, wr, wt, wp)
            if sum(w) == 0:
                continue
            mrr, _ = _evaluate_weights(train, w)
            grid.append({"w_distance": wd, "w_recency": wr, "w_touch": wt, "w_period30": wp, "train_mrr": mrr})
            if mrr > best_mrr + 1e-12:
                best_mrr = mrr
                best = w
            elif abs(mrr - best_mrr) <= 1e-12 and best is not None:
                # Prefer simpler/tighter weights when train MRR ties.
                if (sum(w), w) < (sum(best), best):
                    best = w
    return best, pd.DataFrame(grid).sort_values(["train_mrr", "w_distance", "w_recency"], ascending=[False, True, True])


def _queries_from_weights(c, weights):
    z = c.copy()
    z["score"] = _score(z, weights)
    return _query_eval(z, "score")


def _novel_holdout(hold_q: pd.DataFrame, train_source_keys: set[tuple]):
    if hold_q.empty:
        return hold_q
    keep = [
        (str(r.pair), str(r.tf), round(float(r.published_level), 12)) not in train_source_keys
        for r in hold_q.itertuples(index=False)
    ]
    return hold_q[np.asarray(keep, dtype=bool)].copy()


def main():
    a = parse_args()
    v4dir = Path(a.v4dir)
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    parsed_path = v4dir / "digash_public_breakouts.csv"
    cov_path = v4dir / "cache_coverage.csv"
    if not parsed_path.exists() or not cov_path.exists():
        raise FileNotFoundError("Run Fidelity V4 first; V4.5 consumes its frozen public-source snapshot.")

    parsed = pd.read_csv(parsed_path)
    parsed["post_time"] = pd.to_datetime(parsed.post_time, utc=True, errors="coerce")
    cov = pd.read_csv(cov_path)
    ok = cov[cov.status.eq("OK")][["pair", "tf"]].drop_duplicates()
    work = parsed.merge(ok, on=["pair", "tf"], how="inner")
    groups = [g.to_dict("records") for _, g in work.groupby(["pair", "tf"], sort=True)]

    print("=== DIGASH FIDELITY V4.5 — CHRONOLOGICAL SELECTOR HOLDOUT ===", flush=True)
    print("NO PnL. NO entry fitting. NO downloads.", flush=True)
    print("Candidate universe is V4.3 touch-rearmed + side-consistent levels. Fit uses only early LOW_TF public alerts; later alerts are untouched selector holdout.", flush=True)
    print("Features: within-query rank of distance-to-price, last-touch recency, touch count, and a period30 penalty. Coarse integer weights only.", flush=True)
    print(f"covered source rows={len(work)} | groups={len(groups)} | workers={a.workers} | train_frac={a.train_frac:.2f}", flush=True)

    results = []
    metas = []
    t0 = time.monotonic()
    done = 0
    with ProcessPoolExecutor(max_workers=max(1, a.workers)) as ex:
        futs = [ex.submit(audit_group, recs, a.config, a.datadir, a.warmup_days) for recs in groups]
        for f in as_completed(futs):
            rows, meta = f.result()
            done += 1
            results.extend(rows)
            metas.append(meta)
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(futs) - done) / done if done else np.nan
            print(f"V4.5 {done:3d}/{len(futs)} {meta.get('pair','')} {meta.get('tf','')} status={meta.get('status')} candidates={len(rows)} elapsed={elapsed:.1f}s ETA={eta:.1f}s", flush=True)

    pd.DataFrame(metas).to_csv(outdir / "coverage.csv", index=False)
    c = pd.DataFrame(results)
    if c.empty:
        print("No candidate rows.", flush=True)
        return 2
    c["post_time"] = pd.to_datetime(c.post_time, utc=True, errors="coerce")
    c = _add_query_ranks(c)
    c.to_csv(outdir / "candidate_rows.csv", index=False)

    low = c[c.tf.isin(LOW_TFS)].copy()
    uq = low[["post_id", "post_time"]].drop_duplicates().sort_values(["post_time", "post_id"])
    if len(uq) < 10:
        print("Too few LOW_TF posts for chronological split.", flush=True)
        return 2
    cut_i = max(1, min(len(uq) - 1, int(np.floor(len(uq) * a.train_frac))))
    train_post_ids = set(uq.iloc[:cut_i].post_id.astype(int))
    hold_post_ids = set(uq.iloc[cut_i:].post_id.astype(int))
    cutoff_time = uq.iloc[cut_i].post_time
    train = low[low.post_id.astype(int).isin(train_post_ids)].copy()
    hold = low[low.post_id.astype(int).isin(hold_post_ids)].copy()

    print("\n=== CHRONOLOGICAL SPLIT ===", flush=True)
    print(f"LOW_TF unique posts={len(uq)} | train posts={len(train_post_ids)} | holdout posts={len(hold_post_ids)} | first holdout time={cutoff_time}", flush=True)
    print(f"candidate rows train={len(train):,} holdout={len(hold):,}", flush=True)

    baselines = {
        "DISTANCE_ONLY": (1, 0, 0, 0),
        "RECENCY_ONLY": (0, 1, 0, 0),
        "TOUCH_ONLY": (0, 0, 1, 0),
        "DIST+RECENCY": (1, 1, 0, 0),
        "EQUAL_3": (1, 1, 1, 0),
    }
    best, grid = _fit_grid(train)
    grid.to_csv(outdir / "train_weight_grid.csv", index=False)
    print(f"TRAIN-SELECTED WEIGHTS distance={best[0]} recency={best[1]} touch={best[2]} period30_penalty={best[3]}", flush=True)

    print("\n=== SELECTOR TRAIN VS UNTOUCHED HOLDOUT ===", flush=True)
    for name, w in {**baselines, "TRAIN_SELECTED": best}.items():
        tq = _queries_from_weights(train, w)
        hq = _queries_from_weights(hold, w)
        _fmt_metrics(f"TRAIN {name}", tq)
        _fmt_metrics(f"HOLD  {name}", hq)
        if name == "TRAIN_SELECTED":
            tq.to_csv(outdir / "train_queries_selected.csv", index=False)
            hq.to_csv(outdir / "holdout_queries_selected.csv", index=False)

    selected_hold = _queries_from_weights(hold, best)
    train_keys_df = train[["pair", "tf", "published_level"]].drop_duplicates()
    train_keys = {(str(r.pair), str(r.tf), round(float(r.published_level), 12)) for r in train_keys_df.itertuples(index=False)}
    novel = _novel_holdout(selected_hold, train_keys)
    print("\n=== HOLDOUT ROBUSTNESS ===", flush=True)
    _fmt_metrics("HOLDOUT ALL", selected_hold)
    _fmt_metrics("HOLDOUT NOVEL LEVEL", novel)
    for tf in LOW_TFS:
        _fmt_metrics(f"HOLDOUT {tf}", selected_hold[selected_hold.tf.eq(tf)])

    high = c[~c.tf.isin(LOW_TFS)].copy()
    if not high.empty:
        print("\n=== HIGH_TF OUT-OF-DOMAIN DIAGNOSTIC ===", flush=True)
        highq = _queries_from_weights(high, best)
        _fmt_metrics("1h+4h learned LOW_TF", highq)
        for tf in ("1h", "4h"):
            _fmt_metrics(tf, highq[highq.tf.eq(tf)])

    print("\n=== DECISION RULE ===", flush=True)
    print("Promote a selector mechanism only if the TRAIN_SELECTED score materially beats the simple baselines on the later chronological holdout, not merely on train.", flush=True)
    print("If the combined score does not generalize, do not invent more weights: the missing selector likely depends on source context absent from OHLCV (activity sorting, density/orderbook, movement quality, or other screener inputs).", flush=True)
    print("Activity is intentionally not included in within-post candidate ranking because it is common to all candidate levels for that formation timestamp.", flush=True)
    print(f"Reports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
