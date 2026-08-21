#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html as html_lib
import json
import math
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

import digash_v3_common as dc

CHANNEL = "Digash_Formations"
BASE_URL = f"https://t.me/s/{CHANNEL}"
MATCH_BPS = (10.0, 25.0, 50.0, 100.0)
TF_MIN = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240}


def parse_args():
    p = argparse.ArgumentParser(description="Digash Fidelity V4 public-formation level audit")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_fidelity_v4")
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--max-pages", type=int, default=140)
    p.add_argument("--max-posts", type=int, default=3000)
    p.add_argument("--warmup-days", type=int, default=120)
    p.add_argument("--refresh-source", action="store_true")
    return p.parse_args()


def strip_tags(s: str) -> str:
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    s = html_lib.unescape(s)
    s = s.replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", x).strip() for x in s.splitlines()]
    return "\n".join(x for x in lines if x)


def fetch_url(url: str, attempts: int = 4) -> str:
    last = None
    for n in range(attempts):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/131 Safari/537.36",
                    "Accept-Language": "ru,en;q=0.8",
                },
            )
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.read().decode("utf-8", errors="replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
            time.sleep(1.0 + n * 1.5)
    raise RuntimeError(f"failed to fetch {url}: {last}")


def extract_page_posts(page: str) -> list[dict]:
    hits = list(re.finditer(r'data-post="' + re.escape(CHANNEL) + r'/(\d+)"', page, flags=re.I))
    out = []
    seen = set()
    for i, m in enumerate(hits):
        post_id = int(m.group(1))
        if post_id in seen:
            continue
        seen.add(post_id)
        end = hits[i + 1].start() if i + 1 < len(hits) else min(len(page), m.start() + 50000)
        block = page[m.start():end]
        tm = re.search(r'<time[^>]+datetime="([^"]+)"', block, flags=re.I | re.S)
        tx = re.search(r'<div class="[^"]*tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>', block, flags=re.I | re.S)
        text = strip_tags(tx.group(1)) if tx else ""
        out.append({
            "post_id": post_id,
            "post_time": tm.group(1) if tm else "",
            "text": text,
            "url": f"https://t.me/{CHANNEL}/{post_id}",
        })
    return out


def collect_public_posts(max_pages: int, max_posts: int) -> pd.DataFrame:
    rows: dict[int, dict] = {}
    before = None
    previous_min = None
    for page_no in range(1, max_pages + 1):
        url = BASE_URL if before is None else f"{BASE_URL}?before={before}"
        body = fetch_url(url)
        page_rows = extract_page_posts(body)
        if not page_rows:
            print(f"SOURCE page={page_no}: no messages; pagination stopped", flush=True)
            break
        for r in page_rows:
            rows[r["post_id"]] = r
        cur_min = min(r["post_id"] for r in page_rows)
        cur_max = max(r["post_id"] for r in page_rows)
        print(
            f"SOURCE page={page_no:3d} ids={cur_min}..{cur_max} page_posts={len(page_rows):2d} unique={len(rows):4d}",
            flush=True,
        )
        if len(rows) >= max_posts or cur_min <= 1 or cur_min == previous_min:
            break
        previous_min = cur_min
        before = cur_min
        time.sleep(0.20)
    df = pd.DataFrame(sorted(rows.values(), key=lambda x: x["post_id"]))
    if not df.empty:
        df["post_time"] = pd.to_datetime(df["post_time"], utc=True, errors="coerce")
    return df


def parse_breakout_posts(posts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    pair_re = re.compile(r"Бинанс\s+фьючерс\w*\s*[-–—:]\s*([A-Z0-9]+USDT)\b", re.I)
    tf_re = re.compile(r"таймфрейм\w*\s*(1m|5m|15m|1h|4h)\b", re.I)
    lvl_re = re.compile(r"Пробой\s+уров(?:ня|ней)\s+(.+?)\s+на\s+таймфрейм", re.I | re.S)
    num_re = re.compile(r"\d+(?:\.\d+)?")

    for r in posts.itertuples(index=False):
        text = str(r.text or "")
        if "пробой" not in text.lower() or "бинанс" not in text.lower():
            continue
        pm = pair_re.search(text)
        tm = tf_re.search(text)
        lm = lvl_re.search(text)
        if not (pm and tm and lm) or pd.isna(r.post_time):
            continue
        symbol = pm.group(1).upper()
        tf = tm.group(1).lower()
        nums = []
        for x in num_re.findall(lm.group(1)):
            try:
                v = float(x)
            except ValueError:
                continue
            if np.isfinite(v) and v > 0:
                nums.append(v)
        if not nums:
            continue
        base = symbol[:-4]
        pair = f"{base}/USDT:USDT"
        for j, level in enumerate(nums, start=1):
            rows.append({
                "post_id": int(r.post_id),
                "post_time": pd.Timestamp(r.post_time),
                "symbol": symbol,
                "pair": pair,
                "tf": tf,
                "published_level_no": j,
                "published_level": float(level),
                "url": r.url,
                "text": text,
            })
    return pd.DataFrame(rows)


def _resample(raw: pd.DataFrame, rule: str, minutes: int) -> pd.DataFrame:
    if raw.empty:
        return raw
    x = raw[["date", "open", "high", "low", "close", "volume"]].copy()
    x["date"] = pd.to_datetime(x["date"], utc=True)
    y = (
        x.sort_values("date").set_index("date")
        .resample(rule, label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"),
             close=("close", "last"), volume=("volume", "sum"))
        .dropna().reset_index()
    )
    return dc.prep_ohlcv(y, minutes)


def load_published_tf(config: dict, datadir: Path, pair: str, tf: str) -> tuple[pd.DataFrame, str]:
    direct = dc.load_tf(config, datadir, pair, tf)
    if not direct.empty:
        return dc.prep_ohlcv(direct, TF_MIN[tf]), tf

    if tf == "1m":
        return pd.DataFrame(), "none"

    if tf == "5m":
        d5, src = dc.load_5m(config, datadir, pair)
        return (dc.prep_ohlcv(d5, 5), src) if not d5.empty else (pd.DataFrame(), "none")

    if tf == "15m":
        d5, src = dc.load_5m(config, datadir, pair)
        if not d5.empty:
            return _resample(d5, "15min", 15), f"{src}->15m"
        return pd.DataFrame(), "none"

    # 1h / 4h: use cached 15m when available, otherwise causal 5m-derived 15m.
    d15 = dc.load_tf(config, datadir, pair, "15m")
    src = "15m"
    if d15.empty:
        d5, src5 = dc.load_5m(config, datadir, pair)
        if d5.empty:
            return pd.DataFrame(), "none"
        d15 = _resample(d5, "15min", 15)
        src = f"{src5}->15m"
    else:
        d15 = dc.prep_ohlcv(d15, 15)
    rule = "1h" if tf == "1h" else "4h"
    return dc.resample_from_15(d15, rule, TF_MIN[tf]), f"{src}->{tf}"


def nearest_level(levels: list[dc.Level], t: pd.Timestamp, p: float) -> tuple[float, float, str, str, int]:
    best = None
    for lv in levels:
        if pd.Timestamp(lv.formed_time) > t:
            continue
        err_bps = abs(float(lv.price) / p - 1.0) * 10000.0
        if best is None or err_bps < best[0]:
            best = (err_bps, float(lv.price), lv.kind, pd.Timestamp(lv.formed_time).isoformat(), int(lv.counted_touches))
    if best is None:
        return np.nan, np.nan, "", "", 0
    return best


def audit_group(records: list[dict], config_path: str, datadir_s: str, warmup_days: int) -> tuple[list[dict], dict]:
    t0 = time.monotonic()
    config = json.loads(Path(config_path).read_text())
    datadir = Path(datadir_s)
    pair = records[0]["pair"]
    tf = records[0]["tf"]
    dc.TF_MINUTES.setdefault("1m", 1)
    try:
        x, source = load_published_tf(config, datadir, pair, tf)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_DATA", "source": source, "bars": 0, "elapsed_s": time.monotonic()-t0}

        times = pd.to_datetime([r["post_time"] for r in records], utc=True)
        lo = times.min() - pd.Timedelta(days=warmup_days)
        hi = times.max() + pd.Timedelta(days=1)
        x = x[(x.date >= lo) & (x.date < hi)].reset_index(drop=True)
        if x.empty:
            return [], {"pair": pair, "tf": tf, "status": "NO_RANGE", "source": source, "bars": 0, "elapsed_s": time.monotonic()-t0}

        levels20 = dc.build_levels(x, tf, 20, 0)
        levels30 = dc.build_levels(x, tf, 30, len(levels20))
        out = []
        for r in records:
            z = dict(r)
            t = pd.Timestamp(r["post_time"])
            p = float(r["published_level"])
            e20, p20, k20, f20, tc20 = nearest_level(levels20, t, p)
            e30, p30, k30, f30, tc30 = nearest_level(levels30, t, p)
            choices = [(e20, 20, p20, k20, f20, tc20), (e30, 30, p30, k30, f30, tc30)]
            choices = [q for q in choices if np.isfinite(q[0])]
            if choices:
                e, period, nearp, kind, formed, touches = min(choices, key=lambda q: q[0])
            else:
                e, period, nearp, kind, formed, touches = np.nan, 0, np.nan, "", "", 0
            z.update({
                "data_source": source,
                "bars_used": len(x),
                "levels_p20": len(levels20),
                "levels_p30": len(levels30),
                "nearest_p20_bps": e20,
                "nearest_p20_price": p20,
                "nearest_p30_bps": e30,
                "nearest_p30_price": p30,
                "nearest_bps": e,
                "nearest_period": period,
                "nearest_price": nearp,
                "nearest_kind": kind,
                "nearest_formed_time": formed,
                "nearest_counted_touches": touches,
            })
            for b in MATCH_BPS:
                z[f"match_{int(b)}bps"] = bool(np.isfinite(e) and e <= b)
            out.append(z)
        return out, {
            "pair": pair, "tf": tf, "status": "OK", "source": source, "bars": len(x),
            "levels_p20": len(levels20), "levels_p30": len(levels30), "alerts": len(records),
            "elapsed_s": time.monotonic()-t0,
        }
    except Exception as e:
        return [], {"pair": pair, "tf": tf, "status": "ERROR", "error": f"{type(e).__name__}: {e}", "elapsed_s": time.monotonic()-t0}


def pct(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").mean() * 100.0) if len(x) else np.nan


def print_summary(parsed: pd.DataFrame, matched: pd.DataFrame, coverage: pd.DataFrame, outdir: Path) -> None:
    unique_alerts = parsed[["post_id", "pair", "tf"]].drop_duplicates()
    covered_levels = len(matched)
    covered_posts = matched[["post_id", "pair", "tf"]].drop_duplicates().shape[0] if not matched.empty else 0
    print("\n=== SOURCE / CACHE COVERAGE ===", flush=True)
    print(f"parsed public breakout posts={unique_alerts.shape[0]:,} | published levels={len(parsed):,}", flush=True)
    print(f"covered posts={covered_posts:,} | covered published levels={covered_levels:,}", flush=True)
    if not coverage.empty:
        print(f"pair×TF groups OK={(coverage.status == 'OK').sum()}/{len(coverage)} | NO_DATA={(coverage.status == 'NO_DATA').sum()} | ERROR={(coverage.status == 'ERROR').sum()}", flush=True)

    if matched.empty:
        print("No source alerts overlap the local OHLCV cache; fidelity cannot be measured.", flush=True)
        return

    print("\n=== CAUSAL LEVEL MATCH CURVE ===", flush=True)
    for b in MATCH_BPS:
        c = f"match_{int(b)}bps"
        print(f"nearest reconstructed level <= {int(b):3d} bps: {pct(matched[c]):5.1f}% ({int(matched[c].sum())}/{len(matched)})", flush=True)
    finite = pd.to_numeric(matched.nearest_bps, errors="coerce").dropna()
    print(f"nearest error median={finite.median():.1f} bps | p75={finite.quantile(.75):.1f} | p90={finite.quantile(.90):.1f}", flush=True)

    print("\n=== MATCH BY PUBLISHED TIMEFRAME ===", flush=True)
    tf_rows = []
    for tf, g in matched.groupby("tf"):
        row = {"tf": tf, "levels": len(g), "median_bps": pd.to_numeric(g.nearest_bps, errors="coerce").median()}
        for b in MATCH_BPS:
            row[f"match_{int(b)}bps_pct"] = pct(g[f"match_{int(b)}bps"])
        tf_rows.append(row)
        print(
            f"{tf:4s} levels={len(g):4d} median={row['median_bps']:7.1f}bps | "
            + " | ".join(f"<={int(b)}:{row[f'match_{int(b)}bps_pct']:5.1f}%" for b in MATCH_BPS),
            flush=True,
        )
    pd.DataFrame(tf_rows).to_csv(outdir / "match_by_timeframe.csv", index=False)

    print("\n=== PERIOD DIAGNOSTIC (NOT PNL TUNING) ===", flush=True)
    for period in (20, 30):
        s = pd.to_numeric(matched[f"nearest_p{period}_bps"], errors="coerce")
        print(
            f"p{period}: finite={s.notna().sum():4d} median={s.median():7.1f}bps "
            f"<=25bps={(s <= 25).mean()*100:5.1f}% <=100bps={(s <= 100).mean()*100:5.1f}%",
            flush=True,
        )

    per_post = matched.groupby(["post_id", "pair", "tf"], as_index=False).agg(
        published_levels=("published_level", "size"),
        matched_levels_25=("match_25bps", "sum"),
        matched_levels_100=("match_100bps", "sum"),
        max_nearest_bps=("nearest_bps", "max"),
    )
    per_post["all_25"] = per_post.matched_levels_25 == per_post.published_levels
    per_post["any_25"] = per_post.matched_levels_25 > 0
    per_post["all_100"] = per_post.matched_levels_100 == per_post.published_levels
    per_post["any_100"] = per_post.matched_levels_100 > 0
    per_post.to_csv(outdir / "match_by_post.csv", index=False)
    print("\n=== POST-LEVEL REPRODUCTION ===", flush=True)
    print(f"25bps: any published level={per_post.any_25.mean()*100:.1f}% | all published levels={per_post.all_25.mean()*100:.1f}%", flush=True)
    print(f"100bps: any published level={per_post.any_100.mean()*100:.1f}% | all published levels={per_post.all_100.mean()*100:.1f}%", flush=True)

    worst = matched.sort_values("nearest_bps", ascending=False).head(12)
    print("\n=== WORST COVERED MISMATCHES ===", flush=True)
    for r in worst.itertuples(index=False):
        print(
            f"post={r.post_id} {r.symbol:16s} tf={r.tf:3s} published={r.published_level:g} "
            f"nearest={r.nearest_price:g} err={r.nearest_bps:.1f}bps p{r.nearest_period}",
            flush=True,
        )


def main() -> int:
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_path = outdir / "public_posts_raw.csv"
    parsed_path = outdir / "digash_public_breakouts.csv"

    print("=== DIGASH FIDELITY V4 — PUBLIC BREAKOUT LEVEL AUDIT ===", flush=True)
    print("Goal: reproduce Digash public breakout levels first; NO PnL optimization and NO market-data downloads.", flush=True)
    print(f"source={BASE_URL} | max_pages={a.max_pages} | warmup={a.warmup_days}d | workers={a.workers}", flush=True)

    if raw_path.exists() and not a.refresh_source:
        posts = pd.read_csv(raw_path)
        posts["post_time"] = pd.to_datetime(posts.post_time, utc=True, errors="coerce")
        print(f"Using cached public-source snapshot: {raw_path} ({len(posts):,} posts)", flush=True)
    else:
        posts = collect_public_posts(a.max_pages, a.max_posts)
        posts.to_csv(raw_path, index=False)
        print(f"Public-source snapshot saved: {raw_path} ({len(posts):,} posts)", flush=True)

    parsed = parse_breakout_posts(posts)
    parsed.to_csv(parsed_path, index=False)
    if parsed.empty:
        print("No Binance breakout alerts parsed. Inspect public_posts_raw.csv before changing any detector rule.", flush=True)
        return 2

    up = parsed[["post_id", "pair", "tf"]].drop_duplicates()
    print(
        f"Parsed Binance breakout alerts: posts={len(up):,} levels={len(parsed):,} "
        f"pairs={parsed.pair.nunique()} TFs={','.join(sorted(parsed.tf.unique()))}",
        flush=True,
    )

    groups = []
    for (pair, tf), g in parsed.groupby(["pair", "tf"], sort=True):
        groups.append(g.to_dict("records"))

    results = []
    coverage = []
    workers = max(1, min(a.workers, len(groups)))
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(audit_group, recs, a.config, a.datadir, a.warmup_days): (recs[0]["pair"], recs[0]["tf"])
            for recs in groups
        }
        done = 0
        for f in as_completed(futs):
            rows, meta = f.result()
            done += 1
            coverage.append(meta)
            results.extend(rows)
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(groups) - done) / done if done else np.nan
            print(
                f"FIDELITY {done:3d}/{len(groups)} | {meta.get('pair')} {meta.get('tf')} "
                f"status={meta.get('status')} | matched_levels={len(rows)} | elapsed={elapsed:.1f}s | ETA={eta:.1f}s",
                flush=True,
            )

    cov = pd.DataFrame(coverage)
    cov.to_csv(outdir / "cache_coverage.csv", index=False)
    matched = pd.DataFrame(results)
    matched.to_csv(outdir / "level_matches.csv", index=False)
    print_summary(parsed, matched, cov, outdir)

    print("\n=== INTERPRETATION RULE ===", flush=True)
    print("Do not optimize PF from this run. First decide whether the detector reproduces Digash's published levels.", flush=True)
    print("If fidelity is weak, fix source fidelity and validate changes on held-out public alerts before returning to PnL.", flush=True)
    print(f"Reports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
