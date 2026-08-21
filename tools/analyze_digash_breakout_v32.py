#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from freqtrade.data.history import get_datahandler, load_pair_history
from freqtrade.enums import CandleType, TradingMode

VOL_FLOOR_24H = 70_000_000.0
BASE_COST_BPS = 8.0
STRESS_COST_BPS = 12.0


def parse_args():
    p = argparse.ArgumentParser(description="Digash breakout V3.2 fidelity / robustness audit")
    p.add_argument("--config", default="/freqtrade/user_data/v7/config-v7-core-backtest.json")
    p.add_argument("--datadir", default="/freqtrade/user_data/data/binance")
    p.add_argument("--events", default="/freqtrade/user_data/digash_replication_v31/events.csv")
    p.add_argument("--outdir", default="/freqtrade/user_data/digash_breakout_v32")
    p.add_argument("--workers", type=int, default=16)
    return p.parse_args()


def as_ns(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=True).astype("datetime64[ns, UTC]")


def _bool_col(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


def _ct_value(v) -> str:
    return str(getattr(v, "value", v)).lower()


def discover_cached_15m_usdt_futures(config: dict, datadir: Path) -> list[str]:
    dh = get_datahandler(datadir, config.get("dataformat_ohlcv"))
    combos = dh.ohlcv_get_available_data(datadir, TradingMode.FUTURES)
    pairs = {
        pair
        for pair, timeframe, candle_type in combos
        if timeframe == "15m"
        and _ct_value(candle_type) == "futures"
        and pair.endswith("/USDT:USDT")
    }
    return sorted(pairs)


def load_pair_activity_snapshot(
    pair: str,
    config: dict,
    datadir: Path,
    needed_ctx: pd.DatetimeIndex,
    min_ctx: pd.Timestamp,
    max_ctx: pd.Timestamp,
) -> tuple[str, pd.DataFrame | None, str | None]:
    try:
        d = load_pair_history(
            pair=pair,
            timeframe="15m",
            datadir=datadir,
            fill_up_missing=False,
            drop_incomplete=False,
            data_format=config.get("dataformat_ohlcv"),
            candle_type=CandleType.FUTURES,
        )
        if d.empty:
            return pair, None, "empty"
        x = d[["date", "open", "high", "low", "close", "volume"]].copy()
        x["date"] = as_ns(x["date"])
        # 24h features need 96 x 15m warmup. Two days keeps the first requested bucket causal.
        x = x[(x.date >= min_ctx - pd.Timedelta(days=2)) & (x.date <= max_ctx)].copy()
        if len(x) < 20:
            return pair, None, "too_short"
        x = x.sort_values("date").drop_duplicates("date").reset_index(drop=True)

        q = x["volume"].to_numpy(float) * x["close"].to_numpy(float)
        close = x["close"].astype(float)
        prev = close.shift()
        tr = pd.concat([
            x["high"].astype(float) - x["low"].astype(float),
            (x["high"].astype(float) - prev).abs(),
            (x["low"].astype(float) - prev).abs(),
        ], axis=1).max(axis=1)

        x["ret_24h"] = close.pct_change(96)
        x["quote_vol_24h"] = pd.Series(q, index=x.index).rolling(96, min_periods=48).sum()
        x["natr_15m"] = tr.rolling(14, min_periods=7).mean() / close * 100.0
        x["ctx_time"] = x["date"] + pd.Timedelta(minutes=15)
        x = x[x["ctx_time"].isin(needed_ctx)][
            ["ctx_time", "ret_24h", "quote_vol_24h", "natr_15m"]
        ].copy()
        if x.empty:
            return pair, None, "no_requested_times"
        x["pair"] = pair
        return pair, x, None
    except Exception as e:
        return pair, None, f"{type(e).__name__}: {e}"


def fmt_seconds(v: float | None) -> str:
    if v is None or not np.isfinite(v) or v < 0:
        return "--:--"
    v = int(v)
    h, r = divmod(v, 3600)
    m, s = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def build_broad_activity(
    events: pd.DataFrame,
    config: dict,
    datadir: Path,
    workers: int,
    outdir: Path,
) -> tuple[pd.DataFrame, dict]:
    needed_ctx = pd.DatetimeIndex(events["ctx_time"].drop_duplicates().sort_values())
    min_ctx, max_ctx = needed_ctx.min(), needed_ctx.max()
    pairs = discover_cached_15m_usdt_futures(config, datadir)
    if not pairs:
        raise RuntimeError("No cached 15m USDT futures pairs discovered")

    print(f"Cached 15m USDT perpetual universe discovered: {len(pairs)} pairs", flush=True)
    print("CACHE ONLY: broad ranking will not download any missing market data.", flush=True)
    frames: list[pd.DataFrame] = []
    failures: list[tuple[str, str]] = []
    workers = max(1, min(int(workers), len(pairs)))
    t0 = time.monotonic()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(load_pair_activity_snapshot, p, config, datadir, needed_ctx, min_ctx, max_ctx): p
            for p in pairs
        }
        done = 0
        for f in as_completed(futures):
            p = futures[f]
            pair, x, err = f.result()
            done += 1
            if x is not None and not x.empty:
                frames.append(x)
            elif err not in ("no_requested_times", "too_short", "empty"):
                failures.append((pair, err or "unknown"))
            elapsed = time.monotonic() - t0
            eta = elapsed * (len(pairs) - done) / done if done else None
            width = 24
            nbar = int(round(width * done / len(pairs)))
            bar = "[" + "█" * nbar + "░" * (width - nbar) + "]"
            sys.stdout.write(
                "\r\033[K"
                + f"{bar} broad-cache {done}/{len(pairs)} | loaded {len(frames)} | "
                + f"elapsed {fmt_seconds(elapsed)} | ETA {fmt_seconds(eta)}"
            )
            sys.stdout.flush()
    sys.stdout.write("\r\033[K")
    sys.stdout.flush()
    print(
        f"Broad cache prepared: {len(frames)}/{len(pairs)} pairs contributed | failures={len(failures)} | "
        f"elapsed={fmt_seconds(time.monotonic()-t0)}",
        flush=True,
    )
    if failures:
        pd.DataFrame(failures, columns=["pair", "error"]).to_csv(outdir / "broad_activity_failures.csv", index=False)
    if not frames:
        raise RuntimeError("No cached pairs contributed broad activity rows")

    panel = pd.concat(frames, ignore_index=True)
    panel["liquid70"] = panel["quote_vol_24h"] >= VOL_FLOOR_24H
    panel["universe_n"] = panel.groupby("ctx_time")["pair"].transform("nunique")
    panel["liquid_n"] = panel.groupby("ctx_time")["liquid70"].transform("sum")

    # Rank *inside* the liquid universe. This mirrors a volume floor followed by the screener lists.
    rank_specs = {
        "growth_rank": ("ret_24h", False),
        "decline_rank": ("ret_24h", True),
        "volatility_rank": ("natr_15m", False),
        "volume_rank": ("quote_vol_24h", False),
    }
    for rank_name, (col, ascending) in rank_specs.items():
        masked = panel[col].where(panel["liquid70"])
        panel[rank_name] = masked.groupby(panel["ctx_time"]).rank(ascending=ascending, method="first")

    for n in (5, 10):
        cols = []
        for stem in ("growth", "decline", "volatility", "volume"):
            c = f"broad_top{n}_{stem}"
            panel[c] = panel["liquid70"] & (panel[f"{stem}_rank"] <= n)
            cols.append(c)
        panel[f"broad_active_top{n}"] = panel[cols].any(axis=1)

    needed_pairs = set(events["pair"].unique())
    keep = [
        "pair", "ctx_time", "universe_n", "liquid_n", "liquid70",
        "growth_rank", "decline_rank", "volatility_rank", "volume_rank",
        "broad_top5_growth", "broad_top5_decline", "broad_top5_volatility", "broad_top5_volume",
        "broad_top10_growth", "broad_top10_decline", "broad_top10_volatility", "broad_top10_volume",
        "broad_active_top5", "broad_active_top10",
    ]
    flags = panel[panel["pair"].isin(needed_pairs)][keep].copy()

    coverage = panel.groupby("ctx_time", as_index=False).agg(
        universe_n=("pair", "nunique"),
        liquid_n=("liquid70", "sum"),
    )
    coverage.to_csv(outdir / "broad_universe_coverage.csv", index=False)
    meta = {
        "discovered_pairs": len(pairs),
        "contributing_pairs": len(frames),
        "failed_pairs": len(failures),
        "median_universe_n": float(coverage["universe_n"].median()),
        "median_liquid_n": float(coverage["liquid_n"].median()),
        "min_universe_n": int(coverage["universe_n"].min()),
        "max_universe_n": int(coverage["universe_n"].max()),
    }
    return flags, meta


def pf(x: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce").dropna()
    pos = x.clip(lower=0).sum()
    neg = -x.clip(upper=0).sum()
    return float(pos / neg) if neg > 0 else math.inf


def stats(z: pd.DataFrame) -> dict:
    if z.empty:
        return {
            "n": 0, "wk": np.nan, "gross": np.nan, "gpf": np.nan, "net": np.nan, "npf": np.nan,
            "stress": np.nan, "spf": np.nan, "risk": np.nan, "cost": np.nan,
            "h1": np.nan, "h2": np.nan, "h3": np.nan, "pos_years": 0, "years": 0,
        }
    span_days = max((z.entry_time.max() - z.entry_time.min()).total_seconds() / 86400.0, 7.0)
    y = z.assign(year=z.entry_time.dt.year).groupby("year")["net_3r"].mean()
    return {
        "n": int(len(z)),
        "wk": float(len(z) / (span_days / 7.0)),
        "gross": float(z.gross_3r.mean()),
        "gpf": pf(z.gross_3r),
        "net": float(z.net_3r.mean()),
        "npf": pf(z.net_3r),
        "stress": float(z.stress_net_3r.mean()),
        "spf": pf(z.stress_net_3r),
        "risk": float(z.risk_pct.median()),
        "cost": float(z.cost_r.median()),
        "h1": float(z.hit_1r.mean() * 100),
        "h2": float(z.hit_2r.mean() * 100),
        "h3": float(z.hit_3r.mean() * 100),
        "pos_years": int((y > 0).sum()),
        "years": int(len(y)),
    }


def fmt_stat(label: str, s: dict) -> str:
    if s["n"] == 0:
        return f"{label:24s} N=0"
    return (
        f"{label:24s} N={s['n']:5d} wk={s['wk']:4.1f} "
        f"gross={s['gross']:+.3f}R gPF={s['gpf']:.2f} "
        f"net={s['net']:+.3f}R PF={s['npf']:.2f} stress={s['stress']:+.3f}R/{s['spf']:.2f} "
        f"risk={s['risk']:.3f}% cost={s['cost']:.2f}R years={s['pos_years']}/{s['years']}"
    )


def variant_masks(s: pd.DataFrame) -> dict[str, pd.Series]:
    fact = s.fact_proxy
    return {
        "FACT_ALL": fact,
        "LOCAL_ACTIVE": fact & s.local_active_any,
        "BROAD_TOP10": fact & s.broad_active_top10,
        "BROAD_TOP5": fact & s.broad_active_top5,
    }


def print_tf_period_table(df: pd.DataFrame, outdir: Path) -> None:
    rows = []
    for (tf, period), s in df.groupby(["tf", "period"]):
        for name, mask in variant_masks(s).items():
            z = s.loc[mask]
            rows.append({"tf": tf, "period": int(period), "variant": name, **stats(z)})
    q = pd.DataFrame(rows)
    q.to_csv(outdir / "tf_period_variants.csv", index=False)
    print("\n=== BREAKOUT FIDELITY BY TF / PERIOD ===", flush=True)
    for tf in ("1h", "4h"):
        for period in (20, 30):
            z = q[(q.tf == tf) & (q.period == period)]
            for r in z.itertuples(index=False):
                print(
                    fmt_stat(f"{tf} p{period} {r.variant}", {
                        "n": r.n, "wk": r.wk, "gross": r.gross, "gpf": r.gpf,
                        "net": r.net, "npf": r.npf, "stress": r.stress, "spf": r.spf,
                        "risk": r.risk, "cost": r.cost, "pos_years": r.pos_years, "years": r.years,
                    }), flush=True,
                )


def split_report(z: pd.DataFrame, label: str, outdir: Path) -> None:
    print(f"\n=== {label} TIME ROBUSTNESS ===", flush=True)
    rows = []
    splits = {
        "2022-2024": z[(z.entry_time >= "2022-01-01") & (z.entry_time < "2025-01-01")],
        "2025": z[(z.entry_time >= "2025-01-01") & (z.entry_time < "2026-01-01")],
        "2026": z[z.entry_time >= "2026-01-01"],
    }
    for name, g in splits.items():
        s = stats(g)
        rows.append({"split": name, **s})
        print(fmt_stat(name, s), flush=True)
    for year, g in z.groupby(z.entry_time.dt.year):
        s = stats(g)
        rows.append({"split": str(int(year)), **s})
        print(fmt_stat(str(int(year)), s), flush=True)
    pd.DataFrame(rows).to_csv(outdir / "primary_time_splits.csv", index=False)

    if not z.empty:
        month_key = z.entry_time.dt.tz_localize(None).dt.to_period("M").astype(str)
        m = z.assign(month=month_key).groupby("month").agg(
            n=("net_3r", "size"), net=("net_3r", "mean"), gross=("gross_3r", "mean")
        ).reset_index()
        m["positive"] = m["net"] > 0
        m.to_csv(outdir / "primary_monthly.csv", index=False)
        print(
            f"positive months={int(m.positive.sum())}/{len(m)} ({(m.positive.mean()*100 if len(m) else 0):.1f}%)",
            flush=True,
        )


def pair_robustness(z: pd.DataFrame, outdir: Path) -> None:
    print("\n=== PRIMARY PAIR / CONCENTRATION ROBUSTNESS ===", flush=True)
    if z.empty:
        print("No primary sample.", flush=True)
        return
    rows = []
    for pair, g in z.groupby("pair"):
        s = stats(g)
        rows.append({"pair": pair, "sum_net_r": float(g.net_3r.sum()), **s})
    p = pd.DataFrame(rows).sort_values("sum_net_r", ascending=False)
    p.to_csv(outdir / "primary_by_pair.csv", index=False)
    print(f"pairs with trades={len(p)}", flush=True)
    for r in p.head(8).itertuples(index=False):
        print(f"{r.pair:20s} N={r.n:4d} net={r.net:+.3f}R PF={r.npf:.2f} sum={r.sum_net_r:+.2f}R", flush=True)

    loo = []
    for pair in p.pair:
        g = z[z.pair != pair]
        s = stats(g)
        loo.append({"excluded": pair, **s})
    l = pd.DataFrame(loo)
    l.to_csv(outdir / "primary_leave_one_pair_out.csv", index=False)
    if not l.empty:
        worst_mean = l.loc[l.net.idxmin()]
        best_mean = l.loc[l.net.idxmax()]
        print(
            f"LOO net range {l.net.min():+.3f}R .. {l.net.max():+.3f}R; "
            f"worst exclusion={worst_mean.excluded}, best exclusion={best_mean.excluded}",
            flush=True,
        )
        print(f"LOO PF range {l.npf.min():.2f} .. {l.npf.max():.2f}", flush=True)


def diagnostic_partitions(z: pd.DataFrame, outdir: Path) -> None:
    print("\n=== PRIMARY BREAKOUT DIAGNOSTIC PARTITIONS (NOT NEW RULES) ===", flush=True)
    if z.empty:
        print("No primary sample.", flush=True)
        return
    x = z.copy()
    x["approach_bucket"] = np.where(x.approach_no <= 1, "1", np.where(x.approach_no == 2, "2", "3+"))
    bd = pd.to_numeric(x.break_distance_atr, errors="coerce")
    x["break_mode"] = np.select(
        [x.impulse_proxy & (bd >= 0.10), x.impulse_proxy & (bd < 0.10), (~x.impulse_proxy) & (bd >= 0.10)],
        ["IMPULSE+>=0.10ATR", "IMPULSE_ONLY", "CLOSE_ONLY"],
        default="OTHER",
    )
    age = pd.to_numeric(x.level_age_h, errors="coerce")
    x["age_bucket"] = pd.cut(
        age, bins=[-1e-9, 24, 72, 168, np.inf], labels=["<1d", "1-3d", "3-7d", "7d+"], include_lowest=True
    ).astype(str)

    rows = []
    for field in ("approach_bucket", "break_mode", "stop_source", "age_bucket"):
        print(f"-- {field} --", flush=True)
        for value, g in x.groupby(field, dropna=False):
            s = stats(g)
            rows.append({"field": field, "value": str(value), **s})
            print(fmt_stat(str(value), s), flush=True)
    pd.DataFrame(rows).to_csv(outdir / "primary_diagnostic_partitions.csv", index=False)


def main() -> int:
    a = parse_args()
    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    config = json.loads(Path(a.config).read_text())
    events_path = Path(a.events)
    if not events_path.exists():
        raise RuntimeError(f"V3.1 events not found: {events_path}. Run V3.1 first.")

    df = pd.read_csv(events_path)
    df = df[df.setup == "H_BREAK"].copy()
    if df.empty:
        raise RuntimeError("V3.1 events contain no H_BREAK rows")
    df["signal_time"] = pd.to_datetime(df["signal_time"], utc=True)
    df["entry_time"] = pd.to_datetime(df["entry_time"], utc=True)
    for c in ["fact_proxy", "active_any", "impulse_proxy", "protor_proxy", "hit_1r", "hit_2r", "hit_3r"]:
        if c in df:
            df[c] = _bool_col(df[c])
    df["local_active_any"] = df["active_any"].astype(bool)
    df["ctx_time"] = df["signal_time"].dt.floor("15min")

    print("=== DIGASH BREAKOUT V3.2 ===", flush=True)
    print(f"Input V3.1 breakout events: {len(df):,}", flush=True)
    print("This is a follow-up fidelity/robustness audit, NOT independent OOS evidence.", flush=True)
    print("Public material confirms multiple breakout types, but their exact private machine rules are not public enough to invent.", flush=True)

    broad, meta = build_broad_activity(df, config, Path(a.datadir), a.workers, outdir)
    merged = df.merge(broad, on=["pair", "ctx_time"], how="left", validate="many_to_one")
    bool_cols = [c for c in broad.columns if c.startswith("broad_top") or c.startswith("broad_active") or c == "liquid70"]
    for c in bool_cols:
        merged[c] = merged[c].fillna(False).astype(bool)
    coverage_pct = float(merged["universe_n"].notna().mean() * 100)
    merged.to_csv(outdir / "breakout_events_v32.csv", index=False)

    meta["breakout_rows"] = int(len(merged))
    meta["event_broad_rank_coverage_pct"] = coverage_pct
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))

    print("\n=== BROAD CACHE UNIVERSE ===", flush=True)
    print(
        f"discovered={meta['discovered_pairs']} contributing={meta['contributing_pairs']} "
        f"median universe={meta['median_universe_n']:.0f} median liquid70={meta['median_liquid_n']:.0f} "
        f"event coverage={coverage_pct:.1f}%",
        flush=True,
    )
    if meta["median_universe_n"] <= 25:
        print("WARNING: cache universe is still close to the old 20-pair whitelist; broad-ranking fidelity is not materially improved.", flush=True)

    print_tf_period_table(merged, outdir)

    # V3.1 made 4h/p30 the follow-up candidate. This is explicitly post-selection and must not be called fresh OOS.
    base = merged[(merged.tf == "4h") & (merged.period == 30) & merged.fact_proxy].copy()
    primary10 = base[base.broad_active_top10].copy()
    primary5 = base[base.broad_active_top5].copy()
    local = base[base.local_active_any].copy()

    print("\n=== 4H P30 FOLLOW-UP CANDIDATE (POST-SELECTION, NOT OOS) ===", flush=True)
    print(fmt_stat("FACT_ALL", stats(base)), flush=True)
    print(fmt_stat("LOCAL_ACTIVE", stats(local)), flush=True)
    print(fmt_stat("BROAD_TOP10", stats(primary10)), flush=True)
    print(fmt_stat("BROAD_TOP5", stats(primary5)), flush=True)

    split_report(primary10, "4H P30 BROAD_TOP10", outdir)
    pair_robustness(primary10, outdir)
    diagnostic_partitions(primary10, outdir)

    # Save a compact primary top5/top10 comparison by split for later no-retuning review.
    comp = []
    for name, z in (("BROAD_TOP10", primary10), ("BROAD_TOP5", primary5)):
        for split, g in {
            "ALL": z,
            "2022-2024": z[(z.entry_time >= "2022-01-01") & (z.entry_time < "2025-01-01")],
            "2025": z[(z.entry_time >= "2025-01-01") & (z.entry_time < "2026-01-01")],
            "2026": z[z.entry_time >= "2026-01-01"],
        }.items():
            comp.append({"variant": name, "split": split, **stats(g)})
    pd.DataFrame(comp).to_csv(outdir / "primary_top5_top10_comparison.csv", index=False)

    print(f"\nReports: {outdir}", flush=True)
    print("=== DONE ===", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
