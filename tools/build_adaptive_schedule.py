from __future__ import annotations

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd


LOOKBACKS = [4, 6, 8, 10, 12, 16]
THRESHOLDS = [0.015, 0.025, 0.035, 0.05, 0.075]
ATR_MULTS = [2.0, 2.5, 3.0, 3.5]
ATR_PERIOD = 14
ANNUAL_PERIODS = 365 * 4  # 6h bars
RISK_FREE_ANNUAL = 0.045
DEFAULT_FEE = 0.0004

KNOWN_CG = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "BNB": "binancecoin",
    "XRP": "ripple",
    "SOL": "solana",
    "DOGE": "dogecoin",
    "ADA": "cardano",
    "LINK": "chainlink",
    "LTC": "litecoin",
    "BCH": "bitcoin-cash",
    "ETC": "ethereum-classic",
    "TRX": "tron",
    "DOT": "polkadot",
    "AVAX": "avalanche-2",
    "ATOM": "cosmos",
    "NEAR": "near",
    "UNI": "uniswap",
    "AAVE": "aave",
    "XLM": "stellar",
    "FIL": "filecoin",
    "ZEC": "zcash",
    "SUI": "sui",
    "WLD": "worldcoin-wld",
    "HYPE": "hyperliquid",
    "PEPE": "pepe",
}


def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def normalize_date_col(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        med = pd.to_numeric(s, errors="coerce").dropna().abs().median()
        unit = "ms" if med > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def pair_from_filename(path: Path) -> str | None:
    name = path.name
    marker = "-6h-futures.feather"
    if not name.endswith(marker):
        return None
    raw = name[: -len(marker)]
    if raw.endswith("_USDT_USDT"):
        base = raw[: -len("_USDT_USDT")]
        return f"{base}/USDT:USDT"
    return None


def load_6h_frames(data_root: Path, whitelist: list[str]) -> dict[str, pd.DataFrame]:
    wanted = set(whitelist)
    index: dict[str, Path] = {}
    for p in data_root.rglob("*-6h-futures.feather"):
        pair = pair_from_filename(p)
        if pair and pair in wanted:
            index[pair] = p

    out: dict[str, pd.DataFrame] = {}
    for pair in whitelist:
        p = index.get(pair)
        if not p:
            print(f"WARN no 6h feather for {pair}")
            continue
        df = pd.read_feather(p)
        required = {"date", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            print(f"WARN bad columns for {pair}: {p}")
            continue
        df = df[["date", "open", "high", "low", "close", "volume"]].copy()
        df["date"] = normalize_date_col(df["date"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna().sort_values("date").drop_duplicates("date").reset_index(drop=True)
        if df.empty:
            continue

        prev_close = df["close"].shift(1)
        tr = pd.concat(
            [
                df["high"] - df["low"],
                (df["high"] - prev_close).abs(),
                (df["low"] - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        # Wilder-style ATR.
        df["atr"] = tr.ewm(alpha=1 / ATR_PERIOD, adjust=False, min_periods=ATR_PERIOD).mean()
        df["quote_volume"] = df["close"] * df["volume"]
        for lb in LOOKBACKS:
            df[f"mom_{lb}"] = df["close"].pct_change(lb)
        out[pair] = df
        print(f"Loaded {pair}: {df['date'].min()} -> {df['date'].max()} ({len(df)} bars)")
    return out


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) < 20:
        return float("-inf")
    rf_step = (1.0 + RISK_FREE_ANNUAL) ** (1.0 / ANNUAL_PERIODS) - 1.0
    excess = returns - rf_step
    sd = float(np.std(excess, ddof=1))
    if not np.isfinite(sd) or sd <= 1e-12:
        return float("-inf")
    return float(np.mean(excess) / sd * math.sqrt(ANNUAL_PERIODS))


@dataclass
class OptResult:
    sharpe: float
    lookback: int
    theta: float
    alpha: float
    trades: int


def simulate_side(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    side: str,
    lookback: int,
    theta: float,
    alpha: float,
    fee: float,
) -> tuple[float, int]:
    # Include warm-up before train_start, but score only the strict preceding-month window.
    warm_start = train_start - pd.Timedelta(days=10)
    sub = df[(df["date"] >= warm_start) & (df["date"] < train_end)].copy()
    if len(sub) < 40:
        return float("-inf"), 0

    dates = sub["date"].to_numpy()
    close = sub["close"].to_numpy(dtype=float)
    atr = sub["atr"].to_numpy(dtype=float)
    mom = sub[f"mom_{lookback}"].to_numpy(dtype=float)
    scoring = (sub["date"] >= train_start).to_numpy()

    rets = np.zeros(len(sub), dtype=float)
    pos = False
    stop = np.nan
    trades = 0
    sign = 1.0 if side == "long" else -1.0

    for i in range(1, len(sub)):
        if not scoring[i]:
            continue

        if pos:
            rets[i] += sign * (close[i] / close[i - 1] - 1.0)
            if np.isfinite(atr[i]):
                if side == "long":
                    candidate = close[i] - alpha * atr[i]
                    stop = max(stop, candidate)
                    crossed = close[i] < stop
                else:
                    candidate = close[i] + alpha * atr[i]
                    stop = min(stop, candidate)
                    crossed = close[i] > stop
                if crossed:
                    rets[i] -= fee
                    pos = False
                    stop = np.nan
                    continue

        if not pos and np.isfinite(mom[i]) and np.isfinite(atr[i]) and atr[i] > 0:
            fire = mom[i] > theta if side == "long" else mom[i] < -theta
            if fire:
                pos = True
                trades += 1
                rets[i] -= fee
                stop = close[i] - alpha * atr[i] if side == "long" else close[i] + alpha * atr[i]

    scored = rets[scoring]
    return annualized_sharpe(scored), trades


def optimize_side(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    side: str,
    fee: float,
) -> OptResult | None:
    best: OptResult | None = None
    for lb in LOOKBACKS:
        for theta in THRESHOLDS:
            for alpha in ATR_MULTS:
                sr, trades = simulate_side(df, train_start, train_end, side, lb, theta, alpha, fee)
                if not np.isfinite(sr):
                    continue
                candidate = OptResult(sr, lb, theta, alpha, trades)
                if best is None or candidate.sharpe > best.sharpe:
                    best = candidate
    return best


def http_json(url: str, retries: int = 4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RMV5-AdaptiveTrend/1.0"})
            with urllib.request.urlopen(req, timeout=45) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            if attempt + 1 >= retries:
                raise
            wait = 6 * (attempt + 1)
            print(f"CoinGecko retry in {wait}s: {type(e).__name__}")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def cg_symbol(base: str) -> str:
    base = base.upper()
    if base.startswith("1000"):
        base = base[4:]
    return base


def resolve_coingecko_ids(bases: list[str], cache_dir: Path) -> dict[str, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    list_cache = cache_dir / "coins-list.json"
    try:
        if list_cache.exists() and time.time() - list_cache.stat().st_mtime < 7 * 86400:
            coins = json.loads(list_cache.read_text())
        else:
            coins = http_json("https://api.coingecko.com/api/v3/coins/list")
            list_cache.write_text(json.dumps(coins))
    except Exception as e:
        print(f"WARN CoinGecko coins/list unavailable: {e}")
        coins = []

    by_symbol: dict[str, list[str]] = {}
    for c in coins:
        sym = str(c.get("symbol", "")).upper()
        cid = str(c.get("id", ""))
        if sym and cid:
            by_symbol.setdefault(sym, []).append(cid)

    out: dict[str, str] = {}
    for base in bases:
        norm = cg_symbol(base)
        if norm in KNOWN_CG:
            out[base] = KNOWN_CG[norm]
            continue
        candidates = by_symbol.get(norm, [])
        if len(candidates) == 1:
            out[base] = candidates[0]
            continue
        if not candidates:
            continue

        # Ambiguous symbols: choose the currently highest-market-cap exact-symbol coin.
        try:
            query = urllib.parse.urlencode({"vs_currency": "usd", "ids": ",".join(candidates[:50])})
            rows = http_json(f"https://api.coingecko.com/api/v3/coins/markets?{query}")
            rows = sorted(rows, key=lambda x: float(x.get("market_cap") or 0), reverse=True)
            if rows:
                out[base] = str(rows[0]["id"])
            time.sleep(1.5)
        except Exception:
            out[base] = candidates[0]
    return out


def load_market_caps(
    pairs: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> dict[str, pd.DataFrame]:
    bases = [p.split("/")[0] for p in pairs]
    ids = resolve_coingecko_ids(bases, cache_dir)
    out: dict[str, pd.DataFrame] = {}
    from_ts = int((start - pd.Timedelta(days=40)).timestamp())
    to_ts = int((end + pd.Timedelta(days=2)).timestamp())

    for pair in pairs:
        base = pair.split("/")[0]
        cid = ids.get(base)
        if not cid:
            print(f"WARN no CoinGecko id for {pair}")
            continue
        cache = cache_dir / f"marketcap-{cid}-{from_ts}-{to_ts}.json"
        try:
            if cache.exists():
                obj = json.loads(cache.read_text())
            else:
                q = urllib.parse.urlencode({"vs_currency": "usd", "from": from_ts, "to": to_ts})
                obj = http_json(f"https://api.coingecko.com/api/v3/coins/{urllib.parse.quote(cid)}/market_chart/range?{q}")
                cache.write_text(json.dumps(obj))
                time.sleep(2.0)
            caps = obj.get("market_caps", [])
            if not caps:
                continue
            df = pd.DataFrame(caps, columns=["ts", "market_cap"])
            df["date"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
            df["market_cap"] = pd.to_numeric(df["market_cap"], errors="coerce")
            out[pair] = df[["date", "market_cap"]].dropna().sort_values("date")
        except Exception as e:
            print(f"WARN CoinGecko history failed for {pair}/{cid}: {e}")
    return out


def cap_before(cap_df: pd.DataFrame, cutoff: pd.Timestamp) -> float | None:
    x = cap_df[cap_df["date"] <= cutoff]
    if x.empty:
        return None
    v = float(x.iloc[-1]["market_cap"])
    return v if np.isfinite(v) and v > 0 else None


def ranking_for_month(
    frames: dict[str, pd.DataFrame],
    caps: dict[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    month_start: pd.Timestamp,
    ranking_mode: str,
) -> tuple[list[str], str]:
    eligible: list[str] = []
    volume_scores: dict[str, float] = {}
    cap_scores: dict[str, float] = {}

    for pair, df in frames.items():
        train = df[(df["date"] >= train_start) & (df["date"] < train_end)]
        if len(train) < 80:
            continue
        eligible.append(pair)
        volume_scores[pair] = float(train["quote_volume"].replace([np.inf, -np.inf], np.nan).dropna().mean())
        if pair in caps:
            v = cap_before(caps[pair], month_start - pd.Timedelta(days=1))
            if v is not None:
                cap_scores[pair] = v

    if ranking_mode == "coingecko" and len(cap_scores) >= max(5, math.ceil(len(eligible) * 0.6)):
        ranked = sorted([p for p in eligible if p in cap_scores], key=lambda p: cap_scores[p], reverse=True)
        return ranked, "coingecko_market_cap"

    ranked = sorted(eligible, key=lambda p: volume_scores.get(p, 0.0), reverse=True)
    return ranked, "binance_quote_volume_proxy"


def result_dict(r: OptResult) -> dict:
    return {
        "sharpe": round(float(r.sharpe), 6),
        "lookback": int(r.lookback),
        "theta": float(r.theta),
        "alpha": float(r.alpha),
        "train_trades": int(r.trades),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="/freqtrade/user_data/data/binance/futures")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--output", default="/freqtrade/user_data/v5/adaptive-schedule.json")
    ap.add_argument("--cache", default="/freqtrade/user_data/v5/coingecko-cache")
    ap.add_argument("--ranking", choices=["coingecko", "volume"], default="coingecko")
    ap.add_argument("--fee", type=float, default=DEFAULT_FEE)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    whitelist = list(cfg.get("exchange", {}).get("pair_whitelist", []))
    if not whitelist:
        raise SystemExit("No pair_whitelist in config")

    start = pd.Timestamp(args.start, tz="UTC")
    end = pd.Timestamp(args.end, tz="UTC") + pd.Timedelta(days=1)
    frames = load_6h_frames(Path(args.data_root), whitelist)
    if not frames:
        raise SystemExit("No 6h frames found")

    caps: dict[str, pd.DataFrame] = {}
    if args.ranking == "coingecko":
        print("Fetching/caching keyless public CoinGecko market-cap history...")
        caps = load_market_caps(list(frames), start, end, Path(args.cache))
        print(f"CoinGecko market-cap coverage: {len(caps)}/{len(frames)} pairs")

    months: dict[str, dict] = {}
    for m in month_starts(start, end - pd.Timedelta(days=1)):
        # Strictly preceding month with a 24h gap before OOS month, as described in the paper.
        train_start = m - pd.offsets.MonthBegin(1)
        train_end = m - pd.Timedelta(hours=24)
        ranked, rank_source = ranking_for_month(frames, caps, train_start, train_end, m, args.ranking)
        if not ranked:
            print(f"{m:%Y-%m}: no eligible previous-month universe")
            continue

        k_long = min(15, len(ranked))
        # The paper states top-15 for long and bottom-Ks for short but does not publish Ks.
        # Keep legs disjoint in a small 20-pair research universe; with >=30 pairs this becomes bottom-15.
        k_short = min(15, max(0, len(ranked) - k_long))
        long_candidates = ranked[:k_long]
        short_candidates = ranked[-k_short:] if k_short else []

        long_selected: dict[str, dict] = {}
        short_selected: dict[str, dict] = {}

        for pair in long_candidates:
            r = optimize_side(frames[pair], train_start, train_end, "long", args.fee)
            if r and r.sharpe >= 1.3:
                long_selected[pair] = result_dict(r)

        for pair in short_candidates:
            r = optimize_side(frames[pair], train_start, train_end, "short", args.fee)
            if r and r.sharpe >= 1.7:
                short_selected[pair] = result_dict(r)

        key = m.strftime("%Y-%m")
        months[key] = {
            "train_start": train_start.isoformat(),
            "train_end_exclusive": train_end.isoformat(),
            "ranking_source": rank_source,
            "ranked_universe": ranked,
            "long_candidates": long_candidates,
            "short_candidates": short_candidates,
            "long": long_selected,
            "short": short_selected,
            "n_long": len(long_selected),
            "n_short": len(short_selected),
        }
        print(
            f"{key} | rank={rank_source} | candidates L/S={len(long_candidates)}/{len(short_candidates)} "
            f"| selected L/S={len(long_selected)}/{len(short_selected)}"
        )

    output = {
        "meta": {
            "strategy": "AdaptiveTrend20x",
            "source_core": "AdaptiveTrend arXiv:2602.11708",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "start": args.start,
            "end": args.end,
            "timeframe": "6h",
            "lookbacks": LOOKBACKS,
            "thresholds": THRESHOLDS,
            "atr_multipliers": ATR_MULTS,
            "atr_period": ATR_PERIOD,
            "selection_sharpe_long": 1.3,
            "selection_sharpe_short": 1.7,
            "long_allocation": 0.70,
            "short_allocation": 0.30,
            "fee_per_fill": args.fee,
            "risk_free_annual": RISK_FREE_ANNUAL,
            "ranking_requested": args.ranking,
            "note": "20x leverage is intentionally NOT used during previous-month parameter/Sharpe selection; it is applied only by the execution strategy.",
        },
        "months": months,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"Schedule written: {out} ({len(months)} months)")


if __name__ == "__main__":
    main()
