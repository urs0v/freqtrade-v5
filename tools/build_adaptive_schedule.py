from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
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


def month_starts(start: pd.Timestamp, end: pd.Timestamp):
    cur = pd.Timestamp(start.year, start.month, 1, tz="UTC")
    last = pd.Timestamp(end.year, end.month, 1, tz="UTC")
    while cur <= last:
        yield cur
        cur = cur + pd.offsets.MonthBegin(1)


def normalize_date_col(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce").dropna()
        if vals.empty:
            return pd.to_datetime(s, utc=True, errors="coerce")
        unit = "ms" if vals.abs().median() > 1e11 else "s"
        return pd.to_datetime(s, unit=unit, utc=True, errors="coerce")
    return pd.to_datetime(s, utc=True, errors="coerce")


def pair_from_filename(path: Path) -> str | None:
    marker = "-6h-futures.feather"
    name = path.name
    if not name.endswith(marker):
        return None
    raw = name[: -len(marker)]
    if raw.endswith("_USDT_USDT"):
        return f"{raw[:-len('_USDT_USDT')]}/USDT:USDT"
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
    emergency_stops: int
    stop_rate: float


def simulate_side(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    side: str,
    lookback: int,
    theta: float,
    alpha: float,
    fee: float,
    leverage: float,
    emergency_stop: float,
) -> tuple[float, int, int]:
    """Previous-month simulation using the SAME 20x survival constraint as execution.

    The original paper optimizes the unlevered H6 trend model. Our execution mandate is
    fixed 20x isolated leverage, so selecting parameters without modelling the 20x
    emergency boundary strongly favors trends that cannot survive normal adverse
    excursions. This simulator therefore keeps the H6 AdaptiveTrend signal/trailing
    core, but prices fees and the fixed entry-anchored emergency stop in margin-return
    space before computing the selection Sharpe.
    """
    warm_start = train_start - pd.Timedelta(days=10)
    sub = df[(df["date"] >= warm_start) & (df["date"] < train_end)].copy()
    if len(sub) < 40:
        return float("-inf"), 0, 0

    close = sub["close"].to_numpy(dtype=float)
    high = sub["high"].to_numpy(dtype=float)
    low = sub["low"].to_numpy(dtype=float)
    atr = sub["atr"].to_numpy(dtype=float)
    mom = sub[f"mom_{lookback}"].to_numpy(dtype=float)
    scoring = (sub["date"] >= train_start).to_numpy()

    rets = np.zeros(len(sub), dtype=float)
    pos = False
    entry = np.nan
    trail = np.nan
    trades = 0
    emergency_stops = 0
    sign = 1.0 if side == "long" else -1.0
    fill_cost = fee * leverage

    for i in range(1, len(sub)):
        if not scoring[i]:
            continue

        if pos:
            emergency_price = entry * (1.0 - emergency_stop if side == "long" else 1.0 + emergency_stop)
            emergency_hit = low[i] <= emergency_price if side == "long" else high[i] >= emergency_price

            if emergency_hit:
                # Conservative stop fill at the configured boundary. Return is on
                # collateral/margin, matching the effect of fixed leverage.
                step = sign * (emergency_price / close[i - 1] - 1.0) * leverage
                rets[i] += step - fill_cost
                emergency_stops += 1
                pos = False
                entry = np.nan
                trail = np.nan
                continue

            rets[i] += sign * (close[i] / close[i - 1] - 1.0) * leverage

            # Paper-style trailing decision only at completed H6 closes.
            if np.isfinite(atr[i]):
                if side == "long":
                    candidate = close[i] - alpha * atr[i]
                    trail = max(trail, candidate)
                    crossed = close[i] < trail
                else:
                    candidate = close[i] + alpha * atr[i]
                    trail = min(trail, candidate)
                    crossed = close[i] > trail
                if crossed:
                    rets[i] -= fill_cost
                    pos = False
                    entry = np.nan
                    trail = np.nan
                    continue

        if not pos and np.isfinite(mom[i]) and np.isfinite(atr[i]) and atr[i] > 0:
            fire = mom[i] > theta if side == "long" else mom[i] < -theta
            if fire:
                pos = True
                trades += 1
                entry = close[i]
                rets[i] -= fill_cost
                trail = close[i] - alpha * atr[i] if side == "long" else close[i] + alpha * atr[i]

    scored = rets[scoring]
    return annualized_sharpe(scored), trades, emergency_stops


def optimize_side(
    df: pd.DataFrame,
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
    side: str,
    fee: float,
    leverage: float,
    emergency_stop: float,
    max_stop_rate: float,
    min_trades: int,
) -> OptResult | None:
    best: OptResult | None = None
    for lb in LOOKBACKS:
        for theta in THRESHOLDS:
            for alpha in ATR_MULTS:
                sr, trades, stops = simulate_side(
                    df, train_start, train_end, side, lb, theta, alpha,
                    fee, leverage, emergency_stop,
                )
                if not np.isfinite(sr) or trades < min_trades:
                    continue
                stop_rate = stops / trades if trades else 1.0
                # A parameter set that repeatedly touches the 20x survival boundary
                # is not deployable even if one outlier trend makes its Sharpe look good.
                if stop_rate > max_stop_rate:
                    continue
                candidate = OptResult(sr, lb, theta, alpha, trades, stops, stop_rate)
                if best is None or candidate.sharpe > best.sharpe:
                    best = candidate
    return best


def ranking_for_month(
    frames: dict[str, pd.DataFrame],
    train_start: pd.Timestamp,
    train_end: pd.Timestamp,
) -> list[str]:
    scores: dict[str, float] = {}
    for pair, df in frames.items():
        train = df[(df["date"] >= train_start) & (df["date"] < train_end)]
        if len(train) < 80:
            continue
        qv = train["quote_volume"].replace([np.inf, -np.inf], np.nan).dropna()
        if not qv.empty:
            scores[pair] = float(qv.mean())
    return sorted(scores, key=scores.get, reverse=True)


def result_dict(r: OptResult) -> dict:
    return {
        "sharpe": round(float(r.sharpe), 6),
        "lookback": int(r.lookback),
        "theta": float(r.theta),
        "alpha": float(r.alpha),
        "train_trades": int(r.trades),
        "emergency_stops": int(r.emergency_stops),
        "stop_rate": round(float(r.stop_rate), 6),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-root", default="/freqtrade/user_data/data/binance/futures")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--output", default="/freqtrade/user_data/v5/adaptive-schedule.json")
    ap.add_argument("--ranking", choices=["volume"], default="volume")
    ap.add_argument("--fee", type=float, default=DEFAULT_FEE)
    ap.add_argument("--leverage", type=float, default=20.0)
    ap.add_argument("--emergency-stop", type=float, default=0.035)
    ap.add_argument("--max-stop-rate", type=float, default=0.25)
    ap.add_argument("--min-trades", type=int, default=2)
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

    print(
        f"20x-aware optimizer: leverage={args.leverage:g}x | emergency price stop={args.emergency_stop*100:.2f}% "
        f"| max training stop-rate={args.max_stop_rate*100:.1f}%"
    )

    months: dict[str, dict] = {}
    for m in month_starts(start, end - pd.Timedelta(days=1)):
        train_start = m - pd.offsets.MonthBegin(1)
        train_end = m - pd.Timedelta(hours=24)
        ranked = ranking_for_month(frames, train_start, train_end)
        if not ranked:
            print(f"{m:%Y-%m}: no eligible previous-month universe")
            continue

        k_long = min(15, len(ranked))
        k_short = min(15, max(0, len(ranked) - k_long))
        long_candidates = ranked[:k_long]
        short_candidates = ranked[-k_short:] if k_short else []

        long_selected: dict[str, dict] = {}
        short_selected: dict[str, dict] = {}

        for pair in long_candidates:
            r = optimize_side(
                frames[pair], train_start, train_end, "long", args.fee,
                args.leverage, args.emergency_stop, args.max_stop_rate, args.min_trades,
            )
            if r and r.sharpe >= 1.3:
                long_selected[pair] = result_dict(r)

        for pair in short_candidates:
            r = optimize_side(
                frames[pair], train_start, train_end, "short", args.fee,
                args.leverage, args.emergency_stop, args.max_stop_rate, args.min_trades,
            )
            if r and r.sharpe >= 1.7:
                short_selected[pair] = result_dict(r)

        key = m.strftime("%Y-%m")
        months[key] = {
            "train_start": train_start.isoformat(),
            "train_end_exclusive": train_end.isoformat(),
            "ranking_source": "binance_quote_volume_proxy",
            "ranked_universe": ranked,
            "long_candidates": long_candidates,
            "short_candidates": short_candidates,
            "long": long_selected,
            "short": short_selected,
            "n_long": len(long_selected),
            "n_short": len(short_selected),
        }

        stop_rates = [x["stop_rate"] for x in list(long_selected.values()) + list(short_selected.values())]
        sr_txt = f" | avg selected stop-rate={np.mean(stop_rates)*100:.1f}%" if stop_rates else ""
        print(
            f"{key} | candidates L/S={len(long_candidates)}/{len(short_candidates)} "
            f"| selected L/S={len(long_selected)}/{len(short_selected)}{sr_txt}"
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
            "ranking_requested": "volume",
            "optimizer_leverage": args.leverage,
            "optimizer_emergency_price_stop": args.emergency_stop,
            "optimizer_max_stop_rate": args.max_stop_rate,
            "optimizer_min_trades": args.min_trades,
            "note": "Project adaptation: previous-month parameter/Sharpe selection now models the same fixed-leverage emergency survival boundary used by execution.",
        },
        "months": months,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, indent=2))
    print(f"Schedule written: {out} ({len(months)} months)")


if __name__ == "__main__":
    main()
