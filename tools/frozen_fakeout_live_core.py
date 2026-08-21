#!/usr/bin/env python3
from __future__ import annotations

from array import array
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import time

import numpy as np
import pandas as pd

import breakout_retest_profit_v1 as v1
import digash_v3_common as dc
import frozen_fakeout_incremental as inc
import prospective_fakeout_v2 as p2
import frozen_fakeout_signal_feed_v2 as feed2


TAIL5 = 720
TAIL15 = 3100


def _norm_raw(x: pd.DataFrame) -> pd.DataFrame:
    if x is None or x.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    y = x[["date", "open", "high", "low", "close", "volume"]].copy()
    y["date"] = pd.to_datetime(y["date"], utc=True).astype("datetime64[ns, UTC]")
    for c in ["open", "high", "low", "close", "volume"]:
        y[c] = pd.to_numeric(y[c], errors="coerce")
    return y.dropna().drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)


def _row_frame(row: dict) -> pd.DataFrame:
    return pd.DataFrame([{
        "date": pd.Timestamp(row["date"]),
        "open": float(row["open"]),
        "high": float(row["high"]),
        "low": float(row["low"]),
        "close": float(row["close"]),
        "volume": float(row.get("volume", 0.0)),
    }])


def _replace_last_or_append(df: pd.DataFrame, row: dict) -> tuple[pd.DataFrame, bool]:
    r = _row_frame(row)
    dt = pd.Timestamp(r.iloc[0]["date"])
    if len(df) and pd.Timestamp(df.iloc[-1]["date"]) == dt:
        out = df.copy()
        out.iloc[-1] = r.iloc[0]
        return out, False
    if len(df) and pd.Timestamp(df.iloc[-1]["date"]) > dt:
        raise RuntimeError(f"NON_MONOTONIC_BAR {dt} <= {df.iloc[-1]['date']}")
    return pd.concat([df, r], ignore_index=True), True


class TFSeries:
    """Minimal append-only history needed by the frozen level builder."""

    def __init__(self, tf: str, minutes: int, x: pd.DataFrame):
        self.tf = tf
        self.minutes = int(minutes)
        self.n = int(len(x))
        self.closes = array("d", pd.to_numeric(x["close"], errors="coerce").astype(float).tolist())
        self.high5 = deque(pd.to_numeric(x["high"].tail(5), errors="coerce").astype(float).tolist(), maxlen=5)
        self.low5 = deque(pd.to_numeric(x["low"].tail(5), errors="coerce").astype(float).tolist(), maxlen=5)
        self.last_signal_time = pd.Timestamp(x.iloc[-1]["signal_time"]) if len(x) else None

    def close_view(self) -> np.ndarray:
        return np.frombuffer(self.closes, dtype=np.float64)

    def append(self, bar: dict) -> list[dict]:
        signal_time = pd.Timestamp(bar["signal_time"])
        if self.last_signal_time is not None:
            if signal_time == self.last_signal_time:
                return []
            if signal_time < self.last_signal_time:
                raise RuntimeError(
                    f"{self.tf} NON_MONOTONIC_SIGNAL_TIME {signal_time} < {self.last_signal_time}"
                )
        self.closes.append(float(bar["close"]))
        self.high5.append(float(bar["high"]))
        self.low5.append(float(bar["low"]))
        idx = self.n
        self.n += 1
        self.last_signal_time = signal_time

        if len(self.high5) < 5:
            return []
        pivot_idx = idx - dc.PIVOT_RIGHT
        highs = np.asarray(self.high5, dtype=float)
        lows = np.asarray(self.low5, dtype=float)
        out = []
        center = dc.PIVOT_RIGHT
        if highs[center] >= np.max(highs):
            out.append({
                "kind": "R", "idx": int(pivot_idx), "avail": int(idx),
                "price": float(highs[center]), "formed": signal_time,
            })
        if lows[center] <= np.min(lows):
            out.append({
                "kind": "S", "idx": int(pivot_idx), "avail": int(idx),
                "price": float(lows[center]), "formed": signal_time,
            })
        return sorted(out, key=lambda r: (r["avail"], r["idx"], r["kind"]))


class LevelBuilder:
    def __init__(self, series: TFSeries, tf: str, period: int):
        self.series = series
        self.tf = tf
        self.period = int(period)
        self.pending = {"R": [], "S": []}
        self.formed_state: list[dict] = []
        self.levels: list[dc.Level] = []

    def _one_sided(self, a: dict, b: dict, price: float) -> bool:
        if b["idx"] <= a["idx"] + 1:
            return False
        closes_all = self.series.close_view()
        closes = closes_all[int(a["idx"]) + 1:int(b["idx"])]
        if len(closes) == 0:
            return False
        band = price * dc.TOUCH_TOL_PCT
        if a["kind"] == "R":
            departed = np.min(closes) <= price - band
            wrong = np.mean(closes > price + 0.25 * band)
        else:
            departed = np.max(closes) >= price + band
            wrong = np.mean(closes < price - 0.25 * band)
        return bool(departed and wrong <= 0.05)

    def accept(self, p: dict, level_id: int) -> tuple[dc.Level | None, int]:
        best_existing = None
        best_err = float("inf")
        for st in self.formed_state:
            lv = st["level"]
            if lv.kind != p["kind"] or p["idx"] - st["last_touch_idx"] < self.period:
                continue
            err = abs(p["price"] - lv.price) / max(abs(lv.price), 1e-12)
            if err <= dc.TOUCH_TOL_PCT and err < best_err:
                best_existing, best_err = st, err
        if best_existing is not None:
            best_existing["last_touch_idx"] = int(p["idx"])
            best_existing["level"].counted_touches += 1
            return None, level_id

        arr = self.pending[p["kind"]]
        best = None
        best_err = float("inf")
        best_pos = None
        for pos in range(len(arr) - 1, -1, -1):
            q = arr[pos]
            if p["idx"] - q["idx"] < self.period:
                continue
            err = abs(p["price"] - q["price"]) / max(abs(q["price"]), 1e-12)
            if err <= dc.TOUCH_TOL_PCT and err < best_err:
                center = (p["price"] + q["price"]) / 2.0
                if self._one_sided(q, p, center):
                    best, best_err, best_pos = q, err, pos
                    if err <= dc.TOUCH_TOL_PCT * 0.25:
                        break

        if best is not None:
            center = (p["price"] + best["price"]) / 2.0
            lv = dc.Level(
                level_id=int(level_id), tf=self.tf, tf_minutes=dc.TF_MINUTES[self.tf],
                period=self.period, kind=p["kind"], price=float(center),
                init_price=float(best["price"]), touch_price=float(p["price"]),
                touch_error_pct=float(best_err * 100.0), init_idx=int(best["idx"]),
                touch_idx=int(p["idx"]), formed_time=pd.Timestamp(p["formed"]), clean_between=True,
            )
            self.levels.append(lv)
            self.formed_state.append({"level": lv, "last_touch_idx": int(p["idx"])})
            if best_pos is not None:
                arr.pop(best_pos)
            return lv, level_id + 1

        arr.append(dict(p))
        if len(arr) > 5000:
            del arr[:2000]
        return None, level_id


class LevelSystem:
    def __init__(self, frames: dict[str, pd.DataFrame]):
        self.series: dict[str, TFSeries] = {}
        self.builders: dict[tuple[str, int], LevelBuilder] = {}
        self.levels: list[dc.Level] = []
        self.next_id = 0

        for tf in p2.LEVEL_TFS:
            frame = frames[tf]
            s = TFSeries(tf, dc.TF_MINUTES[tf], frame)
            self.series[tf] = s
            pivots = dc.local_pivots(frame)
            for period in p2.LEVEL_PERIODS:
                b = LevelBuilder(s, tf, period)
                for p in pivots:
                    lv, self.next_id = b.accept(p, self.next_id)
                    if lv is not None:
                        self.levels.append(lv)
                self.builders[(tf, period)] = b

    def append(self, tf: str, bar: dict) -> list[dc.Level]:
        pivots = self.series[tf].append(bar)
        made = []
        for period in p2.LEVEL_PERIODS:
            b = self.builders[(tf, period)]
            for p in pivots:
                lv, self.next_id = b.accept(p, self.next_id)
                if lv is not None:
                    self.levels.append(lv)
                    made.append(lv)
        return made


def _agg15(raw15: pd.DataFrame, boundary: pd.Timestamp, minutes: int) -> dict:
    n = minutes // 15
    start = boundary - pd.Timedelta(minutes=minutes)
    q = raw15[(raw15["date"] >= start) & (raw15["date"] < boundary)].tail(n)
    expected = [start + pd.Timedelta(minutes=15 * i) for i in range(n)]
    got = [pd.Timestamp(v) for v in q["date"].tolist()]
    if len(q) != n or got != expected:
        raise RuntimeError(f"INCOMPLETE_{minutes}M_AGG boundary={boundary} got={got}")
    return {
        "date": start, "open": float(q.iloc[0]["open"]), "high": float(q["high"].max()),
        "low": float(q["low"].min()), "close": float(q.iloc[-1]["close"]),
        "volume": float(q["volume"].sum()), "signal_time": boundary,
    }


@dataclass
class BootstrapMeta:
    pair: str
    parity: dict
    rows5: int
    rows15: int
    levels: int
    elapsed_sec: float


class LivePairState:
    def __init__(
        self, pair: str, raw5: pd.DataFrame, raw15: pd.DataFrame, level_system: LevelSystem,
        detector_state: dict, dedup_seen: set, last_processed_global: int,
        prior_signal_time: pd.Timestamp, parity: dict,
    ):
        raw5n = _norm_raw(raw5)
        raw15n = _norm_raw(raw15)
        self.pair = pair
        self.raw5 = raw5n.tail(TAIL5).reset_index(drop=True)
        self.raw15 = raw15n.tail(TAIL15).reset_index(drop=True)
        self.total5 = int(len(raw5n))
        self.level_system = level_system
        self.detector_state = detector_state
        self.dedup_seen = set(dedup_seen)
        self.last_processed_global = int(last_processed_global)
        self.prior_signal_time = pd.Timestamp(prior_signal_time)
        self.parity = dict(parity)
        self.signals: list[dict] = []

    @classmethod
    def bootstrap(
        cls, pair: str, raw5: pd.DataFrame, raw15: pd.DataFrame, reference_csv: str | Path,
    ) -> tuple["LivePairState", BootstrapMeta]:
        t0 = time.monotonic()
        raw5 = _norm_raw(raw5)
        raw15 = _norm_raw(raw15)
        if raw5.empty or raw15.empty:
            raise RuntimeError("NO_BOOTSTRAP_DATA")

        x15 = dc.prep_ohlcv(raw15, 15)
        x5 = v1._prep_exec(raw5)
        x5 = v1._add_activity(x5, v1._activity15(x15))
        frames = {
            "15m": x15,
            "1h": dc.resample_from_15(x15, "1h", 60),
            "4h": dc.resample_from_15(x15, "4h", 240),
        }
        levels = LevelSystem(frames)
        events, detector_state, end_i = inc.detect_events_incremental(x5, levels.levels, start_i=1)
        selected, seen = inc.causal_dedup_incremental(events)
        rows = feed2._rows_from_events(pair, selected, x5, p2.HISTORY_START)
        parity = feed2._bootstrap_parity(pair, rows, Path(reference_csv))
        if not parity.get("pass"):
            raise RuntimeError(f"LIVE_BOOTSTRAP_PARITY_FAIL {pair} {parity}")
        if end_i < 1:
            raise RuntimeError("BOOTSTRAP_TOO_SHORT")

        state = cls(
            pair=pair, raw5=raw5, raw15=raw15, level_system=levels,
            detector_state=detector_state, dedup_seen=seen, last_processed_global=end_i,
            prior_signal_time=pd.Timestamp(x5.iloc[end_i]["signal_time"]), parity=parity,
        )
        meta = BootstrapMeta(
            pair=pair, parity=parity, rows5=len(raw5), rows15=len(raw15),
            levels=len(levels.levels), elapsed_sec=round(time.monotonic() - t0, 3),
        )
        return state, meta

    def _update_15m_context(self, closed15: dict | None, boundary: pd.Timestamp) -> int:
        if boundary.minute % 15 != 0:
            return 0
        if closed15 is None:
            raise RuntimeError(f"MISSING_15M_CLOSE {boundary}")

        expected_date = boundary - pd.Timedelta(minutes=15)
        if pd.Timestamp(closed15["date"]) != expected_date:
            raise RuntimeError(f"BAD_15M_CLOSE expected={expected_date} got={closed15['date']}")
        old_last = pd.Timestamp(self.raw15.iloc[-1]["date"]) if len(self.raw15) else None
        self.raw15, appended = _replace_last_or_append(self.raw15, closed15)
        if not appended and old_last == expected_date:
            return 0
        self.raw15 = self.raw15.tail(TAIL15).reset_index(drop=True)

        made = len(self.level_system.append("15m", {**closed15, "signal_time": boundary}))
        if boundary.minute == 0:
            made += len(self.level_system.append("1h", _agg15(self.raw15, boundary, 60)))
        if boundary.minute == 0 and boundary.hour % 4 == 0:
            made += len(self.level_system.append("4h", _agg15(self.raw15, boundary, 240)))
        return made

    def _prep_tail(self) -> tuple[pd.DataFrame, int]:
        x15 = dc.prep_ohlcv(self.raw15, 15)
        activity = v1._activity15(x15)
        x5 = v1._prep_exec(self.raw5)
        x5 = v1._add_activity(x5, activity)
        return x5, int(self.total5 - len(self.raw5))

    def process_boundary(
        self, closed5: dict, open5: dict, *, closed15: dict | None = None,
        received_at: pd.Timestamp | None = None,
    ) -> tuple[list[dict], dict]:
        t0 = time.monotonic()
        boundary = pd.Timestamp(open5["date"])
        closed_date = boundary - pd.Timedelta(minutes=5)
        if pd.Timestamp(closed5["date"]) != closed_date:
            raise RuntimeError(f"BAD_5M_CLOSE expected={closed_date} got={closed5['date']}")
        if not len(self.raw5) or pd.Timestamp(self.raw5.iloc[-1]["date"]) != closed_date:
            raise RuntimeError(
                f"5M_SEQUENCE_GAP pair={self.pair} expected_stub={closed_date} "
                f"last={self.raw5.iloc[-1]['date'] if len(self.raw5) else None}"
            )

        self.raw5, _ = _replace_last_or_append(self.raw5, closed5)
        stub = {
            "date": boundary, "open": float(open5["open"]), "high": float(open5["open"]),
            "low": float(open5["open"]), "close": float(open5["open"]), "volume": 0.0,
        }
        self.raw5, appended_open = _replace_last_or_append(self.raw5, stub)
        if not appended_open:
            raise RuntimeError(f"OPEN_STUB_ALREADY_EXISTS {boundary}")
        self.total5 += 1
        self.raw5 = self.raw5.tail(TAIL5).reset_index(drop=True)

        new_levels = self._update_15m_context(closed15, boundary)
        x5, offset = self._prep_tail()
        signal_local = len(x5) - 2
        signal_global = offset + signal_local
        expected_global = self.last_processed_global + 1
        if signal_global != expected_global:
            raise RuntimeError(
                f"GLOBAL_INDEX_GAP expected={expected_global} got={signal_global} offset={offset} local={signal_local}"
            )

        events, detector_state, _ = inc.detect_events_incremental(
            x5, self.level_system.levels, start_i=signal_local, stop_i=signal_local,
            initial_state=self.detector_state, prior_signal_time=self.prior_signal_time,
            index_offset=offset,
        )
        selected, seen = inc.causal_dedup_incremental(events, self.dedup_seen)
        self.detector_state = detector_state
        self.dedup_seen = seen
        self.last_processed_global = signal_global
        self.prior_signal_time = pd.Timestamp(x5.iloc[signal_local]["signal_time"])

        rows = []
        for e in selected:
            if e.setup != "H_FAKEOUT":
                continue
            si = int(e.signal_idx) - offset
            if si < 0 or si >= len(x5):
                raise RuntimeError("LIVE_EVENT_OUTSIDE_TAIL")
            activity = float(x5.iloc[si].get("activity_score", np.nan))
            if not np.isfinite(activity) or activity < p2.THRESH:
                continue
            entry = float(open5["open"])
            stop = float(e.stop)
            side = int(e.side)
            risk_abs = side * (entry - stop)
            if not np.isfinite(risk_abs) or risk_abs <= 0:
                continue
            risk_bps = risk_abs / entry * 10000.0
            if risk_bps < p2.RISK_MIN_BPS or risk_bps > 3000.0:
                continue
            target = entry + side * p2.RR * risk_abs
            signal_time = pd.Timestamp(x5.iloc[si]["signal_time"])
            d = {
                "pair": self.pair, "tf": str(e.tf), "period": int(e.period),
                "level_price": float(e.level_price), "level_kind": str(e.level_kind),
                "approach_no": int(e.approach_no), "confluence_tfs": int(e.confluence_tfs),
                "touch_error_pct": float(e.touch_error_pct), "activity_score": activity,
                "natr_ratio30d": float(x5.iloc[si].get("natr_ratio30d", np.nan)),
                "qvol24_ratio30d": float(x5.iloc[si].get("qvol24_ratio30d", np.nan)),
                "stop_source": str(e.stop_source), "signal_time": signal_time,
                "entry_time": boundary, "entry_price": entry, "side": side,
                "stop_price": stop, "target_price": target, "risk_bps": risk_bps,
                "status": "OPEN", "exit_time": pd.NaT, "exit_price": np.nan,
                "exit_reason": "OPEN", "net8_r": np.nan, "stress12_r": np.nan,
            }
            d["signal_id"] = (
                f"{p2.symbol(self.pair)}|{signal_time.isoformat()}|{side}|"
                f"{d['tf']}|{d['period']}|{d['level_price']:.10g}"
            )
            rows.append(d)

        self.signals.extend(rows)
        received_at = pd.Timestamp(received_at) if received_at is not None else pd.Timestamp.now(tz="UTC")
        meta = {
            "pair": self.pair, "boundary": boundary.isoformat(),
            "received_at": received_at.isoformat(),
            "transport_to_ready_ms": round(float((received_at - boundary).total_seconds() * 1000.0), 3),
            "compute_ms": round(float((time.monotonic() - t0) * 1000.0), 3),
            "events": len(events), "selected": len(selected), "signals": len(rows),
            "new_levels": int(new_levels), "levels_total": len(self.level_system.levels),
            "parity_pass": bool(self.parity.get("pass")),
        }
        return rows, meta
