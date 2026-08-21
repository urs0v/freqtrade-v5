from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute


class FrozenFakeoutV1(IStrategy):
    """Execution-only Freqtrade bridge for the frozen fully-causal FAKEOUT signal.

    Alpha generation stays in the parity-tested signal feed. Freqtrade is responsible
    for realistic dry-run order timing, wallet / slot constraints, fills, fees,
    structural stop, 3R target and the 4h time exit.
    """

    INTERFACE_VERSION = 3
    timeframe = "5m"
    can_short = True
    startup_candle_count = 20

    # The feed can finish after Freqtrade's first loop for a newly closed candle.
    # Re-evaluating the current candle lets the dry-run consume the signal later in
    # that same 5m entry window instead of silently missing it.
    process_only_new_candles = False

    minimal_roi = {}
    stoploss = -0.99
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    position_adjustment_enable = False

    RR = 3.0
    RISK_PCT = 0.01
    FIXED_LEVERAGE = 5.0
    RISK_MIN_BPS = 160.0
    RISK_MAX_BPS = 3000.0
    MAINT_MARGIN_FRAC = 0.005
    MAX_SIGNAL_AGE_SECONDS = 285.0

    _feed_mtime_ns: int | None = None
    _feed: DataFrame | None = None

    @property
    def feed_path(self) -> Path:
        return Path(
            os.environ.get(
                "FROZEN_FAKEOUT_FEED",
                "/freqtrade/user_data/frozen_fakeout_feed/signals.csv",
            )
        )

    def _load_feed(self, force: bool = False) -> DataFrame:
        path = self.feed_path
        try:
            st = path.stat()
        except OSError:
            self._feed = pd.DataFrame()
            self._feed_mtime_ns = None
            return self._feed

        if not force and self._feed is not None and self._feed_mtime_ns == st.st_mtime_ns:
            return self._feed

        try:
            z = pd.read_csv(path)
            if not z.empty:
                z["signal_time"] = pd.to_datetime(z["signal_time"], utc=True, errors="coerce")
                z["entry_time"] = pd.to_datetime(z["entry_time"], utc=True, errors="coerce")
                for c in ["side", "stop_price", "activity_score", "risk_bps", "entry_price"]:
                    if c in z:
                        z[c] = pd.to_numeric(z[c], errors="coerce")
                z = z.dropna(subset=["signal_id", "pair", "entry_time", "side", "stop_price"])
                z = z.drop_duplicates("signal_id", keep="last").sort_values("entry_time")
            self._feed = z
            self._feed_mtime_ns = st.st_mtime_ns
        except Exception:
            # Never fall back to a stale entry decision when the feed cannot be read.
            self._feed = pd.DataFrame()
            self._feed_mtime_ns = st.st_mtime_ns
        return self._feed

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        self._load_feed(force=False)

    def _signal_by_id(self, signal_id: str | None) -> dict | None:
        if not signal_id:
            return None
        z = self._load_feed(force=False)
        if z is None or z.empty:
            return None
        g = z[z["signal_id"].astype(str).eq(str(signal_id))]
        if g.empty:
            return None
        return g.iloc[-1].to_dict()

    @staticmethod
    def _signal_id_from_tag(entry_tag: str | None) -> str | None:
        if not entry_tag or not str(entry_tag).startswith("ffv1:"):
            return None
        return str(entry_tag)[5:]

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        x = dataframe.copy()
        x["ff_signal_id"] = None
        x["ff_side"] = np.nan
        x["ff_stop_abs"] = np.nan
        x["ff_activity"] = np.nan
        x["ff_model_entry"] = pd.NaT

        z = self._load_feed(force=False)
        if z is None or z.empty:
            return x
        pair = str(metadata.get("pair", ""))
        g = z[z["pair"].astype(str).eq(pair)].copy()
        if g.empty:
            return x

        # Only signals that were still alive when the executable feed published
        # them are eligible. Historical CLOSED rows remain in the feed for audit.
        if "feed_eligible" in g:
            g = g[g["feed_eligible"].astype(bool)]
        if g.empty:
            return x

        g["signal_candle"] = g["entry_time"] - pd.Timedelta(minutes=5)
        g = g.sort_values("entry_time").drop_duplicates("signal_candle", keep="last")
        lookup = g.set_index("signal_candle")
        dates = pd.to_datetime(x["date"], utc=True, errors="coerce")

        for idx, dt in dates.items():
            if dt not in lookup.index:
                continue
            r = lookup.loc[dt]
            if isinstance(r, DataFrame):
                r = r.iloc[-1]
            x.at[idx, "ff_signal_id"] = str(r["signal_id"])
            x.at[idx, "ff_side"] = float(r["side"])
            x.at[idx, "ff_stop_abs"] = float(r["stop_price"])
            x.at[idx, "ff_activity"] = float(r.get("activity_score", np.nan))
            x.at[idx, "ff_model_entry"] = pd.Timestamp(r["entry_time"])
        return x

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        valid = dataframe["ff_signal_id"].notna() & dataframe["ff_stop_abs"].notna()
        long_sig = valid & (pd.to_numeric(dataframe["ff_side"], errors="coerce") > 0)
        short_sig = valid & (pd.to_numeric(dataframe["ff_side"], errors="coerce") < 0)
        dataframe.loc[long_sig, "enter_long"] = 1
        dataframe.loc[short_sig, "enter_short"] = 1
        dataframe.loc[valid, "enter_tag"] = "ffv1:" + dataframe.loc[valid, "ff_signal_id"].astype(str)
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        return float(min(self.FIXED_LEVERAGE, max_leverage))

    def _entry_geometry(self, entry_tag: str | None, rate: float, side: str) -> tuple[dict | None, float, float]:
        sig = self._signal_by_id(self._signal_id_from_tag(entry_tag))
        if sig is None or not np.isfinite(rate) or rate <= 0:
            return sig, np.nan, np.nan
        stop = float(sig.get("stop_price", np.nan))
        sgn = -1 if side == "short" else 1
        risk_abs = sgn * (float(rate) - stop)
        risk_bps = risk_abs / float(rate) * 10000.0 if risk_abs > 0 else np.nan
        return sig, risk_abs, risk_bps

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        current_time: datetime,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> bool:
        sig, risk_abs, risk_bps = self._entry_geometry(entry_tag, rate, side)
        if sig is None or not np.isfinite(risk_abs) or risk_abs <= 0 or not np.isfinite(risk_bps):
            return False
        if str(sig.get("pair")) != pair:
            return False
        sig_side = int(float(sig.get("side", 0)))
        if (side == "short" and sig_side >= 0) or (side == "long" and sig_side <= 0):
            return False
        if not (self.RISK_MIN_BPS <= risk_bps <= self.RISK_MAX_BPS):
            return False

        model_entry = pd.Timestamp(sig["entry_time"])
        now = pd.Timestamp(current_time)
        if model_entry.tzinfo is None:
            model_entry = model_entry.tz_localize("UTC")
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        age = (now - model_entry).total_seconds()
        if age < -5.0 or age > self.MAX_SIGNAL_AGE_SECONDS:
            return False

        # Same execution-feasibility rule used in the $100 portfolio realism audit.
        stop_frac = risk_bps / 10000.0
        liq_room = 1.0 / self.FIXED_LEVERAGE - self.MAINT_MARGIN_FRAC
        if stop_frac >= liq_room:
            return False
        return True

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        sig, risk_abs, risk_bps = self._entry_geometry(entry_tag, current_rate, side)
        if sig is None or not np.isfinite(risk_abs) or risk_abs <= 0 or not np.isfinite(risk_bps):
            return 0.0
        if not (self.RISK_MIN_BPS <= risk_bps <= self.RISK_MAX_BPS):
            return 0.0

        try:
            equity = float(self.wallets.get_total_stake_amount())
        except Exception:
            equity = float(self.config.get("dry_run_wallet", 100.0))
        if not np.isfinite(equity) or equity <= 0:
            return 0.0

        stop_frac = risk_bps / 10000.0
        lev = max(float(leverage), 1.0)
        collateral = equity * self.RISK_PCT / max(stop_frac * lev, 1e-12)
        collateral = min(float(collateral), float(max_stake))
        if min_stake is not None and collateral < float(min_stake):
            return 0.0
        return max(0.0, collateral)

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order,
        current_time: datetime,
        **kwargs,
    ) -> None:
        try:
            first_entry = trade.nr_of_successful_entries == 1 and order.ft_order_side == trade.entry_side
        except Exception:
            first_entry = False
        if not first_entry:
            return None

        signal_id = self._signal_id_from_tag(trade.enter_tag)
        sig = self._signal_by_id(signal_id)
        if sig is None:
            return None

        fill = float(trade.open_rate)
        stop = float(sig["stop_price"])
        side = -1 if trade.is_short else 1
        risk_abs = side * (fill - stop)
        if not np.isfinite(risk_abs) or risk_abs <= 0:
            return None
        target = fill + side * self.RR * risk_abs
        risk_bps = risk_abs / fill * 10000.0

        trade.set_custom_data(key="ff_signal_id", value=str(signal_id))
        trade.set_custom_data(key="ff_stop_abs", value=float(stop))
        trade.set_custom_data(key="ff_target_abs", value=float(target))
        trade.set_custom_data(key="ff_initial_risk_bps", value=float(risk_bps))
        trade.set_custom_data(key="ff_model_entry", value=str(sig.get("entry_time")))
        trade.set_custom_data(key="ff_model_open", value=float(sig.get("entry_price", np.nan)))
        trade.set_custom_data(key="ff_activity", value=float(sig.get("activity_score", np.nan)))
        return None

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool,
        **kwargs,
    ) -> float | None:
        stop_abs = trade.get_custom_data(key="ff_stop_abs", default=None)
        if stop_abs is None:
            return None
        try:
            stop_abs = float(stop_abs)
        except Exception:
            return None
        if not np.isfinite(stop_abs) or stop_abs <= 0:
            return None
        return stoploss_from_absolute(
            stop_abs,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        target = trade.get_custom_data(key="ff_target_abs", default=None)
        if target is not None:
            try:
                target = float(target)
                if np.isfinite(target):
                    if trade.is_short and current_rate <= target:
                        return "ff_target_3r"
                    if not trade.is_short and current_rate >= target:
                        return "ff_target_3r"
            except Exception:
                pass

        age = (current_time - trade.open_date_utc).total_seconds()
        if age >= 4.0 * 3600.0:
            return "ff_time_4h"
        return None
