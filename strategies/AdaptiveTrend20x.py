from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, stoploss_from_absolute


class AdaptiveTrend20x(IStrategy):
    """AdaptiveTrend execution core with project-required fixed 20x leverage.

    Source architecture (arXiv:2602.11708):
      - native 6h momentum signals
      - previous-month parameter optimization / Sharpe selection
      - monthly active universe
      - 70/30 long-short capital allocation
      - ATR trailing exit updated on completed H6 candles only

    Project modification:
      - every trade uses up to 20x isolated leverage
      - an emergency price-space stop protects against liquidation between H6 closes
    """

    INTERFACE_VERSION = 3
    timeframe = "6h"
    can_short = True
    startup_candle_count = 40
    process_only_new_candles = True

    minimal_roi = {}
    # At 20x, -0.70 corresponds to roughly a 3.5% adverse price move.
    # custom_stoploss sets the same constraint explicitly in price space.
    stoploss = -0.70
    use_custom_stoploss = True
    trailing_stop = False
    # custom_exit is used for the paper-style H6 trailing / monthly rebalance.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    schedule_path = os.environ.get("ADAPTIVE_SCHEDULE", "/freqtrade/user_data/v5/adaptive-schedule.json")

    def _ensure_schedule(self) -> dict:
        if hasattr(self, "_adaptive_schedule"):
            return self._adaptive_schedule
        path = Path(self.schedule_path)
        if not path.exists():
            self._adaptive_schedule = {"meta": {}, "months": {}}
            return self._adaptive_schedule
        self._adaptive_schedule = json.loads(path.read_text())
        return self._adaptive_schedule

    def bot_start(self, **kwargs) -> None:
        self._ensure_schedule()

    @staticmethod
    def _month_from_tag(tag: str | None, fallback_time: datetime) -> str:
        if tag and tag.startswith("AT_"):
            bits = tag.split("_")
            if len(bits) >= 3 and len(bits[2]) == 7:
                return bits[2]
        return pd.Timestamp(fallback_time).strftime("%Y-%m")

    def _month_block(self, month: str) -> dict:
        return self._ensure_schedule().get("months", {}).get(month, {})

    def _entry_params(self, pair: str, month: str, side: str) -> dict | None:
        return self._month_block(month).get(side, {}).get(pair)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        schedule = self._ensure_schedule()
        atr_period = int(schedule.get("meta", {}).get("atr_period", 14))
        lookbacks = schedule.get("meta", {}).get("lookbacks", [4, 6, 8, 10, 12, 16])

        dataframe["atr"] = ta.ATR(dataframe, timeperiod=atr_period)
        for lb in sorted({int(x) for x in lookbacks}):
            dataframe[f"mom_{lb}"] = dataframe["close"].pct_change(lb)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata["pair"]
        schedule = self._ensure_schedule()
        months = schedule.get("months", {})

        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        dates = pd.to_datetime(dataframe["date"], utc=True)
        valid_base = (dataframe["volume"] > 0) & dataframe["atr"].notna() & (dataframe["atr"] > 0)

        for month, block in months.items():
            month_mask = dates.dt.strftime("%Y-%m") == month
            if not month_mask.any():
                continue

            lp = block.get("long", {}).get(pair)
            if lp:
                lb = int(lp["lookback"])
                col = f"mom_{lb}"
                if col in dataframe.columns:
                    mask = month_mask & valid_base & dataframe[col].notna() & (dataframe[col] > float(lp["theta"]))
                    dataframe.loc[mask, "enter_long"] = 1
                    dataframe.loc[mask, "enter_tag"] = f"AT_L_{month}"

            sp = block.get("short", {}).get(pair)
            if sp:
                lb = int(sp["lookback"])
                col = f"mom_{lb}"
                if col in dataframe.columns:
                    mask = month_mask & valid_base & dataframe[col].notna() & (dataframe[col] < -float(sp["theta"]))
                    dataframe.loc[mask, "enter_short"] = 1
                    dataframe.loc[mask, "enter_tag"] = f"AT_S_{month}"

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
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
        force = os.environ.get("ADAPTIVE_FORCE_20X", "true").lower() in {"1", "true", "yes", "on"}
        requested = 20.0 if force else proposed_leverage
        return float(min(requested, max_leverage))

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
        # Compound automatically: allocations are fractions of CURRENT account equity.
        month = self._month_from_tag(entry_tag, current_time)
        block = self._month_block(month)
        count = int(block.get("n_long", 0) if side == "long" else block.get("n_short", 0))
        if count <= 0:
            return 0.0

        try:
            equity = float(self.wallets.get_total_stake_amount())
        except Exception:
            equity = float(max_stake)

        leg = 0.70 if side == "long" else 0.30
        stake = equity * leg / count
        stake = min(stake, float(max_stake))
        if min_stake is not None and stake < float(min_stake):
            return 0.0
        return float(max(0.0, stake))

    def _last_closed_h6(self, pair: str, current_time: datetime):
        # timeframe-detail may call callbacks every 5m. The paper updates its trailing
        # stop only once per COMPLETED H6 candle, so ignore intra-H6 price noise here.
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty:
            return None
        x = df.copy()
        x["date"] = pd.to_datetime(x["date"], utc=True)
        now = pd.Timestamp(current_time)
        if now.tzinfo is None:
            now = now.tz_localize("UTC")
        else:
            now = now.tz_convert("UTC")
        boundary = now.floor("6h")
        closed = x[x["date"] < boundary]
        if closed.empty:
            return None
        return closed.iloc[-1]

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        side = "short" if trade.is_short else "long"
        entry_month = self._month_from_tag(trade.enter_tag, trade.open_date_utc)
        current_month = pd.Timestamp(current_time).strftime("%Y-%m")

        # Monthly portfolio rebalance: if the side is no longer selected, leave it.
        if current_month != entry_month:
            if self._entry_params(pair, current_month, side) is None:
                return "monthly_rebalance"

        params = self._entry_params(pair, entry_month, side)
        if params is None:
            return None
        alpha = float(params["alpha"])

        candle = self._last_closed_h6(pair, current_time)
        if candle is None:
            return None
        candle_ts = pd.Timestamp(candle["date"]).isoformat()
        last_ts = trade.get_custom_data(key="at_last_h6")
        if last_ts == candle_ts:
            return None

        close = float(candle["close"])
        atr = float(candle["atr"])
        if not np.isfinite(close) or not np.isfinite(atr) or atr <= 0:
            trade.set_custom_data(key="at_last_h6", value=candle_ts)
            return None

        previous = trade.get_custom_data(key="at_trailing_stop")
        if previous is None:
            stop = close + alpha * atr if trade.is_short else close - alpha * atr
        else:
            previous = float(previous)
            candidate = close + alpha * atr if trade.is_short else close - alpha * atr
            stop = min(previous, candidate) if trade.is_short else max(previous, candidate)

        trade.set_custom_data(key="at_trailing_stop", value=float(stop))
        trade.set_custom_data(key="at_last_h6", value=candle_ts)

        if trade.is_short and close > stop:
            return "adaptive_h6_trailing"
        if not trade.is_short and close < stop:
            return "adaptive_h6_trailing"
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
        # This is NOT the paper's trailing stop. It is a 20x-only emergency guard
        # against liquidation between H6 decisions. It stays anchored to entry.
        price_stop = float(os.environ.get("ADAPTIVE_EMERGENCY_PRICE_STOP", "0.035"))
        price_stop = min(max(price_stop, 0.01), 0.045)
        absolute = trade.open_rate * (1.0 + price_stop if trade.is_short else 1.0 - price_stop)
        return stoploss_from_absolute(
            absolute,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
