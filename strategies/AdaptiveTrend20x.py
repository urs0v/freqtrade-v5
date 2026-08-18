from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, IntParameter, DecimalParameter, stoploss_from_absolute


class AdaptiveTrend20x(IStrategy):
    """AdaptiveTrend-style H6 momentum strategy with a forced 20x research mode.

    Core intentionally follows the paper's signal architecture much more closely than
    RegimeMomentumV5:
      - native 6h signal generation
      - momentum threshold entries
      - volatility-adaptive ATR trailing stop
      - asymmetric long/short treatment

    The paper itself does not use 20x leverage. 20x is an explicit project-level
    stress-test modification, so the ATR stop distance is capped in price space to
    keep the stop inside a practical isolated-20x liquidation boundary.
    """

    INTERFACE_VERSION = 3
    timeframe = "6h"
    can_short = True
    startup_candle_count = 140
    process_only_new_candles = True

    minimal_roi = {}
    stoploss = -0.95
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = False
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    # Paper parameters are re-optimized monthly. These ranges are deliberately broad
    # so the research scripts can later perform strict prior-month walk-forward tuning.
    mom_lookback = IntParameter(4, 20, default=10, space="buy")
    long_mom_threshold = DecimalParameter(0.01, 0.12, default=0.03, decimals=3, space="buy")
    short_mom_threshold = DecimalParameter(0.015, 0.18, default=0.045, decimals=3, space="buy")
    atr_period = IntParameter(10, 24, default=14, space="sell")
    atr_mult = DecimalParameter(2.0, 3.5, default=2.5, decimals=2, space="sell")

    # 20x-specific execution guard. The source strategy does not need this because it
    # is not specified as a 20x isolated-margin strategy.
    max_price_stop_pct = DecimalParameter(0.012, 0.035, default=0.025, decimals=3, space="sell")

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lb = int(self.mom_lookback.value)
        atr_p = int(self.atr_period.value)
        dataframe["mom"] = dataframe["close"].pct_change(lb)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=atr_p)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        valid = (
            (dataframe["volume"] > 0)
            & dataframe["mom"].notna()
            & dataframe["atr"].notna()
            & (dataframe["atr"] > 0)
        )

        long_signal = valid & (dataframe["mom"] > float(self.long_mom_threshold.value))
        short_signal = valid & (dataframe["mom"] < -float(self.short_mom_threshold.value))

        dataframe.loc[long_signal, ["enter_long", "enter_tag"]] = (1, "adaptive_h6_long")
        dataframe.loc[short_signal, ["enter_short", "enter_tag"]] = (1, "adaptive_h6_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # AdaptiveTrend exits are managed by the monotonic ATR trailing stop.
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
        # User-requested research constraint: keep 20x.
        force = os.environ.get("ADAPTIVE_FORCE_20X", "true").lower() in {"1", "true", "yes", "on"}
        return float(min(20.0 if force else proposed_leverage, max_leverage))

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
        # Approximate the paper's 70/30 long-short capital asymmetry in Freqtrade.
        # With max_open_trades=10 this caps a full all-long book near 70% margin and
        # a full all-short book near 30% margin, while preserving compounding.
        try:
            total = float(self.wallets.get_total_stake_amount())
        except Exception:
            total = float(max_stake)
        fraction = 0.07 if side == "long" else 0.03
        stake = total * fraction
        if min_stake is not None and stake < float(min_stake):
            return 0.0
        return float(min(stake, max_stake))

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
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty or "atr" not in df.columns:
            return None

        atr = float(df.iloc[-1]["atr"])
        if not np.isfinite(atr) or atr <= 0 or current_rate <= 0:
            return None

        raw_distance = atr * float(self.atr_mult.value)
        cap_distance = current_rate * float(self.max_price_stop_pct.value)
        distance = min(raw_distance, cap_distance)

        if trade.is_short:
            # Freqtrade's custom stop can only tighten, so using the best price reached
            # yields the paper's monotonic short-side analogue.
            anchor = float(trade.min_rate or current_rate)
            absolute_stop = anchor + distance
            # A stale anchor can put the candidate below current rate after a rebound.
            absolute_stop = max(absolute_stop, current_rate * 1.0005)
        else:
            anchor = float(trade.max_rate or current_rate)
            absolute_stop = anchor - distance
            absolute_stop = min(absolute_stop, current_rate * 0.9995)

        return stoploss_from_absolute(
            absolute_stop,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
