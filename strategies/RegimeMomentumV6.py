from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute


class RegimeMomentumV6(IStrategy):
    """Aggressive price-only research strategy for 2026 testing.

    Architecture:
      - 4h trend/regime filter
      - 1h breakout + pullback-reclaim entries
      - fixed 10x isolated leverage by default
      - ATR/volatility-based account-risk sizing
      - entry-anchored initial stop with profit locking / trailing

    This intentionally does not use the derivatives feature DB yet. The goal of the
    first V6 test is to establish whether the faster price-only core has edge before
    reintroducing OI/funding/liquidation filters.
    """

    INTERFACE_VERSION = 3
    timeframe = "1h"
    can_short = True
    startup_candle_count = 300
    process_only_new_candles = True

    minimal_roi = {}
    stoploss = -0.40
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema60"] = ta.EMA(dataframe, timeperiod=60)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["ret3"] = dataframe["close"].pct_change(3)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema12"] = ta.EMA(dataframe, timeperiod=12)
        dataframe["ema36"] = ta.EMA(dataframe, timeperiod=36)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["ret1"] = dataframe["close"].pct_change()
        dataframe["mom3"] = dataframe["close"].pct_change(3)
        dataframe["vol_sma"] = dataframe["volume"].rolling(24).mean()
        dataframe["breakout_high"] = dataframe["high"].rolling(12).max().shift(1)
        dataframe["breakout_low"] = dataframe["low"].rolling(12).min().shift(1)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        liquid = (
            (dataframe["volume"] > 0)
            & dataframe["atr_pct"].between(0.0025, 0.08)
            & dataframe["ret1"].abs().lt(0.10)
        )

        bull = (
            (dataframe["ema20_4h"] > dataframe["ema60_4h"])
            & (dataframe["close_4h"] > dataframe["ema20_4h"])
            & (dataframe["adx_4h"] >= 17)
            & (dataframe["rsi_4h"] >= 48)
            & (dataframe["ret3_4h"] > -0.03)
        )
        bear = (
            (dataframe["ema20_4h"] < dataframe["ema60_4h"])
            & (dataframe["close_4h"] < dataframe["ema20_4h"])
            & (dataframe["adx_4h"] >= 17)
            & (dataframe["rsi_4h"] <= 52)
            & (dataframe["ret3_4h"] < 0.03)
        )

        base_long = (
            (dataframe["ema12"] > dataframe["ema36"])
            & (dataframe["adx"] >= 16)
            & dataframe["rsi"].between(50, 79)
            & (dataframe["mom3"] > 0)
        )
        base_short = (
            (dataframe["ema12"] < dataframe["ema36"])
            & (dataframe["adx"] >= 16)
            & dataframe["rsi"].between(21, 50)
            & (dataframe["mom3"] < 0)
        )

        breakout_long = (
            bull
            & base_long
            & (dataframe["close"] > dataframe["breakout_high"])
            & (dataframe["volume"] >= dataframe["vol_sma"] * 1.05)
            & liquid
        )
        breakout_short = (
            bear
            & base_short
            & (dataframe["close"] < dataframe["breakout_low"])
            & (dataframe["volume"] >= dataframe["vol_sma"] * 1.05)
            & liquid
        )

        reclaim_long = (
            bull
            & base_long
            & (dataframe["close"].shift(1) <= dataframe["ema12"].shift(1) * 1.003)
            & (dataframe["close"] > dataframe["ema12"])
            & (dataframe["close"] > dataframe["open"])
            & (dataframe["volume"] >= dataframe["vol_sma"] * 0.85)
            & liquid
        )
        reclaim_short = (
            bear
            & base_short
            & (dataframe["close"].shift(1) >= dataframe["ema12"].shift(1) * 0.997)
            & (dataframe["close"] < dataframe["ema12"])
            & (dataframe["close"] < dataframe["open"])
            & (dataframe["volume"] >= dataframe["vol_sma"] * 0.85)
            & liquid
        )

        dataframe.loc[breakout_long, ["enter_long", "enter_tag"]] = (1, "v6_breakout_long")
        dataframe.loc[breakout_short, ["enter_short", "enter_tag"]] = (1, "v6_breakout_short")
        dataframe.loc[reclaim_long & ~breakout_long, ["enter_long", "enter_tag"]] = (1, "v6_reclaim_long")
        dataframe.loc[reclaim_short & ~breakout_short, ["enter_short", "enter_tag"]] = (1, "v6_reclaim_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None

        long_invalid = (
            (dataframe["ema12"] < dataframe["ema36"])
            | (dataframe["ema20_4h"] < dataframe["ema60_4h"])
            | ((dataframe["rsi"] < 43) & (dataframe["close"] < dataframe["ema12"]))
        )
        short_invalid = (
            (dataframe["ema12"] > dataframe["ema36"])
            | (dataframe["ema20_4h"] > dataframe["ema60_4h"])
            | ((dataframe["rsi"] > 57) & (dataframe["close"] > dataframe["ema12"]))
        )

        dataframe.loc[long_invalid, ["exit_long", "exit_tag"]] = (1, "v6_regime_exit_long")
        dataframe.loc[short_invalid, ["exit_short", "exit_tag"]] = (1, "v6_regime_exit_short")
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
        requested = float(os.environ.get("RMV6_LEVERAGE", "10"))
        return float(min(max(requested, 1.0), max_leverage))

    def _current_atr_pct(self, pair: str, current_rate: float) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty or "atr" not in df.columns:
            return 0.015
        atr = float(df.iloc[-1]["atr"])
        if not np.isfinite(atr) or atr <= 0:
            return 0.015
        return float(atr / max(float(current_rate), 1e-12))

    def _initial_price_stop_pct(self, pair: str, current_rate: float) -> float:
        atr_pct = self._current_atr_pct(pair, current_rate)
        mult = float(os.environ.get("RMV6_ATR_STOP_MULT", "2.2"))
        return float(min(max(atr_pct * mult, 0.010), 0.035))

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
        try:
            equity = float(self.wallets.get_total_stake_amount())
        except Exception:
            equity = float(max_stake)

        risk_fraction = float(os.environ.get("RMV6_ACCOUNT_RISK", "0.03"))
        collateral_cap = float(os.environ.get("RMV6_COLLATERAL_CAP", "0.20"))
        stop_pct = self._initial_price_stop_pct(pair, current_rate)
        lev = max(float(leverage), 1.0)

        # Approximate account loss at initial stop:
        # collateral * leverage * adverse_price_move ~= equity * risk_fraction.
        stake = equity * risk_fraction / max(stop_pct * lev, 1e-9)
        stake = min(stake, equity * collateral_cap, float(max_stake))
        return float(max(stake, 0.0))

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
        stop_pct = trade.get_custom_data(key="v6_initial_stop_pct")
        if stop_pct is None:
            stop_pct = self._initial_price_stop_pct(pair, trade.open_rate)
            trade.set_custom_data(key="v6_initial_stop_pct", value=float(stop_pct))
        stop_pct = float(stop_pct)

        if trade.is_short:
            absolute_stop = trade.open_rate * (1.0 + stop_pct)
        else:
            absolute_stop = trade.open_rate * (1.0 - stop_pct)

        atr_pct = self._current_atr_pct(pair, current_rate)

        # current_profit is leverage-aware in futures mode. Lock profit early enough to
        # avoid turning every fast 1h move into another large leveraged stop.
        if current_profit >= 0.12:
            if trade.is_short:
                absolute_stop = min(absolute_stop, trade.open_rate * 0.999)
            else:
                absolute_stop = max(absolute_stop, trade.open_rate * 1.001)

        if current_profit >= 0.22:
            if trade.is_short:
                absolute_stop = min(absolute_stop, trade.open_rate * 0.997)
            else:
                absolute_stop = max(absolute_stop, trade.open_rate * 1.003)

        if current_profit >= 0.35:
            trail_pct = min(max(atr_pct * 1.25, 0.007), 0.020)
            if trade.is_short:
                absolute_stop = min(absolute_stop, current_rate * (1.0 + trail_pct))
            else:
                absolute_stop = max(absolute_stop, current_rate * (1.0 - trail_pct))

        return stoploss_from_absolute(
            absolute_stop,
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
        age_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        if age_hours >= 72 and current_profit < 0.05:
            return "v6_stale_72h"
        if age_hours >= 168:
            return "v6_timeout_7d"
        return None
