from __future__ import annotations

import os
from datetime import datetime

import numpy as np
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import IStrategy, informative, stoploss_from_absolute


class AdaptivePerp15mV7(IStrategy):
    """V7-Core from the deep-research design.

    P1 intentionally contains only two price/volume alpha engines:
      1. multi-hour trend continuation
      2. pullback / reclaim continuation

    Both long and short scores are always computed. 1h/4h context changes
    confidence continuously and never hard-disables a direction. OI, funding,
    order-flow, liquidation and depth modules are added only after this price
    core passes validation.
    """

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True
    startup_candle_count = 800
    process_only_new_candles = True

    minimal_roi = {}
    # Safety backstop. The normal initial stop is ATR/structure-derived and tighter.
    stoploss = -0.60
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    _equity_hwm: float | None = None

    @staticmethod
    def _env_float(name: str, default: float) -> float:
        try:
            return float(os.environ.get(name, str(default)))
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _env_flag(name: str, default: bool = True) -> bool:
        raw = os.environ.get(name)
        if raw is None:
            return default
        return raw.lower() in {"1", "true", "yes", "on"}

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -12.0, 12.0)
        return 1.0 / (1.0 + np.exp(-x))

    @property
    def protections(self):
        if not self._env_flag("RMV7_ENABLE_PROTECTIONS", True):
            return []
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 1,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 32,
                "trade_limit": 4,
                "stop_duration_candles": 8,
                "required_profit": 0.0,
                "only_per_pair": False,
                "only_per_side": False,
            },
            {
                "method": "MaxDrawdown",
                "calculation_mode": "equity",
                "lookback_period_candles": 96,
                "trade_limit": 10,
                "stop_duration_candles": 16,
                "max_allowed_drawdown": 0.15,
            },
        ]

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema24"] = ta.EMA(dataframe, timeperiod=24)
        dataframe["ema72"] = ta.EMA(dataframe, timeperiod=72)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=20)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ret4"] = dataframe["close"].pct_change(4)
        dataframe["ret12"] = dataframe["close"].pct_change(12)
        dataframe["ema24_slope6"] = dataframe["ema24"].pct_change(6)
        return dataframe

    @informative("4h")
    def populate_indicators_4h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["ema18"] = ta.EMA(dataframe, timeperiod=18)
        dataframe["ema54"] = ta.EMA(dataframe, timeperiod=54)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=20)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ret6"] = dataframe["close"].pct_change(6)
        dataframe["ret18"] = dataframe["close"].pct_change(18)
        dataframe["ema18_slope4"] = dataframe["ema18"].pct_change(4)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        eps = 1e-12

        dataframe["ema20"] = ta.EMA(dataframe, timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=20)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"].replace(0, np.nan)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=14)
        dataframe["ret1"] = dataframe["close"].pct_change()
        dataframe["ret4"] = dataframe["close"].pct_change(4)
        dataframe["ret16"] = dataframe["close"].pct_change(16)

        # Past-only local structure. Shifted so the signal candle itself cannot
        # define the level it is compared with.
        dataframe["donch_high"] = dataframe["high"].rolling(32).max().shift(1)
        dataframe["donch_low"] = dataframe["low"].rolling(32).min().shift(1)
        dataframe["swing_low"] = dataframe["low"].rolling(16).min().shift(1)
        dataframe["swing_high"] = dataframe["high"].rolling(16).max().shift(1)

        donch_range = (dataframe["donch_high"] - dataframe["donch_low"]).replace(0, np.nan)
        dataframe["donch_pos"] = (
            (dataframe["close"] - dataframe["donch_low"]) / donch_range
        ).clip(0.0, 1.0)

        # Past/current closed-candle rolling normalisation.
        logv = np.log1p(dataframe["volume"].clip(lower=0))
        vol_med = logv.rolling(96, min_periods=48).median()
        vol_dev = (logv - vol_med).abs()
        vol_mad = vol_dev.rolling(96, min_periods=48).median()
        dataframe["volume_z"] = (
            (logv - vol_med) / (1.4826 * vol_mad).replace(0, np.nan)
        ).clip(-5, 5)

        atr_med = dataframe["atr_pct"].rolling(192, min_periods=96).median()
        dataframe["vol_ratio"] = dataframe["atr_pct"] / atr_med.replace(0, np.nan)
        dataframe["vol_stress"] = (
            (dataframe["vol_ratio"] - 1.0) / 1.5
        ).clip(0.0, 1.0).fillna(0.0)

        # @informative merges only completed higher-timeframe candles.
        atr1 = dataframe["atr_1h"].replace(0, np.nan)
        atr4 = dataframe["atr_4h"].replace(0, np.nan)
        atrp1 = dataframe["atr_pct_1h"].replace(0, np.nan)
        atrp4 = dataframe["atr_pct_4h"].replace(0, np.nan)

        dataframe["trend_dir_1h"] = np.tanh(
            (dataframe["ema24_1h"] - dataframe["ema72_1h"]) / (2.0 * atr1)
        )
        dataframe["trend_dir_4h"] = np.tanh(
            (dataframe["ema18_4h"] - dataframe["ema54_4h"]) / (2.5 * atr4)
        )
        dataframe["momentum_1h"] = np.tanh(
            dataframe["ret4_1h"] / (2.5 * atrp1 + eps)
        )
        dataframe["momentum_4h"] = np.tanh(
            dataframe["ret6_4h"] / (3.0 * atrp4 + eps)
        )
        dataframe["trend_signed"] = (
            0.38 * dataframe["trend_dir_1h"]
            + 0.32 * dataframe["trend_dir_4h"]
            + 0.18 * dataframe["momentum_1h"]
            + 0.12 * dataframe["momentum_4h"]
        ).clip(-1.0, 1.0)
        dataframe["trend_strength"] = dataframe["trend_signed"].abs()
        dataframe["adx_quality"] = (
            0.60 * ((dataframe["adx_1h"] - 15.0) / 25.0).clip(0.0, 1.0)
            + 0.40 * ((dataframe["adx_4h"] - 15.0) / 25.0).clip(0.0, 1.0)
        )

        mom15 = np.tanh(
            dataframe["ret4"] / (2.0 * dataframe["atr_pct"].replace(0, np.nan) + eps)
        )
        donch_signed = (2.0 * dataframe["donch_pos"] - 1.0).fillna(0.0)
        dataframe["continuation_signed"] = (
            0.55 * mom15.fillna(0.0) + 0.45 * donch_signed
        ).clip(-1.0, 1.0)
        dataframe["volume_quality"] = (
            (dataframe["volume_z"].fillna(0.0) + 1.5) / 3.0
        ).clip(0.0, 1.0)

        # Alpha engine 1: trend continuation.
        common_quality = (
            0.65 * dataframe["adx_quality"]
            + 0.18 * dataframe["volume_quality"]
            - 0.45 * dataframe["vol_stress"]
            - 0.40
        )
        trend_raw_long = (
            2.00 * dataframe["trend_signed"]
            + 0.45 * dataframe["continuation_signed"]
            + common_quality
        )
        trend_raw_short = (
            -2.00 * dataframe["trend_signed"]
            - 0.45 * dataframe["continuation_signed"]
            + common_quality
        )
        dataframe["trend_long_score"] = self._sigmoid(trend_raw_long)
        dataframe["trend_short_score"] = self._sigmoid(trend_raw_short)

        # Alpha engine 2: trend pullback + reclaim near 1h dynamic value.
        value_atr_low = (dataframe["low"] - dataframe["ema24_1h"]) / atr1
        value_atr_high = (dataframe["high"] - dataframe["ema24_1h"]) / atr1
        recent_low_dev = value_atr_low.rolling(6, min_periods=2).min()
        recent_high_dev = value_atr_high.rolling(6, min_periods=2).max()

        cross_up = (
            (dataframe["close"] > dataframe["ema20"])
            & (dataframe["close"].shift(1) <= dataframe["ema20"].shift(1) * 1.004)
            & (dataframe["close"] > dataframe["open"])
        )
        cross_down = (
            (dataframe["close"] < dataframe["ema20"])
            & (dataframe["close"].shift(1) >= dataframe["ema20"].shift(1) * 0.996)
            & (dataframe["close"] < dataframe["open"])
        )
        dataframe["pull_event_long"] = (
            recent_low_dev.between(-2.0, 0.60)
            & cross_up
            & (dataframe["ret4"] > -0.02)
        ).astype(int)
        dataframe["pull_event_short"] = (
            recent_high_dev.between(-0.60, 2.0)
            & cross_down
            & (dataframe["ret4"] < 0.02)
        ).astype(int)

        pull_raw_long = (
            1.80 * dataframe["trend_signed"]
            + 0.45 * dataframe["adx_quality"]
            + 0.55 * mom15.fillna(0.0)
            + 0.20 * dataframe["volume_quality"]
            - 0.25 * dataframe["vol_stress"]
            - 0.25
        )
        pull_raw_short = (
            -1.80 * dataframe["trend_signed"]
            + 0.45 * dataframe["adx_quality"]
            - 0.55 * mom15.fillna(0.0)
            + 0.20 * dataframe["volume_quality"]
            - 0.25 * dataframe["vol_stress"]
            - 0.25
        )
        pull_long_active = self._sigmoid(pull_raw_long)
        pull_short_active = self._sigmoid(pull_raw_short)
        # 0.5 is neutral evidence when a pullback event is absent.
        dataframe["pull_long_score"] = np.where(
            dataframe["pull_event_long"] > 0, pull_long_active, 0.5
        )
        dataframe["pull_short_score"] = np.where(
            dataframe["pull_event_short"] > 0, pull_short_active, 0.5
        )

        # Soft meta-layer. State adjusts weights rather than banning a side.
        dataframe["w_trend"] = (
            1.00
            + 0.30 * dataframe["trend_strength"]
            - 0.15 * dataframe["vol_stress"]
        ).clip(0.70, 1.35)
        dataframe["w_pull"] = (
            0.75
            + 0.35 * dataframe["trend_strength"]
            + 0.15 * (1.0 - dataframe["vol_stress"])
        ).clip(0.65, 1.25)
        wsum = (dataframe["w_trend"] + dataframe["w_pull"]).replace(0, np.nan)
        dataframe["long_score"] = (
            dataframe["w_trend"] * dataframe["trend_long_score"]
            + dataframe["w_pull"] * dataframe["pull_long_score"]
        ) / wsum
        dataframe["short_score"] = (
            dataframe["w_trend"] * dataframe["trend_short_score"]
            + dataframe["w_pull"] * dataframe["pull_short_score"]
        ) / wsum
        dataframe["confidence"] = dataframe[["long_score", "short_score"]].max(axis=1)
        dataframe["score_gap"] = (dataframe["long_score"] - dataframe["short_score"]).abs()

        # P1 execution-quality proxy. Spread/depth arrives in later research stages.
        dataframe["eligible_core"] = (
            (dataframe["volume"] > 0)
            & dataframe["atr_pct"].between(0.0015, 0.060)
            & dataframe["ret1"].abs().lt(0.12)
            & dataframe["long_score"].notna()
            & dataframe["short_score"].notna()
        ).astype(int)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        threshold = self._env_float("RMV7_ENTRY_THRESHOLD", 0.64)
        gap = self._env_float("RMV7_SCORE_GAP", 0.08)
        long_signal = (
            (dataframe["eligible_core"] > 0)
            & (dataframe["long_score"] >= threshold)
            & ((dataframe["long_score"] - dataframe["short_score"]) >= gap)
        )
        short_signal = (
            (dataframe["eligible_core"] > 0)
            & (dataframe["short_score"] >= threshold)
            & ((dataframe["short_score"] - dataframe["long_score"]) >= gap)
        )
        long_pull = long_signal & (dataframe["pull_event_long"] > 0)
        short_pull = short_signal & (dataframe["pull_event_short"] > 0)

        dataframe.loc[long_signal & ~long_pull, ["enter_long", "enter_tag"]] = (1, "v7_trend_long")
        dataframe.loc[short_signal & ~short_pull, ["enter_short", "enter_tag"]] = (1, "v7_trend_short")
        dataframe.loc[long_pull, ["enter_long", "enter_tag"]] = (1, "v7_pullback_long")
        dataframe.loc[short_pull, ["enter_short", "enter_tag"]] = (1, "v7_pullback_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # No old-style "regime changed => market exit".
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None
        return dataframe

    def _last_candle(self, pair: str):
        try:
            df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if df is None or df.empty:
                return None
            return df.iloc[-1]
        except Exception:
            return None

    def _score_for_side(self, candle, side: str) -> float:
        if candle is None:
            return 0.5
        key = "short_score" if side == "short" else "long_score"
        try:
            value = float(candle.get(key, 0.5))
            return value if np.isfinite(value) else 0.5
        except Exception:
            return 0.5

    def _initial_stop_distance(self, current_rate: float, candle, side: str) -> float:
        floor = self._env_float("RMV7_STOP_FLOOR", 0.006)
        cap = self._env_float("RMV7_STOP_CAP", 0.040)
        atr_mult = self._env_float("RMV7_ATR_STOP_MULT", 1.8)
        structure_buffer_atr = self._env_float("RMV7_STRUCTURE_BUFFER_ATR", 0.15)
        if candle is None:
            return min(max(0.015, floor), cap)

        try:
            atr = float(candle.get("atr", np.nan))
        except Exception:
            atr = np.nan
        if not np.isfinite(atr) or atr <= 0 or current_rate <= 0:
            return min(max(0.015, floor), cap)

        atr_dist = atr_mult * atr / current_rate
        struct_dist = 0.0
        if side == "short":
            try:
                swing = float(candle.get("swing_high", np.nan))
            except Exception:
                swing = np.nan
            if np.isfinite(swing) and swing > current_rate:
                struct_dist = (
                    (swing - current_rate) / current_rate
                    + structure_buffer_atr * atr / current_rate
                )
        else:
            try:
                swing = float(candle.get("swing_low", np.nan))
            except Exception:
                swing = np.nan
            if np.isfinite(swing) and swing < current_rate:
                struct_dist = (
                    (current_rate - swing) / current_rate
                    + structure_buffer_atr * atr / current_rate
                )
        return float(min(max(atr_dist, struct_dist, floor), cap))

    def _initial_stop_price(self, current_rate: float, candle, side: str) -> float:
        d = self._initial_stop_distance(current_rate, candle, side)
        if side == "short":
            return current_rate * (1.0 + d)
        return current_rate * (1.0 - d)

    def _equity(self) -> float:
        try:
            value = float(self.wallets.get_total_stake_amount())
            if np.isfinite(value) and value > 0:
                return value
        except Exception:
            pass
        try:
            return float(self.config.get("dry_run_wallet", 100.0))
        except Exception:
            return 100.0

    def _drawdown_multiplier(self, equity: float) -> float:
        if self._equity_hwm is None or not np.isfinite(self._equity_hwm):
            self._equity_hwm = max(equity, 1e-9)
        self._equity_hwm = max(float(self._equity_hwm), equity)
        dd = max(0.0, 1.0 - equity / max(float(self._equity_hwm), 1e-9))
        if dd >= 0.30:
            return 0.25
        if dd >= 0.20:
            return 0.50
        if dd >= 0.12:
            return 0.75
        return 1.00

    def _open_heat(self, side: str) -> tuple[float, float]:
        total = 0.0
        same_side = 0.0
        default_risk = self._env_float("RMV7_DEFAULT_OPEN_RISK", 0.015)
        try:
            trades = Trade.get_trades_proxy(is_open=True)
        except Exception:
            return 0.0, 0.0

        for trade in trades:
            try:
                risk = float(trade.get_custom_data(key="v7_risk_fraction", default=default_risk))
            except Exception:
                risk = default_risk
            if not np.isfinite(risk) or risk <= 0:
                risk = default_risk
            total += risk
            trade_side = "short" if getattr(trade, "is_short", False) else "long"
            if trade_side == side:
                same_side += risk
        return total, same_side

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
        candle = self._last_candle(pair)
        threshold = self._env_float("RMV7_ENTRY_THRESHOLD", 0.64)
        confidence = self._score_for_side(candle, side)
        conf_norm = np.clip((confidence - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0)

        lev_floor = self._env_float("RMV7_LEVERAGE_MIN", 3.0)
        lev_cap = self._env_float("RMV7_LEVERAGE_MAX", 10.0)
        base = lev_floor + (lev_cap - lev_floor) * conf_norm

        vol_stress = 0.0
        if candle is not None:
            try:
                vol_stress = float(candle.get("vol_stress", 0.0))
            except Exception:
                vol_stress = 0.0
        vol_stress = float(np.clip(vol_stress, 0.0, 1.0))
        dd_mult = self._drawdown_multiplier(self._equity())

        target = base * (1.0 - 0.35 * vol_stress) * max(0.50, dd_mult)
        target = float(np.clip(target, lev_floor, lev_cap))
        return float(min(target, max_leverage))

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
        candle = self._last_candle(pair)
        threshold = self._env_float("RMV7_ENTRY_THRESHOLD", 0.64)
        confidence = self._score_for_side(candle, side)
        conf_norm = float(np.clip((confidence - threshold) / max(1.0 - threshold, 1e-9), 0.0, 1.0))

        equity = self._equity()
        dd_mult = self._drawdown_multiplier(equity)
        risk_min = self._env_float("RMV7_RISK_MIN", 0.0075)
        risk_max = self._env_float("RMV7_RISK_MAX", 0.0200)
        gamma = self._env_float("RMV7_RISK_GAMMA", 1.6)
        raw_risk = (risk_min + (risk_max - risk_min) * (conf_norm ** gamma)) * dd_mult

        heat_cap = self._env_float("RMV7_PORTFOLIO_HEAT", 0.08)
        side_cap = self._env_float("RMV7_SIDE_HEAT", 0.05)
        used_heat, used_side_heat = self._open_heat(side)
        allowed = min(heat_cap - used_heat, side_cap - used_side_heat)
        risk_fraction = min(raw_risk, max(0.0, allowed))

        if risk_fraction < self._env_float("RMV7_MIN_EFFECTIVE_RISK", 0.004):
            return 0.0

        stop_dist = self._initial_stop_distance(current_rate, candle, side)
        lev = max(float(leverage), 1.0)
        # notional = equity*risk/price_stop_distance; Freqtrade stake is collateral.
        stake = equity * risk_fraction / max(stop_dist * lev, 1e-9)
        collateral_cap = self._env_float("RMV7_COLLATERAL_CAP", 0.25)
        stake = min(stake, equity * collateral_cap, float(max_stake))

        if min_stake is not None and stake < float(min_stake):
            return 0.0
        return float(max(stake, 0.0))

    def order_filled(
        self,
        pair: str,
        trade: Trade,
        order,
        current_time: datetime,
        **kwargs,
    ) -> None:
        try:
            is_first_entry = (
                trade.nr_of_successful_entries == 1
                and order.ft_order_side == trade.entry_side
            )
        except Exception:
            is_first_entry = False
        if not is_first_entry:
            return None

        candle = self._last_candle(pair)
        side = "short" if trade.is_short else "long"
        stop_abs = self._initial_stop_price(float(trade.open_rate), candle, side)
        stop_dist = abs(float(trade.open_rate) - stop_abs) / max(float(trade.open_rate), 1e-12)
        trade.set_custom_data(key="v7_initial_stop_abs", value=float(stop_abs))
        trade.set_custom_data(key="v7_initial_stop_pct", value=float(stop_dist))
        trade.set_custom_data(key="v7_entry_score", value=float(self._score_for_side(candle, side)))

        equity = self._equity()
        try:
            actual_risk = (
                float(trade.stake_amount)
                * float(trade.leverage)
                * stop_dist
                / max(equity, 1e-9)
            )
        except Exception:
            actual_risk = self._env_float("RMV7_DEFAULT_OPEN_RISK", 0.015)
        trade.set_custom_data(key="v7_risk_fraction", value=float(actual_risk))
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
        candle = self._last_candle(pair)
        side = "short" if trade.is_short else "long"
        stop_abs = trade.get_custom_data(key="v7_initial_stop_abs", default=None)
        stop_pct = trade.get_custom_data(key="v7_initial_stop_pct", default=None)

        if stop_abs is None or stop_pct is None:
            stop_abs = self._initial_stop_price(float(trade.open_rate), candle, side)
            stop_pct = abs(float(trade.open_rate) - float(stop_abs)) / max(float(trade.open_rate), 1e-12)
            trade.set_custom_data(key="v7_initial_stop_abs", value=float(stop_abs))
            trade.set_custom_data(key="v7_initial_stop_pct", value=float(stop_pct))

        stop_abs = float(stop_abs)
        stop_pct = max(float(stop_pct), 1e-6)
        if side == "short":
            price_move = (float(trade.open_rate) - current_rate) / max(float(trade.open_rate), 1e-12)
        else:
            price_move = (current_rate - float(trade.open_rate)) / max(float(trade.open_rate), 1e-12)
        r_multiple = price_move / stop_pct

        atr = np.nan
        swing = np.nan
        score = self._score_for_side(candle, side)
        if candle is not None:
            try:
                atr = float(candle.get("atr", np.nan))
                swing = float(candle.get("swing_high" if trade.is_short else "swing_low", np.nan))
            except Exception:
                pass

        if r_multiple >= 1.0:
            if trade.is_short:
                stop_abs = min(stop_abs, float(trade.open_rate) * 0.9992)
            else:
                stop_abs = max(stop_abs, float(trade.open_rate) * 1.0008)

        # Preserve right-tail winners: trailing only starts after 1.5 initial R.
        if np.isfinite(atr) and atr > 0 and r_multiple >= 1.5:
            trail_mult = 2.5 if r_multiple < 3.0 else 2.0
            trail_dist = max(atr * trail_mult, current_rate * 0.006)
            if trade.is_short:
                stop_abs = min(stop_abs, current_rate + trail_dist)
                if np.isfinite(swing) and swing > current_rate:
                    stop_abs = min(stop_abs, swing + 0.15 * atr)
            else:
                stop_abs = max(stop_abs, current_rate - trail_dist)
                if np.isfinite(swing) and swing < current_rate:
                    stop_abs = max(stop_abs, swing - 0.15 * atr)

        # Score deterioration only tightens a profitable trade. It never market-exits
        # merely because a regime indicator flipped.
        if np.isfinite(atr) and atr > 0 and r_multiple >= 0.5 and score < 0.45:
            if trade.is_short:
                stop_abs = min(stop_abs, current_rate + 1.6 * atr)
            else:
                stop_abs = max(stop_abs, current_rate - 1.6 * atr)

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
        if not self._env_flag("RMV7_ENABLE_TIME_EXIT", True):
            return None

        age_hours = (current_time - trade.open_date_utc).total_seconds() / 3600.0
        candle = self._last_candle(pair)
        side = "short" if trade.is_short else "long"
        score = self._score_for_side(candle, side)

        # Multi-hour signals that never develop stop consuming portfolio heat.
        # Strong trends remain open and are managed by the structural trail.
        if age_hours >= 12.0 and score < 0.45 and current_profit < 0.05:
            return "v7_time_decay_12h"
        if age_hours >= 36.0 and score < 0.55 and current_profit < 0.15:
            return "v7_stale_36h"
        return None
