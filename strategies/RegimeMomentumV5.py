from pathlib import Path
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame
import talib.abstract as ta

from freqtrade.persistence import Trade
from freqtrade.strategy import (
    IStrategy,
    IntParameter,
    DecimalParameter,
    informative,
    stoploss_from_absolute,
)


class RegimeMomentumV5(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = "15m"
    can_short = True
    startup_candle_count = 900

    minimal_roi = {}
    stoploss = -0.60
    use_custom_stoploss = True
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    process_only_new_candles = True

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    h6_mom_lb = IntParameter(6, 18, default=10, space="buy")
    h6_mom_thr = DecimalParameter(0.01, 0.08, default=0.03, decimals=3, space="buy")
    h6_adx_min = IntParameter(16, 35, default=22, space="buy")
    h1_breakout = IntParameter(12, 72, default=36, space="buy")
    h1_vol_mult = DecimalParameter(1.0, 2.2, default=1.2, decimals=2, space="buy")
    atr_stop_mult = DecimalParameter(1.5, 4.5, default=2.6, decimals=2, space="sell")
    profit_lock_start = DecimalParameter(0.05, 0.30, default=0.12, decimals=2, space="sell")
    oi_confirm = DecimalParameter(-0.01, 0.05, default=0.00, decimals=3, space="buy")
    max_abs_funding = DecimalParameter(0.0002, 0.0030, default=0.0010, decimals=4, space="buy")
    panic_return = DecimalParameter(0.015, 0.080, default=0.035, decimals=3, space="buy")
    panic_oi_drop = DecimalParameter(0.005, 0.08, default=0.025, decimals=3, space="buy")
    panic_liq_z = DecimalParameter(1.0, 5.0, default=2.0, decimals=1, space="buy")

    feature_db = os.environ.get("RMV5_FEATURE_DB", "/freqtrade/user_data/v5/features.sqlite")

    plot_config = {
        "main_plot": {"ema50_1h": {}, "ema200_1h": {}},
        "subplots": {
            "6h": {"mom_6h": {}, "adx_6h": {}},
            "Derivatives": {"oi_chg": {}, "funding": {}, "liq_z": {}},
        },
    }

    @informative("1h")
    def populate_indicators_1h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        w = int(self.h1_breakout.value)
        dataframe["breakout_high"] = dataframe["high"].rolling(w).max().shift(1)
        dataframe["breakout_low"] = dataframe["low"].rolling(w).min().shift(1)
        dataframe["vol_sma"] = dataframe["volume"].rolling(48).mean()
        dataframe["ret"] = dataframe["close"].pct_change()
        return dataframe

    @informative("6h")
    def populate_indicators_6h(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        lb = int(self.h6_mom_lb.value)
        dataframe["mom"] = dataframe["close"].pct_change(lb)
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=14)
        dataframe["ema50"] = ta.EMA(dataframe, timeperiod=50)
        dataframe["ema200"] = ta.EMA(dataframe, timeperiod=200)
        dataframe["rv"] = dataframe["close"].pct_change().rolling(20).std()
        dataframe["rv_med"] = dataframe["rv"].rolling(60).median()
        dataframe["high_vol"] = (dataframe["rv"] > dataframe["rv_med"] * 1.8).astype(int)
        dataframe["chop"] = (dataframe["adx"] < 18).astype(int)
        return dataframe

    def _load_external_features(self, pair: str, dataframe: DataFrame) -> DataFrame:
        out = dataframe.copy()
        for c in ["oi", "funding", "long_liq", "short_liq", "taker_ratio", "top_ls_ratio"]:
            out[c] = np.nan
        out["liq_observed"] = 0.0

        db = Path(self.feature_db)
        if not db.exists() or dataframe.empty:
            return out

        symbol = pair.split("/")[0].replace(":USDT", "") + "USDT"
        start_ms = int(pd.to_datetime(dataframe["date"].iloc[0], utc=True).timestamp() * 1000)
        end_ms = int(pd.to_datetime(dataframe["date"].iloc[-1], utc=True).timestamp() * 1000) + 15 * 60 * 1000

        try:
            with sqlite3.connect(db) as conn:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(features)")}
                liq_expr = "liq_observed" if "liq_observed" in cols else "1 AS liq_observed"
                feat = pd.read_sql_query(
                    f"""
                    SELECT bucket_ms AS ts,
                           oi,
                           funding_rate AS funding,
                           long_liq_usdt AS long_liq,
                           short_liq_usdt AS short_liq,
                           taker_ratio,
                           top_ls_ratio,
                           {liq_expr}
                    FROM features
                    WHERE symbol = ? AND bucket_ms BETWEEN ? AND ?
                    ORDER BY bucket_ms
                    """,
                    conn,
                    params=(symbol, start_ms, end_ms),
                )
        except Exception:
            return out

        if feat.empty:
            return out

        feat["date"] = pd.to_datetime(feat["ts"], unit="ms", utc=True)
        feat = feat.drop(columns=["ts"])
        out["date"] = pd.to_datetime(out["date"], utc=True)
        out = pd.merge_asof(
            out.sort_values("date"),
            feat.sort_values("date"),
            on="date",
            direction="backward",
            tolerance=pd.Timedelta("30min"),
            suffixes=("", "_ext"),
        )

        for c in ["oi", "funding", "long_liq", "short_liq", "taker_ratio", "top_ls_ratio", "liq_observed"]:
            ext = f"{c}_ext"
            if ext in out.columns:
                out[c] = out[ext]
                out.drop(columns=[ext], inplace=True)
        return out

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=14)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]
        dataframe["ret_1"] = dataframe["close"].pct_change()
        dataframe["vol_sma"] = dataframe["volume"].rolling(96).mean()
        dataframe = self._load_external_features(metadata["pair"], dataframe)

        dataframe["oi_chg"] = dataframe["oi"].pct_change(fill_method=None)
        liq_total = dataframe[["long_liq", "short_liq"]].fillna(0).sum(axis=1)
        liq_mean = liq_total.rolling(96).mean()
        liq_std = liq_total.rolling(96).std().replace(0, np.nan)
        dataframe["liq_z"] = (liq_total - liq_mean) / liq_std
        dataframe["liq_imbalance"] = (
            dataframe["short_liq"].fillna(0) - dataframe["long_liq"].fillna(0)
        ) / liq_total.replace(0, np.nan)
        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = None

        oi_ok = dataframe["oi_chg"].isna() | (dataframe["oi_chg"] >= float(self.oi_confirm.value))
        funding_ok = dataframe["funding"].isna() | (dataframe["funding"].abs() <= float(self.max_abs_funding.value))

        normal_long = (
            (dataframe["mom_6h"] > float(self.h6_mom_thr.value))
            & (dataframe["adx_6h"] >= int(self.h6_adx_min.value))
            & (dataframe["ema50_6h"] > dataframe["ema200_6h"])
            & (dataframe["close_1h"] > dataframe["breakout_high_1h"])
            & (dataframe["volume_1h"] > dataframe["vol_sma_1h"] * float(self.h1_vol_mult.value))
            & (dataframe["high_vol_6h"] == 0)
            & oi_ok
            & funding_ok
            & (dataframe["volume"] > 0)
        )

        normal_short = (
            (dataframe["mom_6h"] < -float(self.h6_mom_thr.value))
            & (dataframe["adx_6h"] >= int(self.h6_adx_min.value))
            & (dataframe["ema50_6h"] < dataframe["ema200_6h"])
            & (dataframe["close_1h"] < dataframe["breakout_low_1h"])
            & (dataframe["volume_1h"] > dataframe["vol_sma_1h"] * float(self.h1_vol_mult.value))
            & (dataframe["high_vol_6h"] == 0)
            & oi_ok
            & funding_ok
            & (dataframe["volume"] > 0)
        )

        observed = dataframe["liq_observed"].fillna(0) >= 0.5
        live_panic_long = (
            observed
            & (dataframe["liq_z"] >= float(self.panic_liq_z.value))
            & (dataframe["liq_imbalance"] > 0.35)
            & (dataframe["oi_chg"] <= -float(self.panic_oi_drop.value))
            & (dataframe["ret_1"] >= float(self.panic_return.value))
        )
        live_panic_short = (
            observed
            & (dataframe["liq_z"] >= float(self.panic_liq_z.value))
            & (dataframe["liq_imbalance"] < -0.35)
            & (dataframe["oi_chg"] <= -float(self.panic_oi_drop.value))
            & (dataframe["ret_1"] <= -float(self.panic_return.value))
        )

        proxy = ~observed
        proxy_panic_long = (
            proxy
            & (dataframe["oi_chg"] <= -float(self.panic_oi_drop.value))
            & (dataframe["ret_1"] >= float(self.panic_return.value))
            & (dataframe["volume"] > dataframe["vol_sma"] * 2.0)
            & (dataframe["taker_ratio"].isna() | (dataframe["taker_ratio"] > 1.05))
        )
        proxy_panic_short = (
            proxy
            & (dataframe["oi_chg"] <= -float(self.panic_oi_drop.value))
            & (dataframe["ret_1"] <= -float(self.panic_return.value))
            & (dataframe["volume"] > dataframe["vol_sma"] * 2.0)
            & (dataframe["taker_ratio"].isna() | (dataframe["taker_ratio"] < 0.95))
        )

        dataframe.loc[normal_long, ["enter_long", "enter_tag"]] = (1, "normal_momentum_long")
        dataframe.loc[normal_short, ["enter_short", "enter_tag"]] = (1, "normal_momentum_short")
        dataframe.loc[live_panic_long | proxy_panic_long, ["enter_long", "enter_tag"]] = (1, "panic_long")
        dataframe.loc[live_panic_short | proxy_panic_short, ["enter_short", "enter_tag"]] = (1, "panic_short")
        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = None

        long_invalid = (
            (dataframe["mom_6h"] < 0)
            | (dataframe["ema50_6h"] < dataframe["ema200_6h"])
            | (dataframe["close_1h"] < dataframe["ema50_1h"])
        )
        short_invalid = (
            (dataframe["mom_6h"] > 0)
            | (dataframe["ema50_6h"] > dataframe["ema200_6h"])
            | (dataframe["close_1h"] > dataframe["ema50_1h"])
        )

        dataframe.loc[long_invalid, ["exit_long", "exit_tag"]] = (1, "regime_invalid_long")
        dataframe.loc[short_invalid, ["exit_short", "exit_tag"]] = (1, "regime_invalid_short")
        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        tag = trade.enter_tag or ""
        age = current_time - trade.open_date_utc
        if tag.startswith("panic_"):
            if age.total_seconds() >= 6 * 3600:
                return "panic_timeout"
            if age.total_seconds() >= 30 * 60 and current_profit > 0.20:
                return "panic_takeprofit"
        return None

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
        if os.environ.get("RMV5_FORCE_20X", "false").lower() in {"1", "true", "yes", "on"}:
            return min(20.0, max_leverage)

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if df is None or df.empty or "atr" not in df.columns:
            return min(3.0, max_leverage)

        atr = float(df.iloc[-1]["atr"])
        stop_price_pct = max((atr * float(self.atr_stop_mult.value)) / max(current_rate, 1e-12), 0.002)
        target_margin_risk = 0.10
        lev = target_margin_risk / stop_price_pct
        return float(min(max(1.0, lev), 20.0, max_leverage))

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
        if not np.isfinite(atr) or atr <= 0:
            return None

        mult = float(self.atr_stop_mult.value)
        if (trade.enter_tag or "").startswith("panic_"):
            mult = max(1.2, mult * 0.65)

        if trade.is_short:
            absolute_stop = current_rate + atr * mult
        else:
            absolute_stop = current_rate - atr * mult

        if current_profit >= float(self.profit_lock_start.value):
            lock_distance = atr * max(0.8, mult * 0.45)
            if trade.is_short:
                absolute_stop = min(absolute_stop, current_rate + lock_distance)
            else:
                absolute_stop = max(absolute_stop, current_rate - lock_distance)

        return stoploss_from_absolute(
            absolute_stop,
            current_rate=current_rate,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )
