import os
from typing import Tuple

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


class VolumeProfileMeanReversionV0(IStrategy):
    """
    Independent replication of the rules described in:
    L. N. H. Perera (2026), "Volume Profile Mean Reversion Strategy
    with Tape Speed Confirmation for Cryptocurrency Futures Markets".

    Important deviations from the paper are intentional and remove known bias:
    - Real Binance futures candles are used by the backtest script.
    - The current-day POC is DEVELOPING: at every 5m candle it is calculated
      only from candles available up to that point. No full-day lookahead.
    - Signals execute according to normal Freqtrade next-candle semantics.
    - Leverage is fixed at 1x while we test whether the underlying edge exists.
    """

    INTERFACE_VERSION = 3

    timeframe = "5m"
    can_short = True
    startup_candle_count = 300
    process_only_new_candles = True

    minimal_roi = {}
    stoploss = -float(os.environ.get("VPMR_STOPLOSS", "0.02"))
    trailing_stop = False
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    profile_bins = 50
    value_area_fraction = 0.70
    tape_threshold = 0.50

    @staticmethod
    def _volume_profile(
        lows: np.ndarray,
        highs: np.ndarray,
        closes: np.ndarray,
        volumes: np.ndarray,
        bins: int,
        value_area_fraction: float,
    ) -> Tuple[float, float, float]:
        """Return (POC, VAL, VAH) using the paper's 50-bin / 70% method."""
        valid = (
            np.isfinite(lows)
            & np.isfinite(highs)
            & np.isfinite(closes)
            & np.isfinite(volumes)
        )
        if not np.any(valid):
            return np.nan, np.nan, np.nan

        lows = lows[valid]
        highs = highs[valid]
        closes = closes[valid]
        volumes = np.maximum(volumes[valid], 0.0)

        lo = float(np.min(lows))
        hi = float(np.max(highs))

        if not np.isfinite(lo) or not np.isfinite(hi):
            return np.nan, np.nan, np.nan

        if hi <= lo:
            price = float(closes[-1])
            return price, price, price

        edges = np.linspace(lo, hi, bins + 1, dtype=float)
        typical = (highs + lows + closes) / 3.0

        bin_idx = np.searchsorted(edges, typical, side="right") - 1
        bin_idx = np.clip(bin_idx, 0, bins - 1)

        hist = np.bincount(bin_idx, weights=volumes, minlength=bins).astype(float)
        poc_idx = int(np.argmax(hist))
        poc = float((edges[poc_idx] + edges[poc_idx + 1]) / 2.0)

        total_volume = float(hist.sum())
        if total_volume <= 0:
            return poc, float(edges[poc_idx]), float(edges[poc_idx + 1])

        threshold = total_volume * value_area_fraction
        order = np.argsort(hist)[::-1]

        selected = []
        accumulated = 0.0
        for idx in order:
            selected.append(int(idx))
            accumulated += float(hist[idx])
            if accumulated >= threshold:
                break

        val = float(edges[min(selected)])
        vah = float(edges[max(selected) + 1])
        return poc, val, vah

    def _add_profile_levels(self, dataframe: DataFrame) -> DataFrame:
        """
        Previous-day VAH/VAL are calculated from the completed previous UTC day.
        Developing POC is recalculated from day-start through the current candle only.
        """
        df = dataframe.copy()
        dates = pd.to_datetime(df["date"], utc=True)
        df["_vp_day"] = dates.dt.floor("D")

        df["prev_poc"] = np.nan
        df["prev_val"] = np.nan
        df["prev_vah"] = np.nan
        df["developing_poc"] = np.nan

        full_day_profiles = {}
        grouped_positions = {}

        for day, idx_labels in df.groupby("_vp_day", sort=True).groups.items():
            positions = df.index.get_indexer_for(idx_labels)
            positions = np.sort(positions)
            grouped_positions[day] = positions

            chunk = df.iloc[positions]
            full_day_profiles[day] = self._volume_profile(
                chunk["low"].to_numpy(dtype=float),
                chunk["high"].to_numpy(dtype=float),
                chunk["close"].to_numpy(dtype=float),
                chunk["volume"].to_numpy(dtype=float),
                self.profile_bins,
                self.value_area_fraction,
            )

            lows = chunk["low"].to_numpy(dtype=float)
            highs = chunk["high"].to_numpy(dtype=float)
            closes = chunk["close"].to_numpy(dtype=float)
            volumes = chunk["volume"].to_numpy(dtype=float)

            developing = np.full(len(chunk), np.nan, dtype=float)
            for i in range(len(chunk)):
                developing[i] = self._volume_profile(
                    lows[: i + 1],
                    highs[: i + 1],
                    closes[: i + 1],
                    volumes[: i + 1],
                    self.profile_bins,
                    self.value_area_fraction,
                )[0]

            df.iloc[positions, df.columns.get_loc("developing_poc")] = developing

        ordered_days = sorted(grouped_positions.keys())
        for i in range(1, len(ordered_days)):
            day = ordered_days[i]
            previous = ordered_days[i - 1]

            # Do not bridge missing UTC calendar days.
            if day - previous != pd.Timedelta(days=1):
                continue

            poc, val, vah = full_day_profiles[previous]
            positions = grouped_positions[day]
            df.iloc[positions, df.columns.get_loc("prev_poc")] = poc
            df.iloc[positions, df.columns.get_loc("prev_val")] = val
            df.iloc[positions, df.columns.get_loc("prev_vah")] = vah

        return df

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = self._add_profile_levels(dataframe)

        price_change = df["close"].diff()
        df["price_momentum_5"] = price_change.rolling(5, min_periods=5).sum()

        volume_ma_5 = df["volume"].rolling(5, min_periods=5).mean()
        volume_ratio = df["volume"] / volume_ma_5.replace(0.0, np.nan)
        df["volume_ratio_5"] = volume_ratio

        raw_long = np.sign(df["price_momentum_5"]) * volume_ratio
        raw_short = np.sign(-df["price_momentum_5"]) * volume_ratio

        df["tape_speed_long"] = raw_long.rolling(3, min_periods=3).mean()
        df["tape_speed_short"] = raw_short.rolling(3, min_periods=3).mean()

        return df

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        long_candidate = (
            df["prev_val"].notna()
            & df["developing_poc"].notna()
            & (df["close"] <= df["prev_val"])
            & (df["tape_speed_long"] >= self.tape_threshold)
            & (df["developing_poc"] > df["close"])
            & (df["volume"] > 0)
        )

        short_candidate = (
            df["prev_vah"].notna()
            & df["developing_poc"].notna()
            & (df["close"] >= df["prev_vah"])
            & (df["tape_speed_short"] >= self.tape_threshold)
            & (df["developing_poc"] < df["close"])
            & (df["volume"] > 0)
        )

        # Paper rule: maximum one trade per UTC day, first valid setup only.
        short_candidate = short_candidate & ~long_candidate
        any_candidate = (long_candidate | short_candidate).astype(int)
        day = pd.to_datetime(df["date"], utc=True).dt.floor("D")
        candidate_number = any_candidate.groupby(day).cumsum()
        first_candidate = candidate_number == 1

        df.loc[long_candidate & first_candidate, ["enter_long", "enter_tag"]] = (
            1,
            "vpmr_long",
        )
        df.loc[short_candidate & first_candidate, ["enter_short", "enter_tag"]] = (
            1,
            "vpmr_short",
        )

        return df

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        df = dataframe

        long_target = df["developing_poc"].notna() & (
            df["close"] >= df["developing_poc"]
        )
        short_target = df["developing_poc"].notna() & (
            df["close"] <= df["developing_poc"]
        )

        # Signal on 23:50 so normal next-candle execution closes at the
        # 23:55 candle rather than carrying the position into the next UTC day.
        dt = pd.to_datetime(df["date"], utc=True)
        eod = (dt.dt.hour == 23) & (dt.dt.minute == 50)

        df.loc[long_target | eod, ["exit_long", "exit_tag"]] = (1, "poc_or_eod")
        df.loc[short_target | eod, ["exit_short", "exit_tag"]] = (1, "poc_or_eod")

        return df

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        # V0 tests alpha only. Leverage comes later if the strategy has edge.
        return 1.0
