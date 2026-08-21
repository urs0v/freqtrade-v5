# Digash Fidelity V4

## Goal

V4 changes the research question. It does **not** optimize PnL and does **not** promote the V3 `4h/p30/approach>=3` slice.

The goal is to test whether our causal horizontal-level reconstruction can reproduce the levels published by Digash's public breakout-formation feed (`@Digash_Formations`). Public Digash material states that the free channel contains the **level breakout** formation, while the wider product contains multiple formation families and can be filtered by timeframe.

## Ground truth

The collector reads only the public Telegram web preview at `https://t.me/s/Digash_Formations` and paginates backwards. It parses posts of the form:

- Binance futures pair
- `Пробой уровня/уровней`
- one or more published price levels
- published timeframe
- Telegram publication timestamp

Raw posts and parsed alerts are saved before any comparison with our detector.

## What is compared

For every parsed Binance-USDT breakout alert for which local OHLCV exists:

1. load the **same published timeframe** (1m/5m/15m/1h/4h), causally resampling only from a finer cached timeframe when necessary;
2. construct our existing V3/V3.1 horizontal levels with the already documented public-source settings/proxies (`period=20/30`, touch tolerance implementation unchanged);
3. allow only levels whose `formed_time <= Telegram post time`;
4. for each Digash-published level, measure the nearest causal reconstructed level in basis points;
5. report match curves at fixed diagnostic thresholds: 10, 25, 50 and 100 bps.

These thresholds are **diagnostics**, not strategy parameters and not PnL optimization.

## Important limitations

- Telegram publication time can lag the screener's internal detection time.
- We only audit alerts whose pair/timeframe exists in the local cache. Missing market history is reported, never downloaded.
- A finite warm-up is needed for runtime. Default is 120 days before the earliest public alert in each pair/timeframe group. This is an analysis window, **not** a claimed Digash level-lifetime rule. Lifetime remains source-documented as `0`; V4 does not invent an expiry.
- V4.0 audits **level fidelity first**. It deliberately does not judge PnL, stops, exits, or the three breakout execution styles. If level fidelity is poor, trade simulation is premature. If level fidelity is good, the next audit should compare formation timing/type and then execution.

## Interpretation

The key output is not PF. It is whether the existing detector sees approximately the same levels as the public Digash feed, on the same assets and timeframes, without looking into the future.

If match rate is poor, inspect the mismatch distribution by timeframe/period and reconstruct the detector more faithfully. Parameter changes are allowed only to improve **source fidelity**, and must be evaluated on a held-out set of public alerts before returning to PnL testing.
