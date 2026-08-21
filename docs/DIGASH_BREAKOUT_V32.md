# Digash Breakout V3.2 — follow-up fidelity / robustness audit

V3.2 does **not** introduce a new trading strategy. It follows the signal exposed by V3.1: horizontal breakout quality improved with level timeframe, and the 4h / period-30 slice was the first Digash-replication slice to remain slightly positive after the assumed 8 bps round-trip cost.

That 4h/p30 observation is already post-selection. V3.2 therefore labels it a **follow-up candidate**, not fresh OOS proof.

## Public facts motivating the audit

Public Digash material explicitly says:

- breakout trading contains several different breakout types rather than one generic crossing;
- coin selection is a separate part of the workflow;
- the screener covers all USDT pairs / broad market lists rather than a fixed 20-name research universe;
- the product separates formations into scalping, medium-term and long-term timeframe groups;
- active coins and the quality of the move into/through a level are important.

References:
- https://t.me/s/DigashLive?before=39 — public post linking the breakout video and saying it covers different breakout types and coin selection.
- https://t.me/s/DigashLive?before=130 — public formations post separating scalping / medium-term / long-term formation groups.
- https://t.me/s/DigashLive?before=28 — public screener description saying all USDT pairs are represented.
- https://t.me/s/DigashLive — public post describing second-chart inspection of whether a level break was impulsive or choppy.

The exact private machine definitions of each breakout subtype are not public enough to reproduce faithfully. V3.2 therefore does **not** invent named Digash subtypes. Instead it partitions already-recorded breakouts by observable diagnostics (`approach_no`, impulse vs close-through, stop source, level age) and labels those partitions as diagnostics only.

## Main V3.2 correction: broad cached-universe activity

V3.1 ranked activity inside the configured 20-pair whitelist. That made `top-5` the top 25% of the research universe, which is not comparable to a broad-market screener.

V3.2 discovers every already-cached Binance USDT perpetual with 15m OHLCV through Freqtrade's data handler. It performs **no downloads**.

At each 15m context timestamp needed by a V3.1 breakout event, it computes causal 24h return, 24h quote volume and 15m NATR, applies the public ~$70m 24h quote-volume floor, then ranks the liquid cached universe into separate lists:

- top growth;
- top decline;
- top volatility;
- top quote volume.

Both top-5 and top-10 membership are reported. Historical trade-count remains unavailable and is not replaced with a fake variable.

The script prints the discovered and contributing cache universe. If the median available universe is still near 20 pairs, V3.2 explicitly warns that this fidelity problem has not actually been solved.

## Predeclared robustness views

V3.2 consumes the frozen V3.1 event file; it does not regenerate entries or stops.

It reports:

1. 1h / 4h breakouts, period 20 / 30, under four selection views: factual only, old local-active filter, broad top-10, broad top-5.
2. The already-selected 4h/p30 follow-up candidate under factual-only / local-active / broad-top10 / broad-top5.
3. 2022–2024, 2025, 2026 and per-year stability for 4h/p30 broad-top10.
4. Monthly positive-rate and saved monthly table.
5. Pair concentration and leave-one-pair-out results.
6. Diagnostic-only partitions by first/second/third+ approach, impulse/close-through mode, structural stop source and level age.

No diagnostic partition is automatically promoted into a new rule from this same sample.

## Interpretation gate

A promising result would not be `PF > 1` in one tiny cell. We want the 4h/p30 effect to survive a materially broader cached activity universe, remain positive or near-positive across time splits, avoid dependence on one pair, and retain similar behavior under top-5 and top-10 activity definitions.

If it fails those checks, the V3.1 4h/p30 result is likely noise or sample-specific. If it survives, the next step is to reconstruct the actual public breakout subtype logic more precisely and then validate prospectively rather than retuning this historical sample.
