# Digash Replication V3 — frozen research spec

Purpose: test the reproducible core of the public Digash/Digahka level-trading process without silently replacing it with a different strategy.

## Publicly supported rules used in V3

1. Horizontal levels are configured with a **period** (number of candles between touches), **touch count**, **tolerance/error**, and **lifetime**. The first extremum is an initialization point and is not counted as a touch. In the published screener walkthrough Digash says he normally uses period **20–30 candles**, usually **1 counted touch**, tolerance setting **1**, and lifetime **0**; he raises tolerance for cascades.
2. Levels from other timeframes can be displayed together. The walkthrough explicitly demonstrates local levels together with 15m / 1h / 4h levels.
3. Active-coin selection uses top growth, top decline, number of trades and volatility. For top lists he shows a 24h window and commonly uses a roughly $70m 24h volume floor to remove illiquid names. Volatility is viewed on local windows (roughly 5m–6h depending on the setup).
4. Volume-spike sorting compares a short current window (example: 5m) to average volume over about the previous 2h. For alerts he gives an example of a 5x volume increase together with about a 3% price move.
5. Horizontal-level breakout setups are not “any crossing”: he repeatedly describes waiting for an approach / trading near the level ("проторговка") and then taking the breakout. Retest and rebound are separate strategies.
6. Daily high/low and round numbers are treated as liquidity locations; trend levels and order-book densities are also important in his full workflow.

Primary public references used while freezing this spec:
- https://ru.scribd.com/document/871237922/Screener-Deegash (transcript of the full screener walkthrough)
- https://t.me/s/DigashLive (official screener channel posts linking the breakout, retest, rebound, structure-break and activity videos)
- https://videohighlight.com/v/LrktB-HyfDA (secondary summary of the public breakout video; used only as corroboration)

## What V3 intentionally does NOT claim to replicate

- Historical order-book densities: not present in the existing cache and the user explicitly does not want new data downloaded.
- Historical trade-count ranking: not present in OHLCV; V3 does not fake it.
- The screener's closed trend-line algorithm: not public enough to reproduce faithfully.
- The four exact “structure break” formations: public summaries confirm they exist, but exact machine rules are not sufficiently available here.
- The exact unit/implementation behind screener tolerance setting “1”. A third-party walkthrough interprets it as 1%; V3 therefore freezes `touch_tolerance_pct = 1%` and labels this as the largest replication uncertainty rather than optimizing it after results.

## Mechanical implementation choices (not attributed to Digash)

These are necessary to make the public rules causal and testable:

- Local extrema primitive: a pivot is known only after 2 candles on the right have closed.
- A horizontal resistance/support is formed when an initializer and a later same-side extremum are separated by at least 20 or 30 candles and differ by no more than 1%.
- Between initializer and counted touch, price must leave the 1% touch band and resistance/support must remain predominantly one-sided; otherwise it is rejected as a noisy back-and-forth level.
- `protor_proxy`: at least 3 of the previous 6 execution candles close within 0.5% of the level. This is reported as a proxy and is never described as an exact Digash rule.
- Entry is only after a closed 5m fact; execution is next 5m open. Stops are structural (reaction/break/retest extreme), not a fixed percentage.
- 1R/2R/3R path outcomes are reported; 8 bps round-trip is base cost and 12 bps stress.

## Tested setup families

- `H_BREAK`: first confirmed horizontal-level crossing after the level was already known.
- `H_RETEST`: first successful retest after a confirmed break.
- `H_BOUNCE`: touch/rejection followed by a small 5m confirmation away from the level.
- `H_FAKEOUT`: crossing/sweep followed by a quick reclaim back through the level.

Every event stores timeframe (5m/15m/1h/4h), period profile (20/30), touch error, approach number, activity ranks, protor proxy, impulse/volume features, nearest already-known opposing level and structural RR.

## Frequency is a fidelity diagnostic, not an optimization target

The user recalls Digash doing roughly 15 trades/week. V3 does not cap trades at 15. It prints events/week for every variant. If the detector still produces hundreds of trades/week, that is evidence the replication is still too broad, not a reason to cherry-pick the best 15 after the fact.
