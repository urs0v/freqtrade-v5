# Digash Replication V3.1 — fidelity corrections

V3.1 is not a new strategy and does not optimize V3 parameters after seeing its result. It corrects implementation mismatches and adds diagnostics where the public rule is ambiguous.

## Source-backed points kept unchanged

- Horizontal level: initialization extremum + later counted touch; initialization is not itself a counted touch.
- Period profiles 20 / 30 candles.
- Public walkthrough uses one counted touch, tolerance UI setting `1`, and horizontal-level lifetime filter `0`.
- Multi-timeframe levels are visible together (local / 15m / 1h / 4h).
- Active-name workflow uses top growth, top decline, volatility, volume/trade activity, and volume spikes.
- Breakout is not any crossing: the public process waits for approach / protorgovka and then a breakout fact.
- Retest, rebound and fakeout/pierce are separate families.

## Important correction: no invented level expiry

The walkthrough explicitly shows the horizontal-level lifetime filter set to `0`. Therefore V3.1 does **not** impose an arbitrary 1d/7d/14d expiry. `level_age_h` is only reported as a diagnostic so we can see whether age matters without silently changing the strategy.

## V3 implementation problems corrected / diagnostics added

1. **Structural target fidelity.** A horizontal support can become resistance after a break and vice versa, so V3.1 does not freeze a level's role to its original S/R label. It keeps the causal V3 idea of the nearest already-known horizontal level in trade direction, excludes the source zone, and additionally reports the nearest level on the same-or-higher timeframe. This lets us test whether tiny lower-TF duplicates were incorrectly defining structural room without inventing a private role algorithm.
2. **Breakout protorgovka proxy.** V3 only counted closes near the level. V3.1 additionally requires the pre-break closes to be predominantly on the approach side and their distance to the level to contract. Thresholds remain explicit mechanical proxies, not claimed private Digash rules.
3. **Breakout stop.** V3 used a generic six-bar extreme. V3.1 first uses the closest causally confirmed 5m structural swing high/low, then a recent-structure fallback. Bounce/retest/fakeout stops remain behind their actual reaction/sweep extreme.
4. **Activity lists.** V3.1 uses top-5 growth, decline, volatility and quote-volume lists plus the existing volume-spike alert. Quote volume is kept as its own real list and is **not** mislabeled as historical trade-count. Historical trade-count remains unavailable.
5. **RR/fee diagnosis.** V3.1 does not invent a minimum stop width to improve fees. It reports median risk %, fee load in R, gross and net expectancy separately. If tiny structural stops make costs fatal, that is visible rather than hidden.
6. **Tolerance uncertainty.** The public UI setting `1` is still mechanically interpreted as 1% for continuity, but that unit is not proven. V3.1 reports touch-error buckets as a diagnostic and does not choose the best bucket as a new parameter.

## V3.1 research variant

`ACTIVE_FACT_RR3_HTF` means:

- coin is on at least one causal active list available from our cached data;
- the setup has its required factual confirmation (for breakout: one-sided/contracting protorgovka plus either a measurable close through the level or an impulse proxy; retest/bounce/fakeout are already post-fact definitions);
- structural room to the next already-known horizontal level in trade direction on the same-or-higher timeframe is at least 3R.

This is a research gate, not a claim that every missing private Digash component has been replicated. Order-book densities, historical trade-count ranking, exact trend-line algorithm and private formation logic remain missing rather than fabricated.
