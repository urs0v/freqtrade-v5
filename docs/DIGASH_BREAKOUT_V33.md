# Digash Breakout V3.3 — repeated-approach robustness

V3.3 is a report-only follow-up. It does not change V3.1 entries, stops, targets, level detection, activity features or trade outcomes.

## Why this test exists

V3.2 found that the post-hoc `approach_no >= 3` partition of 4h/p30 factual breakouts had materially stronger results than the aggregate sample. Because that partition was discovered after looking at the data, it is not promoted directly into a strategy.

There is source context for repeated approaches, but not an exact equivalence to our cached-OHLCV counter:

- Digash's public screener walkthrough describes a level/density example where repeated approaches and protorgovka precede a breakout.
- The same public material says a density can be considered for a breakout after two or more approaches, while first approach can be considered for a bounce.
- A public density-trading example explicitly discusses the third approach and waiting for absorption.

Historical order-book density snapshots are absent from the cache, so `approach_no` is only a horizontal-price approach proxy. `APPROACH_2PLUS` is therefore labeled source-proxy, not exact Digash. `APPROACH_3PLUS` is labeled post-hoc.

## Frozen V3.3 checks

For the already selected 4h/p30 factual breakout family, V3.3 reports:

- `APPROACH_1`, exact `APPROACH_2`, `APPROACH_2PLUS`, `APPROACH_3PLUS`;
- FACT_ALL, LOCAL_ACTIVE, BROAD_TOP10 and BROAD_TOP5 cohorts;
- 8 / 12 / 16 bps cost stress;
- 2022–2024, 2025, 2026 and individual-year splits;
- long vs short;
- pair concentration, leave-one-pair-out and leave-one-year-out;
- month-clustered bootstrap confidence intervals for mean R;
- pair×year composition-adjusted diagnostics;
- level-age overlap only as a diagnostic, not a selection rule.

## Post-selection promotion gate

Because `APPROACH_3PLUS` came from V3.2 inspection, V3.3 uses a deliberately strict gate before it may become a separately frozen formation test:

- N >= 100;
- 8 bps PF >= 1.20 and positive expectancy;
- 12 bps PF >= 1.10 and positive expectancy;
- at least 4 positive years at 8 bps;
- positive 2025 and positive 2026 at 8 bps;
- minimum leave-one-pair-out PF at 12 bps >= 1.00.

Passing this gate still is not independent OOS proof. It only means the repeated-approach hypothesis deserves a new, explicitly frozen formation test rather than more post-hoc filtering.
