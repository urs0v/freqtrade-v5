# DIGASH V4 — Research Specification

Status: **research-only design freeze v0.1**

Purpose: reproduce the trading logic described in the supplied Digash / "decisive places" transcripts closely enough to test it causally. This is **not** a claim that the strategy is profitable. The first obligation is visual/semantic parity with the described setups, not parameter optimization.

## 1. Core thesis

The strategy is not "trade every horizontal level" and not "trade every local pivot".

The repeated decision chain is:

```text
ACTIVE COIN
    -> clear higher-timeframe direction / liquidity objective
    -> decisive place
    -> local structure on 5m / 1m
    -> one explicit scenario
    -> structural invalidation
    -> real market target
    -> dynamic management as new structure appears
```

"Decisive places" are the locations where other participants are expected to make decisions and therefore create executable liquidity / forced flow:

- horizontal levels / highs / lows;
- cascades of nearby levels;
- protorgovka / accumulation ranges;
- retests;
- sweeps / liquidity grabs;
- trendline-like structures only when reinforced by a horizontal price object;
- order-book densities (Stage 2 only; unavailable in current historical OHLCV core).

## 2. What V4 explicitly rejects from older research

The following are **not valid Digash V4 approximations**:

1. Treating every +/-2-bar pivot as a meaningful level.
2. Using a fixed +/-1% level zone across all assets/timeframes.
3. Counting every 5m close near a level as a new "touch".
4. Entering simply because price touched or crossed a level.
5. Entering on a trendline break with no horizontal high/low/range behind which stops plausibly sit.
6. Taking entries in the middle of a structure between decisive boundaries.
7. Optimizing fixed 1R/1.5R/2R take-profits as if those were the author's main target logic.
8. Treating stop and target as static for the entire position.
9. Treating activity as a late optional filter instead of the first universe gate.
10. Mixing breakout, retest, reaction, sweep-reclaim and structural-break entries into one statistic.

## 3. Source-derived hard rules

These are repeated rules from the supplied transcripts and are considered the closest thing to "author-stated" constraints.

### 3.1 Active coin gate

Trade only active coins. Source descriptions repeatedly use one or more of:

- top gainers;
- top losers;
- top by number of trades;
- top volatility;
- examples explicitly mention >= 1,000,000 trades in 24h as a suitable activity threshold.

A coin qualifying in multiple activity categories is considered stronger / "super-active".

**Important:** very high activity is not unconditionally positive. The author also describes super-active coins as more manipulative, with more sweeps of obvious entries. V4 must therefore distinguish ACTIVE from EXTREME_ACTIVE rather than assume monotonic benefit.

### 3.2 Timeframe hierarchy

Repeated hierarchy:

```text
4h / 1h : broad trend and major targets / levels
5m      : pullback, retest, accumulation, local construction
1m      : precise structure break / reaction / entry
```

30m is not part of the frozen V4 core until a transcript rule requires it. It can be tested later as an additional structural timeframe.

### 3.3 Meaningful targets must exist before entry

A trade needs a clear reason for price to travel in the intended direction.

Targets can include:

- multi-touch horizontal level;
- prominent isolated high/low;
- cascade of levels;
- accumulated range boundary / protorgovka;
- later, order-book density.

A tiny noisy local high/low that is not visually salient should not qualify merely because it is a mathematical pivot.

### 3.4 Trade with higher-timeframe direction

The author repeatedly describes entries as local reversals / breaks **back into the intended higher-timeframe direction**.

Typical long structure:

```text
HTF bullish context / bullish target ahead
local 1m-5m countertrend: lower highs / lower lows
protorgovka / retest / compression
break first meaningful local high / range boundary
local trend flips long
entry
```

Short is mirrored.

### 3.5 No trade inside the construction

If the meaningful boundaries are a high, low, retest or range edge, arbitrary micro-pivots in the interior do not create a valid trade.

This is a hard veto.

### 3.6 A slope line alone is insufficient

A "naklonka" may describe local direction, but the author explicitly emphasizes that stops are normally behind a horizontal wall / level / high / low / cascade / accumulation / density rather than behind an abstract diagonal line.

Therefore V4 only accepts a slope-line scenario if there is also a horizontal liquidity object that is being crossed, defended, swept or reclaimed.

### 3.7 Available reward must be structurally large

The repeated risk/reward rule is:

```text
minimum available RR >= 3
prefer larger RR when the real target permits it
```

RR is evaluated from:

```text
entry -> structural invalidation
entry -> real structural/liquidity target
```

The target is **not** defined as "3R because 3R was requested". 3R is a pre-trade quality gate.

### 3.8 Stops are structural

Initial stop is behind the current invalidation point, for example:

- retest extreme;
- reaction low/high;
- structural swing;
- sweep extreme;
- opposite side of a valid local accumulation when that is the actual invalidation.

If a distant sweep would make RR poor, the author sometimes uses the nearer retest / protorgovka structure instead when that is the real local invalidation.

### 3.9 Position management is dynamic

The strategy is explicitly not:

```text
enter -> static stop -> static TP -> walk away
```

As price creates new decisive places:

- initial stop may move to breakeven;
- later to a protected profitable swing;
- target may extend if a new valid continuation formation appears near the original target;
- trade may exit early if price stalls at a decisive place or order-book resistance appears;
- failure to make the expected next high/low after a breakout can be an exit condition.

## 4. Formal market objects

The following are **our causal proxies** for the source concepts. They are intentionally separated from source-derived hard rules.

### 4.1 Higher-timeframe trend

Do not use a single moving-average sign as the primary definition.

Use confirmed causal swing structure on 1h and 4h.

Candidate proxy:

```text
LONG HTF:
    last two confirmed significant swing highs are non-decreasing
    AND last two significant swing lows are non-decreasing
    OR a confirmed bullish structure break has occurred and has not been invalidated

SHORT HTF: mirrored
```

The first implementation should expose the component flags rather than collapse everything to one opaque score.

### 4.2 Prominent high / low

Old +/-2 candle pivots are forbidden as final levels.

A candidate swing must satisfy all of:

1. causally confirmed after right-side bars close;
2. local prominence relative to ATR / surrounding range;
3. separation from neighboring candidate levels;
4. visible price departure after formation.

Initial small research grid (not final truth):

```text
pivot span:              3 / 5 HTF bars each side
minimum prominence:      0.5 / 0.75 / 1.0 ATR(HTF)
minimum departure:       0.5 / 1.0 ATR(HTF)
```

The purpose of the grid is to find a robust neighborhood, not a single magic value.

### 4.3 Horizontal level / zone

Represent a level as a narrow ATR-normalized zone, not a fixed percent.

Initial candidate width:

```text
zone half-width = 0.10 / 0.15 / 0.20 * ATR(HTF)
```

Two separate prominent touches are enough to make a multi-touch level valid, but additional touches may increase breakout relevance.

A new touch only counts if:

- price previously departed the zone materially;
- enough time elapsed to make it a distinct market visit;
- the same candle cluster is not counted repeatedly.

### 4.4 Cascade

A cascade is a sequence of 2+ valid same-direction target levels ahead of price.

Implementation requirement:

- preserve every constituent level;
- do not merge the entire cascade into one average price;
- compute available RR to each stage and to the final stage;
- allow management to react as each stage is crossed.

### 4.5 Protorgovka / accumulation

A local 1m/5m construction with high overlap and reduced directional progress before a break.

Candidate measurable components:

- range compression relative to recent ATR;
- high candle overlap;
- low net displacement / high total path length;
- repeated interaction with both sides or one defended side;
- duration long enough to be a real local construction, not 2 random candles.

Initial 1m windows to inspect visually:

```text
5 / 10 / 15 / 20 minutes
```

No single compression formula is frozen before visual parity review.

### 4.6 Clean vs noisy structure

Create a veto-oriented `structure_quality` feature set.

Bad / noisy candidates include:

- excessive wick-to-range ratio;
- heavy overlap with no identifiable boundary;
- many false crossings of the same micro-level;
- no clear local swing sequence;
- no meaningful target;
- "entry" based on an interior micro-pivot.

This is a quality veto, not an indicator-based alpha score.

## 5. Entry families — never aggregate them initially

Every candidate must have exactly one `entry_family`.

### A. BOS_BREAK

Use when local countertrend / accumulation has a clear first decisive boundary.

Long example:

```text
HTF long context
local short structure
protorgovka / retest
first meaningful local high is crossed
=> bullish BOS
```

Conservative backtest entry:

```text
first executable 1m price after a closed 1m bar confirms the crossing
```

### B. RETEST_LIMIT

After valid BOS, price returns to the broken decisive zone before invalidation.

Research separately because fill quality and adverse selection differ from reaction entry.

### C. RETEST_REACTION

Preferred high-confidence interpretation from several examples:

```text
valid BOS
price revisits broken area
retest does not invalidate
observable reaction begins in intended direction
enter only after reaction confirmation
stop behind reaction extreme
```

This must not be simulated as a blind limit fill.

### D. SWEEP_RETURN

```text
price sweeps a meaningful high/low / decisive place
liquidity beyond it is taken
price fails to continue through the sweep
price returns / reclaims
entry only after factual return or local confirmation
stop behind sweep or nearer valid retest structure
```

No "catch the knife" blind entry.

### E. LEVEL_BREAK / CASCADE_BREAK

For active coins with a well-developed multi-touch level / cascade:

```text
level already valid
price trades/accumulates near it
cross first decisive boundary
enter into stop-flow / continuation
manage through next cascade levels
```

This is separate from BOS_BREAK because the target structure and liquidity logic are different.

### F. PRE_ENTRY

Author sometimes enters slightly early based on a coin's observed movement character.

This is **not Stage 1**. It is discretionary and easy to overfit. Only evaluate after the deterministic families above work.

## 6. Activity model for historical research

Author-stated data include number of trades, gainers/losers and volatility rankings. Historical Freqtrade OHLCV may not expose exact exchange trade-count data.

Therefore V4 must preserve two fields:

```text
activity_source = AUTHOR_EQUIVALENT | OHLCV_PROXY
activity_quality = exact | proxy
```

Stage-1 OHLCV proxy can use cross-sectional 24h ranks:

- absolute 24h return;
- 24h volume / quote-volume proxy;
- realized volatility.

Suggested initial classification across the available research universe:

```text
ACTIVE:
    top 20-25% in at least one activity dimension

SUPER_ACTIVE:
    top 20-25% in at least two dimensions
```

Do not silently describe this as the author's exact "top by number of trades" rule.

## 7. Target and RR logic

For each entry candidate calculate:

```text
stop_distance = distance(entry, structural_invalidation)
target_distance = distance(entry, next_valid_target)
available_R = target_distance / stop_distance
```

Reject if:

```text
available_R < 3.0
```

Store targets as ordered stages:

```text
target_1
target_2
...
final_target
```

Baseline strategy exits can be compared using:

1. `STRUCTURAL_TARGET`: exit at the planned target / final cascade condition;
2. `FIXED_3R_CONTROL`: diagnostic only, not the primary strategy;
3. `STRUCTURAL_TRAIL`: move invalidation with new confirmed reaction swings;
4. later `DYNAMIC_TARGET_EXTENSION` after parity is proven.

## 8. Dynamic management model — Stage 1 graph-only

The first graph-only management engine should be deterministic.

### 8.1 Initial state

```text
stop = source-specific structural invalidation
target = first planned structural target / cascade plan
```

### 8.2 Protected retest

If a valid breakout is retested and the retest holds, the new retest reaction becomes the protected structure.

Candidate action:

```text
move stop to reaction extreme
```

This often puts the trade near breakeven, but the rule is structural rather than "always BE at +1R".

### 8.3 Favorable structure continuation

As new valid swings form in the trade direction:

```text
trail behind the most recent confirmed protected swing
```

Never move the stop farther from price.

### 8.4 Failure at the next decisive place

At an important target / level:

```text
break or test occurs
but expected new high/low cannot be made
and price loses the reaction / retest structure
=> exit
```

This is the graph-only proxy for the author's "activity died / resistance appeared" decision.

### 8.5 Stage-2 order-book override

Historical L2/tape data are required for true density / tape confirmation. Until then, this condition must be labelled `UNAVAILABLE`, not invented from candles.

## 9. Research vetoes

Reject candidate before simulation if any applies:

```text
NOT_ACTIVE
NO_HTF_DIRECTION
NO_MEANINGFUL_TARGET
TARGET_NOT_VISIBLE / LOW_PROMINENCE
INTERIOR_ENTRY
SLOPE_ONLY_NO_HORIZONTAL_LIQUIDITY
NO_LOCAL_STRUCTURE
NO_VALID_INVALIDATION
AVAILABLE_R_LT_3
TOO_NOISY
ALREADY_INVALIDATED
```

Do not optimize weak candidates after a veto.

## 10. Causality contract

Every feature must be available at `asof_time`.

Examples:

```text
confirmed pivot -> usable only after required right bars have closed
rolling ranks -> trailing data only
level touches -> no future touch used to validate historical entry
HTF trend -> only closed HTF bars
1m BOS -> entry no earlier than first executable price after confirmation
retest reaction -> reaction must occur before entry
```

No hindsight drawing of levels.

## 11. Required event schema

Every detected setup should write one row with at least:

```text
pair
asof_time
entry_time
side
entry_family
activity_class
activity_components
htf_direction
htf_tf
level_type
level_id
level_center
level_width
level_touch_count
level_prominence_atr
cascade_count
local_structure_type
protorgovka_duration_min
bos_price
retest_price
sweep_depth_atr
entry_price
initial_stop
initial_stop_bps
initial_stop_atr
target_1
final_target
available_R
structure_quality flags
veto flags
```

Exit replay adds:

```text
exit_time
exit_price
exit_reason
gross_R
net_R_8bps
net_R_12bps
net_R_20bps
MFE_R
MAE_R
stop_moves
target_moves
```

## 12. Validation order — mandatory

### Stage 0 — Visual parity before PnL

1. Detect candidate levels and setups.
2. Randomly sample at least 100 events across pairs, directions and months.
3. Render charts showing:
   - HTF context;
   - target levels / cascades;
   - local 5m construction;
   - 1m entry structure;
   - stop and target;
   - which entry family fired.
4. Manually inspect whether they resemble the supplied Digash examples.

If visual parity is bad, **do not backtest**. Fix detector semantics first.

### Stage 1 — Event-level graph-only edge

Only after Stage 0 passes.

Report separately by:

```text
BOS_BREAK
RETEST_LIMIT
RETEST_REACTION
SWEEP_RETURN
LEVEL_BREAK
```

And slice by:

```text
ACTIVE vs SUPER_ACTIVE
long vs short
HTF timeframe
single level vs cascade
2-touch vs 3+ touch
clean vs borderline structure
```

Primary metrics:

```text
N
gross expectancy R
8/12/20bps expectancy R
PF
win rate
median MFE/MAE
trades/month
monthly consistency
symbol concentration
```

### Stage 2 — Management

Only for Stage-1 families with evidence of edge.

Compare:

```text
STRUCTURAL_TARGET
FIXED_3R_CONTROL
STRUCTURAL_TRAIL
FAILURE_AT_DECISIVE_PLACE_EXIT
```

### Stage 3 — Microstructure

If historical L2 / aggTrade data are later available, add:

- order-book density / resistance;
- taker activity / tape acceleration;
- absorption / failed aggression;
- spread and depth state;
- execution slippage.

Do not retrofit candle proxies and call them true OFI / density.

## 13. Initial success criteria

The first V4 milestone is **not 50% monthly ROI**.

Milestone A: semantic parity.

```text
>= 80% of random reviewed candidates look like legitimate strategy setups
```

Milestone B: graph-only alpha basis.

At least one entry family should show, before aggressive sizing:

```text
positive gross expectancy across multiple months
positive net expectancy at realistic cost
PF preferably >= 1.25-1.35 before deeper optimization
no single symbol / month explaining most profit
reasonable parameter-neighborhood stability
```

Only then run the full $100 / leverage / portfolio model.

## 14. Small research grid — no brute-force explosion

Detector semantics first. After visual parity, only vary a few interpretable dimensions:

```text
level prominence ATR: 0.5 / 0.75 / 1.0
level zone ATR:       0.10 / 0.15 / 0.20
protorgovka window:   5 / 10 / 15 / 20m
retest window:        3 / 5 / 10m
activity percentile:  top 20% / 25%
```

Entry family, target logic and management type are categorical experiments, not parameters to blend together.

## 15. First implementation deliverable

The next code change should **not** begin with a huge PnL search.

It should implement:

```text
Digash V4 Stage-0 Detector
```

Deliverables:

1. causal HTF prominent-level / cascade detector;
2. activity proxy labelling;
3. 5m/1m local structure objects;
4. candidate classification by entry family;
5. veto reasons;
6. chart export for random visual review;
7. event CSV/Parquet for later replay.

Only after the visual sample is approved do we add the full trade simulator.
