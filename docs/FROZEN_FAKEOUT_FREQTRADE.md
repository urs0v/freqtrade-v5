# FrozenFakeoutV1 Freqtrade prospective dry-run

This is the execution stage for the fully-causal frozen FAKEOUT candidate validated through V1.6.

## Frozen alpha

No alpha parameter is tuned here:

- setup: `FAKEOUT`
- activity score: `>= 1.5`
- initial risk: `>= 160 bps` and executable domain `<= 3000 bps`
- target: `3R`
- maximum holding time: `48 x 5m = 4h`
- causal 15m activity timing from V1.5
- causal first-signal-bar 3-candle dedup from V1.6
- fixed 20-pair universe

Historical implementation parity before this stage is `604/604`, Jaccard `1.0`, with matched 8/12bps R differing only by floating-point noise.

## Why there is a signal-feed process

The frozen level/event lifecycle starts from the original historical warmup and contains hundreds of thousands of 5m bars per pair. Recomputing that full lifecycle synchronously inside each Freqtrade pair callback would make the trading loop too slow.

`frozen_fakeout_signal_feed.py` therefore runs the already parity-tested causal detector in a 16-worker process pool. For the current 5m candle it keeps only the new candle's immutable open as an entry stub; `detect_events()` never processes that final incomplete row as a signal candle. Immediately before publishing a signal, the feed refetches the current candle and rejects it if the model stop or 3R target was already touched while the scan was running.

Freqtrade remains the execution simulator: it decides the actual market fill, 5x leverage, 1% equity risk sizing, three-position portfolio capacity, fees, futures mechanics, structural stop, 3R target and four-hour exit. This also exposes execution differences that the historical portfolio simulator could not model, such as one open Freqtrade trade per pair and real feed-to-order delay.

## Start

```bash
bash /opt/rmv5/tools/run_frozen_fakeout_freqtrade.sh start
```

The runner:

1. validates Python/config syntax and that Freqtrade can load `FrozenFakeoutV1`;
2. requires the existing `PARITY_PASS` (reruns the gate only if needed);
3. stops the old CPU-heavy custom prospective loop without deleting its cutoff/results;
4. inherits the original prospective cutoff from `/freqtrade/user_data/prospective_fakeout_v2/state.json`;
5. starts the executable signal feed;
6. starts a separate Freqtrade dry-run database at `/freqtrade/user_data/trades-frozen-fakeout.sqlite`.

No exchange orders are sent because `dry_run=true`.

## Commands

```bash
bash /opt/rmv5/tools/run_frozen_fakeout_freqtrade.sh status
bash /opt/rmv5/tools/run_frozen_fakeout_freqtrade.sh report
bash /opt/rmv5/tools/run_frozen_fakeout_freqtrade.sh log
bash /opt/rmv5/tools/run_frozen_fakeout_freqtrade.sh stop
```

The main checkpoints remain 50 closed trades (preliminary) and 100 closed trades (primary). Do not tune pairs, regimes, times, thresholds, RR, stops or holding time from prospective results before those checkpoints.
