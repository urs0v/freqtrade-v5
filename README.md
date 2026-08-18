# RegimeMomentumV5

Research/dry-run Freqtrade strategy:

- 6h trend/momentum regime
- 1h breakout + volume confirmation
- 15m execution
- ATR stop / profit lock
- OI + funding + taker/top-trader flow
- live Binance liquidation stream
- historical liquidation/cascade proxy from OI + price + volume + taker flow
- long + short
- dynamic leverage capped at 20x, or forced 20x stress mode
- crypto-only Binance USD-M universe

## Current dry-run setup

- wallet: 100 USDT
- stake: 50 USDT
- max open trades: 1
- `RMV5_FORCE_20X=true`

This is an aggressive research setup, not a live-risk recommendation.

## Coolify

The container selects 20 crypto perpetuals, starts the derivatives collector, then starts Freqtrade.
Persistent files include:

- `/freqtrade/user_data/config-v5.generated.json`
- `/freqtrade/user_data/v5/universe.json`
- `/freqtrade/user_data/v5/features.sqlite` (live derivatives)
- `/freqtrade/user_data/trades-rmv5.sqlite`

## Free historical backtest

No Tardis, trial, paid feed, or free-tier provider is required.
The historical derivatives layer is built from Binance public archives:

- daily USD-M `metrics`: OI + taker/top-trader ratios
- monthly USD-M `fundingRate`: funding history
- historical liquidation mode: OI/price/volume/taker-flow cascade proxy
- live liquidation mode: actual Binance force-order WebSocket collected by this bot

Run inside the deployed container:

```bash
bash /opt/rmv5/tools/free_backtest.sh 2022-01-01 2026-08-17
```

The script uses a separate database so it does not overwrite live data:

- `/freqtrade/user_data/v5/features-backtest.sqlite`

It performs:

1. Free derivatives backfill.
2. Freqtrade OHLCV download for `5m 15m 1h 6h`.
3. Full RegimeMomentumV5 backtest at forced 20x with `--timeframe-detail 5m`.

Downloaded Binance archive ZIPs are cached under:

- `/freqtrade/user_data/v5/free-cache`

Re-running the backfill reuses the cache.

## Backfill only

```bash
python /opt/rmv5/tools/backfill_free.py \
  --start 2022-01-01 \
  --end 2026-08-17 \
  --db /freqtrade/user_data/v5/features-backtest.sqlite
```

By default it reads the current RMV5 `universe.json`. A custom Binance-symbol list can be supplied with `--symbols`.

## Dynamic leverage later

Set:

```text
RMV5_FORCE_20X=false
```

Then leverage is derived from ATR stop distance and capped at 20x.

## Research process

First get a working full V5 backtest. Then compare ablations (price-only, +OI, +funding, +cascade) and finally use untouched out-of-sample periods. A high historical return is only useful if it survives costs and out-of-sample testing with tolerable drawdown.
