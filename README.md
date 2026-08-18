# RegimeMomentumV5

A research/dry-run Freqtrade strategy combining:

- 6h momentum/trend regime backbone inspired by AdaptiveTrend.
- 1h breakout + volume confirmation.
- ATR volatility regime and custom ATR trailing stop.
- Funding + open-interest filters.
- Live Binance liquidation-stream cascade detector.
- Historical cascade proxy when true liquidation history is unavailable.
- Long + short.
- Dynamic leverage, capped at 20x.
- Optional forced 20x stress-test mode.
- Automatic top-N Binance USD-M **crypto-only** universe using `underlyingType == COIN`.

## Supplied stress-test setup

- dry wallet: 100 USDT
- stake: 50 USDT
- max open trades: 1
- `RMV5_FORCE_20X=true`

This is deliberately aggressive. It is not a live-risk recommendation.

## Coolify deployment

Use this folder as a Git/Docker Compose project. Change the API-server secrets/password
in Coolify before deploy. The container will:

1. Query Binance `exchangeInfo` + 24h tickers.
2. Select the top 20 USDT perpetuals where `underlyingType == COIN`.
3. Exclude 2026 TradFi perpetuals (stocks, commodities, etc.).
4. Start the derivatives collector.
5. Start Freqtrade with a separate RMV5 database.

Files created in the persistent volume:

- `/freqtrade/user_data/config-v5.generated.json`
- `/freqtrade/user_data/v5/universe.json`
- `/freqtrade/user_data/v5/features.sqlite`
- `/freqtrade/user_data/trades-rmv5.sqlite`

## Dynamic leverage later

Change:

    RMV5_FORCE_20X=false

Then leverage is estimated from ATR stop distance and capped at 20x.

## Backtest

Inside the container:

    freqtrade download-data       -c /freqtrade/user_data/config-v5.generated.json       --trading-mode futures       -t 15m 1h 6h       --timerange 20210101-

Then:

    freqtrade backtesting       -c /freqtrade/user_data/config-v5.generated.json       --strategy RegimeMomentumV5       --strategy-path /opt/rmv5/strategies       --timerange 20210101-       --timeframe-detail 5m       --breakdown month year

## Historical derivatives features

`tools/import_binance_metrics.py` imports Binance public USD-M futures metrics archives.
Those archives can provide historical open interest and flow ratios.

True historical liquidations are not in the current Binance public metrics archive.
For exact liquidation history, `tools/import_tardis.py` imports normalized Tardis
`derivative_ticker` and `liquidations` CSV(.gz) files.

Without true liquidation history the strategy uses an OI-collapse + price + volume +
taker-flow proxy for historical cascade entries.

## Hyperopt / +200% research target

Training example:

    freqtrade hyperopt       -c /freqtrade/user_data/config-v5.generated.json       --strategy RegimeMomentumV5       --strategy-path /opt/rmv5/strategies       --spaces buy sell       --timerange 20210101-20231231       --epochs 500

Untouched OOS test:

    freqtrade backtesting       -c /freqtrade/user_data/config-v5.generated.json       --strategy RegimeMomentumV5       --strategy-path /opt/rmv5/strategies       --timerange 20240101-       --timeframe-detail 5m       --breakdown month year

The research target can be +200%, but only count it as interesting if it survives an
untouched out-of-sample period with acceptable drawdown and costs.
