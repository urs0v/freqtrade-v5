#!/usr/bin/env bash
set -euo pipefail

START="${1:-2022-01-01}"
END="${2:-$(date -u -d 'yesterday' +%F)}"
LIVE_CONFIG="${RMV5_CONFIG:-/freqtrade/user_data/config-v5.generated.json}"
HIST_CONFIG="${RMV5_HIST_CONFIG:-/freqtrade/user_data/config-v5.backtest.json}"
HIST_DB="${RMV5_HIST_DB:-/freqtrade/user_data/v5/features-backtest.sqlite}"
HIST_UNIVERSE="${RMV5_HIST_UNIVERSE:-/opt/rmv5/tools/historical_universe.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"

START_FT="${START//-/}"
END_FT="${END//-/}"
TIMERANGE="${START_FT}-${END_FT}"

python /opt/rmv5/tools/make_backtest_config.py \
  --source "$LIVE_CONFIG" \
  --universe "$HIST_UNIVERSE" \
  --output "$HIST_CONFIG"

SYMBOLS="$(python - <<'PY'
import json
p='/opt/rmv5/tools/historical_universe.json'
print(','.join(json.load(open(p))['symbols']))
PY
)"

echo "=== 1/3 FREE BINANCE DERIVATIVES BACKFILL ==="
python /opt/rmv5/tools/backfill_free_fixed.py \
  --start "$START" \
  --end "$END" \
  --symbols "$SYMBOLS" \
  --db "$HIST_DB"

echo "=== 2/3 OHLCV DOWNLOAD ==="
freqtrade download-data \
  --config "$HIST_CONFIG" \
  --trading-mode futures \
  --timeframes 5m 15m 1h 6h \
  --timerange "$TIMERANGE"

echo "=== 3/3 FULL V5 BACKTEST ==="
RMV5_FEATURE_DB="$HIST_DB" RMV5_FORCE_20X=true \
freqtrade backtesting \
  --config "$HIST_CONFIG" \
  --strategy RegimeMomentumV5 \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe-detail 5m \
  --breakdown month year
