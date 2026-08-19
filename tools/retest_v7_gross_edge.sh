#!/usr/bin/env bash
set -euo pipefail

START="${1:-2026-01-01}"
END_INCLUSIVE="${2:-2026-08-18}"
CONFIG="${RMV7_TEST_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing $CONFIG. Run retest_v7_core_2026.sh once first."
  exit 2
fi

END_EXCLUSIVE="$(python - "$END_INCLUSIVE" <<'PY'
import sys
from datetime import datetime, timedelta
x = datetime.strptime(sys.argv[1], '%Y-%m-%d') + timedelta(days=1)
print(x.strftime('%Y%m%d'))
PY
)"
START_FT="${START//-/}"
TIMERANGE="${START_FT}-${END_EXCLUSIVE}"

# This is deliberately NOT a realistic deployment backtest.
# It isolates signal/exit quality from leverage, fees, DD sizing and protections.
echo "=== V7 GROSS-EDGE DIAGNOSTIC ==="
echo "1x leverage | fixed 10 USDT stake | fee=0 | no protections | no detail timeframe"

RMV7_ENABLE_PROTECTIONS=false \
RMV7_ENABLE_TIME_EXIT=true \
freqtrade backtesting \
  --config "$CONFIG" \
  --strategy AdaptivePerp15mV7Audit \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe 15m \
  --max-open-trades 5 \
  --stake-amount 10 \
  --dry-run-wallet 100 \
  --fee 0 \
  --cache none \
  --export trades \
  --breakdown month year
