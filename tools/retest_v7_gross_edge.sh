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

# Deliberately not a realistic deployment backtest.
# This isolates signal/exit quality from leverage, fees, DD sizing, protections
# and most portfolio-slot competition.
echo "=== V7 GROSS-EDGE DIAGNOSTIC ==="
echo "1x leverage | fixed 20 USDT stake | fee=0 | no protections | no detail timeframe"

RMV7_ENABLE_PROTECTIONS=false \
RMV7_ENABLE_TIME_EXIT=true \
freqtrade backtesting \
  --config "$CONFIG" \
  --strategy AdaptivePerp15mV7Audit \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe 15m \
  --max-open-trades 20 \
  --stake-amount 20 \
  --dry-run-wallet 1000 \
  --fee 0 \
  --cache none \
  --export trades \
  --breakdown month year
