#!/usr/bin/env bash
set -euo pipefail

START="${1:-2026-01-01}"
END_INCLUSIVE="${2:-2026-08-18}"
CONFIG="${RMV7_TEST_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"
STRATEGY="${RMV7_STRATEGY:-AdaptivePerp15mV7R1}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing $CONFIG"
  echo "Run retest_v7_core_2026.sh once first so the reproducible config exists."
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

freqtrade lookahead-analysis \
  --config "$CONFIG" \
  --strategy "$STRATEGY" \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe 15m
