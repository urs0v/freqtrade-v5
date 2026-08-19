#!/usr/bin/env bash
set -euo pipefail

CONFIG="${RMV8_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
START="${1:-2022-01-01}"
END="${2:-2025-10-03}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing config: $CONFIG"
  exit 2
fi

START_FT="${START//-/}"
END_FT="${END//-/}"

echo "=== V8 15m FUTURES HISTORY BACKFILL ==="
echo "Range requested: ${START} -> ${END}"
echo "Pairs come from: $CONFIG"

freqtrade download-data \
  -c "$CONFIG" \
  --trading-mode futures \
  --candle-types futures \
  -t 15m \
  --prepend \
  --timerange "${START_FT}-${END_FT}"

echo "Backfill complete. Re-run: bash /opt/rmv5/tools/run_v8_empirical_research.sh"
