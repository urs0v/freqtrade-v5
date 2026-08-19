#!/usr/bin/env bash
set -euo pipefail

START="${1:-2026-01-01}"
END_INCLUSIVE="${2:-2026-08-18}"
CONFIG="${RMV7_TEST_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"
RESULTS_DIR="${RMV7_RESULTS_DIR:-/freqtrade/user_data/backtest_results}"
OUTDIR="${RMV7_ALPHA_OUTDIR:-/freqtrade/user_data/v7/alpha_audit}"

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

mkdir -p "$OUTDIR"

BEFORE="$(ls -1t "$RESULTS_DIR"/*_signals.pkl 2>/dev/null | head -1 || true)"

echo "=== V7 ALPHA AUDIT: EXPORT RAW SIGNALS ==="
echo "No detail timeframe, no protections, 1x, fee=0. This run is diagnostic only."

RMV7_ENABLE_PROTECTIONS=false \
RMV7_ENABLE_TIME_EXIT=true \
freqtrade backtesting \
  --config "$CONFIG" \
  --strategy AdaptivePerp15mV7Audit \
  --strategy-path "$STRATEGY_PATH" \
  --timerange "$TIMERANGE" \
  --timeframe 15m \
  --max-open-trades 20 \
  --stake-amount 1 \
  --dry-run-wallet 1000 \
  --fee 0 \
  --cache none \
  --export signals

SIGNALS="$(ls -1t "$RESULTS_DIR"/*_signals.pkl 2>/dev/null | head -1 || true)"
if [ -z "$SIGNALS" ] || [ "$SIGNALS" = "$BEFORE" ]; then
  echo "No new *_signals.pkl found in $RESULTS_DIR"
  exit 3
fi

echo "Signals file: $SIGNALS"
python /opt/rmv5/tools/audit_v7_alpha.py \
  --signals "$SIGNALS" \
  --config "$CONFIG" \
  --outdir "$OUTDIR"
