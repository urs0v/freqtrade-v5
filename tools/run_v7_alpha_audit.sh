#!/usr/bin/env bash
set -euo pipefail

START="${1:-2026-01-01}"
END_INCLUSIVE="${2:-2026-08-18}"
CONFIG="${RMV7_TEST_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
STRATEGY_PATH="${RMV5_STRATEGY_PATH:-/opt/rmv5/strategies}"
RESULTS_DIR="${RMV7_RESULTS_DIR:-/freqtrade/user_data/backtest_results}"
OUTDIR="${RMV7_ALPHA_OUTDIR:-/freqtrade/user_data/v7/alpha_audit}"
EXISTING_ZIP="${RMV7_ALPHA_EXISTING_ZIP:-}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing $CONFIG. Run retest_v7_core_2026.sh once first."
  exit 2
fi

mkdir -p "$OUTDIR"

if [ -n "$EXISTING_ZIP" ]; then
  if [ ! -f "$EXISTING_ZIP" ]; then
    echo "Missing existing backtest ZIP: $EXISTING_ZIP"
    exit 4
  fi
  echo "=== V7 ALPHA AUDIT: REUSE EXISTING ZIP ==="
  echo "ZIP: $EXISTING_ZIP"
  python /opt/rmv5/tools/audit_v7_alpha.py \
    --signals "$EXISTING_ZIP" \
    --config "$CONFIG" \
    --outdir "$OUTDIR"
  exit $?
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

BEFORE="$(ls -1t "$RESULTS_DIR"/backtest-result-*.zip 2>/dev/null | head -1 || true)"

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
  --stake-amount 20 \
  --dry-run-wallet 1000 \
  --fee 0 \
  --cache none \
  --export signals

RESULT_ZIP="$(ls -1t "$RESULTS_DIR"/backtest-result-*.zip 2>/dev/null | head -1 || true)"
if [ -z "$RESULT_ZIP" ] || [ "$RESULT_ZIP" = "$BEFORE" ]; then
  echo "No new backtest ZIP found in $RESULTS_DIR"
  exit 3
fi

if ! unzip -l "$RESULT_ZIP" | grep -q '_signals.pkl'; then
  echo "Backtest ZIP has no *_signals.pkl member: $RESULT_ZIP"
  exit 5
fi

echo "Backtest ZIP: $RESULT_ZIP"
python /opt/rmv5/tools/audit_v7_alpha.py \
  --signals "$RESULT_ZIP" \
  --config "$CONFIG" \
  --outdir "$OUTDIR"
