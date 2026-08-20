#!/usr/bin/env bash
set -euo pipefail

START="${START:-2026-05-01}"
END="${END:-2026-08-18}"
ROOT="${ROOT:-/freqtrade/user_data/qh_edge}"

mkdir -p "$ROOT"

echo "=== QH EDGE RUNNER ==="
echo "This is NOT a Freqtrade strategy backtest."
echo "It downloads Binance USD-M daily aggTrades, reduces them to 10-second bins, and audits the quarter-hour orderflow effect."
echo "START=$START END=$END ROOT=$ROOT"
echo

python /opt/rmv5/tools/qh_edge_audit.py \
  --start "$START" \
  --end "$END" \
  --root "$ROOT" \
  --symbols BTCUSDT ETHUSDT XRPUSDT SOLUSDT DOGEUSDT ADAUSDT \
  2>&1 | tee "$ROOT/run.log"

echo
echo "=== DONE ==="
echo "Paste everything from '=== RESULT ===' plus the six symbol summary blocks above it."
