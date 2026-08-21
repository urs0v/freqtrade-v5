#!/usr/bin/env bash
set -euo pipefail

START="${START:-2026-08-01}"
END="${END:-2026-08-07}"
ROOT="${ROOT:-/freqtrade/user_data/qh_edge}"
SYMBOLS="${SYMBOLS:-BTCUSDT}"

read -r -a SYMBOL_LIST <<< "$SYMBOLS"
mkdir -p "$ROOT"

echo "=== QH EDGE PILOT ==="
echo "This is a lightweight pipeline/signal smoke test, not the final replication."
echo "START=$START END=$END ROOT=$ROOT SYMBOLS=$SYMBOLS"
echo

python /opt/rmv5/tools/qh_edge_audit.py \
  --start "$START" \
  --end "$END" \
  --root "$ROOT" \
  --symbols "${SYMBOL_LIST[@]}" \
  2>&1 | tee "$ROOT/pilot.log"

echo
echo "=== DONE ==="
echo "Paste everything from '=== RESULT ===' plus the symbol summary block above it."
