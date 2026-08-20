#!/usr/bin/env bash
set -euo pipefail

SRC="/freqtrade/user_data/reversal8w_perp/weekly_results.csv"
OUT="/freqtrade/user_data/reversal8w_mirror"
mkdir -p "$OUT"

echo "=== 8-WEEK MIRROR AUDIT RUNNER ==="
echo "Source: $SRC"
echo "No downloads. No recomputation of signals. Exact mirror of saved weekly PnL."
echo

python /opt/rmv5/tools/reversal8w_mirror_audit.py \
  --input "$SRC" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "Paste the block starting at '=== MIRROR RESULT ==='."
