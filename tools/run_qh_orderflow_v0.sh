#!/usr/bin/env bash
set -euo pipefail

START="${1:-2025-01-01}"
END="${2:-2026-07-31}"
QH_WORKERS="${QH_WORKERS:-64}"
OUT="/freqtrade/user_data/qh_orderflow_v0"
DB="$OUT/qh.sqlite"

mkdir -p "$OUT"
export QH_WORKERS

echo "=== QH ORDER FLOW V0 RUNNER ==="
echo "Evaluation: $START -> $END"
echo "Workers:    $QH_WORKERS"
echo "DB:         $DB"
echo

python /opt/rmv5/tools/qh_orderflow_v0.py \
  --db "$DB" \
  --start "$START" \
  --end "$END" \
  --workers "$QH_WORKERS" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "CSV: $OUT/asset_breakdown.csv, $OUT/year_breakdown.csv, $OUT/gates.csv, $OUT/qh_events_with_8h_returns.csv"
echo "Paste the block starting at '=== QH ORDER FLOW V0 RESULT ==='."
