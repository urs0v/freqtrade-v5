#!/usr/bin/env bash
set -euo pipefail

START="${1:-2025-01-01}"
END="${2:-2026-07-31}"
DB="/freqtrade/user_data/qh_orderflow_v0/qh.sqlite"
OUT="/freqtrade/user_data/qh_flow_structure"

mkdir -p "$OUT"

echo "=== QH FLOW STRUCTURE AUDIT RUNNER ==="
echo "Evaluation: $START -> $END"
echo "DB:         $DB"
echo "No downloads. Reuses completed QH aggTrade SQLite."
echo

python /opt/rmv5/tools/qh_flow_structure_audit.py \
  --db "$DB" \
  --start "$START" \
  --end "$END" \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "CSV: $OUT/summary.csv, $OUT/year_breakdown.csv, $OUT/residual_asset_breakdown.csv, $OUT/common_flow_panel.csv, $OUT/relative_flow_panel.csv, $OUT/cross_asset_rank_ic.csv, $OUT/gates.csv"
echo "Paste the block starting at '=== QH FLOW STRUCTURE RESULT ==='."
