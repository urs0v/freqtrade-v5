#!/usr/bin/env bash
set -euo pipefail

START="${1:-2021-01-01}"
END="${2:-2026-07-31}"
DB="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
OUT="/freqtrade/user_data/reversal8w_perp"

mkdir -p "$OUT"

echo "=== 8-WEEK REVERSAL PERP AUDIT RUNNER ==="
echo "Evaluation: $START -> $END"
echo "DB:         $DB"
echo "No downloads. Reuses completed Binance 6h/funding archive."
echo

python /opt/rmv5/tools/reversal8w_perp_test_v2.py \
  --db "$DB" \
  --start "$START" \
  --end "$END" \
  --universe 70 \
  --tail-frac 0.20 \
  --high-vol-frac 0.50 \
  --side-cost-bps 7.0 \
  --min-cross-section 25 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "CSV: $OUT/summary.csv, $OUT/year_breakdown.csv, $OUT/weekly_results.csv, $OUT/asset_contributions.csv, $OUT/gates.csv"
echo "Paste the block starting at '=== 8-WEEK REVERSAL PERP RESULT ==='."
