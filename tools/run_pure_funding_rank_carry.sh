#!/usr/bin/env bash
set -euo pipefail

START="${1:-2021-01-01}"
END="${2:-2026-07-31}"
DB="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
OUT="/freqtrade/user_data/pure_funding_rank_carry"

mkdir -p "$OUT"

echo "=== PURE FUNDING-RANK CARRY RUNNER ==="
echo "Evaluation: $START -> $END"
echo "DB:         $DB"
echo "No downloads. Reuses existing Binance 6h/funding archive."
echo

python /opt/rmv5/tools/pure_funding_rank_carry.py \
  --db "$DB" \
  --start "$START" \
  --end "$END" \
  --universe 70 \
  --side-cost-bps 7.0 \
  --min-cross-section 25 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "CSV: $OUT/summary.csv, $OUT/year_breakdown.csv, $OUT/weekly_results.csv, $OUT/asset_results.csv, $OUT/tail10_stress_weekly.csv, $OUT/gates.csv"
echo "Paste MAIN RESULT, YEAR BREAKDOWN, POST-PAPER, TAIL-CONCENTRATION STRESS and PRE-REGISTERED PURE-CARRY GATES."
