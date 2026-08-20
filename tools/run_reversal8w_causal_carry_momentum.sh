#!/usr/bin/env bash
set -euo pipefail

DB="/freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite"
ASSETS="/freqtrade/user_data/reversal8w_perp/asset_contributions.csv"
OUT="/freqtrade/user_data/reversal8w_causal_carry_momentum"

mkdir -p "$OUT"

echo "=== CAUSAL FUNDING-CONFIRMED HIGH-VOL MOMENTUM RUNNER ==="
echo "DB:     $DB"
echo "Assets: $ASSETS"
echo "No downloads. Reuses saved high-vol mirror assets + historical funding."
echo

python /opt/rmv5/tools/reversal8w_causal_carry_momentum.py \
  --db "$DB" \
  --assets "$ASSETS" \
  --side-cost-bps 7.0 \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "CSV: $OUT/summary.csv, $OUT/year_breakdown.csv, $OUT/weekly_results.csv, $OUT/asset_results.csv, $OUT/tail10_stress_weekly.csv, $OUT/gates.csv"
echo "Paste MAIN RESULT, YEAR BREAKDOWN, POST-PAPER, TAIL-CONCENTRATION STRESS and CAUSAL CANDIDATE GATES."
