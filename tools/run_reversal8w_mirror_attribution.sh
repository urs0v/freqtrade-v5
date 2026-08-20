#!/usr/bin/env bash
set -euo pipefail

OUT="/freqtrade/user_data/reversal8w_mirror_attribution"
mkdir -p "$OUT"

echo "=== 8-WEEK MIRROR ATTRIBUTION RUNNER ==="
echo "No downloads. Reads saved reversal weekly/asset CSVs."
echo

python /opt/rmv5/tools/reversal8w_mirror_attribution.py \
  --weekly /freqtrade/user_data/reversal8w_perp/weekly_results.csv \
  --assets /freqtrade/user_data/reversal8w_perp/asset_contributions.csv \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "Paste OVERALL ATTRIBUTION, YEAR ATTRIBUTION, LEG ATTRIBUTION and HIGH_VOL DIAGNOSTIC."
