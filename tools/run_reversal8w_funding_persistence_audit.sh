#!/usr/bin/env bash
set -euo pipefail

OUT="/freqtrade/user_data/reversal8w_funding_persistence"
mkdir -p "$OUT"

echo "=== HIGH-VOL MIRROR FUNDING PERSISTENCE RUNNER ==="
echo "No downloads. Reads saved mirror selections + existing funding SQLite."
echo

python /opt/rmv5/tools/reversal8w_funding_persistence_audit.py \
  --assets /freqtrade/user_data/reversal8w_mirror_attribution/asset_mirror_attribution.csv \
  --db /freqtrade/user_data/strategy_build/adaptivetrend/core.sqlite \
  --output-dir "$OUT" \
  2>&1 | tee "$OUT/run.log"

echo
echo "=== DONE ==="
echo "Log: $OUT/run.log"
echo "Paste OVERALL PERSISTENCE, YEAR PERSISTENCE, LEG PERSISTENCE, quantiles, and INTERPRETATION GATES."
