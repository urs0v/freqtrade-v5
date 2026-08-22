#!/bin/bash
set -euo pipefail

OUT="/freqtrade/user_data/digash_v4_2_fidelity"
mkdir -p "$OUT"

echo "=== DIGASH V4.2 QUALITY-FIRST FIDELITY RUNNER ==="
echo "No PnL. Frequency is NOT an objective. Standalone BOS entries are disabled."
echo "Output: $OUT"

python /opt/rmv5/tools/digash_v4_2_fidelity.py \
  --outdir "$OUT" \
  --gold /opt/rmv5/tools/digash_v4_2_gold.csv \
  --start 2025-11-01 \
  --end 2026-08-19 \
  --workers "${WORKERS:-12}" \
  --sample 100
