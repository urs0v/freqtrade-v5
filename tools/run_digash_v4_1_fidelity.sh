#!/bin/bash
set -euo pipefail

OUT="/freqtrade/user_data/digash_v4_1_fidelity"
mkdir -p "$OUT"

echo "=== DIGASH V4.1 SEQUENTIAL FIDELITY RUNNER ==="
echo "Stage-0 is preserved. No PnL is computed."
echo "Output: $OUT"

python /opt/rmv5/tools/digash_v4_1_fidelity.py \
  --outdir "$OUT" \
  --gold /opt/rmv5/tools/digash_v4_1_gold.csv \
  --start 2025-11-01 \
  --end 2026-08-19 \
  --workers "${WORKERS:-12}" \
  --sample 100
