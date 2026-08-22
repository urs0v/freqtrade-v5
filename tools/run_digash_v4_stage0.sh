#!/usr/bin/env bash
set -euo pipefail

echo "=== DIGASH V4 STAGE-0 RUNNER ==="
echo "Visual-parity detector only; no PnL is computed."

python /opt/rmv5/tools/digash_v4_stage0.py \
  --datadir /freqtrade/user_data/data/binance/futures \
  --outdir /freqtrade/user_data/digash_v4_stage0 \
  --start 2025-11-01 \
  --end 2026-08-19 \
  --workers 12 \
  --sample 100

echo
cat /freqtrade/user_data/digash_v4_stage0/summary.json
