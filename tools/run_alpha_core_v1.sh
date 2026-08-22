#!/usr/bin/env bash
set -euo pipefail

echo "=== ALPHA CORE V1 RUNNER ==="
echo "Causal 5m volatility-breakout research only; production realtime strategy is untouched."

python /opt/rmv5/tools/alpha_core_v1.py \
  --datadir /freqtrade/user_data/data/binance/futures \
  --outdir /freqtrade/user_data/alpha_core_v1 \
  --start 2022-01-01 \
  --end 2026-08-19 \
  --workers 7

echo "=== SUMMARY ==="
cat /freqtrade/user_data/alpha_core_v1/summary.json
