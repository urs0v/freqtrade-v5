#!/usr/bin/env bash
set -euo pipefail

echo "=== LEVEL EDGE HIGH-ROI V1 RUNNER ==="
echo "Causal level-event research only; production frozen WS strategy is untouched."

python /opt/rmv5/tools/level_edge_highroi_v1.py \
  --config /freqtrade/user_data/v7/config-v7-core-backtest.json \
  --datadir /freqtrade/user_data/data/binance \
  --outdir /freqtrade/user_data/level_edge_highroi_v1 \
  --start 2022-01-01 \
  --end 2026-08-19 \
  --workers 16

echo "=== SUMMARY ==="
cat /freqtrade/user_data/level_edge_highroi_v1/summary.json
echo "=== DONE ==="
