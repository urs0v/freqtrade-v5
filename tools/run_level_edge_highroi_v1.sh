#!/usr/bin/env bash
set -euo pipefail

OUT=/freqtrade/user_data/level_edge_highroi_v1

echo "=== LEVEL EDGE HIGH-ROI V1 RUNNER ==="
echo "Causal level-event research only; production frozen WS strategy is untouched."

if [[ ! -s "$OUT/causal_events.csv" || ! -s "$OUT/stage2_train.csv" ]]; then
  python /opt/rmv5/tools/level_edge_highroi_v1_prepare.py \
    --config /freqtrade/user_data/v7/config-v7-core-backtest.json \
    --datadir /freqtrade/user_data/data/binance \
    --outdir "$OUT" \
    --start 2022-01-01 \
    --end 2026-08-19 \
    --workers 16
else
  echo "Reusing completed causal scan and stage2 shortlist; skipping expensive market rescan."
fi

python /opt/rmv5/tools/level_edge_highroi_v1_resume.py \
  --outdir "$OUT" \
  --workers 16

echo "=== SUMMARY ==="
cat "$OUT/summary.json"
echo "=== DONE ==="
