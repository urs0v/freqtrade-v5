#!/usr/bin/env bash
set -euo pipefail

CONFIG="${RMV9_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${RMV9_DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${RMV9_OUTDIR:-/freqtrade/user_data/v9/research}"
COST_BPS="${RMV9_ROUNDTRIP_COST_BPS:-8}"

mkdir -p "$OUTDIR"

echo "=== V9 CROSS-SECTIONAL RESEARCH ==="
echo "Train: 2022-2024 | validation: 2025 | 2026 diagnostic"
echo "Market-neutral long/short ranking. Cost assumption: ${COST_BPS} bps roundtrip."

python /opt/rmv5/tools/research_v9_cross_sectional.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --outdir "$OUTDIR" \
  --train-start 2022-01-01 \
  --train-end 2025-01-01 \
  --val-start 2025-01-01 \
  --val-end 2026-01-01 \
  --test-start 2026-01-01 \
  --test-end 2026-08-19 \
  --roundtrip-cost-bps "$COST_BPS"
