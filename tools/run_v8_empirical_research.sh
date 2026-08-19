#!/usr/bin/env bash
set -euo pipefail

CONFIG="${RMV8_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${RMV8_DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${RMV8_OUTDIR:-/freqtrade/user_data/v8/research}"
SAMPLE_STEP="${RMV8_SAMPLE_STEP:-2}"

if [ ! -f "$CONFIG" ]; then
  echo "Missing config: $CONFIG"
  exit 2
fi

mkdir -p "$OUTDIR"

echo "=== V8 EMPIRICAL ALPHA RESEARCH ==="
echo "Train: 2022-2024 | validation: 2025 | frozen OOS test: 2026"
echo "No Freqtrade trade simulation is run here."

python /opt/rmv5/tools/research_v8_empirical_alpha.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --train-start 2022-01-01 \
  --train-end 2025-01-01 \
  --val-start 2025-01-01 \
  --val-end 2026-01-01 \
  --test-start 2026-01-01 \
  --test-end 2026-08-19 \
  --sample-step "$SAMPLE_STEP" \
  --outdir "$OUTDIR"
