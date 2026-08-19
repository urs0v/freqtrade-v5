#!/usr/bin/env bash
set -euo pipefail

CONFIG="${QH_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${QH_DATADIR:-/freqtrade/user_data/data/binance}"
DB="${QH_DB:-/freqtrade/user_data/alpha_lab/qh_orderflow.sqlite}"
OUTDIR="${QH_ALPHA_OUTDIR:-/freqtrade/user_data/alpha_lab/qh_alpha}"

mkdir -p "$OUTDIR"

echo "=== QUARTER-HOUR RAW ORDER-FLOW ALPHA LAB ==="
echo "2024 calibration | 2025 validation | 2026 diagnostic"
echo "Factors: imbalance_notional, imbalance_qty | horizons: 4h,8h,12h | cost: 8bps"

python -u /opt/rmv5/tools/research_quarter_hour_orderflow.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --db "$DB" \
  --outdir "$OUTDIR"
