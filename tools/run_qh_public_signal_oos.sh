#!/usr/bin/env bash
set -euo pipefail

CONFIG="${QH_PUBLIC_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${QH_PUBLIC_DATADIR:-/freqtrade/user_data/data/binance}"
DB="${QH_DB:-/freqtrade/user_data/alpha_lab/qh_orderflow.sqlite}"
OUT="${QH_PUBLIC_OUT:-/freqtrade/user_data/alpha_lab/qh_public_oos}"

mkdir -p "$OUT"

echo "=== QH PUBLIC-SIGNAL OOS LAB ==="
echo "2024 fit only | 2025 validation | 2026 diagnostic"
echo "12h public-component signal | 8bps | no leverage yet"

python -u /opt/rmv5/tools/research_qh_public_signal_oos_safe.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --qh-db "$DB" \
  --outdir "$OUT"
