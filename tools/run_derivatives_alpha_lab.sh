#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DERIV_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DERIV_DATADIR:-/freqtrade/user_data/data/binance}"
DB="${DERIV_DB:-/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite}"
OUTDIR="${DERIV_OUTDIR:-/freqtrade/user_data/alpha_lab/results}"
COST_BPS="${DERIV_COST_BPS:-8.0}"

if [ ! -f "$DB" ]; then
  echo "Missing point-in-time DB: $DB"
  echo "Run first: bash /opt/rmv5/tools/backfill_derivatives_lab.sh"
  exit 2
fi

mkdir -p "$OUTDIR"

echo "=== DERIVATIVES ALPHA LAB ==="
echo "Train: 2022-2024 | validation: 2025 | 2026 diagnostic only"
echo "DB: $DB"
echo "Cost: $COST_BPS bps roundtrip"

python /opt/rmv5/tools/research_derivatives_alpha.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --db "$DB" \
  --outdir "$OUTDIR" \
  --train-start 2022-01-01 \
  --train-end 2025-01-01 \
  --val-start 2025-01-01 \
  --val-end 2026-01-01 \
  --test-start 2026-01-01 \
  --test-end 2026-08-19 \
  --cost-bps "$COST_BPS"
