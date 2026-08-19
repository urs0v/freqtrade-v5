#!/usr/bin/env bash
set -euo pipefail

DB="${DERIV_DB:-/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite}"
OUTDIR="${DERIV_ROBUST_OUTDIR:-/freqtrade/user_data/alpha_lab/robustness}"
CONFIG="${DERIV_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DERIV_DATADIR:-/freqtrade/user_data/data/binance}"

if [ ! -f "$DB" ]; then
  echo "Missing DB: $DB"
  exit 2
fi
mkdir -p "$OUTDIR"

echo "=== DERIVATIVES ROBUSTNESS AUDIT ==="
echo "Candidates: funding_z, taker_minus_funding | horizon: 12h"
echo "2026 is diagnostic only"

python /opt/rmv5/tools/audit_derivatives_robustness.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --db "$DB" \
  --outdir "$OUTDIR"
