#!/usr/bin/env bash
set -euo pipefail

CONFIG="${DERIV_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DERIV_DATADIR:-/freqtrade/user_data/data/binance}"
DB="${DERIV_DB:-/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite}"
OUTDIR="${DERIV_OUTDIR:-/freqtrade/user_data/alpha_lab/portfolio}"

if [ ! -f "$DB" ]; then
  echo "Missing derivatives DB: $DB"
  exit 2
fi

mkdir -p "$OUTDIR"

echo "=== FLOW-FUNDING EXECUTABLE PORTFOLIO AUDIT ==="
echo "Factor: taker_minus_funding | horizon: 12h"
echo "Canonical: q25, 8bps, 20 equal pair slots, 1x"
echo "2026 is diagnostic only"

python /opt/rmv5/tools/audit_flow_funding_portfolio.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --db "$DB" \
  --outdir "$OUTDIR"
