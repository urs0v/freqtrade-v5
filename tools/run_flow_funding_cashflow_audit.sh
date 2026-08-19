#!/usr/bin/env bash
set -euo pipefail

OUT="${FUNDING_CASHFLOW_OUT:-/freqtrade/user_data/alpha_lab/funding_cashflow}"
mkdir -p "$OUT"

echo "=== FLOW-FUNDING CASHFLOW AUDIT RUNNER ==="
echo "Uses existing derivatives DB + cached Binance fundingRate archives; no new large downloads."

python -u /opt/rmv5/tools/audit_flow_funding_cashflows.py \
  --config "${FUNDING_CASHFLOW_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}" \
  --datadir "${FUNDING_CASHFLOW_DATADIR:-/freqtrade/user_data/data/binance}" \
  --db "${FUNDING_CASHFLOW_DB:-/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite}" \
  --funding-cache "${FUNDING_CASHFLOW_CACHE:-/freqtrade/user_data/v5/free-cache}" \
  --outdir "$OUT"
