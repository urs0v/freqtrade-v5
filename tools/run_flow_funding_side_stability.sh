#!/usr/bin/env bash
set -euo pipefail

OUT="${SIDE_STABILITY_OUT:-/freqtrade/user_data/alpha_lab/side_stability}"
mkdir -p "$OUT"

echo "=== FLOW-FUNDING SIDE STABILITY RUNNER ==="

echo "Year-by-year long vs short robustness; actual funding cashflows; no new downloads."

python -u /opt/rmv5/tools/audit_flow_funding_side_stability.py \
  --config "${SIDE_STABILITY_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}" \
  --datadir "${SIDE_STABILITY_DATADIR:-/freqtrade/user_data/data/binance}" \
  --db "${SIDE_STABILITY_DB:-/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite}" \
  --funding-cache "${SIDE_STABILITY_CACHE:-/freqtrade/user_data/v5/free-cache}" \
  --outdir "$OUT"
