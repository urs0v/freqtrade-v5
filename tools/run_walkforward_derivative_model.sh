#!/usr/bin/env bash
set -euo pipefail

OUT="${WF_DERIV_OUT:-/freqtrade/user_data/alpha_lab/walkforward_derivatives}"
mkdir -p "$OUT"

echo "=== WALK-FORWARD DERIVATIVES RUNNER ==="
echo "365d rolling fit -> next month | state-conditioned flow/funding | actual funding | no leverage"

python -u /opt/rmv5/tools/research_walkforward_derivative_model.py \
  --config "${WF_DERIV_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}" \
  --datadir "${WF_DERIV_DATADIR:-/freqtrade/user_data/data/binance}" \
  --db "${WF_DERIV_DB:-/freqtrade/user_data/alpha_lab/derivatives_pti.sqlite}" \
  --funding-cache "${WF_DERIV_FUNDING_CACHE:-/freqtrade/user_data/v5/free-cache}" \
  --outdir "$OUT"
