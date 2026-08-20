#!/usr/bin/env bash
set -euo pipefail

OUT="${TREND_V1_OUT:-/freqtrade/user_data/strategy_build/trend_v1_fixed20}"
mkdir -p "$OUT"

echo "=== FROZEN TREND V1 RUNNER ==="
echo "Fixed Core-20 | existing data only | H6 | no tuning | 1x"

python -u /opt/rmv5/tools/backtest_trend_v1_fixed20.py \
  --config "${TREND_V1_CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}" \
  --datadir "${TREND_V1_DATADIR:-/freqtrade/user_data/data/binance}" \
  --funding-cache "${TREND_V1_FUNDING_CACHE:-/freqtrade/user_data/v5/free-cache}" \
  --start "${TREND_V1_START:-2022-01-01}" \
  --end "${TREND_V1_END:-2026-08-19}" \
  --outdir "$OUT" \
  2>&1 | tee "$OUT/run.log"
