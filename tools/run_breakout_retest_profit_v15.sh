#!/usr/bin/env bash
set -euo pipefail

echo "=== BREAKOUT / RETEST PROFIT V1.5 RUNNER ==="
echo "Audits activity timing causality for the frozen FAKEOUT_RISK160P signal. No tuning."

V1DIR="${V1DIR:-/freqtrade/user_data/breakout_retest_profit_v1}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/breakout_retest_profit_v15}"
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
mkdir -p "$OUTDIR"

python /opt/rmv5/tools/breakout_retest_profit_v15.py \
  --v1dir "$V1DIR" \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --outdir "$OUTDIR" \
  "$@" | tee "$OUTDIR/run.log"

echo "=== DONE ==="
