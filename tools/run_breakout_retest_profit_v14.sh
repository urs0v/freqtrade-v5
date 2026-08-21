#!/usr/bin/env bash
set -euo pipefail

echo "=== BREAKOUT / RETEST PROFIT V1.4 RUNNER ==="
echo 'Frozen V1.3 signal -> exact 5m exits -> $100 capital/concurrency/leverage/funding realism. No signal tuning.'

V13DIR="${V13DIR:-/freqtrade/user_data/breakout_retest_profit_v13}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/breakout_retest_profit_v14}"
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
mkdir -p "$OUTDIR"

echo "V13DIR=$V13DIR OUTDIR=$OUTDIR"
python /opt/rmv5/tools/breakout_retest_profit_v14.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --v13dir "$V13DIR" \
  --outdir "$OUTDIR" \
  "$@" | tee "$OUTDIR/run.log"

echo "=== DONE ==="
