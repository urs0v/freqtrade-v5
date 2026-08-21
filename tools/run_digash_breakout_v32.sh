#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
EVENTS="${EVENTS:-/freqtrade/user_data/digash_replication_v31/events.csv}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_breakout_v32}"
WORKERS="${WORKERS:-16}"
mkdir -p "$OUTDIR"
echo "=== DIGASH BREAKOUT V3.2 RUNNER ==="
echo "CACHE ONLY: no market downloads."
echo "WORKERS=$WORKERS"
echo "V3.1 EVENTS=$EVENTS"
echo "OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/analyze_digash_breakout_v32.py \
  --config "$CONFIG" --datadir "$DATADIR" --events "$EVENTS" --outdir "$OUTDIR" --workers "$WORKERS" \
  2>&1 | tee "$OUTDIR/run.log"
