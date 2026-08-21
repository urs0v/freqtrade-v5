#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_replication_v31}"
START="${START:-2022-01-01}"
END="${END:-2026-08-19}"
WORKERS="${WORKERS:-16}"
mkdir -p "$OUTDIR"
echo "=== DIGASH REPLICATION V3.1 RUNNER ==="
echo "CACHE ONLY: no market downloads."
echo "WORKERS=$WORKERS"
echo "START=$START END=$END"
echo "OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/run_digash_replication_v31_parallel.py \
  --config "$CONFIG" --datadir "$DATADIR" --outdir "$OUTDIR" \
  --start "$START" --end "$END" --workers "$WORKERS" 2>&1 | tee "$OUTDIR/run.log"
echo
echo "=== DONE ==="
echo "Paste from '=== DIGASH REPLICATION V3.1 AGGREGATE ===' through the end."
