#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/level_structure_v2}"
START="${START:-2022-01-01}"
END="${END:-2026-08-19}"
WORKERS="${WORKERS:-16}"

mkdir -p "$OUTDIR"

echo "=== LEVEL / STRUCTURE EDGE V2 RUNNER ==="
echo "CACHE ONLY: no market downloads."
echo "WORKERS=$WORKERS"
echo "CONFIG=$CONFIG"
echo "DATADIR=$DATADIR"
echo "START=$START END=$END"
echo "OUTDIR=$OUTDIR"
echo

PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/run_level_structure_v2_parallel.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --outdir "$OUTDIR" \
  --start "$START" \
  --end "$END" \
  --workers "$WORKERS" \
  2>&1 | tee "$OUTDIR/run.log"

echo
echo "=== DONE ==="
echo "Paste from '=== LEVEL/STRUCTURE EDGE V2 AGGREGATE ===' through the end."
