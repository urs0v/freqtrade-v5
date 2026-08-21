#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/level_edge_audit_16w}"
START="${START:-2022-01-01}"
END="${END:-2026-08-19}"
WORKERS="${WORKERS:-16}"

if [[ ! -f "$CONFIG" ]]; then
  echo "CONFIG_MISSING: $CONFIG"
  exit 2
fi

mkdir -p "$OUTDIR"

echo "=== LEVEL EDGE AUDIT / ${WORKERS} WORKERS ==="
echo "CACHE ONLY: no market downloads."
echo "Parallelism: one Python process per pair, up to ${WORKERS} simultaneously."
echo "CONFIG=$CONFIG"
echo "DATADIR=$DATADIR"
echo "START=$START END=$END"
echo "OUTDIR=$OUTDIR"
echo

PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/run_level_edge_parallel.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --outdir "$OUTDIR" \
  --start "$START" \
  --end "$END" \
  --workers "$WORKERS" \
  2>&1 | tee "$OUTDIR/run.log"

echo
echo "=== DONE ==="
echo "Paste from '=== PARALLEL AGGREGATE ===' through the end."
