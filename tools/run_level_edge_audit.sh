#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/level_edge_audit}"
START="${START:-2022-01-01}"
END="${END:-2026-08-19}"

if [[ ! -f "$CONFIG" ]]; then
  echo "CONFIG_MISSING: $CONFIG"
  exit 2
fi

mkdir -p "$OUTDIR"

echo "=== LEVEL BOUNCE / BREAK-RETEST AUDIT ==="
echo "CACHE ONLY: this runner does not download market data."
echo "It uses existing Binance futures 15m plus existing 5m, or aggregates existing 1m -> 5m."
echo "CONFIG=$CONFIG"
echo "DATADIR=$DATADIR"
echo "START=$START END=$END"
echo

python /opt/rmv5/tools/audit_level_edge.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --outdir "$OUTDIR" \
  --start "$START" \
  --end "$END" \
  2>&1 | tee "$OUTDIR/run.log"

echo
echo "=== DONE ==="
echo "Paste from '=== SUMMARY: ACTIVE SUBSET / 8 BPS ===' through the end."
