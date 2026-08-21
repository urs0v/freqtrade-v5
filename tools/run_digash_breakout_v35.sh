#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_breakout_v35}"
START="${START:-2022-01-01}"
END="${END:-2026-08-19}"
WORKERS="${WORKERS:-16}"
BOOTSTRAP="${BOOTSTRAP:-5000}"
mkdir -p "$OUTDIR"
echo "=== DIGASH BREAKOUT V3.5 RUNNER ==="
echo "NEW-ASSET CACHE UNION: frozen candidate; no tuning and no downloads."
echo "WORKERS=$WORKERS START=$START END=$END"
echo "OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/analyze_digash_breakout_v35.py \
  --config "$CONFIG" --datadir "$DATADIR" --outdir "$OUTDIR" \
  --start "$START" --end "$END" --workers "$WORKERS" --bootstrap "$BOOTSTRAP" \
  2>&1 | tee "$OUTDIR/run.log"
