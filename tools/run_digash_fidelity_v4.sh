#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_fidelity_v4}"
WORKERS="${WORKERS:-16}"
MAX_PAGES="${MAX_PAGES:-140}"
WARMUP_DAYS="${WARMUP_DAYS:-120}"
mkdir -p "$OUTDIR"
echo "=== DIGASH FIDELITY V4 RUNNER ==="
echo "PUBLIC FORMATION GROUND TRUTH: source-level reconstruction audit; no PnL tuning and no market downloads."
echo "WORKERS=$WORKERS MAX_PAGES=$MAX_PAGES WARMUP_DAYS=$WARMUP_DAYS"
echo "OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/audit_digash_fidelity_v4.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --outdir "$OUTDIR" \
  --workers "$WORKERS" \
  --max-pages "$MAX_PAGES" \
  --warmup-days "$WARMUP_DAYS" \
  "$@" 2>&1 | tee "$OUTDIR/run.log"
