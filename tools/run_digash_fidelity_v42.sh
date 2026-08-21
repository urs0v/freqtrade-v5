#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
V4DIR="${V4DIR:-/freqtrade/user_data/digash_fidelity_v4}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_fidelity_v42}"
WORKERS="${WORKERS:-16}"
WARMUP_DAYS="${WARMUP_DAYS:-120}"
BAND_PCT="${BAND_PCT:-0.10}"
mkdir -p "$OUTDIR"
echo "=== DIGASH FIDELITY V4.2 RUNNER ==="
echo "LIVE LEVEL LIFECYCLE: no PnL, no tuning, no downloads."
echo "WORKERS=$WORKERS WARMUP_DAYS=$WARMUP_DAYS BAND_PCT=$BAND_PCT"
echo "V4DIR=$V4DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/audit_digash_fidelity_v42.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --v4dir "$V4DIR" \
  --outdir "$OUTDIR" \
  --workers "$WORKERS" \
  --warmup-days "$WARMUP_DAYS" \
  --band-pct "$BAND_PCT" \
  "$@" 2>&1 | tee "$OUTDIR/run.log"
