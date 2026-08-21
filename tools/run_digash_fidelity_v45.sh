#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
V4DIR="${V4DIR:-/freqtrade/user_data/digash_fidelity_v4}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_fidelity_v45}"
WORKERS="${WORKERS:-16}"
WARMUP_DAYS="${WARMUP_DAYS:-120}"
TRAIN_FRAC="${TRAIN_FRAC:-0.70}"
mkdir -p "$OUTDIR"
echo "=== DIGASH FIDELITY V4.5 RUNNER ==="
echo "CHRONOLOGICAL SELECTOR HOLDOUT: no PnL, no entry fitting, no downloads."
echo "WORKERS=$WORKERS WARMUP_DAYS=$WARMUP_DAYS TRAIN_FRAC=$TRAIN_FRAC"
echo "V4DIR=$V4DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/audit_digash_fidelity_v45.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --v4dir "$V4DIR" \
  --outdir "$OUTDIR" \
  --workers "$WORKERS" \
  --warmup-days "$WARMUP_DAYS" \
  --train-frac "$TRAIN_FRAC" \
  "$@" 2>&1 | tee "$OUTDIR/run.log"
