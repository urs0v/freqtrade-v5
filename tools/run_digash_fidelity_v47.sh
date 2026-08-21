#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
V46DIR="${V46DIR:-/freqtrade/user_data/digash_fidelity_v46}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_fidelity_v47}"
WORKERS="${WORKERS:-16}"
TRAIN_FRAC="${TRAIN_FRAC:-0.70}"
mkdir -p "$OUTDIR"
echo "=== DIGASH FIDELITY V4.7 RUNNER ==="
echo "SOURCE BREAKOUT QUALITY: exact public levels, finest cached detail, no PnL fitting, no stops/targets, no downloads."
echo "WORKERS=$WORKERS TRAIN_FRAC=$TRAIN_FRAC"
echo "V46DIR=$V46DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/audit_digash_fidelity_v47.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --v46dir "$V46DIR" \
  --outdir "$OUTDIR" \
  --workers "$WORKERS" \
  --train-frac "$TRAIN_FRAC" \
  "$@" 2>&1 | tee "$OUTDIR/run.log"
