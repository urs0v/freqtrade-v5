#!/usr/bin/env bash
set -euo pipefail
CONFIG="${CONFIG:-/freqtrade/user_data/v7/config-v7-core-backtest.json}"
DATADIR="${DATADIR:-/freqtrade/user_data/data/binance}"
V45DIR="${V45DIR:-/freqtrade/user_data/digash_fidelity_v45}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/digash_fidelity_v46}"
WORKERS="${WORKERS:-16}"
CONTROLS="${CONTROLS:-3}"
EXCLUDE_SOURCE_BPS="${EXCLUDE_SOURCE_BPS:-25}"
TRAIN_FRAC="${TRAIN_FRAC:-0.70}"
mkdir -p "$OUTDIR"
echo "=== DIGASH FIDELITY V4.6 RUNNER ==="
echo "SOURCE FOLLOW-THROUGH vs MATCHED CONTROLS: no PnL fitting, no stops/targets, no downloads."
echo "WORKERS=$WORKERS CONTROLS=$CONTROLS EXCLUDE_SOURCE_BPS=$EXCLUDE_SOURCE_BPS TRAIN_FRAC=$TRAIN_FRAC"
echo "V45DIR=$V45DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/audit_digash_fidelity_v46.py \
  --config "$CONFIG" \
  --datadir "$DATADIR" \
  --v45dir "$V45DIR" \
  --outdir "$OUTDIR" \
  --workers "$WORKERS" \
  --controls "$CONTROLS" \
  --exclude-source-bps "$EXCLUDE_SOURCE_BPS" \
  --train-frac "$TRAIN_FRAC" \
  "$@" 2>&1 | tee "$OUTDIR/run.log"
