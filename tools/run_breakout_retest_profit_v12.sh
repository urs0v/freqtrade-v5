#!/usr/bin/env bash
set -euo pipefail
V1DIR="${V1DIR:-/freqtrade/user_data/breakout_retest_profit_v1}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/breakout_retest_profit_v12}"
mkdir -p "$OUTDIR"
echo "=== BREAKOUT / RETEST PROFIT V1.2 RUNNER ==="
echo "Validates TRAIN-derived structural hypotheses on VALID only. HOLDOUT remains untouched."
echo "V1DIR=$V1DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/breakout_retest_profit_v12.py \
  --v1dir "$V1DIR" --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/run.log"
echo "=== DONE ==="
