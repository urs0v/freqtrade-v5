#!/usr/bin/env bash
set -euo pipefail
V1DIR="${V1DIR:-/freqtrade/user_data/breakout_retest_profit_v1}"
V12DIR="${V12DIR:-/freqtrade/user_data/breakout_retest_profit_v12}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/breakout_retest_profit_v13}"
mkdir -p "$OUTDIR"
echo "=== BREAKOUT / RETEST PROFIT V1.3 RUNNER ==="
echo "Evaluates exactly one frozen VALID-selected hypothesis on previously untouched HOLDOUT."
echo "V1DIR=$V1DIR V12DIR=$V12DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/breakout_retest_profit_v13.py \
  --v1dir "$V1DIR" --v12dir "$V12DIR" --outdir "$OUTDIR" 2>&1 | tee "$OUTDIR/run.log"
echo "=== DONE ==="
