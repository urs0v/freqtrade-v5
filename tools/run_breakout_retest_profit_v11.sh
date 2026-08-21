#!/usr/bin/env bash
set -euo pipefail
V1DIR="${V1DIR:-/freqtrade/user_data/breakout_retest_profit_v1}"
OUTDIR="${OUTDIR:-/freqtrade/user_data/breakout_retest_profit_v11}"
mkdir -p "$OUTDIR"
echo "=== BREAKOUT / RETEST PROFIT V1.1 RUNNER ==="
echo "TRAIN-ONLY failure diagnostics. VALID/HOLDOUT intentionally not used. No downloads."
echo "V1DIR=$V1DIR OUTDIR=$OUTDIR"
PYTHONUNBUFFERED=1 python -u /opt/rmv5/tools/analyze_breakout_retest_profit_v11.py \
  --v1dir "$V1DIR" --outdir "$OUTDIR" "$@" 2>&1 | tee "$OUTDIR/run.log"
